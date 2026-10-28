"""Worker process: imports one moe_ffn package and runs commands for the scorer.

The scorer sends commands over a socket and owns the clock. For the
submission this process runs as an unprivileged user. Outputs leave this
process as per-row digests (squared norm and a random projection) of every
timed call, plus full rows chosen by the scorer after the timed block.
"""

import json
import os
import pathlib
import socket
import struct
import sys
import traceback

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import jax  # pylint: disable=g-import-not-at-top
import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
import numpy as np  # pylint: disable=g-import-not-at-top
import reference  # pylint: disable=g-import-not-at-top

MAX_IN_FLIGHT = 2


def receive(sock):
  def exact(n):
    buf = bytearray()
    while len(buf) < n:
      chunk = sock.recv(n - len(buf))
      if not chunk:
        raise EOFError("scorer disconnected")
      buf.extend(chunk)
    return bytes(buf)
  n = struct.unpack("!I", exact(4))[0]
  return json.loads(exact(n))


def send(sock, obj):
  data = json.dumps(obj, allow_nan=False).encode()
  sock.sendall(struct.pack("!I", len(data)) + data)


def tree_nbytes(tree):
  return sum(int(x.nbytes) for x in jax.tree.leaves(tree) if hasattr(x, "nbytes"))


class State:
  def __init__(self):
    self.api = None
    self.model = None
    self.weights = None
    self.handle = None
    self.xs = []
    self.gss = []
    self.r = None
    self.m = 0
    self.digests = []
    self.last = None
    self.bad_output = None


def check_output(st, out):
  if st.bad_output is None and (
      getattr(out, "shape", None) != (st.m, st.model["H"])
      or getattr(out, "dtype", None) != jnp.bfloat16
      or not isinstance(out, jax.Array)):
    st.bad_output = (f"moe_ffn returned {type(out).__name__} "
                     f"{getattr(out, 'shape', None)} {getattr(out, 'dtype', None)}; "
                     f"expected bf16 jax.Array of shape {(st.m, st.model['H'])}")
  return st.bad_output is None


def handle(st, cmd):
  op = cmd["op"]
  if op == "hello":
    dev = jax.devices()[0]  # Initialise the backend before importing user code.
    sys.path.insert(1, os.getcwd())
    from moe_ffn import api  # pylint: disable=g-import-not-at-top
    for fn in ("prepare", "moe_ffn"):
      if not callable(getattr(api, fn, None)):
        raise AttributeError(f"moe_ffn/api.py has no function {fn}()")
    st.api = api
    return {"device": dev.device_kind, "jax": jax.__version__}
  if op == "model":
    st.model = cmd["model"]
    # Free the previous model before allocating the next one.
    st.handle = st.weights = None
    st.xs, st.gss, st.digests, st.last = [], [], [], None
    st.weights = jax.block_until_ready(reference.make_weights(st.model, cmd["seed"]))
    return {"weight_bytes": tree_nbytes(st.weights)}
  if op == "prepare":
    st.handle = jax.block_until_ready(st.api.prepare(*st.weights))
    st.weights = None  # Only the handle stays alive in this process.
    return {"handle_bytes": tree_nbytes(st.handle)}
  if op == "inputs":
    st.m = cmd["m"]
    st.xs = []  # Free the previous workload's inputs first.
    st.xs = [reference.make_x(st.model, st.m, s) for s in cmd["x_seeds"]]
    gs = np.load(cmd["gs_path"], allow_pickle=False)
    st.gss = [jnp.asarray(g) for g in gs]
    st.r = reference.make_projection(st.model, cmd["r_seed"])
    jax.block_until_ready((st.xs, st.gss, st.r))
    st.digests, st.last, st.bad_output = [], None, None
    return {}
  if op in ("warm", "time"):
    digests = []
    out = None
    for xi, gi in cmd["calls"]:
      out = st.api.moe_ffn(st.handle, st.xs[xi], st.gss[gi])
      if not check_output(st, out):
        break
      digests.append(reference.digest(out, st.r))
      if len(digests) > MAX_IN_FLIGHT:
        # Bound queued calls (and their buffers); the GPU stays busy.
        digests[-1 - MAX_IN_FLIGHT].block_until_ready()
    jax.block_until_ready(digests)
    if op == "time":
      st.digests, st.last = digests, out
    if st.bad_output:
      raise ValueError(st.bad_output)
    return {"n": len(digests)}
  if op == "fetch":
    path = pathlib.Path(cmd["dir"])
    np.save(path / "digests.npy", np.asarray(jnp.stack(st.digests)))
    rows = jnp.asarray(np.load(cmd["rows_path"], allow_pickle=False))
    np.save(path / "rows.npy", np.asarray(st.last[rows].astype(jnp.float32)))
    st.digests, st.last = [], None
    return {}
  if op == "run":
    m = cmd["m"]
    x = reference.make_x(st.model, m, cmd["x_seed"])
    gs = jnp.asarray(np.asarray(cmd["gs"], np.int32))
    st.m = m
    out = st.api.moe_ffn(st.handle, x, gs)
    if not check_output(st, out):
      raise ValueError(st.bad_output)
    np.save(pathlib.Path(cmd["dir"]) / "out.npy", np.asarray(out.astype(jnp.float32)))
    st.bad_output = None
    return {}
  if op == "memory":
    stats = jax.devices()[0].memory_stats() or {}
    return {"peak_bytes": int(stats.get("peak_bytes_in_use", 0))}
  raise ValueError(f"unknown op {op}")


def main():
  sock = socket.socket(fileno=int(sys.argv[1]))
  st = State()
  while True:
    try:
      cmd = receive(sock)
    except EOFError:
      return
    if cmd.get("op") == "stop":
      return
    try:
      resp = handle(st, cmd)
    except Exception as ex:  # pylint: disable=broad-except
      traceback.print_exc()
      resp = {"error": f"{type(ex).__name__}: {str(ex)[:1500]}"}
    send(sock, resp)


if __name__ == "__main__":
  main()
