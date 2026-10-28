"""Dev bench: ws kernel correctness and speed vs the starter."""
import sys, time, itertools, json, os
sys.path.insert(0, "/dev_ws")
sys.path.insert(0, "/workspace/validation")
import jax, jax.numpy as jnp, numpy as np
import reference
from moe_ffn import kernel_ws, kernel_sm90, common

STARTER = common.Config(block_m=64, block_n=64, block_k=128, num_stages=2,
    persistent=False, grid_minor_dim=common.MatmulDimension.M, grid_tile_width=1)

def timeit(f, *a, n=None):
  f(*a).block_until_ready()
  t0 = time.perf_counter(); f(*a).block_until_ready(); one = time.perf_counter() - t0
  n = n or max(3, min(200, int(0.3 / max(one, 1e-5))))
  t0 = time.perf_counter()
  for _ in range(n): o = f(*a)
  o.block_until_ready()
  return (time.perf_counter() - t0) / n

def starter_ffn(wgu, wd, x, gs):
  i = wd.shape[1]
  h = kernel_sm90.ragged_dot_kernel(x, wgu, gs, jnp.bfloat16, STARTER).astype(jnp.float32)
  a = (jax.nn.silu(h[:, :i]) * h[:, i:]).astype(jnp.bfloat16)
  return kernel_sm90.ragged_dot_kernel(a, wd, gs, jnp.bfloat16, STARTER)

def ws_ffn(wgu, wd, x, gs, c1, c2):
  a = kernel_ws.ragged_dot_ws(x, wgu, gs, c1, swiglu=True)
  return kernel_ws.ragged_dot_ws(a, wd, gs, c2)

models = {"qwen": dict(E=128, H=2048, I=768), "ds": dict(E=64, H=7168, I=2048)}
which = sys.argv[1] if len(sys.argv) > 1 else "qwen"
Ms = [int(v) for v in sys.argv[2].split(",")] if len(sys.argv) > 2 else [16384]
sweep = len(sys.argv) > 3 and sys.argv[3] == "sweep"
mdl = models[which]
w = reference.make_weights(mdl, 0)
for M in Ms:
  rng = np.random.default_rng(M)
  for routing in ["balanced", "skewed"]:
    e = mdl["E"]
    p = np.ones(e) if routing == "balanced" else (1 + rng.permutation(e)) ** -1.2
    gs = jnp.asarray(rng.multinomial(M, p / p.sum()).astype(np.int32))
    x = reference.make_x(mdl, M, 1)
    ref = reference.moe_ffn_reference(*w, x, gs)
    f_s = jax.jit(starter_ffn)
    ts = timeit(f_s, *w, x, gs)
    flops = 6 * M * mdl["H"] * mdl["I"]
    print(f"{which} M={M} {routing}: starter {ts*1e6:.0f}us {flops/ts/1e12:.0f}TF/s", flush=True)
    c1s = [kernel_ws.WsConfig(128, 64, 64, 4), kernel_ws.WsConfig(128, 64, 64, 3), kernel_ws.WsConfig(64, 128, 64, 3), kernel_ws.WsConfig(128, 128, 64, 2)] if not sweep else [
        kernel_ws.WsConfig(tm, tn, 64, st, gmd, gw) for tm, tn, st, gmd, gw in itertools.product(
            (64, 128), (64, 128), (3, 4, 5), (common.MatmulDimension.M, common.MatmulDimension.N), (1, 4, 8))]
    c2s = [kernel_ws.WsConfig(128, 128, 64, 3), kernel_ws.WsConfig(128, 64, 64, 4), kernel_ws.WsConfig(128, 64, 64, 5)] if not sweep else c1s
    best = None
    for c1, c2 in ([(c, c2s[0]) for c in c1s] + [(c1s[0], c) for c in c2s[1:]]):
      try:
        f = jax.jit(lambda a, b, c, d: ws_ffn(a, b, c, d, c1, c2))
        out = f(*w, x, gs)
        g_err = float(jnp.linalg.norm(out.astype(jnp.float32) - ref) / jnp.linalg.norm(ref))
        t = timeit(f, *w, x, gs)
        print(f"  c1={c1.tile_m}x{c1.tile_n}x{c1.tile_k}s{c1.stages}{c1.grid_minor_dim.name}{c1.grid_tile_width} "
              f"c2={c2.tile_m}x{c2.tile_n}x{c2.tile_k}s{c2.stages}{c2.grid_minor_dim.name}{c2.grid_tile_width} "
              f"{t*1e6:.0f}us x{ts/t:.2f} {flops/t/1e12:.0f}TF/s err={g_err:.1e}", flush=True)
        if best is None or t < best[0]: best = (t, c1, c2)
      except Exception as ex:
        print(f"  {c1} {c2} FAIL {type(ex).__name__}: {str(ex)[:400]}", flush=True)
    if best: print(f"  BEST x{ts/best[0]:.2f}", best[1], best[2], flush=True)
