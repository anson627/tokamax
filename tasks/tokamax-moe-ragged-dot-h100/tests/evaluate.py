"""Scorer for a moe_ffn submission: build, correctness, timing and reward.

This process is trusted. It never imports submitted code. The starter and the
submission each run in their own worker process (see `harness.py` and
`worker.py`); the submission's worker runs as UID 2000 when this process runs
as root. This process creates the inputs from seeds, computes the fp32
reference, measures time with its own clock and writes the reward.

Timing, per workload: one discarded block per worker, then blocks S X S X S, where
S is the starter and X the submission. Every timed call uses its own routing (`group_sizes`), so outputs
cannot be reused between calls. While one worker is timed, the other is
stopped with SIGSTOP, and every block starts after the same 0.5 s idle gap.
The order is S X S X S X S. Each X block is compared with the mean of the two
S blocks around it; the median of the three ratios is the workload's speedup.

Correctness: every timed submission call returns per-row digests (squared
norm and a random projection) that are compared with the reference. After
each block the last output's rows at random positions, chosen after the block
ran, are compared in full. Edge cases (full mode) are compared in full.
"""

import argparse
import json
import math
import os
import pathlib
import secrets
import shutil
import sys
import tempfile
import time
import traceback

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.30")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness  # pylint: disable=g-import-not-at-top
from harness import InfrastructureError, SubmissionError  # pylint: disable=g-import-not-at-top

TOL_GLOBAL = 1e-2
TOL_ROW = 5e-2
PROJ_SIGMAS = 6.0  # Gaussian projection of the error: |e.r| <= 6 ||e||.
S_TARGET = 2.0
PEAK_BF16_FLOPS = 989e12  # H100 SXM dense bf16
PEAK_HBM_BW = 3.35e12  # H100 SXM HBM3 B/s
MAX_PREPARE_S = 120.0
BUILD_TIMEOUT_S = 900.0
SUB_MEM_FRACTION = 0.40  # ~32 GiB of the 80 GiB GPU for the submission
STARTER_MEM_FRACTION = 0.22
NUM_X_SETS = 2
NUM_X_BLOCKS = 3
CHECK_ROWS = 2048
N_MIN, N_MAX = 5, 400
GAP_S = 0.5
BLOCK_S = {"quick": 0.15, "full": 0.4}
LOGS = pathlib.Path("/logs/verifier")
# The image installs world-readable copies of worker.py and reference.py here,
# so the unprivileged worker can import them without reading the scorer's dir.
WORKER_DIR = pathlib.Path("/opt/moe_eval")
METRICS = ("speedup_geomean", "latency_us_geomean", "roofline_fraction_geomean",
           "max_normalized_error")


def log(msg):
  print(msg, flush=True)


def geomean(xs):
  return float(math.exp(sum(math.log(x) for x in xs) / len(xs))) if xs else float("nan")


def load_npy(path, shape):
  import numpy as np  # pylint: disable=g-import-not-at-top
  try:
    arr = np.load(path, allow_pickle=False)
  except (OSError, ValueError) as ex:
    raise SubmissionError(f"worker output {path.name} is unreadable") from ex
  if arr.shape != shape or arr.dtype != np.float32:
    raise SubmissionError(f"worker output {path.name} has {arr.shape} {arr.dtype}, "
                          f"expected {shape} float32")
  return arr


class Scorer:
  """Holds the workers and runs one evaluation."""

  def __init__(self, args):
    import jax  # pylint: disable=g-import-not-at-top
    import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
    import numpy as np  # pylint: disable=g-import-not-at-top
    import reference  # pylint: disable=g-import-not-at-top
    self.jax, self.jnp, self.np, self.ref = jax, jnp, np, reference
    self.args = args
    self.cfg = json.loads((HERE / "workloads.json").read_text())
    self.block_s = BLOCK_S[args.mode]
    self.workers = []
    self.results = []
    self.edge = []
    self.failures = []
    self.notes = {}

    @jax.jit
    def digest_errors(d_sub, d_ref):
      rn_s = jnp.sqrt(jnp.maximum(d_sub[0], 0.0))
      rn_r = jnp.sqrt(d_ref[0])
      den = jnp.maximum(rn_r, 1e-3 * jnp.mean(rn_r))
      dp = d_sub[1] - d_ref[1]
      return jnp.stack([
          jnp.linalg.norm(dp) / jnp.linalg.norm(d_ref[1]),  # global (estimate)
          jnp.max(jnp.abs(rn_s - rn_r) / den),  # row norm
          jnp.max(jnp.abs(dp) / den) / PROJ_SIGMAS,  # row projection
          jnp.all(jnp.isfinite(d_sub)).astype(jnp.float32),
      ])

    @jax.jit
    def full_errors(out, ref):
      diff = out - ref
      row_ref = jnp.linalg.norm(ref, axis=1)
      den = jnp.maximum(row_ref, 1e-3 * jnp.mean(row_ref))
      return jnp.stack([
          jnp.linalg.norm(diff) / jnp.linalg.norm(ref),
          jnp.max(jnp.linalg.norm(diff, axis=1) / den),
          jnp.all(jnp.isfinite(out)).astype(jnp.float32),
      ])

    self.digest_errors, self.full_errors = digest_errors, full_errors

  # -- setup ----------------------------------------------------------------

  def setup(self):
    dev = self.jax.devices()[0]
    if dev.platform != "gpu" or "H100" not in dev.device_kind:
      harness.fail_unsupported(f"device {dev.device_kind} is not an H100", LOGS)
    log(f"jax {self.jax.__version__} on {dev.device_kind}; "
        f"isolation {'on (UID %d)' % harness.UID if harness.isolated() else 'off (not root)'}")
    self.work = pathlib.Path(tempfile.mkdtemp(prefix="moe_eval_"))
    self.work.chmod(0o711)
    self.shared = self.work / "shared"
    self.shared.mkdir()
    self.shared.chmod(0o755)
    sub_root = self.work / "submission"
    homes = {n: self.work / f"home_{n}" for n in ("starter", "submission")}
    self.xchg = self.work / "xchg"
    for d in (*homes.values(), self.xchg):
      d.mkdir()
    n_files = harness.stage_submission(pathlib.Path(self.args.submission), sub_root)
    for d in (sub_root, homes["submission"], self.xchg):
      harness.own_tree(d)
    log(f"staged {n_files} source files from {self.args.submission}")
    build_log = self.work / "build.log"
    with open(build_log, "w") as f:
      t0 = time.perf_counter()
      try:
        harness.run_build(sub_root / "moe_ffn", homes["submission"], f, BUILD_TIMEOUT_S)
      except SubmissionError:
        log(build_log.read_text()[-4000:])
        raise
    if (sub_root / "moe_ffn" / "build.sh").exists():
      log(f"build.sh finished in {time.perf_counter() - t0:.0f} s")
    self.logs = {n: open(self.work / f"worker_{n}.log", "w") for n in homes}
    worker_py = (WORKER_DIR if (WORKER_DIR / "worker.py").exists() else HERE) / "worker.py"
    self.starter = harness.Worker(
        "starter", HERE / "starter", worker_py, homes["starter"],
        STARTER_MEM_FRACTION, self.logs["starter"], untrusted=False, deadline_s=600)
    self.workers.append(self.starter)
    self.sub = harness.Worker(
        "submission", sub_root, worker_py, homes["submission"],
        SUB_MEM_FRACTION, self.logs["submission"], untrusted=True, deadline_s=600)
    self.workers.append(self.sub)

  def close(self):
    for w in self.workers:
      w.close()
    for f in getattr(self, "logs", {}).values():
      f.close()

  def worker_log_tail(self):
    path = self.work / "worker_submission.log"
    return path.read_text()[-4000:] if path.exists() else ""

  # -- evaluation -----------------------------------------------------------

  def run(self):
    jnp, np, ref = self.jnp, self.np, self.ref
    workloads = self.cfg["workloads"]
    if self.args.mode == "quick":
      workloads = [w for w in workloads if w["name"] in self.cfg["quick"]]
    if self.args.only:
      workloads = [w for w in workloads if any(s in w["name"] for s in self.args.only)]
    if not workloads:
      raise SystemExit("no workloads selected")
    do_edges = self.args.mode == "full" and not self.args.only
    for model_name, model in self.cfg["models"].items():
      wls = [w for w in workloads if w["model"] == model_name]
      if not wls:
        continue
      weights = self.jax.block_until_ready(ref.make_weights(model, model["seed"]))
      w_bytes = sum(int(a.nbytes) for a in weights)
      for wk in (self.starter, self.sub):
        wk.call({"op": "model", "model": model, "seed": model["seed"]}, 600)
      resp, prep_s = self.sub.timed_call({"op": "prepare"}, MAX_PREPARE_S + 60)
      self.starter.call({"op": "prepare"}, 600)
      ratio = resp["handle_bytes"] / w_bytes
      log(f"[{model_name}] E={model['E']} H={model['H']} I={model['I']}, weights "
          f"{w_bytes / 2**30:.2f} GiB; prepare {prep_s:.1f} s, handle {ratio:.2f}x weights")
      if prep_s > MAX_PREPARE_S:
        raise SubmissionError(f"prepare took {prep_s:.0f} s (limit {MAX_PREPARE_S:.0f} s)")
      if do_edges:
        self.edge_cases(model_name, model, weights)
      for w in wls:
        self.workload(model, weights, w)
      del weights
    mem = self.sub.call({"op": "memory"}, 60)["peak_bytes"] / 2**30
    self.notes["submission_peak_mem_gib"] = mem

  def edge_cases(self, model_name, model, weights):
    jnp, np, ref = self.jnp, self.np, self.ref
    for case, gs in ref.edge_case_group_sizes(model).items():
      m = int(gs.sum())
      self.sub.call({"op": "run", "m": m, "x_seed": 7, "gs": gs.tolist(),
                     "dir": str(self.xchg)}, 600)
      out = load_npy(self.xchg / "out.npy", (m, model["H"]))
      expect = ref.moe_ffn_reference(*weights, ref.make_x(model, m, 7), jnp.asarray(gs))
      g, row, finite = (float(v) for v in self.full_errors(jnp.asarray(out), expect))
      ok = bool(finite) and g <= TOL_GLOBAL and row <= TOL_ROW
      self.edge.append(dict(name=f"{model_name}/{case}", M=m, ok=ok, global_err=g, row_err=row))
      log(f"  edge {model_name}/{case:28s} M={m:5d} {'OK  ' if ok else 'FAIL'} "
          f"err={g:.2e}/{row:.2e}{'' if finite else ' non-finite'}")
      if not ok:
        self.failures.append(f"edge case {model_name}/{case}")

  def workload(self, model, weights, w):
    jnp, np, ref = self.jnp, self.np, self.ref
    e, h, i = model["E"], model["H"], model["I"]
    m = w["M"]
    rng = np.random.default_rng(w["seed"])
    flops = 6.0 * m * h * i
    est = max(flops / 250e12, 3 * e * h * i * 2 / 2.5e12) + 50e-6  # starter-like
    n = int(min(N_MAX, max(N_MIN, math.ceil(self.block_s / est))))
    gs_all = np.stack([ref.make_group_sizes(model, m, w["routing"], rng)
                       for _ in range(NUM_X_BLOCKS * n + 2)])
    gs_path = self.shared / "gs.npy"
    np.save(gs_path, gs_all)
    gs_path.chmod(0o644)
    x_seeds = [w["seed"] * 10 + s for s in range(NUM_X_SETS)]
    r_seed = w["seed"] * 10 + 9
    inputs = {"op": "inputs", "m": m, "x_seeds": x_seeds, "gs_path": str(gs_path),
              "r_seed": r_seed}
    for wk in (self.starter, self.sub):
      wk.call(inputs, 600)
      wk.call({"op": "warm", "calls": [[0, NUM_X_BLOCKS * n],
                                         [1 % NUM_X_SETS, NUM_X_BLOCKS * n + 1]]}, 900)
    calls = [[[c % NUM_X_SETS, b * n + c] for c in range(n)] for b in range(NUM_X_BLOCKS)]
    deadline = 120 + 100 * est * n

    def block(worker, other, cl):
      # The same idle gap before every block: back-to-back blocks otherwise
      # run the second one hotter (measured 2-6% on weight-bound shapes).
      time.sleep(GAP_S)
      other.pause()
      try:
        _, dt = worker.timed_call({"op": "time", "calls": cl}, deadline)
      finally:
        other.resume()
      return dt / len(cl)

    # Discarded blocks first: GPU clocks ramp up after idling, and each
    # worker's allocator grows to this workload's steady-state size.
    block(self.starter, self.sub, calls[0])
    block(self.sub, self.starter, calls[0])
    s_t, x_t, fetched = [], [], []
    for rep in range(2 * NUM_X_BLOCKS + 1):
      if rep % 2 == 0:
        s_t.append(block(self.starter, self.sub, calls[min(rep // 2, NUM_X_BLOCKS - 1)]))
        continue
      x_t.append(block(self.sub, self.starter, calls[rep // 2]))
      rows = np.sort(np.random.default_rng(secrets.randbits(64)).choice(
          m, min(m, CHECK_ROWS), replace=False)).astype(np.int32)
      rows_path = self.shared / "rows.npy"
      np.save(rows_path, rows)
      rows_path.chmod(0o644)
      self.sub.call({"op": "fetch", "dir": str(self.xchg), "rows_path": str(rows_path)}, 600)
      fetched.append((load_npy(self.xchg / "digests.npy", (n, 2, m)),
                      rows, load_npy(self.xchg / "rows.npy", (len(rows), h))))

    # Correctness of every timed submission call.
    xs = [ref.make_x(model, m, s) for s in x_seeds]
    r = ref.make_projection(model, r_seed)
    errs = []
    for (digs, rows, out_rows), cl in zip(fetched, calls):
      d_sub = jnp.asarray(digs)
      for j, (xi, gi) in enumerate(cl):
        expect = ref.moe_ffn_reference(*weights, xs[xi], jnp.asarray(gs_all[gi]))
        de = self.digest_errors(d_sub[j], ref.digest(expect, r))
        if j == len(cl) - 1:
          fe = self.full_errors(jnp.asarray(out_rows), expect[jnp.asarray(rows)])
          errs.append([float(fe[0]), float(fe[1]), float(fe[2])])
        errs.append([float(de[0]), max(float(de[1]), float(de[2])), float(de[3])])
    errs = np.asarray(errs)
    g_err, row_err, finite = float(errs[:, 0].max()), float(errs[:, 1].max()), bool(errs[:, 2].min())
    ok = finite and g_err <= TOL_GLOBAL and row_err <= TOL_ROW

    ratios = [(s_t[b] + s_t[b + 1]) / 2 / x_t[b] for b in range(NUM_X_BLOCKS)]
    speedup = float(np.median(ratios))
    x_us = float(np.median(x_t)) * 1e6
    active = float(np.mean((gs_all[: NUM_X_BLOCKS * n] > 0).sum(axis=1)))
    min_bytes = active * 3 * h * i * 2 + 2 * m * h * 2
    t_lb = max(flops / PEAK_BF16_FLOPS, min_bytes / PEAK_HBM_BW)
    res = dict(
        name=w["name"], M=m, routing=w["routing"], active_experts=active, ok=ok,
        global_err=g_err, row_err=row_err, finite=finite, calls_per_block=n,
        starter_us=float(np.median(s_t)) * 1e6, sub_us=x_us, speedup=speedup,
        block_ratios=ratios, blocks_us=dict(starter=[t * 1e6 for t in s_t],
                                            submission=[t * 1e6 for t in x_t]),
        tflops=flops / (x_us * 1e-6) / 1e12,
        roofline_frac=t_lb / (x_us * 1e-6))
    self.results.append(res)
    if not ok:
      self.failures.append(f"workload {w['name']}")
    log(f"  {w['name']:30s} M={m:6d} {'OK  ' if ok else 'FAIL'} err={g_err:.2e}/{row_err:.2e} "
        f"starter={res['starter_us']:8.0f}us sub={x_us:8.0f}us x{speedup:.3f} "
        f"({'/'.join(f'{r:.3f}' for r in ratios)}) {res['tflops']:5.0f}TF/s roof={res['roofline_frac']:.2f}")

  def summary(self, error=None):
    res = self.results
    invalid = bool(error or self.failures)
    s = geomean([r["speedup"] for r in res]) if res else 0.0
    errs = [max(r["global_err"] / TOL_GLOBAL, r["row_err"] / TOL_ROW)
            for r in res + self.edge]
    reward = 0.0 if invalid or not res else float(
        min(1.0, max(0.0, math.log(s) / math.log(S_TARGET))))
    out = dict(
        reward=reward, invalid=int(invalid),
        speedup_geomean=s if res else 0.0,
        latency_us_geomean=geomean([r["sub_us"] for r in res]) if res else 0.0,
        roofline_fraction_geomean=geomean([r["roofline_frac"] for r in res]) if res else 0.0,
        max_normalized_error=max(errs) if errs and all(math.isfinite(v) for v in errs) else 0.0,
    )
    detail = dict(out, mode=self.args.mode, split=self.cfg["split"], s_target=S_TARGET,
                  invalid_reasons=([error] if error else []) + self.failures,
                  notes=self.notes, edge_cases=self.edge, workloads=res)
    return out, detail


def write_json(path, obj):
  tmp = path.with_suffix(".tmp")
  tmp.write_text(json.dumps(obj, indent=1, allow_nan=False))
  tmp.replace(path)


def main():
  p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  p.add_argument("mode", nargs="?", default="full", choices=["quick", "full"])
  p.add_argument("--submission", default="/workspace/submission")
  p.add_argument("--only", nargs="*", help="substring filter on workload names")
  p.add_argument("--out", help="also write the detailed result JSON here")
  p.add_argument("--clean", action="store_true",
                 help="accepted for compatibility; every run builds from a clean copy")
  p.add_argument("--keep-workdir", action="store_true")
  args = p.parse_args()
  split = json.loads((HERE / "workloads.json").read_text())["split"]
  if split == "test" and (len(sys.argv) > 2 or args.mode != "full"):
    raise SystemExit("the test evaluator takes no options")
  final = args.mode == "full" and not args.only
  if final:  # Clear any stale reward before evaluating.
    try:
      LOGS.mkdir(parents=True, exist_ok=True)
      for name in ("reward.json", "result.json", "environment_unsupported.json"):
        (LOGS / name).unlink(missing_ok=True)
    except OSError as ex:
      log(f"note: cannot use {LOGS} ({ex}); the reward is printed only")
      final = False

  scorer = None
  error = None
  t0 = time.perf_counter()
  try:
    scorer = Scorer(args)
    scorer.setup()
    scorer.run()
  except SubmissionError as ex:
    error = str(ex)
    log(f"INVALID SUBMISSION: {error}")
    if scorer is not None and hasattr(scorer, "work"):
      log(scorer.worker_log_tail())
  except SystemExit:
    raise
  except Exception:  # pylint: disable=broad-except
    traceback.print_exc()
    log("EVALUATOR ERROR: trusted evaluation code or the machine failed; no reward was written.")
    sys.exit(2)
  finally:
    if scorer is not None:
      scorer.close()
      if hasattr(scorer, "work") and not args.keep_workdir:
        shutil.rmtree(scorer.work, ignore_errors=True)
  out, detail = scorer.summary(error)
  detail["elapsed_s"] = time.perf_counter() - t0
  log(json.dumps({k: v for k, v in detail.items() if k not in ("workloads", "edge_cases")},
                 indent=1))
  if args.out:
    write_json(pathlib.Path(args.out), detail)
  if final:
    write_json(LOGS / "result.json", detail)
    write_json(LOGS / "reward.json", out)
  elif args.mode == "quick" or args.only:
    log("note: partial run (quick or --only); its reward is not the full-run reward.")


if __name__ == "__main__":
  main()
