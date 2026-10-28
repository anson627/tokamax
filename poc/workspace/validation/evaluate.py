"""PoC evaluator: correctness, timing and reward for a moe_ffn submission.

PoC simplifications vs. PROJECT.md section 5: the submission runs in the same
process as the scorer (no UID-2000 isolation) and the starter is loaded from
`validation/starter/`.
"""

import argparse
import importlib.util
import json
import math
import os
import pathlib
import subprocess
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import reference  # pylint: disable=g-import-not-at-top

TOL_GLOBAL = 1e-2
TOL_ROW = 5e-2
S_TARGET = 2.2
PEAK_BF16_FLOPS = 989e12  # H100 SXM dense bf16
PEAK_HBM_BW = 3.35e12  # H100 SXM HBM3 B/s
MAX_PREPARE_RATIO = 1.25
MAX_PREPARE_S = 120.0
NUM_INPUT_SETS = 3


def load_package(pkg_dir: pathlib.Path, alias: str):
  """Imports `<pkg_dir>/api.py` as `<alias>.api` (packages may share a name)."""
  spec = importlib.util.spec_from_file_location(
      alias, pkg_dir / "__init__.py", submodule_search_locations=[str(pkg_dir)]
  )
  mod = importlib.util.module_from_spec(spec)
  sys.modules[alias] = mod
  spec.loader.exec_module(mod)
  return importlib.import_module(f"{alias}.api")


def build(pkg_dir: pathlib.Path, clean: bool):
  if clean:
    subprocess.run(
        ["find", str(pkg_dir), "-name", "__pycache__", "-prune", "-exec",
         "rm", "-rf", "{}", ";"], check=True)
  if (pkg_dir / "build.sh").exists():
    subprocess.run(["bash", "build.sh"], cwd=pkg_dir, check=True)


def tree_nbytes(tree):
  return sum(x.nbytes for x in jax.tree.leaves(tree) if hasattr(x, "nbytes"))


def errors(out, ref):
  """Returns (global rel err, max row rel err, finite)."""
  out = out.astype(jnp.float32)
  diff = out - ref
  finite = bool(jnp.all(jnp.isfinite(out)))
  g = float(jnp.linalg.norm(diff) / jnp.linalg.norm(ref))
  row_ref = jnp.linalg.norm(ref, axis=1)
  floor = 1e-3 * jnp.mean(row_ref)
  row = float(jnp.max(jnp.linalg.norm(diff, axis=1) / jnp.maximum(row_ref, floor)))
  return g, row, finite


def check(out, ref, m, h):
  if out.shape != (m, h) or out.dtype != jnp.bfloat16:
    return dict(ok=False, why=f"bad output {out.shape} {out.dtype}")
  g, row, finite = errors(out, ref)
  ok = finite and g <= TOL_GLOBAL and row <= TOL_ROW
  return dict(ok=ok, global_err=g, row_err=row, finite=finite)


def time_calls(api, handle, inputs, n_iters, n_warmup=3):
  """Mean seconds per call, rotating over input sets. Returns last outputs too."""
  for i in range(n_warmup):
    api.moe_ffn(handle, *inputs[i % len(inputs)]).block_until_ready()
  outs = [None] * len(inputs)
  t0 = time.perf_counter()
  for i in range(n_iters):
    outs[i % len(inputs)] = api.moe_ffn(handle, *inputs[i % len(inputs)])
  jax.block_until_ready(outs)
  return (time.perf_counter() - t0) / n_iters, outs


def peak_mem_gib():
  stats = jax.devices()[0].memory_stats() or {}
  return stats.get("peak_bytes_in_use", 0) / 2**30


def run(args):
  cfg = json.loads((HERE / "workloads.json").read_text())
  workloads = cfg["workloads"]
  if args.mode == "quick":
    workloads = [w for w in workloads if w["name"] in cfg["quick"]]
  if args.only:
    workloads = [w for w in workloads if any(s in w["name"] for s in args.only)]

  sub_dir = pathlib.Path(args.submission).resolve() / "moe_ffn"
  starter_dir = pathlib.Path(args.starter).resolve()
  build(sub_dir, args.clean)
  sub = load_package(sub_dir, "submission_moe_ffn")
  starter = None if args.no_starter else load_package(starter_dir, "starter_moe_ffn")
  for fn in ("prepare", "moe_ffn"):
    if not hasattr(sub, fn):
      raise SystemExit(f"INVALID: submission api has no {fn}()")

  print(f"jax {jax.__version__} on {jax.devices()[0].device_kind}", flush=True)
  results, invalid = [], []
  for model_name, model in cfg["models"].items():
    wls = [w for w in workloads if w["model"] == model_name]
    if not wls and not (args.mode == "full" and not args.only):
      continue
    e, h, i = model["E"], model["H"], model["I"]
    weights = reference.make_weights(model, seed=0)
    jax.block_until_ready(weights)
    w_bytes = tree_nbytes(weights)

    t0 = time.perf_counter()
    handle = jax.block_until_ready(sub.prepare(*weights))
    prep_s = time.perf_counter() - t0
    prep_ratio = tree_nbytes(handle) / w_bytes
    print(f"[{model_name}] weights {w_bytes / 2**30:.2f} GiB, prepare "
          f"{prep_s:.1f}s, size ratio {prep_ratio:.3f}", flush=True)
    if prep_s > MAX_PREPARE_S or prep_ratio > MAX_PREPARE_RATIO:
      invalid.append(f"{model_name}: prepare limits")
    s_handle = starter.prepare(*weights) if starter else None

    if args.mode == "full" and not args.only:
      for case, gs in reference.edge_case_group_sizes(model).items():
        m = int(gs.sum())
        x = reference.make_x(model, m, seed=7)
        gs = jnp.asarray(gs)
        ref = reference.moe_ffn_reference(*weights, x, gs)
        try:
          c = check(sub.moe_ffn(handle, x, gs), ref, m, h)
        except Exception as ex:  # pylint: disable=broad-except
          c = dict(ok=False, why=f"{type(ex).__name__}: {str(ex)[:200]}")
        print(f"  edge {model_name}/{case} M={m}: {fmt_check(c)}", flush=True)
        if not c["ok"]:
          invalid.append(f"edge {model_name}/{case}")

    for w in wls:
      t = w["tokens"]
      m = t * model["top_k"]
      rng = np.random.default_rng(w["seed"])
      inputs, refs = [], []
      for s in range(NUM_INPUT_SETS):
        gs = jnp.asarray(reference.make_group_sizes(model, t, w["routing"], rng))
        x = reference.make_x(model, m, seed=w["seed"] * 10 + s)
        inputs.append((x, gs))
        refs.append(reference.moe_ffn_reference(*weights, x, gs))
      n_iters = 100 if t <= 256 else 20

      # Interleave starter / submission timing: S, X, S, X, S.
      s_times, x_times, checks = [], [], []
      try:
        sub.moe_ffn(handle, *inputs[0]).block_until_ready()
      except Exception as ex:  # pylint: disable=broad-except
        print(f"  {w['name']:32s} FAIL {type(ex).__name__}: {str(ex)[:300]}",
              flush=True)
        invalid.append(w["name"])
        continue
      for rep in range(5):
        if rep % 2 == 0:
          if starter is None:
            continue
          dt, outs = time_calls(starter, s_handle, inputs, n_iters)
          s_times.append(dt)
        else:
          dt, outs = time_calls(sub, handle, inputs, n_iters)
          x_times.append(dt)
          checks += [check(o, r, m, h) for o, r in zip(outs, refs)]
      x_t = max(x_times)  # The slower submission run is used.
      s_t = float(np.median(s_times)) if s_times else float("nan")

      active = int(sum(int((g > 0).sum()) for _, g in inputs) / NUM_INPUT_SETS)
      flops = 6.0 * m * h * i
      min_bytes = active * 3 * h * i * 2 + 2 * m * h * 2
      t_lb = max(flops / PEAK_BF16_FLOPS, min_bytes / PEAK_HBM_BW)
      ok = all(c["ok"] for c in checks)
      r = dict(
          name=w["name"], M=m, active_experts=active, ok=ok,
          global_err=max(c.get("global_err", math.inf) for c in checks),
          row_err=max(c.get("row_err", math.inf) for c in checks),
          starter_us=s_t * 1e6, sub_us=x_t * 1e6, speedup=s_t / x_t,
          tflops=flops / x_t / 1e12, weight_tbps=active * 3 * h * i * 2 / x_t / 1e12,
          roofline_frac=t_lb / x_t,
          x_spread=(max(x_times) - min(x_times)) / min(x_times),
      )
      results.append(r)
      if not ok:
        invalid.append(w["name"])
      print(f"  {r['name']:32s} M={m:6d} act={active:3d} "
            f"{'OK ' if ok else 'BAD'} err={r['global_err']:.2e}/"
            f"{r['row_err']:.2e} starter={r['starter_us']:9.1f}us "
            f"sub={r['sub_us']:9.1f}us x{r['speedup']:.3f} "
            f"{r['tflops']:6.1f}TF/s {r['weight_tbps']:.2f}TB/s "
            f"roof={r['roofline_frac']:.2f}", flush=True)
      del inputs, refs
    del weights, handle, s_handle

  geo = lambda xs: float(np.exp(np.mean(np.log(xs)))) if xs else float("nan")
  s = geo([r["speedup"] for r in results])
  reward = 0.0 if invalid else float(np.clip(math.log(s) / math.log(S_TARGET), 0, 1))
  summary = dict(
      invalid=int(bool(invalid)), invalid_reasons=invalid,
      speedup_geomean=s,
      latency_us_geomean=geo([r["sub_us"] for r in results]),
      roofline_fraction_geomean=geo([r["roofline_frac"] for r in results]),
      max_normalized_error=max(
          [max(r["global_err"] / TOL_GLOBAL, r["row_err"] / TOL_ROW)
           for r in results] or [float("nan")]),
      peak_mem_gib=peak_mem_gib(),
      reward=reward, workloads=results,
  )
  print(json.dumps({k: v for k, v in summary.items() if k != "workloads"}, indent=1))
  if args.out:
    pathlib.Path(args.out).write_text(json.dumps(summary, indent=1))
  return summary


def fmt_check(c):
  if "global_err" not in c:
    return f"FAIL {c.get('why')}"
  return (f"{'OK' if c['ok'] else 'FAIL'} err={c['global_err']:.2e}/"
          f"{c['row_err']:.2e} finite={c['finite']}")


def main():
  p = argparse.ArgumentParser()
  p.add_argument("mode", choices=["quick", "full"])
  p.add_argument("--submission", default=os.environ.get(
      "SUBMISSION", "/workspace/submission"))
  p.add_argument("--starter", default=str(HERE / "starter"))
  p.add_argument("--clean", action="store_true")
  p.add_argument("--no-starter", action="store_true")
  p.add_argument("--only", nargs="*", help="substring filter on workload names")
  p.add_argument("--out")
  run(p.parse_args())


if __name__ == "__main__":
  main()
