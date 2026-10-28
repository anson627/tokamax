# Task Proposal: `tokamax-moe-ragged-dot-h100`

> **Status (2026-10-28):** fully implemented in
> `tasks/tokamax-moe-ragged-dot-h100/`, following the Anthology task guide.
> Sections 2–7 are the original proposal. Where they differ from the
> implementation, section 9 and the task's `instruction.md` are current
> (workload mix, DeepSeek-V3 shape, `S_TARGET = 2.0`).

## Source

The starter code is the Tokamax Hopper (sm90) ragged-dot kernel, written in
JAX with Pallas Mosaic GPU. It is pinned to commit
`4bf85ff592e66967efaf9c61c45fe52d0ee8f2f2` of `openxla/tokamax` (Apache-2.0).

| What | Link |
|---|---|
| Kernel | [`pallas_mosaic_gpu_kernel_sm90.py#L35-L171`](https://github.com/openxla/tokamax/blob/4bf85ff592e66967efaf9c61c45fe52d0ee8f2f2/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm90.py#L35-L171) |
| Group tiling and output copy | [`pallas_mosaic_gpu_common.py#L49-L229`](https://github.com/openxla/tokamax/blob/4bf85ff592e66967efaf9c61c45fe52d0ee8f2f2/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_common.py#L49-L229) |
| Shared-memory helpers | [`mosaic_gpu.py#L56-L78`](https://github.com/openxla/tokamax/blob/4bf85ff592e66967efaf9c61c45fe52d0ee8f2f2/tokamax/_src/mosaic_gpu.py#L56-L78) |
| Default H100 config | [`pallas_mosaic_gpu.py#L316-L326`](https://github.com/openxla/tokamax/blob/4bf85ff592e66967efaf9c61c45fe52d0ee8f2f2/tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu.py#L316-L326) |

Changes from the original (the kernel math is unchanged):

1. Copied into one small package with local imports. The `jaxtyping` and
   `pydantic` dependencies are dropped (`Config` is a plain dataclass).
2. Autotuning and the non-H100 code paths are removed. The config is fixed to
   Tokamax's default H100 config (64×64×128 tiles, 2 stages, non-persistent,
   M-minor grid).
3. Two kernel calls with a SwiGLU between them form one MoE expert layer.
4. `api.moe_ffn` pads `M` up to a multiple of 8 and slices the output. The
   kernel's TMA copies fail to lower otherwise (`dimension 0 must have size
   smaller or a multiple of its tiling 8`). Found by the PoC edge cases.
5. The `tokamax` package is not installed. Its H100 tuning cache already has
   tuned configs for DeepSeek-V3 shapes (128×64×128, 4 stages, persistent).
   The PoC measured that config: it is 1.12–1.35× faster on prefill and
   0.76–0.89× on decode (section 8).

Model shapes come from the public configs of Qwen3-30B-A3B and DeepSeek-V3.
Weights are random bf16, generated on the GPU from fixed seeds. Speed does not
depend on weight values, so no checkpoint download is needed.

---

## 1. Task Name

`tokamax-moe-ragged-dot-h100`

## 2. Agent Instructions

#### Background

In a mixture-of-experts (MoE) model, each token is sent to a few experts.
After routing, tokens are sorted by expert, so each expert multiplies one
block of rows. The number of rows per expert varies. This operation is called
a ragged dot (or grouped GEMM).

One MoE expert layer is:

    h   = ragged_dot(x, w_gate_up, group_sizes)   # [M, 2I]
    a   = silu(h[:, :I]) * h[:, I:]               # SwiGLU
    out = ragged_dot(a, w_down, group_sizes)      # [M, H]

| Input | Shape | Type | Notes |
|---|---|---|---|
| `x` | `[M, H]` | bf16 | Rows sorted by expert |
| `w_gate_up` | `[E, H, 2I]` | bf16 | Gate columns first, then up columns |
| `w_down` | `[E, I, H]` | bf16 | |
| `group_sizes` | `[E]` | int32 | Rows per expert; sums to `M`; may contain zeros |

The starter in `/workspace/moe_ffn/` uses the Tokamax H100 kernel with its
default config. It is correct but slow on large batches, where it reaches only
~220–260 TFLOP/s. On small batches it is already close to the weight-read
limit.

Workloads use two model shapes. Each token goes to 8 experts, so `M = 8 × tokens`.

| Model shape | E | H | I | Weight size |
|---|---:|---:|---:|---:|
| Qwen3-30B-A3B | 128 | 2048 | 768 | 1.1 GiB |
| DeepSeek-V3 | 256 | 7168 | 2048 | 21 GiB |

Token counts range from 32 (decode) to 8192 (prefill). Routing is either
balanced or skewed. In skewed routing, a few experts get many rows and some get none.

#### Task

Make the MoE expert layer as fast as possible on one H100. Keep these two
functions in `/workspace/moe_ffn/api.py`:

- `prepare(w_gate_up, w_down) -> handle`: runs once per model shape. Not timed.
  It may change the weight layout. It may use only the weights.
- `moe_ffn(handle, x, group_sizes) -> out`: timed. Returns `[M, H]` bf16. It is
  called many times with different `group_sizes` for the same `M`.

You may change anything behind these functions. For example, you can tune or
rewrite the Pallas kernel, write new Pallas kernels, or write CUDA kernels
called through `jax.ffi`.

Put your submission in `/workspace/submission/`:

- `moe_ffn/`: your package, with `api.py` and all its source files.
- `moe_ffn/build.sh` (optional): build step, run inside `moe_ffn/`.
- `summary.md` with two sections:
  - `## Experiments`: what you tried and what you learned.
  - `## Submitted solution`: your final design, results, and how to reproduce them.

Check your work with `bash /workspace/validation/val.sh <mode>`. Add
`--submission <dir>` to test a different folder.

| Mode | What it does |
|---|---|
| `quick` | Build, correctness, and speed on 2 small workloads |
| `full` | Correctness and speed on all 16 workloads, plus the reward |
| `full --clean` | `full` with a clean rebuild. Run this on your final submission. |

The workloads are listed in `/workspace/validation/workloads.json`.

#### Constraints

- Inputs and outputs are bf16. Do not quantize weights or activations to fp8,
  int8, or lower. Accumulate in fp32 or higher.
- Every row must use its own expert's weights. Do not skip or approximate rows.
- `prepare` output must be at most 1.25× the size of the bf16 weights and must
  finish in 120 s per model shape.
- Do not cache results between `moe_ffn` calls.
- Peak GPU memory must stay below 72 GiB.
- Do not change `/workspace/validation/`. Work on copies elsewhere.
- The submission must build offline from source. Prebuilt binaries are not used.

#### Success Criteria

**Correctness (required).** Each output is compared with an fp32 reference.
For every workload:

- total relative error ≤ [TBD: TOL_GLOBAL, 1e-2]
- relative error of each row ≤ [TBD: TOL_ROW, 5e-2]
- no NaN or Inf

(PoC: the starter measures 3.9e-3 total and ≤ 8.1e-3 per row on every
workload, so it has 2.6× and 6× margin. Not yet measured: whether an fp8
variant fails.)

If any workload fails, the reward is 0.

**Speed.** For each workload, speedup = starter time ÷ your time. The score is
the geometric mean of the speedups. A 1× speedup gives reward 0, and
[TBD: S_TARGET, 2.2×] gives reward 1.

The final test uses the same two model shapes and routing types. It uses
different token counts and different random routings.

## 3. Difficulty & Specification

The starter config leaves a lot of H100 performance unused. Getting it back
takes several separate changes:

| Workload | Problem in the starter | What helps |
|---|---|---|
| Large batches (compute-bound) | Small 64×64 tiles, 2 pipeline stages, one tile per thread block | Larger tiles, deeper pipelines, warp specialization, persistent kernel |
| Small batches (memory-bound) | Little. The PoC measured 2.7–3.0 TB/s, 80–90% of the HBM roofline. Wasted MMA work does not matter while the kernel waits on weight reads | At most ~1.1× from removing the SwiGLU pass and from tail effects |
| Mid batches (2048 tokens) | 1.6–2.3 TB/s: tiles are reloaded per 64-row block of each expert | Larger block_m, so each expert's weights are read once |
| Skewed routing | Fixed work split leaves some SMs idle | Dynamic or load-aware scheduling |
| Whole layer | SwiGLU is a separate pass over `[M, 2I]` | Interleave gate and up columns in `prepare` and fuse SwiGLU into the kernel |

Why this is hard for models:

- One kernel does not fit all cases. Large and small batches need different
  designs, so the agent must profile each case and dispatch by shape.
- A change that helps balanced routing can hurt skewed routing. The score
  counts both.
- Pallas Mosaic GPU is new, with few examples online. Writing CUDA through
  `jax.ffi` is possible but adds build work.

Headroom measured by the PoC (section 8). Each workload's ceiling is the
starter time divided by max(FLOPs / 650 TFLOP/s, active-weight bytes / 3.0 TB/s):

| Tokens | Qwen3 ceiling | DeepSeek-V3 ceiling |
|---:|---:|---:|
| 32 | 1.08–1.12× | 0.99–1.02× |
| 256 | 1.08–1.10× | 1.00–1.02× |
| 2048 | 1.77–1.86× | 1.31–1.46× |
| 8192 | 2.49–2.51× | 2.62–2.85× |

- DeepSeek-V3 large batches: the starter reaches 221 TFLOP/s, about a third of
  ~650 TFLOP/s. This matches the original estimate.
- Qwen3 small batches: the starter already takes 0.25–0.44 ms, near the
  ~0.23–0.40 ms floor set by reading the active experts' weights. The original
  estimate assumed the starter was far from this floor. It is not.

**The geometric-mean ceiling for the current workload mix is ~1.46×,** and
that assumes a perfect kernel on every workload. A 2–2.5× target is not
reachable. Half of the workloads (tokens ≤ 256) have ≤ 1.12× headroom. See
section 8 for options.

Existing fast paths, measured by the PoC on the 16 validation workloads:

| Fast path | Speedup geomean | Notes |
|---|---:|---|
| `jax.lax.ragged_dot` | 0.11 (12 of 16 ran) | XLA:GPU makes a dense `[E, M, K]` masked copy. Out of memory (64 GiB allocation) for DeepSeek-V3 at ≥ 2048 tokens |
| `pallas.ops.gpu.ragged_dot_mgpu` (untuned, 128×128×64, 4 stages) | 0.66 | Fails when `M` is not a multiple of 8 |
| Starter kernel with Tokamax's tuned H100 config | 0.97 | 1.12–1.35× at 8192 tokens, 0.76–0.89× at ≤ 256 tokens |
| Best of starter or tuned config, chosen per shape | ~1.06 | Computed from the two runs above |

`S_TARGET` must sit above ~1.06× (the best dispatch of existing paths) and
below the ~1.46× ceiling.

## 4. Environment

| Item | Value |
|---|---|
| GPU | 1× H100 SXM5 80GB (`H100!`) |
| CPU / RAM / disk | 8 CPUs, 32 GiB RAM, 40 GiB disk (agent and verifier) |
| OS and tools | Ubuntu 24.04, CUDA 12.9 (`nvcc`, Nsight Systems), Python 3.12 |
| Python packages | `jax[cuda12]==0.11.1`, `jaxlib==0.11.1`, `numpy`, `einshape` (pinned) |
| Not installed | `tokamax`, CUTLASS, other kernel libraries |
| Network | Not available during the task |

GPU memory: the weights are 1.1 GiB (Qwen3) and 21 GiB (DeepSeek-V3), plus up
to ~3 GiB for inputs. The PoC evaluator peaked at 36.8 GiB, including the fp32
references for 3 input sets.

PoC check: `jax[cuda12]==0.11.1` from pip on
`nvcr.io/nvidia/cuda:12.9.1-devel-ubuntu24.04` (Python 3.12) runs the
Mosaic GPU kernels on an H100 80GB HBM3 with driver 595.91.07.

Workspace:

- `/workspace/moe_ffn/`: starter package (`api.py`, `kernel_sm90.py`, `common.py`, `mgpu_utils.py`)
- `/workspace/validation/`: `val.sh`, evaluator, reference, a frozen copy of
  the starter, `workloads.json`, and a README
- `/workspace/submission/`: empty, for the final submission

Validation inputs (`val.sh`): 16 workloads, made from 2 model shapes × tokens
{32, 256, 2048, 8192} × {balanced, skewed} routing. It reports correctness,
time, TFLOP/s, weight bandwidth, speedup, and reward. Measured in the PoC:
`quick` takes ~15 s and `full` takes ~1.5–2.5 min, including 10 edge cases
and Mosaic GPU compiles. Slow submissions take longer: the XLA probe took ~4 min.

## 5. Verifiers

`test.sh` builds and runs the submission as a non-root user (UID 2000) in a
separate process. A trusted root process creates the inputs, computes the
reference, measures time, and writes the reward.

**Required checks (any failure: reward 0, `invalid = 1`):**

- The package builds and imports, and both functions exist.
- `prepare` finishes in 120 s, and its output is at most 1.25× the weight size.
- Outputs have shape `[M, H]` and type bf16, with no NaN or Inf.
- Error is within tolerance on every workload. The reference uses fp32
  accumulation. It cannot be `jax.lax.ragged_dot`, which runs out of memory
  on DeepSeek-V3 prefill (section 8). Instead it loops over row tiles, each
  within one expert, and runs plain XLA bf16 × bf16 → fp32 dots. The SwiGLU
  output stays in fp32: it is split into bf16 hi + lo parts for the second
  dot. The tolerance passes normal bf16 rounding differences. It should fail
  fp8 (~2–4% error, not yet measured) and wrong rows (~100% error).
- Edge cases, checked for correctness but not timed: experts with 0 rows, one
  expert with all rows, group edges one row off a tile edge, and `M` not a
  multiple of 8.
- Peak GPU memory is below 72 GiB.

**Speed measurement:**

- Hidden workloads: 2 model shapes × tokens {64, 512, 4096, 6144}
  × {balanced, skewed}, with new seeds. None of these overlap with the
  validation inputs. Time fits easily: a 16-workload `full` run takes ~2 min
  in the PoC. Headroom does not: 64 and 512 tokens are weight-bound like 32
  and 256, so this set has the same low ceiling (section 3).
- Each workload: warm up, then time many calls (100 small, 20 large). The calls
  rotate over 3 different input sets, so cached results do not help.
- The trusted process does the timing. Outputs from timed calls are checked
  against the reference.
- The starter and the submission are timed in turns in the same run
  (starter, submission, starter, submission, starter). The slower submission
  run is used.

**Reported metrics:** `speedup_geomean`, `latency_us_geomean`,
`roofline_fraction_geomean`, `max_normalized_error`.

**Protection against reward hacking:**

| Hack | Defense |
|---|---|
| Fitting to the test inputs | Test inputs are not in the agent image |
| Changing the timer or the score | Submission runs as a non-root user, separate from the scorer |
| Caching outputs | Inputs rotate, and outputs of timed calls are checked |
| Lower precision | Error tolerance |
| Precomputing in `prepare` | `prepare` sees only weights; output size is capped |
| Stale baseline | Starter is re-timed in the same run |

If the trusted code fails (not the submission), `test.sh` exits with an error
and writes no reward.

## 6. Reward

S = geometric mean of the speedups (starter time ÷ submission time) over the
16 hidden workloads. All workloads have equal weight. The geometric mean stops
one case from dominating the score.

For a valid submission:

    R = clip( ln(S) / ln(S_target), 0, 1 )

Speedups multiply, so we use a log scale: going from 1× to 1.48× earns the
same reward as going from 1.48× to 2.2×. The starter gets S = 1 and reward 0.
A slower submission also gets 0.

`S_target` = [TBD: S_TARGET, 2.2×]. It is set using the steps below. The PoC
shows that 2.2× is above the ~1.46× ceiling, so either the target or the
workload mix must change (section 8).

1. the speedup of our reference solution
2. the best existing fast path in the image (the target must be clearly above it)
3. agent trial results, aiming for mean reward ≤ 0.7 and variance > 0.02

## 7. Baseline

`baseline.sh` copies the unchanged starter into `/workspace/submission/moe_ffn/`
and adds a `summary.md`. The starter is the Tokamax kernel with its default
config, called twice with an XLA SwiGLU between, and a `prepare` that returns
the weights unchanged.

It passes all checks. Because it is compared against itself, it gets S ≈ 1 and
reward ≈ 0, and three baseline runs measure the timing noise. It tests the
whole pipeline: build, isolation, correctness, timing, and reward.

`solution/reference/` will hold our optimized solution, used to set
`S_target`. `reference.sh` copies it into `/workspace/submission/`.

## 8. PoC validation (2026-10-26)

A minimal proof of concept is in `poc/`. It ran in one pod on one H100 80GB
HBM3, scheduled as a 1-GPU Kubernetes job.

| PoC file | PROJECT.md item |
|---|---|
| `poc/workspace/moe_ffn/` | Starter package (section "Source"), with the M-padding fix |
| `poc/workspace/validation/evaluate.py`, `val.sh` | `val.sh quick\|full [--clean] [--submission DIR]`: correctness, interleaved S/X/S/X/S timing, metrics, reward |
| `poc/workspace/validation/reference.py` | fp32 tiled reference, weight/routing generation, edge cases |
| `poc/workspace/validation/workloads.json` | The 16 validation workloads |
| `poc/baseline.sh` | Section 7 baseline |
| `poc/probes/` | Existing fast paths and the tuned-config headroom probe |
| `poc/run_in_pod.sh` | Lays out `/workspace` and runs `val.sh` in the pod |
| `poc/k8s_job.yaml` | The 1-GPU Kubernetes job used, with `jax[cuda12]==0.11.1` |
| `poc/results/` | `full` mode JSON and logs for the baseline and each probe |

Not built yet: UID-2000 process isolation (the submission runs in the
scorer's process), hidden workloads, `test.sh`, the reference solution, the
fp8 tolerance check, and the agent image.

### Results

Baseline (`val.sh full --clean` on the unchanged starter): **valid**. All 10
edge cases and 16 workloads pass. Errors are 3.9e-3 total and ≤ 8.1e-3 per
row. S = 0.991, reward = 0.

| Workload | Starter time | Starter throughput | Fraction of roofline |
|---|---:|---|---:|
| Qwen3, 32 / 256 tokens | 253–436 µs | 2.7–2.8 TB/s | 0.80–0.84 |
| Qwen3, 2048 tokens | 712–749 µs | 207–217 TFLOP/s, 1.6–1.7 TB/s | 0.54–0.56 |
| Qwen3, 8192 tokens | 2.37–2.39 ms | 259–261 TFLOP/s | 0.26 |
| DeepSeek-V3, 32 / 256 tokens | 2.7–7.5 ms | 2.9–3.0 TB/s | 0.88–0.90 |
| DeepSeek-V3, 2048 tokens | 9.8–11.0 ms | 132–146 TFLOP/s, 2.1–2.3 TB/s | 0.63–0.70 |
| DeepSeek-V3, 8192 tokens | 23–25 ms | 221 TFLOP/s | 0.28 |

### Findings that change this proposal

1. **`S_TARGET = 2.2×` is not reachable.** Even a perfect kernel caps the
   geomean at ~1.46× (section 3). The starter is already weight-bandwidth
   bound for ≤ 256 tokens. Options:
   - (a) Change the token mix toward prefill, e.g. {1024, 2048, 4096, 8192,
     16384}. 16384 tokens means M = 131k and needs ~4 GiB more for inputs.
   - (b) Keep the mix and set `S_TARGET` between ~1.06× (the best dispatch of
     existing paths) and ~1.3×, after the reference solution is measured.
   - (c) Weight workloads by headroom.

   Option (a) is recommended because it keeps the task about kernel work.
2. **The reference cannot be `jax.lax.ragged_dot`.** It densifies on XLA:GPU
   and needs a 64 GiB buffer for DeepSeek-V3 at ≥ 2048 tokens. The PoC
   reference loops over row tiles instead. It is fast enough: all 16
   workloads plus edge cases are checked in about 2 minutes.
3. **The starter kernel needs `M % 8 == 0`.** The fix is padding in
   `api.py`. `ragged_dot_mgpu` has the same limit.
4. **Timing noise on large workloads is up to ~11%.** The baseline against
   itself measured 0.89× on DeepSeek-V3 at 8192 tokens, from 20 calls with
   the slower run kept. Recommendation: time large workloads for ≥ 1 s per
   run, or use the median of more runs. Also run `baseline.sh` 3× as section
   7 plans, and set the reward's zero point from that noise.
5. **The intermediate `h` precision matters but stays within tolerance.** The
   starter rounds `h` to bf16 before SwiGLU and gets 3.9e-3 error. The XLA
   path keeps it in fp32 and gets 1.7e-3. Both are well under 1e-2.

## 9. Full implementation (2026-10-28)

The task package is `tasks/tokamax-moe-ragged-dot-h100/`. It follows the
Anthology task guide and the Inference Optimization 1 task requirements: five
root entries, `schema_version = "1.4"`, exactly one `H100!` per environment,
a 12-hour agent timeout, an offline agent and verifier, and submission UID 2000.
All runs below used one H100 80GB HBM3 (driver 595.91.07) in one Kubernetes pod. The pod
layout mirrors the two Dockerfiles (`poc/full/stage_task.sh`). The Docker
images themselves were not built in the cluster.

| Path | Contents |
|---|---|
| `instruction.md` | Agent instructions (Background, Task, Constraints, Success Criteria) |
| `task.toml` | Metadata, 4 metrics, resources (8 CPU, 64 GiB RAM, 40 GiB disk), provisional baseline stats |
| `environment/Dockerfile`, `requirements.txt` | CUDA 12.9.1 devel, Nsight Systems/Compute, every Python package pinned |
| `environment/workspace/moe_ffn/` | Starter (PoC package plus a README) |
| `environment/workspace/validation/` | `val.sh`, `evaluate.py` (scorer), `harness.py` (isolation), `worker.py`, `reference.py`, `workloads.json` |
| `tests/` | `test.sh`, the same scorer files, hidden `workloads.json`, a starter copy, `Dockerfile` |
| `solution/` | `baseline.sh`, `solve.sh`, `reference.sh`, `reference/` (optimized solution and `summary.md`) |

### Changes from the proposal

1. **Workload mix (finding 1, option a).** M (rows after routing) is in the
   prefill range: Qwen3 M ∈ {8k, 16k, 32k, 64k, 128k}. Hidden: {6k, 12k, 48k,
   96k, 120k}.
2. **DeepSeek-V3 shape.** 64 local experts, as with 4-way expert parallelism
   (5.3 GiB), not 256 (21 GiB). The scorer, the starter worker and the
   submission worker each hold the weights on the same GPU. 256 experts do
   not fit. DeepSeek M ∈ {4k, 8k, 16k, 32k, 64k}; hidden {3k, 6k, 24k, 48k,
   60k}.
3. **Routing.** Each row picks an expert from a balanced or Zipf(1.2) law
   (multinomial with fixed M), so M is exact.
4. **Isolation.** The scorer never imports submitted code. The starter and
   the submission run in separate `worker.py` processes, the submission under
   `prlimit` and `setpriv` as UID 2000. The scorer times each block round trip
   with its own clock, pauses the idle worker with SIGSTOP, and kills all UID
   2000 processes afterwards. Source files are staged without links,
   binaries or caches. `build.sh` runs as UID 2000.
5. **No result reuse.** Every timed call has its own routing. Outputs stay in
   the worker. Every timed call returns per-row digests (squared norm and a
   random projection), checked against the fp32 reference. After each block,
   2048 rows of its last output, chosen at random after the block ran, are
   compared in full. Edge cases are compared in full.
6. **Memory.** The JAX allocator caps the submission process at 40% of the
   GPU (~32 GiB). The 1.25× prepare-size cap is dropped.
7. **Timing protocol.** One discarded block per worker, then S X S X S X S
   with a 0.5 s idle gap before each block (~0.4 s per block, 5–400 calls).
   Each workload's speedup is the median of the three ratios, each X block
   against the mean of its neighbouring S blocks.
8. **`S_TARGET = 2.0`** (section "Calibration").

### Timing findings

- **Back-to-back blocks bias the second one.** With no idle gap, the
  submission block always ran right after a starter block and measured 2–6%
  slower on identical code (DeepSeek M=8192; `poc/full/results/ab_no_gap.*`).
  The following were ruled out by A/B runs on the pod:
  - memory fraction
  - setpriv/prlimit
  - process start order
  - allocation order
  - SIGSTOP
  A 0.5 s gap before every block removes the bias: 3566 vs 3568 µs
  (`ab_idle_gap.*`).
- **The first block after an idle period is up to 5% slow** (clock ramp,
  allocator growth). Fixed by the discarded warm-up blocks.
- **Occasional single slow blocks (0.87–0.93×) remain.** The median of three
  ratios absorbs them. The earlier "lower of two" rule turned one of these into
  a 0.89× workload score.
- **Bounding in-flight calls to 2.** Without the bound, async dispatch
  allocated buffers for every queued call, and the starter ran out of memory
  at large M.

### Correctness findings

| Submission | Total error | Max row error | Result |
|---|---:|---:|---|
| Starter | 3.9–4.2e-3 | ≤ 6.8e-3 | pass |
| Reference (fp32 SwiGLU from accumulators) | 1.7e-3 | ≤ 3.0e-3 | pass |
| fp8 e4m3 weights, per-column scale (`poc/full/probes/fp8`) | 4.7–4.9e-2 | 5.2–5.8e-2 | **fails** every workload and edge case (≈5× the 1e-2 tolerance) |

### Reference solution

`solution/reference/moe_ffn/kernel_ws.py` is a persistent, warp-specialized
ragged dot, modelled on JAX's `hopper_matmul_mgpu`. One TMA warpgroup and two
wgmma warpgroups split the output columns. It reuses the starter's
`GroupInfo` tiles and its ragged TMA store. The gate/up kernel fuses SwiGLU,
with a gate and an up accumulator per warpgroup, so `[M, 2I]` never reaches
HBM. Configs come from a per-kernel sweep (`poc/full/sweep_ws.py`). The best
configs are 128×(2×64)×64 with 3–4 stages for gate/up, and 128×(2×128)×64 with
3 stages for down. The fused gate/up reaches 633 TF/s and the down kernel
545 TF/s at Qwen M=128k.

### Results (`poc/full/results/`)

| Run | Valid | S | Roofline geomean | Max norm. error | Reward | Time |
|---|---|---:|---:|---:|---:|---:|
| Baseline, `val.sh full` | yes | 1.002 | 0.36 | 0.40 | 0.002 | 385 s |
| Baseline, `test.sh` | yes | 1.002 | 0.37 | 0.42 | 0.003 | 373 s |
| Reference, `val.sh full` | yes | 1.580 | 0.57 | 0.18 | 0.660 | 386 s |
| Reference, `test.sh` | yes | 1.580 | 0.58 | 0.18 | 0.660 | 374 s |
| fp8 probe, `val.sh full` | **no** (tolerance) | — | — | 4.9 | 0 | 302 s |

Baseline workloads sit at 0.996–1.009 against themselves. The reference's
speedup by workload:

| Workload | Reference speedup | Note |
|---|---:|---|
| DeepSeek M=3–4k | 1.03–1.25× | Already 79–88% of roofline |
| Qwen M=6–8k | 1.02–1.16× | Weight-read bound |
| Mid M (12k–32k) | 1.3–1.9× | |
| Large M (≥ 48k) | 1.77–2.04× | |

The validation and hidden mixes give the same S for both solutions.

### Calibration

| Point | S |
|---|---:|
| Starter (reward 0) | 1.0 |
| Best existing fast path (tuned Tokamax config, PoC) | ~1.06 |
| Reference (reward 0.66) | 1.58 |
| `S_TARGET` (reward 1) | 2.0 |
| Perfect-roofline ceiling | ~2.6–2.8 |

Reaching 2.0 needs about 0.72 of roofline on geomean (the reference is at
0.58). That means better large-M kernels (≥ 650 TF/s, deeper pipelines,
clusters/multicast) and less waste at small M (fewer weight re-reads, split-K
or stream-K for skew).

### Verification against the guide (2026-10-28)

The 26 static controls of `anthology-research/inference-optimization-1`
(`checks/static/`) were applied by reading their source; the suite itself was
not executed. Two blocking failures were found:
- `validation-test-contract`: the reward-key comments in `val.sh` and
  `test.sh` lacked the `"key":` form. Fixed.
- `metadata-schema`: `[metadata].organization` is empty. Still open.

Runtime fix: pausing the idle submission worker now stops every UID 2000
process, not just the worker's process group (a `setsid` child escaped it).
This needs a GPU re-run.

Not enforced or not measured:
- The memory cap binds only allocations made through JAX; raw `cudaMalloc`
  from FFI code bypasses it.
- Host RAM and thread counts (sandbox cap 2,048) were not measured.
- The Docker images were not built.

### Still to do

- Build both Docker images and run `/run baseline` (3 paired runs) to replace
  the provisional `baseline_validation` and `baseline_test`.
- Agent calibration rollouts, aiming for mean reward ≤ 0.7 and variance > 0.02.
  Retune `S_TARGET` from them.
- Fill `[metadata].organization` and the PR body (task write-up, sources and
  licences: Tokamax and JAX, Apache-2.0).
