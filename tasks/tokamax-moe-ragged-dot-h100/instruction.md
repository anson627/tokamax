## Background

In a mixture-of-experts (MoE) model, each token is sent to a few experts.
After routing, the rows are sorted by expert, so each expert multiplies one
contiguous block of rows. The number of rows per expert varies from call to
call and can be zero. This operation is called a ragged dot (or grouped GEMM).

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

The starting solution in `/workspace/moe_ffn/` computes this layer with the
Tokamax Hopper ragged-dot kernel, written in JAX with Pallas Mosaic GPU
(copied from `openxla/tokamax` at commit `4bf85ff`, Apache-2.0). It uses
Tokamax's default H100 config (64×64×128 tiles, 2 pipeline stages, one output
tile per thread block). It calls the kernel twice, with the SwiGLU as a
separate XLA step in between. It is correct, but it reaches only about a
third of the H100's bf16 tensor-core throughput on large inputs.

The workloads use two expert shapes, both bf16:

| Model shape | E | H | I | Weights |
|---|---:|---:|---:|---:|
| Qwen3-30B-A3B | 128 | 2048 | 768 | 1.1 GiB |
| DeepSeek-V3, 64 local experts (4-way expert parallelism) | 64 | 7168 | 2048 | 5.3 GiB |

`M` (rows after routing) ranges from 8,192 to 131,072, as in prefill. Routing
is either balanced or skewed: with skewed routing a few experts receive many
rows and some receive very few. Weights and activations are random, generated
on the GPU from fixed seeds.

`/workspace/moe_ffn/README.md` lists the starter files. JAX's own Pallas
Mosaic GPU examples (for example `jax/experimental/pallas/ops/gpu/` in the
installed `jax` package) are available offline.

## Task

Make the MoE expert layer as fast as possible on one H100. Your package must
provide these two functions in `moe_ffn/api.py`:

- `prepare(w_gate_up, w_down) -> handle`: runs once per model shape and is not
  timed. It may change the weight layout. It may use only the weights.
- `moe_ffn(handle, x, group_sizes) -> out`: timed. Returns `[M, H]` bf16 as a
  `jax.Array`. It is called many times with different `x` and `group_sizes`
  for the same `M`.

You may change anything behind these two functions: tune or rewrite the Pallas
kernel, write new Pallas kernels, fuse steps, or write CUDA kernels called
through `jax.ffi`.

Put the submission in `/workspace/submission/`:

- `moe_ffn/`: your package, with `api.py` and all of its source files.
- `moe_ffn/build.sh` (optional): a build step, run inside `moe_ffn/` before
  the package is imported. Use it to compile any native code from source.
- `summary.md` with two sections:
  - `## Experiments`: what you tried, how you measured it, and what worked or
    did not.
  - `## Submitted solution`: your final design, what changed from the starter,
    your results, and how to reproduce them.

To start, copy the starter: `cp -R /workspace/moe_ffn /workspace/submission/`.

Check a submission with `bash /workspace/validation/val.sh <mode>`. It
evaluates `/workspace/submission/` by default; add `--submission <dir>` to test
another directory with the same layout.

| Command | What it does |
|---|---|
| `val.sh quick` | Build, then correctness and speed on 2 workloads with short timing. No edge cases and no reward. |
| `val.sh full` | Build, then 10 edge cases and all 20 workloads in `/workspace/validation/workloads.json`, and the reward. |
| `val.sh full --only <text>` | Only the workloads whose names contain `<text>`; no reward. |

Each workload line shows the error, the starter and submission time per call,
the speedup, TFLOP/s, and the fraction of the H100 roofline. Every run builds
from a clean copy of the submission's source files. Run `val.sh full` on the
final contents of `/workspace/submission/` before you finish.

## Constraints

- Inputs and outputs are bf16. Do not quantize weights, activations or the
  intermediate `a` to fp8, int8 or any lower precision. Accumulate in fp32 or
  higher.
- Every row must be multiplied by its own expert's weights. Do not skip,
  sample or approximate rows.
- `moe_ffn` must compute each call's output from that call's inputs. Do not
  reuse results between calls.
- `prepare` must finish within 120 s per model shape.
- The process that runs your code may use about 32 GiB of GPU memory (40% of
  the GPU, enforced by the JAX allocator), including the weights.
- `moe_ffn` must work for any `M ≥ 1` and any `group_sizes` that sum to `M`,
  including empty experts and `M` that is not a multiple of 8.
- The submission is copied without other files from the workspace. Only source
  files are copied (for example `.py`, `.cu`, `.h`, `.sh`, `.json`); prebuilt
  binaries and caches are ignored. It may use the Python packages and CUDA
  toolkit already installed in `/opt/venv` and the image; no other packages
  can be installed and there is no network access.
- Your code runs as an unprivileged user in its own process. Do not modify
  `/workspace/validation/` or interfere with the evaluator's processes,
  timing or files.

## Success Criteria

**Correctness (required).** Outputs are compared with an fp32 reference that
accumulates in fp32 and keeps the SwiGLU output in fp32. For every workload
and edge case:

- total relative error (Frobenius norm of the difference divided by that of
  the reference) ≤ 1e-2
- relative error of every row ≤ 5e-2
- no NaN or Inf

Every timed call is checked through per-row digests of its output (each row's
norm and its projection onto a random vector). After each timed block, rows at
random positions of its last output are compared in full. Edge cases (empty
experts, one expert with all rows, group edges one row off a tile edge, `M`
not a multiple of 8, `M = 3`) are compared in full. The starter's error is
about 4e-3 total and below 1e-2 per row.

A failed build, a crash, a wrong output shape or type, a failed correctness
check, `prepare` over 120 s, or running out of the memory limit makes the
submission invalid, with reward 0.

**Speed.** For each workload, the starter and your submission are timed in
the same run, alternating starter and submission blocks: S X S X S X S. Each
block calls `moe_ffn` many times (about 0.4 s in total), and every call uses a
new routing. Each submission block is compared with the mean of the two
starter blocks around it, and the median of the three ratios is the
workload's speedup. S is the geometric mean of the speedups over all workloads. For a
valid submission:

    reward = clip( ln(S) / ln(2.0), 0, 1 )

The starter scores about 0; S = 2.0 scores 1. The final evaluation uses the
same two model shapes and the same routing types, with different values of `M`
in the same range and different random inputs and routings.
