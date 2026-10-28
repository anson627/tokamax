## Experiments

- Measured the starter per kernel on both shapes. The gate/up and down
  kernels reach 170–260 TF/s on large M. The 64×64 tiles, 2 pipeline stages
  and one tile per thread block leave the tensor cores idle while waiting on
  TMA. The XLA SwiGLU is an extra pass over `[M, 2I]`.
- Tokamax's tuned H100 config for the same kernel (128×64×128, 4 stages,
  persistent) helps only on large M (about 1.1–1.35×) and is slower for small
  M, so retuning the starter kernel alone is not enough.
- Wrote `kernel_ws.ragged_dot_ws`, a persistent, warp-specialized ragged dot
  modelled on JAX's `hopper_matmul_mgpu`. One warpgroup issues TMA loads and
  two compute warpgroups each own half of the output columns. It reuses the
  starter's `GroupInfo` row tiles (never spanning experts) and its ragged
  power-of-two TMA store.
- Fused the SwiGLU into the gate/up kernel. Each compute warpgroup keeps a
  gate and an up accumulator for the same columns (two rhs TMA streams) and
  writes `silu(gate) * up` as bf16, so `[M, 2I]` never reaches HBM. The SwiGLU
  is computed in fp32 from the accumulators, which also lowers the error
  (1.7e-3 vs the starter's 3.9e-3).
- Swept tile_m ∈ {64, 128}, tile_n ∈ {64, 128} per warpgroup, 2–5 stages, and
  grid order (M or N minor, width 1/4/8) per kernel on M ∈ {8192, 32768,
  131072} for both shapes. Shared memory limits the fused kernel to
  128×(2×64) tiles with 3–4 stages. 128×(2×128) down tiles with 3 stages were
  best on large M. Grid order changes results by under 5%.

## Submitted solution

`moe_ffn/api.py` calls two `ragged_dot_ws` kernels: the fused gate/up +
SwiGLU kernel (`[M, H] × [E, H, 2I] → [M, I]`) and the down projection
(`[M, I] × [E, I, H] → [M, H]`). Configs come from a small table keyed by
`(H, I)` and M. `prepare` returns the weights unchanged. M is padded up to a
multiple of 8 and to at least one row tile; pad rows belong to no expert and
are sliced off.

Results (`val.sh full` on one H100 80GB HBM3): valid on all 20 workloads and
10 edge cases, error 1.7e-3 total and below 2.8e-3 per row. Speedup geomean
1.580 (1.22× at DeepSeek M=4096, already 88% of the weight-read roofline, up
to 2.03× at M=131072/65536), 0.57 of the roofline on geomean, reward 0.66.

Reproduce: `bash /workspace/validation/val.sh full`.
