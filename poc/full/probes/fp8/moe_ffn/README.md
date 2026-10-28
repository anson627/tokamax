# Starter MoE expert layer

| File | Contents |
|---|---|
| `api.py` | `prepare` and `moe_ffn`: two kernel calls with an XLA SwiGLU in between; pads `M` to a multiple of 8 |
| `kernel_sm90.py` | Tokamax's Hopper ragged-dot kernel (`pallas_mosaic_gpu_kernel_sm90.py`) |
| `common.py` | `Config`, `GroupInfo.create_aligned` (row tiles that never span experts), the ragged output copy |
| `mgpu_utils.py` | Shared-memory tiling and swizzle helpers |

Source: `openxla/tokamax` at commit `4bf85ff592e66967efaf9c61c45fe52d0ee8f2f2`
(Apache-2.0). The kernel math is unchanged. Autotuning, the non-H100 code
paths and the `jaxtyping`/`pydantic` dependencies are removed, and the config
is fixed to Tokamax's default H100 config.
