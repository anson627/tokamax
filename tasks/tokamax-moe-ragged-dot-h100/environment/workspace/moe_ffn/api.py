"""Starter MoE expert layer: Tokamax sm90 ragged dot, default H100 config.

    h   = ragged_dot(x, w_gate_up, group_sizes)   # [M, 2I]
    a   = silu(h[:, :I]) * h[:, I:]               # SwiGLU (XLA)
    out = ragged_dot(a, w_down, group_sizes)      # [M, H]
"""

import functools

import jax
import jax.numpy as jnp

from . import common
from . import kernel_sm90

# Tokamax's default sm90 heuristic config for bf16 inputs
# (pallas_mosaic_gpu.py#L316-L326 at openxla/tokamax@4bf85ff).
CONFIG = common.Config(
    block_m=64,
    block_n=64,
    block_k=128,
    num_stages=2,
    split_k=1,
    persistent=False,
    grid_minor_dim=common.MatmulDimension.M,
    grid_tile_width=1,
)


def prepare(w_gate_up: jax.Array, w_down: jax.Array):
  """Runs once per model shape (not timed). Returns the weights unchanged."""
  return (w_gate_up, w_down)


@functools.partial(jax.jit, static_argnames="config")
def _moe_ffn(w_gate_up, w_down, x, group_sizes, config=CONFIG):
  i = w_down.shape[1]
  h = kernel_sm90.ragged_dot_kernel(
      x, w_gate_up, group_sizes, out_dtype=jnp.bfloat16, config=config
  )
  h = h.astype(jnp.float32)
  a = (jax.nn.silu(h[:, :i]) * h[:, i:]).astype(jnp.bfloat16)
  return kernel_sm90.ragged_dot_kernel(
      a, w_down, group_sizes, out_dtype=jnp.bfloat16, config=config
  )


def moe_ffn(handle, x: jax.Array, group_sizes: jax.Array) -> jax.Array:
  """Timed. x: [M, H] bf16 sorted by expert; group_sizes: [E] int32."""
  w_gate_up, w_down = handle
  m = x.shape[0]
  # The kernel's TMA copies need M to be a multiple of 8. Pad rows belong to no
  # group, so the kernel never writes them; they are sliced off.
  pad = -m % 8
  if pad:
    x = jnp.pad(x, ((0, pad), (0, 0)))
  out = _moe_ffn(w_gate_up, w_down, x, group_sizes)
  return out[:m] if pad else out
