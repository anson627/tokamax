"""Headroom probe: starter kernel with Tokamax's tuned H100 DeepSeek-V3 config.

Config from tokamax/data/autotuning/nvidia_h100_80gb_hbm3/
pallas_mosaic_gpu_ragged_dot.json (bf16[131072,7168] x bf16[256,7168,1024]).
kernel_sm90.py / common.py / mgpu_utils.py are copied from the starter by
poc/run_in_pod.sh.
"""

import functools

import jax
import jax.numpy as jnp

from . import common
from . import kernel_sm90

CONFIG = common.Config(
    block_m=128,
    block_n=64,
    block_k=128,
    num_stages=4,
    split_k=1,
    persistent=True,
    grid_minor_dim=common.MatmulDimension.N,
    grid_tile_width=8,
)


def prepare(w_gate_up, w_down):
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


def moe_ffn(handle, x, group_sizes):
  m = x.shape[0]
  pad = -m % 8  # Same M padding as the starter.
  if pad:
    x = jnp.pad(x, ((0, pad), (0, 0)))
  out = _moe_ffn(*handle, x, group_sizes)
  return out[:m] if pad else out
