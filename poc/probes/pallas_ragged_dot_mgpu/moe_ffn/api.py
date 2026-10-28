"""Fast-path probe: jax.experimental.pallas.ops.gpu.ragged_dot_mgpu (Hopper)."""

import functools

import jax
from jax.experimental.pallas.ops.gpu import ragged_dot_mgpu
import jax.numpy as jnp

# One untuned, reasonable config from the module's own sweep space.
KWARGS = dict(block_m=128, block_n=128, block_k=64, max_concurrent_steps=4,
              grid_block_n=1)


def prepare(w_gate_up, w_down):
  return (w_gate_up, w_down)


@jax.jit
def _moe_ffn(w_gate_up, w_down, x, group_sizes):
  i = w_down.shape[1]
  rd = functools.partial(ragged_dot_mgpu.ragged_dot, group_sizes=group_sizes,
                         **KWARGS)
  h = rd(x, w_gate_up).astype(jnp.float32)
  a = (jax.nn.silu(h[:, :i]) * h[:, i:]).astype(jnp.bfloat16)
  return rd(a, w_down).astype(jnp.bfloat16)


def moe_ffn(handle, x, group_sizes):
  return _moe_ffn(*handle, x, group_sizes)
