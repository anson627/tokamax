"""Fast-path probe: jax.lax.ragged_dot (XLA:GPU) for both expert matmuls."""

import jax
import jax.numpy as jnp


def prepare(w_gate_up, w_down):
  return (w_gate_up, w_down)


@jax.jit
def _moe_ffn(w_gate_up, w_down, x, group_sizes):
  i = w_down.shape[1]
  h = jax.lax.ragged_dot(x, w_gate_up, group_sizes,
                         preferred_element_type=jnp.float32)
  a = (jax.nn.silu(h[:, :i]) * h[:, i:]).astype(jnp.bfloat16)
  out = jax.lax.ragged_dot(a, w_down, group_sizes,
                           preferred_element_type=jnp.float32)
  return out.astype(jnp.bfloat16)


def moe_ffn(handle, x, group_sizes):
  return _moe_ffn(*handle, x, group_sizes)
