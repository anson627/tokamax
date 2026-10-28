"""Reference MoE expert layer: warp-specialized ragged dots, fused SwiGLU.

    a   = silu(x @ w_gate[g]) * (x @ w_up[g])     # one kernel, [M, I] bf16
    out = a @ w_down[g]                            # one kernel, [M, H] bf16

Both kernels are `kernel_ws.ragged_dot_ws` (persistent, one TMA warpgroup and
two wgmma warpgroups). Tile configs are chosen per model shape and M from an
offline sweep on the validation shapes (see summary.md).
"""

import functools

import jax
import jax.numpy as jnp

from . import common
from . import kernel_ws

MD = common.MatmulDimension
_C = kernel_ws.WsConfig

# (H, I) -> list of (max M, gate/up config, down config); first match wins.
CONFIGS = {
    (2048, 768): [
        (32768, _C(128, 64, 64, 4, MD.N, 8), _C(64, 128, 64, 4, MD.M, 8)),
        (None, _C(128, 64, 64, 4, MD.N, 8), _C(128, 128, 64, 3, MD.M, 8)),
    ],
    (7168, 2048): [
        (None, _C(128, 64, 64, 3, MD.M, 1), _C(128, 128, 64, 3, MD.M, 1)),
    ],
}
DEFAULT = CONFIGS[(2048, 768)][0][1:]


def pick_configs(h, i, m):
  for max_m, c1, c2 in CONFIGS.get((h, i), []):
    if max_m is None or m <= max_m:
      return c1, c2
  return DEFAULT


def prepare(w_gate_up: jax.Array, w_down: jax.Array):
  return (w_gate_up, w_down)


@functools.partial(jax.jit, static_argnames=("c1", "c2"))
def _moe_ffn(w_gate_up, w_down, x, group_sizes, c1, c2):
  a = kernel_ws.ragged_dot_ws(x, w_gate_up, group_sizes, c1, swiglu=True)
  return kernel_ws.ragged_dot_ws(a, w_down, group_sizes, c2)


def moe_ffn(handle, x: jax.Array, group_sizes: jax.Array) -> jax.Array:
  w_gate_up, w_down = handle
  m, h = x.shape
  i = w_down.shape[1]
  c1, c2 = pick_configs(h, i, m)
  # TMA row offsets must be multiples of 8 and a row tile must fit in M. Pad
  # rows belong to no group, so they are never written; they are sliced off.
  m_pad = max(-(-m // 8) * 8, c1.tile_m, c2.tile_m)
  if m_pad != m:
    x = jnp.pad(x, ((0, m_pad - m), (0, 0)))
  out = _moe_ffn(w_gate_up, w_down, x, group_sizes, c1, c2)
  return out[:m] if m_pad != m else out
