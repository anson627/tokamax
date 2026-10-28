"""Input generation, the fp32 reference and output digests.

Imported by both the scorer and the worker processes, so both build the same
weights and activations from the same seeds on the GPU.

The reference does not use `jax.lax.ragged_dot`: XLA:GPU lowers it to a dense
masked dot over all experts, which does not fit in memory at large M. It walks
row tiles that each lie within one expert and runs plain bf16 x bf16 -> fp32
XLA dots. The SwiGLU output stays fp32; it is split into bf16 hi + lo parts so
the second dot is exact to ~16 mantissa bits.
"""

import functools

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np

_TILE = 256


def _group_tasks(group_sizes, m, tile):
  """Per-task (group, row start, valid end); tiles never span groups."""
  e = group_sizes.shape[0]
  num_tasks = -(-m // tile) + e - 1
  starts = jnp.cumulative_sum(group_sizes, include_initial=True)
  ends = starts[1:]
  starts = starts[:-1]
  nblk = jnp.where(group_sizes == 0, 0, -(-group_sizes // tile))
  task_start = jnp.cumulative_sum(nblk, include_initial=True)
  t = jnp.arange(num_tasks)
  gid = jnp.clip(jnp.searchsorted(task_start[1:], t, side="right"), 0, e - 1)
  row0 = starts[gid] + (t - task_start[gid]) * tile
  valid_end = jnp.minimum(row0 + tile, ends[gid])
  valid_end = jnp.where(t < task_start[-1], valid_end, row0)  # padding tasks
  return gid, row0, valid_end


def _ragged_dot_f32(lhs_parts, rhs, group_sizes, tile=_TILE):
  """sum_p ragged_dot(lhs_p, rhs) with fp32 accumulation. lhs_p: [M, K] bf16."""
  m = lhs_parts[0].shape[0]
  n = rhs.shape[2]
  gid, row0, valid_end = _group_tasks(group_sizes, m, tile)
  lhs_parts = [jnp.pad(p, ((0, tile), (0, 0))) for p in lhs_parts]
  out = jnp.zeros((m + tile, n), jnp.float32)

  def body(out, task):
    g, r0, r1 = task
    w = rhs[g]
    acc = sum(
        jnp.dot(lax.dynamic_slice_in_dim(p, r0, tile), w,
                preferred_element_type=jnp.float32)
        for p in lhs_parts)
    rows = r0 + jnp.arange(tile)
    acc = jnp.where((rows < r1)[:, None], acc, 0.0)
    cur = lax.dynamic_slice_in_dim(out, r0, tile)
    return lax.dynamic_update_slice_in_dim(out, cur + acc, r0, 0), None

  out, _ = lax.scan(body, out, (gid, row0, valid_end))
  return out[:m]


@jax.jit
def moe_ffn_reference(w_gate_up, w_down, x, group_sizes):
  """fp32 MoE expert layer. Returns [M, H] float32."""
  i = w_down.shape[1]
  h = _ragged_dot_f32([x], w_gate_up, group_sizes)
  a = jax.nn.silu(h[:, :i]) * h[:, i:]
  a_hi = a.astype(jnp.bfloat16)
  a_lo = (a - a_hi.astype(jnp.float32)).astype(jnp.bfloat16)
  return _ragged_dot_f32([a_hi, a_lo], w_down, group_sizes)


@jax.jit
def digest(out, r):
  """[2, M] fp32: per-row squared norm and projection onto the vector r."""
  o = out.astype(jnp.float32)  # Fused into the two row reductions.
  return jnp.stack([jnp.sum(o * o, axis=1), jnp.sum(o * r[None, :], axis=1)])


@functools.partial(jax.jit, static_argnames=("e", "k", "n"))
def _random_weight(key, e, k, n):
  keys = jax.random.split(key, e)
  scale = 1.0 / np.sqrt(k)

  def one(key):
    return (jax.random.normal(key, (k, n), jnp.float32) * scale).astype(jnp.bfloat16)

  return lax.map(one, keys)


def make_weights(model, seed):
  e, h, i = model["E"], model["H"], model["I"]
  k1, k2 = jax.random.split(jax.random.key(seed))
  return _random_weight(k1, e, h, 2 * i), _random_weight(k2, e, i, h)


@functools.partial(jax.jit, static_argnames=("m", "h"))
def _random_x(key, m, h):
  return jax.random.normal(key, (m, h), jnp.float32).astype(jnp.bfloat16)


def make_x(model, m, seed):
  return _random_x(jax.random.key(seed), m, model["H"])


def make_projection(model, seed):
  return jax.random.normal(jax.random.key(seed), (model["H"],), jnp.float32)


def make_group_sizes(model, m, routing, rng):
  """int32 [E] summing to m: each row picks an expert from the routing law."""
  e = model["E"]
  if routing == "balanced":
    p = np.ones(e)
  elif routing == "skewed":
    # Zipf-like expert popularity: a few hot experts and a long cold tail.
    p = (1.0 + rng.permutation(e)) ** -1.2
  else:
    raise ValueError(routing)
  return rng.multinomial(m, p / p.sum()).astype(np.int32)


def edge_case_group_sizes(model):
  """Correctness-only cases: name -> group_sizes (numpy int32 [E])."""
  e = model["E"]
  cases = {}
  gs = np.zeros(e, np.int32)
  gs[e // 2] = 2048
  cases["one_expert_all_rows"] = gs
  gs = np.zeros(e, np.int32)
  gs[::3] = 9
  cases["mostly_empty_experts"] = gs
  # Group edges one row off a 64/128-row tile edge.
  gs = np.array([63, 65, 1, 127, 129, 0, 64, 191] * (e // 8), np.int32)
  cases["tile_edge_off_by_one"] = gs
  rng = np.random.default_rng(7)
  gs = np.bincount(rng.integers(0, e, size=1003), minlength=e).astype(np.int32)
  cases["m_not_multiple_of_8"] = gs  # M = 1003
  gs = np.zeros(e, np.int32)
  gs[0] = 1
  gs[-1] = 2
  cases["first_and_last_expert_tiny"] = gs
  return cases
