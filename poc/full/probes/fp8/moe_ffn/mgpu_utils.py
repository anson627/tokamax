# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Mosaic-GPU shared-memory helpers.

Copied from tokamax/_src/mosaic_gpu.py#L56-L78 at openxla/tokamax@4bf85ff.
"""

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
import jax.numpy as jnp


def num_bits(dtype: jax.typing.DTypeLike) -> int:
  fn = jnp.finfo if jnp.issubdtype(dtype, jnp.floating) else jnp.iinfo
  return fn(dtype).bits


def tile_swizzle_transforms(
    shape: tuple[int, ...],
    dtype: jax.typing.DTypeLike,
    what: str = "",
    *,
    swizzle: int | None = None,
    tiling_prefix: tuple[int, ...] = (8,),
) -> tuple[plgpu.TilingTransform, plgpu.SwizzleTransform]:
  """Returns tiling and swizzling transforms."""
  elem_bits = num_bits(dtype)
  if swizzle is None:
    swizzle = plgpu.find_swizzle(shape[-1] * elem_bits, what)
  tiling = (*tiling_prefix, 8 * swizzle // elem_bits)
  return plgpu.TilingTransform(tiling), plgpu.SwizzleTransform(swizzle)


def tiled_swizzled_smem(
    shape: tuple[int, ...],
    dtype: jax.typing.DTypeLike,
    what: str = "",
    *,
    swizzle: int | None = None,
    tiling_prefix: tuple[int, ...] = (8,),
) -> pl.MemoryRef:
  """Returns a memory reference to a tiled and swizzled shared memory array."""
  transforms = tile_swizzle_transforms(
      shape, dtype, what, swizzle=swizzle, tiling_prefix=tiling_prefix
  )
  return plgpu.SMEM(shape, dtype, transforms=transforms)


def tiled_swizzled_block_spec(
    shape, dtype, index_map, what="", **kwargs
) -> plgpu.BlockSpec:
  """Returns a block spec with tiling and swizzling transforms."""
  transforms = tile_swizzle_transforms(shape, dtype, what)
  return plgpu.BlockSpec(shape, index_map, transforms=transforms, **kwargs)
