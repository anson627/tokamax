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
"""Ragged dot Pallas-Mosaic-GPU Non-Quantized Kernel (Hopper).

Copied from tokamax/_src/ops/ragged_dot/pallas_mosaic_gpu_kernel_sm90.py#L35-L171
at openxla/tokamax@4bf85ff. The kernel math is unchanged; only the jaxtyping
decorator is dropped and imports are local.
"""
import functools
from collections.abc import Callable

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.extend import backend
import jax.numpy as jnp

from . import common
from . import mgpu_utils


_tiled_swizzled_block_spec = mgpu_utils.tiled_swizzled_block_spec
_tiled_swizzled_smem = mgpu_utils.tiled_swizzled_smem


def ragged_dot_kernel(
    lhs: jax.Array,  # [M, K]
    rhs: jax.Array,  # [G, K, N]
    group_sizes: jax.Array,  # [G]
    out_dtype: jnp.dtype,
    config: common.Config,
    activation: Callable[[jax.Array], jax.Array] | None = None,
) -> jax.Array:  # [M, N]
  """Pallas kernel for ragged dot with non-quantized inputs."""
  common.check_bf16xbf16_or_f16xf16(lhs, rhs)

  m, k = lhs.shape
  g, _, n = rhs.shape

  block_m = config.block_m
  block_n = config.block_n
  block_k = config.block_k
  m_iters = pl.cdiv(m, block_m) + g - 1
  n_iters = pl.cdiv(n, block_n)
  k_iters = pl.cdiv(k, block_k)
  align_tile = 8

  def kernel(
      lhs_gmem,
      rhs_gmem,
      group_id_gmem,
      start_within_block_gmem,
      actual_size_gmem,
      block_start_gmem,
      o_gmem,
  ):

    def mn_loop_body(m_offset, m_iters, loop_info: plgpu.NDLoopInfo):
      (mni,) = loop_info.index
      mi, ni = plgpu.planar_snake(
          mni, (m_iters, n_iters), config.grid_minor_dim, config.grid_tile_width
      )
      mi += m_offset

      with jax.named_scope("load group_info"):
        gi = group_id_gmem[mi]
        start_within_block = start_within_block_gmem[mi]
        actual_size = actual_size_gmem[mi]
        block_start = block_start_gmem[mi]
        block_start = pl.multiple_of(block_start, align_tile)

      @pl.when(actual_size > 0)
      def body():

        def compute_acc(acc):
          def pipeline_body(_, lhs_smem, rhs_smem):
            plgpu.wgmma(acc, lhs_smem, rhs_smem)
            plgpu.wgmma_wait(1)

          spec = functools.partial(_tiled_swizzled_block_spec, delay_release=1)
          lhs_block_shape = (pl.Element(block_m), block_k)
          rhs_block_shape = (block_k, block_n)
          lhs_index_map = lambda ki: (block_start, ki)
          rhs_index_map = lambda ki: (ki, ni)
          lhs_spec = spec(lhs_block_shape, lhs_gmem.dtype, lhs_index_map, "lhs")
          rhs_spec = spec(rhs_block_shape, rhs_gmem.dtype, rhs_index_map, "rhs")
          plgpu.emit_pipeline(
              pipeline_body,
              grid=(k_iters,),
              in_specs=(lhs_spec, rhs_spec),
              max_concurrent_steps=config.num_stages,
          )(lhs_gmem, rhs_gmem.at[gi])
          return acc[...]

        acc = pl.run_scoped(compute_acc, plgpu.ACC((block_m, block_n)))

        o_smem_type = _tiled_swizzled_smem(
            (block_m, block_n), o_gmem.dtype, tiling_prefix=(1,), what="out"
        )

        @functools.partial(pl.run_scoped, o_smem=o_smem_type)
        def epilogue(o_smem):
          acc_ = acc if activation is None else activation(acc)
          o_smem[...] = acc_.astype(o_smem.dtype)
          plgpu.commit_smem()

          common.copy_ragged_smem_to_gmem(
              o_smem,
              o_gmem,
              block_start=block_start,
              start_within_block=start_within_block,
              actual_size=actual_size,
              ns=pl.ds(ni * block_n, block_n),
              max_rows=min(block_m, m),
          )

    if config.persistent:
      # We stratify the grid: first emit a number of blocks that have definitely
      # work to do. Then schedule blocks that may be no-ops. This way we lower
      # the chances that no-op blocks are scheduled to the same SM.
      def run_mn_loop(m_offset, m_iters):
        mn_iters = m_iters * n_iters
        f = functools.partial(mn_loop_body, m_offset, m_iters)
        plgpu.nd_loop((mn_iters,), collective_axes="sm")(f)

      m0_iters = pl.cdiv(m, block_m)
      run_mn_loop(0, m0_iters)
      run_mn_loop(m0_iters, m_iters - m0_iters)
    else:
      mni = jax.lax.axis_index("mn")
      loop_info = plgpu.NDLoopInfo((mni,), local_index=0, num_local_steps=1)
      mn_loop_body(0, m_iters, loop_info)

  group_info = common.GroupInfo.create_aligned(
      group_sizes, block_m, m_iters, align_tile
  )

  if config.persistent:
    grid = (backend.get_default_device().core_count,)
    grid_names = ("sm",)
  else:
    grid = (m_iters * n_iters,)
    grid_names = ("mn",)

  return plgpu.kernel(
      kernel,
      out_type=jax.ShapeDtypeStruct((m, n), out_dtype),
      grid=grid,
      grid_names=grid_names,
      kernel_name="ragged_dot_sm90",
      compiler_params=plgpu.CompilerParams(
          approx_math=True, unsafe_no_auto_barriers=True
      ),
  )(
      lhs,
      rhs,
      group_info.group_id,
      group_info.start_within_block,
      group_info.actual_size,
      group_info.block_start,
  )
