"""Warp-specialized ragged dot for Hopper, with an optional fused SwiGLU.

One CTA computes a (tile_m, 2 * tile_n) output tile. A memory warpgroup runs
the TMA pipeline and two compute warpgroups each own tile_n output columns.
With `swiglu=True` the rhs is the [E, K, 2I] gate/up weight: every compute
warpgroup keeps a gate and an up accumulator and writes silu(gate) * up, so the
[M, 2I] intermediate never reaches HBM.

Row tiles never span experts (same `GroupInfo` tiling as the starter), and
the output is stored with the starter's ragged power-of-two TMA ladder.
"""

import dataclasses
import functools

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.extend import backend
import jax.numpy as jnp

from . import common
from . import mgpu_utils


@dataclasses.dataclass(frozen=True)
class WsConfig:
  tile_m: int = 128
  tile_n: int = 128  # per compute warpgroup
  tile_k: int = 64
  stages: int = 4
  grid_minor_dim: common.MatmulDimension = common.MatmulDimension.N
  grid_tile_width: int = 1


def ragged_dot_ws(lhs, rhs, group_sizes, config: WsConfig, swiglu=False,
                  out_dtype=jnp.bfloat16):
  m, k = lhs.shape
  g, k2, n_in = rhs.shape
  assert k == k2, (lhs.shape, rhs.shape)
  tile_m, tile_n, tile_k = config.tile_m, config.tile_n, config.tile_k
  n_out = n_in // 2 if swiglu else n_in
  cta_n = 2 * tile_n
  if n_out % cta_n or k % tile_k or m < tile_m:
    raise ValueError(f"bad tiling {config} for {m=} {k=} {n_out=}")
  m_iters = pl.cdiv(m, tile_m) + g - 1
  n_iters = n_out // cta_n
  k_iters = k // tile_k
  n_rhs = 2 if swiglu else 1

  in_transforms = mgpu_utils.tile_swizzle_transforms((tile_k,), lhs.dtype)
  rhs_transforms = mgpu_utils.tile_swizzle_transforms((cta_n,), rhs.dtype)
  lhs_spec = plgpu.BlockSpec(
      (tile_m, tile_k), lambda kk: (0, kk), transforms=in_transforms,
      delay_release=1)
  rhs_spec = plgpu.BlockSpec(
      (tile_k, cta_n), lambda kk: (kk, 0), transforms=rhs_transforms,
      delay_release=1)
  in_specs = [lhs_spec] + [rhs_spec] * n_rhs
  out_smem_type = mgpu_utils.tiled_swizzled_smem(
      (2, tile_m, tile_n), out_dtype, tiling_prefix=(1,), what="out")

  def get_pipeline(body, ctx):
    return plgpu.emit_pipeline_warp_specialized(
        body,
        grid=(k_iters,),
        memory_registers=40,
        in_specs=in_specs,
        wg_axis="wg",
        num_compute_wgs=2,
        max_concurrent_steps=config.stages,
        compute_context=ctx,
    )

  def kernel(lhs_gmem, rhs_gmem, gid_gmem, sw_gmem, size_gmem, bstart_gmem,
             o_gmem):
    ignore = lambda *_, **__: None
    allocs = get_pipeline(ignore, ignore).get_allocations(
        lhs_gmem.at[pl.ds(0, tile_m)],
        *[rhs_gmem.at[0, :, pl.ds(0, cta_n)]] * n_rhs)

    @functools.partial(pl.run_scoped, allocs=allocs, out_smem=out_smem_type,
                       collective_axes="wg")
    def _scope(allocs, out_smem):
      wg = lax.axis_index("wg")
      wg_n = pl.ds(wg * tile_n, tile_n)

      @plgpu.nd_loop((m_iters * n_iters,), collective_axes="sm")
      def _loop(info):
        mi, ni = plgpu.planar_snake(
            info.index[0], (m_iters, n_iters), config.grid_minor_dim,
            config.grid_tile_width)
        gi = gid_gmem[mi]
        start_within_block = sw_gmem[mi]
        actual_size = size_gmem[mi]
        block_start = pl.multiple_of(bstart_gmem[mi], 8)

        @pl.when(actual_size > 0)
        def _work():
          def ctx(eval_pipeline):
            def epilogue(out):
              o = out_smem.at[wg]
              o[...] = out.astype(o.dtype)
              plgpu.commit_smem()
              common.copy_ragged_smem_to_gmem(
                  o, o_gmem, block_start=block_start,
                  start_within_block=start_within_block,
                  actual_size=actual_size,
                  ns=pl.ds(ni * cta_n + wg * tile_n, tile_n),
                  max_rows=tile_m)

            if swiglu:
              @functools.partial(
                  pl.run_scoped, acc_g=plgpu.ACC((tile_m, tile_n)),
                  acc_u=plgpu.ACC((tile_m, tile_n)))
              def _accs(acc_g, acc_u):
                eval_pipeline((acc_g, acc_u))
                gate = acc_g[...]
                up = acc_u[...]
                epilogue(gate * (1.0 / (1.0 + jnp.exp(-gate))) * up)
            else:
              @functools.partial(pl.run_scoped,
                                 acc=plgpu.ACC((tile_m, tile_n)))
              def _acc(acc):
                eval_pipeline((acc,))
                epilogue(acc[...])

          def mma_body(_, lhs_smem, *rest):
            *rhs_smems, accs = rest
            for acc, rhs_smem in zip(accs, rhs_smems):
              plgpu.wgmma(acc, lhs_smem, rhs_smem.at[:, wg_n])
            plgpu.wgmma_wait(n_rhs)
            return accs

          col0 = ni * cta_n
          rhs_refs = [rhs_gmem.at[gi, :, pl.ds(col0, cta_n)]]
          if swiglu:
            rhs_refs.append(rhs_gmem.at[gi, :, pl.ds(n_out + col0, cta_n)])
          get_pipeline(mma_body, ctx)(
              lhs_gmem.at[pl.ds(block_start, tile_m)], *rhs_refs,
              allocations=allocs)

  info = common.GroupInfo.create_aligned(group_sizes, tile_m, m_iters, 8)
  return plgpu.kernel(
      kernel,
      out_type=jax.ShapeDtypeStruct((m, n_out), out_dtype),
      grid=(backend.get_default_device().core_count,),
      grid_names=("sm",),
      num_threads=3,
      thread_name="wg",
      kernel_name="ragged_dot_ws_swiglu" if swiglu else "ragged_dot_ws",
      compiler_params=plgpu.CompilerParams(
          approx_math=True,
          lowering_semantics=plgpu.LoweringSemantics.Warpgroup),
  )(lhs, rhs, info.group_id, info.start_within_block, info.actual_size,
    info.block_start)
