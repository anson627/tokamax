"""Per-kernel config sweep: fused gate/up and down projections separately."""
import sys, time, itertools
sys.path.insert(0, "/dev_ws")
import jax, jax.numpy as jnp, numpy as np
from moe_ffn import kernel_ws, kernel_sm90, common

MD = common.MatmulDimension
STARTER = common.Config(block_m=64, block_n=64, block_k=128, num_stages=2,
    persistent=False, grid_minor_dim=MD.M, grid_tile_width=1)

def timeit(f, *a):
  f(*a).block_until_ready()
  t0 = time.perf_counter(); f(*a).block_until_ready(); one = time.perf_counter() - t0
  n = max(3, min(200, int(0.2 / max(one, 1e-5))))
  t0 = time.perf_counter()
  for _ in range(n): o = f(*a)
  o.block_until_ready()
  return (time.perf_counter() - t0) / n

models = {"qwen": dict(E=128, H=2048, I=768), "ds": dict(E=64, H=7168, I=2048)}
mdl = models[sys.argv[1]]
Ms = [int(v) for v in sys.argv[2].split(",")]
E, H, I = mdl["E"], mdl["H"], mdl["I"]
k = jax.random.key(0)
wgu = (jax.random.normal(k, (E, H, 2 * I), jnp.float32) / np.sqrt(H)).astype(jnp.bfloat16)
wd = (jax.random.normal(k, (E, I, H), jnp.float32) / np.sqrt(I)).astype(jnp.bfloat16)
cfgs = [kernel_ws.WsConfig(tm, tn, tk, st, gmd, gw) for tm, tn, tk, st, gmd, gw in itertools.product(
    (64, 128), (64, 128), (64,), (2, 3, 4, 5), (MD.M, MD.N), (1, 4, 8))]
for M in Ms:
  rng = np.random.default_rng(M)
  for routing in sys.argv[3].split(",") if len(sys.argv) > 3 else ["balanced", "skewed"]:
    p = np.ones(E) if routing == "balanced" else (1 + rng.permutation(E)) ** -1.2
    gs = jnp.asarray(rng.multinomial(M, p / p.sum()).astype(np.int32))
    x = jax.random.normal(k, (M, H), jnp.float32).astype(jnp.bfloat16)
    a = jax.random.normal(k, (M, I), jnp.float32).astype(jnp.bfloat16)
    s1 = timeit(jax.jit(lambda x, w, g: kernel_sm90.ragged_dot_kernel(x, w, g, jnp.bfloat16, STARTER)), x, wgu, gs)
    s2 = timeit(jax.jit(lambda x, w, g: kernel_sm90.ragged_dot_kernel(x, w, g, jnp.bfloat16, STARTER)), a, wd, gs)
    print(f"== {sys.argv[1]} M={M} {routing} starter gate_up {s1*1e6:.0f}us down {s2*1e6:.0f}us", flush=True)
    for name, lhs, rhs, sw, f_fl in (("gu", x, wgu, True, 4 * M * H * I), ("dn", a, wd, False, 2 * M * H * I)):
      res = []
      for c in cfgs:
        try:
          f = jax.jit(lambda l, r, g, c=c, sw=sw: kernel_ws.ragged_dot_ws(l, r, g, c, swiglu=sw))
          t = timeit(f, lhs, rhs, gs)
          res.append((t, c))
        except Exception as ex:
          if "shared memory" not in str(ex) and "bad tiling" not in str(ex):
            print("   FAIL", c, type(ex).__name__, str(ex)[:200], flush=True)
      res.sort(key=lambda r: r[0])
      for t, c in res[:4]:
        print(f"  {name} {c.tile_m}x{c.tile_n}x{c.tile_k} s{c.stages} {c.grid_minor_dim.name}{c.grid_tile_width}: "
              f"{t*1e6:.0f}us {f_fl/t/1e12:.0f}TF/s", flush=True)
