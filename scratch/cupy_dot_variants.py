"""_dot is 17x off bandwidth and two-thirds of a CG iteration.  Fix it.

The reduction keeps the mode axis and sums the other four, so it produces
only nk (=65) outputs from ~19 M inputs.  At one block per output that is 65
blocks on a 108-SM A100 -- most of the card idle -- and `a*b*w` materialises
two full-size temporaries before the reduction even starts.

Four candidates, checked for agreement, then timed.
"""
import sys, time
sys.path.insert(0, '.')
import numpy as np, cupy as cp
import lssem3d; lssem3d.set_backend('cupy')
from lssem2d.mesh import build_channel
from lssem3d import operator as OP, solver3d as S3, bc as BC

ex, ey, nz, N = (int(v) for v in (sys.argv[1:5] or [16, 16, 128, 8]))
L = 2*np.pi
m = build_channel(L, L, ex, ey, N, bcs=(0, 0, 0, 0))
m.periodic_x = L; m.periodic_y = L; m.compute_global_indices()
nk = nz//2 + 1
rng = np.random.default_rng(0)
shape = (m.nelem, N+1, N+1, OP.NVAR_R, nk)
a = cp.asarray(rng.standard_normal(shape))
b = cp.asarray(rng.standard_normal(shape))
w = cp.asarray(np.abs(rng.standard_normal(shape)) + 0.5)
M = int(np.prod(shape[:-1]))
print(f'{a.size/1e6:.2f} M elements -> {nk} outputs '
      f'(reduction ratio {M:,}:1)\n')

def cur(a, b, w):
    return (a*b*w).sum(axis=(0, 1, 2, 3))[None, None, None, None, :]

def resh(a, b, w):
    return (a*b*w).reshape(M, nk).sum(axis=0)[None, None, None, None, :]

ONES = cp.ones((1, M))
def gemv(a, b, w):
    # cuBLAS turns a low-output-count reduction into a GEMM, which is the
    # one kernel on this card guaranteed to use every SM.
    return (ONES @ (a*b*w).reshape(M, nk))[None, None, None, :]

_FUSED = cp.ReductionKernel(
    'float64 x, float64 y, float64 z', 'float64 out',
    'x*y*z', 'a + b', 'out = a', '0', 'lssem_dot3')
def fused(a, b, w):
    # No temporaries at all: the product is computed inside the reduction.
    return _FUSED(a.reshape(M, nk), b.reshape(M, nk), w.reshape(M, nk),
                  axis=0)[None, None, None, None, :]

FUSED_ONES = cp.ones((1, M))
def fused_gemv(a, b, w):
    prod = cp.empty((M, nk))
    cp.multiply(a.reshape(M, nk), b.reshape(M, nk), out=prod)
    prod *= w.reshape(M, nk)
    return (FUSED_ONES @ prod)[None, None, None, :]

ref = cur(a, b, w).ravel()
for name, fn in (('current  sum(axis=0..3)', cur), ('reshape + sum(axis=0)', resh),
                 ('ones @ X   (cuBLAS)', gemv), ('ReductionKernel fused', fused),
                 ('multiply-out + GEMV', fused_gemv)):
    got = fn(a, b, w).ravel()
    err = float(cp.abs(got - ref).max()/cp.abs(ref).max())
    for _ in range(3):
        fn(a, b, w)
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        fn(a, b, w)
    cp.cuda.Stream.null.synchronize()
    ms = (time.perf_counter()-t0)*50
    print(f'  {name:<26} {ms:7.2f} ms   rel err {err:.2e}')
print(f'\n  bandwidth bound (3 reads, no temporaries): '
      f'{3*a.nbytes/1356e9*1e3:.2f} ms')
print(f'  current cost of 2 dots per CG iteration: '
      f'{2*cur.__name__ and 2*0:.0f}', end='')
print(f'\n  a CG iteration is matvec 8.5 + 2 dots + vec 1.7 ms')
