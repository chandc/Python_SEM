"""Where does a CG iteration's GPU time actually go, at production size?

    python scratch/cupy_matvec_profile.py [ex ey nz N]

A CG iteration on the Re = 1600 128^3 case measures 30.6 ms, against ~13 ms
predicted from bandwidth.  Predictions have now been wrong four times, so
this times each PIECE with an explicit synchronise, and prints alongside each
one the time it WOULD take at the card's measured streaming bandwidth.  A
piece far above its bandwidth bound is doing something other than streaming
memory -- which is the thing worth finding.
"""
import sys, time
sys.path.insert(0, '.')
import numpy as np, cupy as cp
import lssem3d; lssem3d.set_backend('cupy')
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem3d import operator as OP, solver3d as S3, bc as BC, fourier as FR, deriv as DV
from lssem3d.kernels_cupy import _L0

ex, ey, nz, N = (int(v) for v in (sys.argv[1:5] or [16, 16, 128, 8]))
L = 2*np.pi
BW = 1356e9      # measured A100 fp64 triad, cell 2

m = build_channel(L, L, ex, ey, N, bcs=(0, 0, 0, 0))
m.periodic_x = L; m.periodic_y = L; m.compute_global_indices()
nk = nz//2 + 1
mask = cp.asarray(BC.build_mask(m, nk, pin_p=False, nz=nz))
D, fx, fy = (cp.asarray(diff_matrix(N)), cp.asarray(m.facx), cp.asarray(m.facy))
kz, wq = cp.asarray(FR.wavenumbers(nz, L)), cp.asarray(m.wq)
U = cp.asarray(np.random.default_rng(0).standard_normal(
    (m.nelem, N+1, N+1, OP.NVAR_R, nk)))
nu, c = 6.25e-4, 1.0/0.0039
rw = cp.asarray(OP.momentum_row_weights(c))
GB = U.nbytes/2**30
print(f'{ex}x{ey} N={N} Nz={nz}: {U.size/1e6:.2f} M dof, '
      f'one field array = {U.nbytes/2**20:.0f} MiB\n')

def t(label, fn, passes):
    """passes = how many array-sized reads+writes the step must do at minimum"""
    for _ in range(3):
        r = fn()
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        r = fn()
    cp.cuda.Stream.null.synchronize()
    ms = (time.perf_counter()-t0)*100
    bound = passes*U.nbytes/BW*1e3
    flag = '  <<<' if ms > 3*bound else ''
    print(f'  {label:<26} {ms:7.2f} ms   (bandwidth bound {bound:5.2f}){flag}')
    return r, ms

print('normal_op internals')
Uc, _ = t('to_complex', lambda: OP.to_complex(U), 2)
Ux, a = t('ddx  (7 fields)', lambda: DV.ddx(Uc, D, fx), 2)
Uy, b = t('ddy  (7 fields)', lambda: DV.ddy(Uc, D, fy), 2)
R,  d = t('_L0 fused (8 rows)', lambda: _L0(Uc, D, fx, fy, kz, nu, c, 0.0), 5)
Rx, e = t('ddxT (8 rows)', lambda: DV.ddxT(R, D, fx), 2.3)
Ry, f = t('ddyT (8 rows)', lambda: DV.ddyT(R, D, fy), 2.3)
_,  g = t('gather-scatter', lambda: S3.gs(m, U), 3)
_,  h = t('mask multiply', lambda: U*mask, 3)
tot, mv = t('FULL normal_op', lambda: S3.normal_op(
    U, D, fx, fy, kz, nu, c, m, mask, wq, 0.0, rw), 20)
print(f'\n  sum of derivative calls alone: {a+b+e+f:.2f} ms '
      f'({100*(a+b+e+f)/mv:.0f}% of the matvec)')
print(f'  a full matvec at bandwidth would be ~{20*U.nbytes/BW*1e3:.1f} ms')
print(f'\nPCG per-iteration extras (on top of one matvec)')
p_ = cp.empty_like(U)
_, v1 = t('axpy  x + a*p', lambda: U + 1.5*p_, 3)
_, v2 = t('dot   (reduction)', lambda: float(cp.sum(U*p_)), 2)
print(f'\n  matvec {mv:.1f} + ~4 axpy {4*v1:.1f} + 3 dot {3*v2:.1f} '
      f'= {mv+4*v1+3*v2:.1f} ms   (measured iteration: 30.6 ms)')
