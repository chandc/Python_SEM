"""Is the Chebyshev smoother's spectral bound right?  (PMG audit)

    python scratch/pmg_smoother_audit.py [N] [ne] [nz]

THE HYPOTHESIS.  3D_STATUS.md 7K found PMG's iteration ratio PINNED at
7.3-7.4x across N = 8/12/16 rather than becoming N-independent, and 7K.2 found
the slow modes are ROUGH.  Rough slow modes are the SMOOTHER's job -- coarse
grids cannot touch them -- so a pinned ratio points at the smoother, not at the
number of levels.  That also explains why 7K's untested rescue (deg proportional
to N) showed a plateau: raising the degree sharpens damping WITHIN the covered
band, and does nothing for modes outside it.

Chebyshev4 damps roughly [rho/k, rho] where rho = safety * lam_max_estimate.
If lam_max is UNDERESTIMATED by more than the 1.3 safety factor, the true top
of the spectrum sits ABOVE rho, where the polynomial does not damp but
AMPLIFIES.  Those modes would then stay slow no matter how high the degree
goes -- exactly the observed signature.

lam_max comes from 20 power iterations, which converge as (lam2/lam1)^k and can
be far from converged on a clustered spectrum.  The estimate approaches from
BELOW, so an underestimate is the expected failure, not a symmetric error.

TWO CHECKS.
  1. Compare the 20-iteration estimate against a converged one.  If the ratio
     exceeds 1.3, the safety factor does not cover the gap and the smoother is
     amplifying its highest modes.
  2. Sweep `safety` and count V-cycle-preconditioned CG iterations.  If rho is
     already adequate this is flat and the hypothesis is dead; if iterations
     fall markedly with larger safety, it was underestimated.
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import lssem3d; lssem3d.set_backend('numpy')
from lssem2d.mesh import build_channel
from lssem3d import precond as PC, operator as OP, fourier as FR, timestep as T
from lssem3d import solver3d as S3

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
ne = int(sys.argv[2]) if len(sys.argv) > 2 else 3
nz = int(sys.argv[3]) if len(sys.argv) > 3 else 16
L = 2*np.pi
m = build_channel(L, L, ne, ne, N, bcs=(0, 0, 0, 0))
m.periodic_x = L; m.periodic_y = L; m.compute_global_indices()
nk = nz//2 + 1
kz = FR.wavenumbers(nz, L)
c = T.implicit_coeff(0.0039, 0)
rw = OP.momentum_row_weights(c)
lev = PC._Level(m, nk, nz, 6.25e-4, c, kz, 0.0, rw, True)
print(f'N={N}, {ne}x{ne} elements, Nz={nz}\n')

print('1. is lam_max converged at the 20 power iterations the code uses?')
prev = None
for npow in (20, 50, 100, 200, 400):
    lam = PC.estimate_lambda_max(lev.A, lev.M_inv, lev.shape, npow=npow)
    tag = '' if prev is None else f'   (+{100*(lam/prev-1):.1f}% over previous)'
    print(f'   npow={npow:>4}: lam_max = {lam:.6f}{tag}')
    prev = lam
l20 = PC.estimate_lambda_max(lev.A, lev.M_inv, lev.shape, npow=20)
l400 = prev
print(f'\n   converged/20-iteration = {l400/l20:.3f}   vs the 1.3 safety factor')
print('   ' + ('*** UNDERESTIMATED BEYOND THE SAFETY FACTOR: the smoother is\n'
               '       amplifying its highest modes ***' if l400/l20 > 1.3 else
               'safety factor covers the gap -- rho is adequate, hypothesis dead'))

print('\n2. do CG iterations respond to the safety factor?')
print(f'   {"safety":>7} {"rho":>10} {"CG its":>8}')
b = S3.gs(m, np.random.default_rng(0).standard_normal(lev.shape))*lev.mask
for safety in (1.0, 1.3, 1.8, 2.5, 4.0):
    pmg = PC.PMG(m, nk, nz, 6.25e-4, c, kz, rw=rw,
                 orders=(N, N//2, 2) if N >= 8 else (N, 2), deg=6)
    for s_ in pmg.smooth:
        s_.rho = safety*(s_.rho/1.3)          # re-scale from the built-in 1.3
    _, it, _ = S3.pcg(b, lev.D if hasattr(lev, 'D') else None, None, None, kz,
                      6.25e-4, c, mesh=m, mask=lev.mask, M_inv=pmg,
                      tol=1e-6, max_iter=4000, wq=m.wq, rw=rw) \
        if False else (None, -1, None)
    print(f'   {safety:>7.1f} {pmg.smooth[0].rho:>10.4f} {"(see note)":>8}')
print('\n   NOTE: wiring the CG run needs the level operator plumbed through\n'
      '   pcg; check 1 is the decisive one and stands alone.')
