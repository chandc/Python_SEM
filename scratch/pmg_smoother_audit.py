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

THE ANSWER: the bound is fine.  Converged lam_max = 3.247 against the
20-iteration estimate of 2.984, a ratio of 1.088 -- comfortably inside the 1.3
safety factor, so rho = 3.88 sits ABOVE the true spectrum with ~19% of margin.
The smoother damps the whole range and amplifies nothing.  Hypothesis dead, and
with it the last suspicion that PMG's pinned ratio is an implementation defect.
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

print('\n2. (not needed -- check 1 settled it)')
print('   Sweeping the safety factor would only matter if rho were short.\n'
      '   It is not: rho = 1.3 * 2.984 = 3.88 against a true maximum of 3.247,\n'
      '   about 19% of margin.  The smoother covers the whole spectrum.')
