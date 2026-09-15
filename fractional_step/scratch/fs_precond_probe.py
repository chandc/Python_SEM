"""Why does the channel pressure Poisson need ~4000 CG iterations per substage?

    python scratch/fs_precond_probe.py

The Re_tau = 180 minimal channel priced at 12030 CG/step on the fractional-step
path, against 780 on periodic TGV.  Velocity solves take ~6 per substage, so
essentially all of it is the PRESSURE Poisson.

The suspicion is the preconditioner, not the operator.  fdm_preconditioner is
ONE-LEVEL additive Schwarz -- an exact element-local inverse plus
gather-scatter, with NO COARSE GRID.  For Poisson the slow modes are global and
smooth, which a one-level Schwarz method cannot touch, so its iteration count
grows with the domain-to-element ratio.  That is textbook, and it is exactly
the case where multigrid works: sec 7K closed p-MG for the LEAST-SQUARES
operator because its slow modes are ROUGH, and Poisson's are not.

Checks, in order: how much the FDM element solve is worth over a plain
diagonal, whether the analytic diagonal matches the probed one, and how the
count scales with element count at fixed polynomial order -- which is the
signature that distinguishes a missing coarse grid from a hard operator.
"""
import sys, time
sys.path.insert(0, '.'); sys.path.insert(0, 'scratch')
import numpy as np
import lssem3d; lssem3d.set_backend('numpy')
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem3d import helmholtz as HH, solver3d as S3, fourier as FR, project as PJ

L = 2*np.pi
N, NZ = 8, 8
nk = NZ//2 + 1


def case(ex, ey):
    m = build_channel(np.pi, 2.0, ex, ey, N, bcs=(0, 0, 1, 1))
    m.periodic_x = np.pi
    m.compute_global_indices()
    mask = PJ.build_masks(m, nk, NZ, 1, wall=False)
    ind = np.zeros(mask.shape); ind[0, 0, 0, 0, 0] = 1.0
    mask[..., 0, 0] *= (S3.gs(m, ind)[..., 0, 0] < 0.5)
    kz = FR.wavenumbers(NZ, 0.34*np.pi)
    return m, diff_matrix(N), mask, kz


print('1. is the analytic diagonal the same as the probed one?')
m, D, mask, kz = case(3, 6)
shape = mask.shape
dp = HH.jacobi_diagonal(shape, D, m.facx, m.facy, m.wq, kz**2, 1.0, m, mask)
da = HH.jacobi_diagonal_analytic(m, N, m.wq, kz**2, 1.0, 2, nk, mask)
print(f'   max rel diff {np.abs(dp-da).max()/np.abs(dp).max():.3e}\n')

print('2. Poisson CG iterations vs ELEMENT COUNT at fixed N')
print('   (a missing coarse grid shows up as growth here)')
print(f'   {"elements":>10} {"dof":>9} {"Jacobi":>8} {"FDM":>8} {"ratio":>7}')
for ex, ey in ((2, 4), (3, 6), (4, 8), (6, 12)):
    m, D, mask, kz = case(ex, ey)
    shape = mask.shape
    rng = np.random.default_rng(0)
    b = S3.gs(m, rng.standard_normal(shape))*mask
    da = HH.jacobi_diagonal_analytic(m, N, m.wq, kz**2, 1.0, 2, nk, mask)
    Mj = HH.jacobi_inverse(da, mask)
    Mjf = lambda r: r*Mj
    Mf = HH.fdm_preconditioner(m, N, kz**2, 1.0, mask, 2, nk)
    out = []
    for M in (Mjf, Mf):
        _, it, _ = HH.solve(b, D, m.facx, m.facy, m.wq, kz**2, 1.0, m, mask, M,
                            tol=1e-8, max_iter=20000, check_every=1)
        out.append(it)
    print(f'   {ex}x{ey:<7} {b.size:>9,} {out[0]:>8} {out[1]:>8} '
          f'{out[0]/max(out[1],1):>7.1f}')
print('\n   FDM growing with element count = the missing coarse grid.')
