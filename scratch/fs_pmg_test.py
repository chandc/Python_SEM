"""Does a coarse grid fix the channel Poisson?

    python scratch/fs_pmg_test.py

One-level Schwarz (FDM) and a plain diagonal both grow like sqrt(elements) on
this operator -- 254 -> 734 as elements go 8 -> 72 -- because the slow modes are
global.  A V-cycle should FLATTEN that: the whole point of a coarse grid is
that the iteration count stops caring how many elements there are.

Read the SCALING, not the single number.  A constant-factor win is worth little
against a V-cycle's cost (roughly 2*deg + 2 matvecs per level); a flat curve is
worth a lot, and is the difference between this being usable at production
element counts and not.
"""
import sys, time
sys.path.insert(0, '.'); sys.path.insert(0, 'scratch')
import numpy as np
import lssem3d; lssem3d.set_backend('numpy')
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem3d import helmholtz as HH, hpmg, solver3d as S3, fourier as FR
from lssem3d import project as PJ

N, NZ = 8, 8
nk = NZ//2 + 1


def case(ex, ey):
    m = build_channel(np.pi, 2.0, ex, ey, N, bcs=(0, 0, 1, 1))
    m.periodic_x = np.pi
    m.compute_global_indices()
    mask = PJ.build_masks(m, nk, NZ, 1, wall=False)
    ind = np.zeros(mask.shape); ind[0, 0, 0, 0, 0] = 1.0
    mask[..., 0, 0] *= (S3.gs(m, ind)[..., 0, 0] < 0.5)
    return m, diff_matrix(N), mask, FR.wavenumbers(NZ, 0.34*np.pi)


print('channel pressure Poisson, N=8, Nz=8')
print(f'{"elements":>9} {"Jacobi":>8} {"FDM":>7} {"PMG":>7} {"setup":>8} '
      f'{"PMG cycles":>11}')
for ex, ey in ((2, 4), (3, 6), (4, 8), (6, 12)):
    m, D, mask, kz = case(ex, ey)
    rng = np.random.default_rng(0)
    b = S3.gs(m, rng.standard_normal(mask.shape))*mask
    dj = HH.jacobi_diagonal_analytic(m, N, m.wq, kz**2, 1.0, 2, nk, mask)
    Mj = HH.jacobi_inverse(dj, mask)
    cands = [('jac', lambda r, Mj=Mj: r*Mj),
             ('fdm', HH.fdm_preconditioner(m, N, kz**2, 1.0, mask, 2, nk))]
    t0 = time.perf_counter()
    P = hpmg.HelmholtzPMG(m, N, kz**2, 1.0, 1, nk, NZ, wall=False,
                          pin_kz0=True, deg=6)
    setup = time.perf_counter() - t0
    cands.append(('pmg', P))
    out = []
    for _, M in cands:
        _, it, _ = HH.solve(b, D, m.facx, m.facy, m.wq, kz**2, 1.0, m, mask, M,
                            tol=1e-8, max_iter=20000, check_every=1)
        out.append(it)
    print(f'{ex}x{ey:<6} {out[0]:>8} {out[1]:>7} {out[2]:>7} {setup:>7.1f}s '
          f'{out[2]:>11}', flush=True)
print('\n  flat PMG column = the coarse grid is doing its job;')
print('  growing = it is not, and the V-cycle cost is wasted.')
