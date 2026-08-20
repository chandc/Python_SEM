"""End-to-end: what does mode-parallelism buy on a REAL stage solve?

    uv run --quiet python scratch/prof3d_endtoend.py [nz]

The 6.7x in prof3d_procs.py is a single-matvec microbenchmark.  A stage solve is
a whole PCG, where two extra effects appear and pull in OPPOSITE directions:

  against -- per-chunk setup (multiplicity_weight rebuilds per chunk) and any
             load imbalance between chunks;
  for     -- serial pcg runs EVERY mode until the WORST mode converges, and
             chunked solves drop that wait.

Only the end-to-end number decides, so measure iterations as well as time: a
speedup that comes from doing less work is worth distinguishing from one that
comes from using more cores, and the iteration columns separate them.
"""
import os, sys, time
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '1'
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem3d import operator as OP, solver3d as S3, bc as BC, fourier as FR
from lssem3d import parallel as PAR

RE, EX, N = 1000.0, 6, 10
NU, C, KAP = 1.0/RE, 2500.0, 2500.0
TOL = 1e-8


def main(nz=64):
    n = N+1
    mesh = build_channel(1.0, 1.0, EX, EX, N, bcs=(1, 1, 1, 2))
    D = diff_matrix(N)
    nk = nz//2 + 1
    kz = FR.wavenumbers(nz, 2*np.pi)
    mask = BC.build_mask(mesh, nk, pin_p=True)
    rng = np.random.default_rng(0)
    U = rng.standard_normal((mesh.nelem, n, n, OP.NVAR_R, nk))*mask
    b = S3.gs(mesh, U)*mask
    kw = dict(mesh=mesh, mask=mask, wq=mesh.wq, kap=KAP)

    Minv = S3.jacobi_inverse(S3.jacobi_diagonal(b.shape, D, mesh.facx,
                                                mesh.facy, kz, NU, C, mesh,
                                                mask, mesh.wq, KAP), mask)

    t0 = time.perf_counter()
    xs, it_s, _ = S3.pcg(b, D, mesh.facx, mesh.facy, kz, NU, C, mesh, mask,
                         Minv, TOL, 4000, None, mesh.wq, KAP)
    t_ser = time.perf_counter()-t0

    print(f'Nz={nz}  modes={nk}  ({EX}x{EX} elements, N={N}, tol={TOL:g})')
    print(f'  serial pcg: {t_ser:6.2f} s   {it_s} iters '
          f'(every mode runs until the worst one converges)\n')
    print(f"{'workers':>8}{'time s':>9}{'speedup':>9}{'max iters':>11}"
          f"{'vs serial':>11}{'max|dx|':>11}")
    for w in [x for x in (2, 4, 6, 8, 12, 16) if x <= nk]:
        t0 = time.perf_counter()
        xp, it_p, _ = PAR.pcg(b, D, mesh.facx, mesh.facy, kz, NU, C,
                              M_inv=Minv, tol=TOL, max_iter=4000, workers=w,
                              **kw)
        t = time.perf_counter()-t0
        dx = np.abs(xp-xs).max()/max(np.abs(xs).max(), 1e-30)
        print(f'{w:>8}{t:>9.2f}{t_ser/t:>8.2f}x{it_p:>11}'
              f'{it_p/it_s:>10.2f}x{dx:>11.2e}')
    print('\n  max|dx| is the relative difference from the serial solution;')
    print(f'  it should sit near the solver tolerance {TOL:g}, not at zero.')
    PAR.shutdown()


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 64)
