"""How does w_con interact with a 3-level p-multigrid preconditioner?

    uv run --quiet python scratch/fosls_wcon_pmg.py

F5 measured w_con against JACOBI and found two competing optima: conditioning
minimised at w_con = 1 (c2/c1 = 1.56e4, 1737 CG its), divergence improving
monotonically above it.  The open question is whether a real preconditioner
changes that trade -- if p-multigrid absorbs the w_con^2 conditioning penalty,
then up-weighting continuity becomes cheap and the divergence gain is free.

HIERARCHY.  Three levels, N -> 4 -> 2, with a DIRECT solve at the coarsest --
what solver_pmg2.f90 runs (p = 10 -> 4 -> 2) and what PMG_ALGORITHM sec 8
recommends.  PMG2 nests: pc=(4, 2) makes the coarse solver itself a V-cycle,
which stays a fixed linear operator and so keeps CG's symmetry requirement.

Compared against the 2-level (N -> 2) and plain Jacobi, so the question
"does depth help, and does it help MORE as w_con rises" is answerable.

Steady (dt=1e4), w_mom=1, nu=1/100, pin_p=True -- the F1 regime.
NUMPY BACKEND ONLY: w_con != 1 raises on numba by design.
"""
import os
import sys
import time

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, 'scratch'))
os.chdir(_R)

import numpy as np
import lssem2d
lssem2d.set_backend('numpy')

import fosls_ellipticity as FE
from lssem2d import precond as P, solver as S
from lssem2d.assembly import gather_scatter
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState
from lssem2d.mesh import build_channel
from lssem2d.solver import apply_A

OUT = os.path.join(_R, 'scratch', 'fosls_wcon_pmg.npz')
WCONS = (0.2, 0.5, 1.0, 2.0, 5.0, 10.0)
N, EX, EY = 8, 4, 4


def case(w_con):
    m = build_channel(2.0, 1.0, EX, EY, N, bcs=(1, 1, 1, 2))
    m.compute_global_indices()
    st = SolverState(m, diff_matrix(N), nu=1/100., dt=1e4, fac1=1.0,
                     w_mom=1.0, w_con=w_con)
    n = N + 1
    z = np.zeros((m.nelem, n, n))
    st.update_linearisation(z, z.copy())
    return m, st, z


def count(st, m, z, pre, seed=0, tol=1e-8, maxit=60000):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((m.nelem, N+1, N+1, 4))
    b = apply_A(st, x, z, z.copy(), pin_p=True)
    mult = gather_scatter(m, np.ones((m.nelem, N+1, N+1, 4)))
    mw = 1.0/np.where(mult < 1e-10, 1.0, mult)
    M_inv = S.compute_jacobi(st, z, z.copy(), pin_p=True)
    t0 = time.perf_counter()
    out = S.pcg_solve(st, b, z, z.copy(), M_inv, mw, pin_p=True,
                      max_iter=maxit, tol=tol, cgsfac=0.0, precond=pre)
    t = time.perf_counter() - t0
    it = out[1] if isinstance(out, tuple) and len(out) > 1 else -1
    return (it[0] if isinstance(it, (list, tuple)) else it), t


def main():
    print(f'w_con vs preconditioner   N={N}, {EX}x{EY} elements, steady, '
          f'pin_p=True, tol=1e-8\n')
    print(f'{"w_con":>6} {"c2/c1":>11} {"sqrt":>7} | {"jacobi":>8} '
          f'{"pmg 2lvl":>9} {"pmg 3lvl":>9} | {"2lvl x":>7} {"3lvl x":>7}')
    rows = []
    for w in WCONS:
        c1, c2, _ = FE.ellipticity(6, 3, 3, dt=1e4, w_mom=1.0, w_con=w)
        m, st, z = case(w)
        M_inv = S.compute_jacobi(st, z, z.copy(), pin_p=True)
        ij, tj = count(st, m, z, None)
        p2 = P.make('pmg2', st, z, z.copy(), M_inv, True, pc=2, deg=4,
                    coarse_solver='direct')
        i2, t2 = count(st, m, z, p2)
        p3 = P.make('pmg2', st, z, z.copy(), M_inv, True, pc=(4, 2), deg=4,
                    coarse_solver='direct')
        i3, t3 = count(st, m, z, p3)
        print(f'{w:6.2f} {c2/c1:11.3e} {np.sqrt(c2/c1):7.0f} | {ij:8d} '
              f'{i2:9d} {i3:9d} | {ij/max(i2,1):6.1f}x {ij/max(i3,1):6.1f}x')
        rows.append((w, c2/c1, ij, i2, i3, tj, t2, t3))
    np.savez_compressed(OUT, rows=np.array(rows),
                        cols=['w_con', 'c2_over_c1', 'jacobi', 'pmg2', 'pmg3',
                              't_jac', 't_pmg2', 't_pmg3'])
    r = np.array(rows)
    print(f'\n  Jacobi growth 1 -> 10 : {r[-1,2]/r[2,2]:.1f}x')
    print(f'  pmg 2-level          : {r[-1,3]/r[2,3]:.1f}x')
    print(f'  pmg 3-level          : {r[-1,4]/r[2,4]:.1f}x')
    print('\n  If the multigrid rows grow MORE SLOWLY than Jacobi, the w_con^2')
    print('  conditioning penalty is partly absorbed and up-weighting gets cheaper.')
    print(f'\nsaved -> {OUT}')


if __name__ == '__main__':
    main()
