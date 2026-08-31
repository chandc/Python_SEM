"""Does an EXACT coarse solve beat the degree-10 polynomial in PMG2?

    uv run --quiet python scratch/pmg_direct_coarse.py

THE HYPOTHESIS.  PMG2 uses a Chebyshev polynomial on the fine level AND a
Chebyshev polynomial on the coarse level, so nothing in the cycle actually
INVERTS the soft direction.  Polynomial smoothers damp a spectral band; the soft
end is what the coarse solve is for -- and the soft outflow pressure mode
(precond.py: "~8e3x softer than a generic direction") is why PMG2 exists.

TEST PROBLEM: plane Poiseuille with the DONG outlet, the shape
dong_obc_test.py's stage0 uses -- [0, L] x [0, 1], bcs = (3, 6, 1, 1), Re = 100.
Chosen as the GATE because it has an exact solution AND, per dong_seeded.py,
"exact Poiseuille zeroes the Dong rows too (p = 0, du/dx = 0, v = 0 at the
exit)".  So the right answer is known and the outflow rows are genuinely
exercised.  Gartling Re = 800 is the follow-on, where the soft mode actually
bites and there is a published reattachment length to check against.

WHAT IS MEASURED.  A single linear solve, b = A x_rand, same b for every
preconditioner, counting CG iterations to a fixed tolerance.  Iteration count is
the honest comparison; wall time here is not, because DirectCoarse pays an
assembly cost that a production path would amortise differently.
"""
import os
import sys
import time

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R)

import numpy as np

from lssem2d import precond as P, solver as S
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState
from lssem2d.mesh import build_channel
from lssem2d.solver import apply_A

NU = 1.0/100.0
OUT = os.path.join(_R, 'scratch', 'pmg_direct_coarse.npz')


def case(N=8, Ex=6, Ey=2, L=12.0, dt=0.5, D0=1.0, delta=0.05):
    m = build_channel(L_x=L, L_y=1.0, E_x=Ex, E_y=Ey, N=N, bcs=(3, 6, 1, 1))
    st = SolverState(m, diff_matrix(N), nu=NU, dt=dt, fac1=1.0,
                     w_mom=1.0, w_mass=1.0)
    st.obc_D0, st.obc_delta = D0, delta
    n = N + 1
    fu = np.zeros((m.nelem, n, n)); fv = np.zeros((m.nelem, n, n))
    st.update_linearisation(fu, fv)
    return m, st, fu, fv


def count(st, b, fu, fv, mw, pre, tol=1e-10, maxit=20000):
    it = [0]
    orig = S.pcg_solve
    x, info = None, None
    t0 = time.perf_counter()
    x = orig(st, b, fu, fv, st._M_inv, mw, pin_p=False, max_iter=maxit,
             tol=tol, cgsfac=0.0, precond=pre)
    t = time.perf_counter() - t0
    if isinstance(x, tuple):
        x, info = x[0], x[1:]
    r = b - apply_A(st, x, fu, fv, pin_p=False)
    return t, float(np.linalg.norm(r)/np.linalg.norm(b)), info


def main(N=8, Ex=6, Ey=2):
    m, st, fu, fv = case(N, Ex, Ey)
    M_inv = S.compute_jacobi(st, fu, fv, pin_p=False)
    st._M_inv = M_inv
    from lssem2d.assembly import gather_scatter
    mult = gather_scatter(m, np.ones((m.nelem, m.N+1, m.N+1, 4)))
    mw = 1.0/np.where(mult < 1e-10, 1.0, mult)

    rng = np.random.default_rng(0)
    xs = rng.standard_normal((m.nelem, m.N+1, m.N+1, 4))
    b = apply_A(st, xs, fu, fv, pin_p=False)

    print(f'Poiseuille + Dong OBC:  N={N}  {Ex}x{Ey} elements  '
          f'{m.nelem*(N+1)**2*4} local DOF\n')

    t0 = time.perf_counter()
    dc = P.DirectCoarse.__new__(P.DirectCoarse)          # probe size only
    pmg_d = P.make('pmg2', st, fu, fv, M_inv, False, pc=2, deg=4,
                   coarse_solver='direct')
    t_setup_d = time.perf_counter() - t0
    print(f'DirectCoarse: {pmg_d.coarse.ndof} free coarse DOF, '
          f'assembly+factorise {t_setup_d:.3f}s, '
          f'asymmetry {pmg_d.coarse.asym:.2e}')

    t0 = time.perf_counter()
    pmg_c = P.make('pmg2', st, fu, fv, M_inv, False, pc=2, deg=4,
                   coarse_deg=10)
    t_setup_c = time.perf_counter() - t0
    print(f'Chebyshev coarse: setup {t_setup_c:.3f}s\n')

    res = {}
    for name, pre in (('jacobi', None),
                      ('pmg2 + cheby10', pmg_c),
                      ('pmg2 + DIRECT', pmg_d)):
        t, rr, info = count(st, b, fu, fv, mw, pre)
        res[name] = (t, rr, info)
        print(f'  {name:16s} wall {t:7.3f}s   true rel residual {rr:.3e}   {info}')

    np.savez_compressed(OUT, N=N, Ex=Ex, Ey=Ey,
                        ndof_coarse=pmg_d.coarse.ndof,
                        asym=pmg_d.coarse.asym,
                        setup_direct=t_setup_d, setup_cheby=t_setup_c,
                        names=list(res), walls=[res[k][0] for k in res],
                        resids=[res[k][1] for k in res])
    print(f'\nsaved -> {OUT}')


if __name__ == '__main__':
    main()
