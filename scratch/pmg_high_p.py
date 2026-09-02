"""p-multigrid with a DIRECT coarse solve, from p = 5 to p = 30.

    uv run --quiet python -u scratch/pmg_high_p.py

WHY THIS RANGE.  FOSLS_2D_PLAN sec F2g measured AMG under p-refinement and found
it degrades (2.16x over N = 4..12) while both AmgX schemes COLLAPSE outright --
aggregation stalls from N=6, classical from N=8.  sec F2h(ii) then found the
published reason: the assembled operator's density grows as O(p^2d), and even
low-order-refined preconditioning trades that for an anisotropy problem.

p-multigrid should be different in kind, because it never assembles the fine
operator.  Only the COARSEST level is assembled, and DirectCoarse's cost scales
with the ELEMENT count, not with p -- so the coarse solve at p_c = 2 costs the
same at N = 30 as at N = 5.  This measures whether that holds.

FOUR PRECONDITIONERS:
  jacobi     the baseline, capped -- it is not expected to survive high p
  2-level    N -> 2                  one big jump
  3-level    N -> 4 -> 2             what solver_pmg2.f90 runs
  ladder     N -> N/2 -> ... -> 2    a real p-multigrid hierarchy

The ladder is the interesting one: a single N=30 -> 2 jump asks one coarse grid
to represent everything the fine grid cannot resolve, which is exactly the
"coarsens too fast" failure Heys et al. (2005) identify for p/2 schemes.

Small 2x2 mesh so N = 30 is affordable: (2*30+1)^2*4 = 14884 global DOF.
Steady, w_mom = 1, w_con = 1, nu = 1/100, pin_p = True -- the F1 regime.
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

from lssem2d import precond as P, solver as S
from lssem2d.assembly import gather_scatter
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState
from lssem2d.mesh import build_channel
from lssem2d.solver import apply_A

OUT = os.path.join(_R, 'scratch', 'pmg_high_p.npz')
ORDERS = (5, 6, 8, 10, 12, 16, 20, 24, 30)
EX = EY = 2
JAC_CAP = 30000


def ladder(N):
    """N -> N//2 -> ... -> 2, the halving hierarchy."""
    seq, p = [], N
    while p > 2:
        p = max(2, p//2)
        seq.append(p)
    return tuple(seq)


def case(N):
    m = build_channel(2.0, 1.0, EX, EY, N, bcs=(1, 1, 1, 2))
    m.compute_global_indices()
    st = SolverState(m, diff_matrix(N), nu=1/100., dt=1e4, fac1=1.0, w_mom=1.0)
    n = N + 1
    z = np.zeros((m.nelem, n, n))
    st.update_linearisation(z, z.copy())
    return m, st, z


def count(st, m, z, N, pre, tol=1e-8, maxit=60000):
    rng = np.random.default_rng(0)
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
    print(f'p-multigrid with direct coarse solve, p = 5..30   ({EX}x{EY} '
          f'elements, steady, pin_p=True, tol=1e-8)\n', flush=True)
    print(f'{"N":>3} {"gDOF":>7} {"ladder":>16} | {"jacobi":>8} {"2-lvl":>7} '
          f'{"3-lvl":>7} {"ladder":>7} | {"jac/lad":>8} {"setup_s":>8}',
          flush=True)
    rows = []
    for N in ORDERS:
        m, st, z = case(N)
        gdof = (int(m.gidx.max())+1)*4
        M_inv = S.compute_jacobi(st, z, z.copy(), pin_p=True)
        lad = ladder(N)

        ij, tj = count(st, m, z, N, None, maxit=JAC_CAP)
        res = {}
        t_setup = 0.0
        for tag, pc in (('p2', 2), ('p3', (4, 2)), ('lad', lad)):
            try:
                t0 = time.perf_counter()
                pre = P.make('pmg2', st, z, z.copy(), M_inv, True, pc=pc,
                             deg=4, coarse_solver='direct')
                ts = time.perf_counter() - t0
                if tag == 'lad':
                    t_setup = ts
                res[tag] = count(st, m, z, N, pre)[0]
            except Exception as e:
                res[tag] = -1
                print(f'    [{tag} at N={N}: {type(e).__name__}: {e}]', flush=True)
        jc = f'{ij}' if ij < JAC_CAP else f'>{JAC_CAP}'
        print(f'{N:3d} {gdof:7d} {str(lad):>16} | {jc:>8} {res["p2"]:7d} '
              f'{res["p3"]:7d} {res["lad"]:7d} | '
              f'{ij/max(res["lad"],1):7.1f}x {t_setup:7.2f}s', flush=True)
        rows.append((N, gdof, ij, res['p2'], res['p3'], res['lad'], t_setup))
        np.savez_compressed(OUT, rows=np.array(rows, dtype=float),
                            cols=['N', 'gdof', 'jacobi', 'p2', 'p3', 'ladder',
                                  'setup_ladder'])
    r = np.array(rows, dtype=float)
    print(f'\n  growth N=5 -> 30:  jacobi {r[-1,2]/r[0,2]:6.1f}x   '
          f'2-lvl {r[-1,3]/r[0,3]:5.2f}x   3-lvl {r[-1,4]/r[0,4]:5.2f}x   '
          f'ladder {r[-1,5]/r[0,5]:5.2f}x')
    print('\n  A hierarchy that is p-INDEPENDENT would show growth ~1.0x.')
    print(f'\nsaved -> {OUT}')


if __name__ == '__main__':
    main()
