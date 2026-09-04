"""WHERE does p-independence break as the mass coefficient grows?

Measured (scratch/p_indep_2d.py): the SAME 2D ladder is p-independent at
fac1/dt = 1 (growth 0.85x, ratio reaching 82x) and p-DEPENDENT at 5405
(growth 4.85x, ratio pinned at ~7x) -- and 3D at c=5405 behaves identically
(5.56x, 6-8x).  So the controlling variable is the mass coefficient, and the
2D/3D difference I chased all day does not exist.

The channel needs c = 1/(beta*dt) = 5405 because dt = 8e-4, so the fix has to
work THERE.  First: locate the transition, in 2D where iteration is cheap.
"""
import os, sys
for _v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '8')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); os.chdir(_R)
import numpy as np
import lssem2d
from lssem2d import precond as P, solver as S
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState
from lssem2d.mesh import build_channel

RE, EX, TOL = 1000.0, 4, 1e-8

def ladder(N):
    seq, p = [N], N
    while p > 2:
        p = max(2, p//2); seq.append(p)
    return tuple(seq[1:])

def main():
    lssem2d.set_backend('numba')
    print(f'2D cavity, Re={RE:.0f}, {EX}x{EX}, cold RHS, tol={TOL:g}\n')
    print(f'{"mass c":>9} | ' + ' '.join(f'{f"N={N}":>9}' for N in (8, 12, 16, 24))
          + f' | {"growth":>7} {"ratio@24":>9}')
    print('-'*72)
    for c in (1.0, 10.0, 50.0, 200.0, 525.0, 2000.0, 5405.0):
        cells, first, last, lastj = [], None, None, None
        for N in (8, 12, 16, 24):
            m = build_channel(1.0, 1.0, EX, EX, N, bcs=(1, 1, 1, 2))
            m.compute_global_indices()
            st = SolverState(m, diff_matrix(N), nu=1.0/RE, dt=1.0/c, fac1=1.0)
            n = N + 1
            fu = np.zeros((m.nelem, n, n)); fv = np.zeros_like(fu)
            st.update_linearisation(fu, fv)
            mult = S.gather_scatter(m, np.ones((m.nelem, n, n, 4)))
            mw = 1.0/np.where(mult < 1e-10, 1.0, mult)
            Mi = S.compute_jacobi(st, fu, fv, pin_p=True)
            x = np.random.default_rng(0).standard_normal((m.nelem, n, n, 4))
            b = S.apply_A(st, x, fu, fv, pin_p=True); b /= np.linalg.norm(b)
            def run(pre):
                out = S.pcg_solve(st, b, fu, fv, Mi, mw, pin_p=True,
                                  max_iter=60000, tol=TOL, precond=pre)
                it = out[1] if isinstance(out, tuple) else 0
                return int(it[0] if isinstance(it, (list, tuple)) else it)
            pmg = P.make('pmg2', st, fu, fv, Mi, True, pc=ladder(N), deg=6,
                         coarse_solver='direct')
            itp = run(pmg)
            cells.append(f'{itp:9d}')
            if first is None: first = itp
            last = itp
            if N == 24: lastj = run(None)
        g = last/max(first, 1)
        flag = '  p-INDEP' if g < 1.5 else ''
        print(f'{c:9.0f} | ' + ' '.join(cells) +
              f' | {g:6.2f}x {lastj/max(last,1):8.1f}x{flag}', flush=True)
    print('\n3D channel runs at c = 5405 (dt=8e-4).  3D measured growth 5.56x.')

if __name__ == '__main__':
    main()
