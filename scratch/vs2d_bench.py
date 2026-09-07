"""Production-path benchmark: matrix-free pcg_solve with Jacobi | PMG2 ladder |
vertex-patch Schwarz (+ p=2 coarse).  Same protocol as p_indep_2d.py (cold
manufactured RHS, tol 1e-8, cavity mesh, Re=1000)."""
import os, sys, time
for _v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '8')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, 'scratch')); os.chdir(_R)
import numpy as np
import lssem2d
from lssem2d import precond as P, solver as S
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState
from lssem2d.mesh import build_channel
from vertex_schwarz2d import VertexSchwarz2D, make_coarse

RE, TOL = 1000.0, 1e-8


def ladder(N):
    seq, p = [N], N
    while p > 2:
        p = max(2, p//2); seq.append(p)
    return tuple(seq[1:])


def case(N, DT, EX):
    m = build_channel(1.0, 1.0, EX, EX, N, bcs=(1, 1, 1, 2)); m.compute_global_indices()
    st = SolverState(m, diff_matrix(N), nu=1.0/RE, dt=DT, fac1=1.0)
    n = N + 1; fu = np.zeros((m.nelem, n, n)); fv = np.zeros_like(fu)
    st.update_linearisation(fu, fv)
    mult = S.gather_scatter(m, np.ones((m.nelem, n, n, 4))); mw = 1.0/np.where(mult < 1e-10, 1.0, mult)
    Mi = S.compute_jacobi(st, fu, fv, pin_p=True)
    x = np.random.default_rng(0).standard_normal((m.nelem, n, n, 4))
    b = S.apply_A(st, x, fu, fv, pin_p=True); b /= np.sqrt(np.sum(b*b*mw))
    def run(pre, cap=40000):
        t0 = time.time()
        _, it = S.pcg_solve(st, b, fu, fv, Mi, mw, pin_p=True, max_iter=cap, tol=TOL, precond=pre)
        return int(it), time.time() - t0
    return st, fu, fv, Mi, run, x.size


def main():
    lssem2d.set_backend('numba')
    print(f'2D cavity, Re={RE:.0f}, cold RHS, tol={TOL:g}; production pcg_solve, matrix-free A\n')
    for DT in (1.85e-4, 1.0):
        print(f'--- dt={DT:g} (c={1/DT:.0f}) ---')
        print(f'{"mesh":>5} {"N":>3} {"gDOF":>7} | {"jacobi":>14} | {"PMG2 ladder":>14} | {"vpatch+coarse":>14} | {"setup":>6} {"patches":>8} {"maxdofs":>7}')
        for EX, N in ((4, 8), (4, 12), (4, 16), (6, 8), (8, 8)):
            st, fu, fv, Mi, run, nd = case(N, DT, EX)
            ij, tj = run(None)
            pmg = P.make('pmg2', st, fu, fv, Mi, True, pc=ladder(N), deg=6, coarse_solver='direct')
            ip, tp = run(pmg)
            vs = VertexSchwarz2D(st, fu, fv, pin_p=True, coarse=make_coarse(st, fu, fv, Mi, True))
            iv, tv = run(vs, cap=3000)
            print(f'{f"{EX}x{EX}":>5} {N:3d} {nd:7d} | {ij:5d} {tj:7.1f}s | {ip:5d} {tp:7.1f}s | {iv:5d} {tv:7.1f}s | {vs.setup_time:5.1f}s {vs.npatch:8d} {vs.maxdofs:7d}', flush=True)
        print()

if __name__ == '__main__':
    main()
