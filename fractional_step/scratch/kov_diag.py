"""Is our N=14 solution the true LS minimiser, or under-solved?

If J(ours) <= J(exact interpolated), we ARE minimising correctly and the 1.66e-10
gap to the exact solution is the discrete minimiser's own distance from it --
i.e. a different discrete problem from Chan's, not a solver failure.
If J(ours) > J(exact), we are under-solved and Chan's 9.22e-13 is reachable.

Also reports the tol actually reaching the CG and the per-solve iteration counts,
since the cg_tol sweep came back bit-identical and that needs explaining.
"""
import sys, os
import numpy as np
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import kov
import lssem2d.solver as S

seen = []
_p = S.pcg_solve
def spy(state, b, fu, fv, M, mw, **kw):
    x, it = _p(state, b, fu, fv, M, mw, **kw)
    seen.append((kw.get('tol'), kw.get('cgsfac'), it,
                 float(np.sqrt(np.sum(b*b*mw)))))
    return x, it

for N in (9, 14):
    seen.clear()
    S.pcg_solve = spy
    try:
        r = kov.run(4, 2, N, 1e-12, cap=60, cg_tol=1e-14)
    finally:
        S.pcg_solve = _p
    print(f"\nN = {N}:  steps {r['steps']}  eps_u {r['eu']:.3e}  res {r['res']:.2e}")
    print(f"  first 6 CG solves (tol, cgsfac, iters, ||b||):")
    for t, c, it, bn in seen[:6]:
        print(f"    tol={t}  cgsfac={c}  iters={it:5d}  ||b||={bn:.3e}")
    print(f"  last 3: " + ", ".join(f"(it={it}, ||b||={bn:.2e})"
                                    for _, _, it, bn in seen[-3:]))

    m = kov.build(4, 2, N)
    from lssem2d.lgl import diff_matrix
    from lssem2d.lssem import SolverState
    st = SolverState(m, diff_matrix(N), nu=kov.NU, dt=1.0, fac1=1.0,
                     w_mom=1.0, w_mass=0.0)
    E, X, Y = kov.fields(m, N)
    res_ex, J_ex = kov.residual(st, E)
    print(f"  J(exact interpolated) = {J_ex:.6e}   rms res {res_ex:.3e}")
    print(f"  J(our converged)      = ?  -> rerun residual on the solved field")
