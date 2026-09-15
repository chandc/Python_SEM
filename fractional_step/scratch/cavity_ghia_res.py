"""Ghia Re=1000 cavity: converge at several resolutions and compare u(y) at x=0.5.

dt=1.0 deliberately: the LSSEM steady state is dt-dependent (momentum rows carry
dt, the constraints do not), and larger dt weights momentum more heavily, which
is the better-conditioned choice for steady-state work.

Preconditioner picked per size from the benchmark: Jacobi below ~40k DOF, p-MG
above, which is where the crossover was measured.
"""
import os
import sys, time, numpy as np
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState
import lssem2d.solver as S
from lssem2d import precond as P

SC = os.path.dirname(os.path.abspath(__file__))
RE, DT, PIN = 1000.0, 1.0, True
CASES = [(3, 6), (4, 8), (6, 10), (8, 12)]
MAXSTEP, STEADY = 500, 1.0e-8

def lagrange(xn, xq):
    n = len(xn); w = np.ones(n)
    for i in range(n):
        for j in range(n):
            if i != j: w[i] /= (xn[i]-xn[j])
    dd = xq-xn
    if np.any(np.abs(dd) < 1e-13):
        L = np.zeros(n); L[np.argmin(np.abs(dd))] = 1.0; return L
    num = w/dd; return num/num.sum()

def centreline_u(mesh, U, n):
    """u(y) along the vertical centreline x=0.5"""
    ys, us = [], []
    for e in range(mesh.nelem):
        xs = mesh.xnod[e]
        if xs[0]-1e-9 <= 0.5 <= xs[-1]+1e-9:
            L = lagrange(xs, 0.5)
            for j in range(n):
                ys.append(mesh.ynod[e, j]); us.append(np.dot(L, U[e, :, j, 0]))
    o = np.argsort(ys); ys, us = np.array(ys)[o], np.array(us)[o]
    k = np.concatenate(([True], np.diff(ys) > 1e-9))
    return ys[k], us[k]

def centreline_v(mesh, U, n):
    """v(x) along the horizontal centreline y=0.5"""
    xs_, vs = [], []
    for e in range(mesh.nelem):
        yn = mesh.ynod[e]
        if yn[0]-1e-9 <= 0.5 <= yn[-1]+1e-9:
            L = lagrange(yn, 0.5)
            for i in range(n):
                xs_.append(mesh.xnod[e, i]); vs.append(np.dot(L, U[e, i, :, 1]))
    o = np.argsort(xs_); xs_, vs = np.array(xs_)[o], np.array(vs)[o]
    k = np.concatenate(([True], np.diff(xs_) > 1e-9))
    return xs_[k], vs[k]

gh = np.load('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo/cavity_re1000_data.npz')
ghia_u, ghia_y = gh['ghia_u'], gh['ghia_y']

out = {}
print(f"Ghia cavity Re={RE:.0f}, dt={DT}, steady tol {STEADY:g}, max {MAXSTEP} steps\n")
print(f"{'mesh':<14}{'DOF':>8}{'precond':>9}{'steps':>7}{'final dU':>11}"
      f"{'RMS vs Ghia':>13}{'% of umax':>11}{'wall s':>9}")
for EX, N in CASES:
    n = N+1
    mesh = build_channel(1.0, 1.0, EX, EX, N, bcs=(1, 1, 1, 2))
    st = SolverState(mesh, diff_matrix(N), nu=1.0/RE, dt=DT, fac1=1.0)
    ndof = mesh.nelem*n*n*4
    use_pmg = ndof > 40000
    _p = S.pcg_solve
    def pcg(state, b, fu, fv, M, mw, pin_p=False, max_iter=5000, tol=1e-6, cgsfac=0.0, precond=None):
        pre = P.make('pmg2', state, fu, fv, M, pin_p, pc=max(2, N//2), deg=4,
                     coarse_deg=10) if use_pmg else None
        return _p(state, b, fu, fv, M, mw, pin_p=pin_p, max_iter=30000, tol=1e-6,
                  cgsfac=1e-3, precond=pre)
    S.pcg_solve = pcg
    U = np.zeros((mesh.nelem, n, n, 4)); hist = [U]
    t0 = time.perf_counter(); dU = np.nan
    try:
        for s in range(MAXSTEP):
            Up = hist[0].copy()
            U = S.step_bdf(st, hist, time=s*DT, max_newton=1, newton_tol=1e-10,
                           newton_factor=0.0, pin_p=PIN, cgsfac=1e-3,
                           cg_max_iter=30000, verbose=False)
            dU = np.max(np.abs(U-Up))
            if not np.all(np.isfinite(U)): break
            if s > 3 and dU < STEADY: break
    finally:
        S.pcg_solve = _p
    tw = time.perf_counter()-t0
    y, u = centreline_u(mesh, U, n)
    x, v = centreline_v(mesh, U, n)
    r = np.sqrt(np.mean((np.interp(ghia_y, y, u)-ghia_u)**2))
    sc = max(abs(ghia_u.max()), abs(ghia_u.min()))
    print(f"{f'{EX}x{EX} p={N}':<14}{ndof:>8}{'p-MG' if use_pmg else 'Jacobi':>9}"
          f"{s+1:>7}{dU:>11.2e}{r:>13.4f}{100*r/sc:>10.2f}%{tw:>9.1f}")
    out[f'{EX}x{EX}_p{N}'] = dict(y=y, u=u, x=x, v=v, ndof=ndof, rms=r, EX=EX, N=N,
                                  steps=s+1, dU=dU, wall=tw)

np.savez_compressed(f'{SC}/cavity_ghia_res.npz',
                    **{f'{k}__{f}': v[f] for k, v in out.items() for f in v},
                    keys=np.array(list(out.keys())), ghia_u=ghia_u, ghia_y=ghia_y)
print(f"\nsaved {SC}/cavity_ghia_res.npz")
