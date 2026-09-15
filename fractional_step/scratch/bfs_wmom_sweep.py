"""BFS steady form (a_mass = 0): sweep w_mom, restarting from a converged flow.

w_mass = 0 makes the momentum row exactly  w_mom * N(U)  with no time-derivative
term, so w_mom is the ONLY weight against the constraints (which carry 1) and dt
is irrelevant.  Starting each run from an already-converged field isolates the
weighting: Newton only has to move to the new minimiser, not develop the flow.

Start state: the converged w_mom = 1 solution (p-MG, loose solve, 11 iterations).
References: Fortran x_r/h 8.154, bubble 2.404, exit p spread 0.113;  Chan 1996
x_r/h 8.11, bubble 1.82.  Exact: Qout/Qin = 1, div = 0, max|u| <= 1.5, rev = 0%.
"""
import os, sys, time
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
sys.path.insert(0, SC)
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
from fgrid import load
from lssem2d.lgl import diff_matrix, lgl_weights
from lssem2d.lssem import SolverState, ls_coeffs
from lssem2d.operators import dUdx, dUdy
import lssem2d.solver as S
from lssem2d import precond as P
from upper_wall import top_shear, crossings

RE, H = 389.0, 0.5
START = f'{SC}/bfs_steadyMG_pmg_1e-06.npz'      # converged w_mom = 1 solution
_p = S.pcg_solve


def build():
    m, _, _ = load('/Users/danielchan/Dropbox/F90_SEM/pmg_clean/cnos_long_grid.dat')
    n = m.N+1
    pin = next((e, n-1, 0) for e in range(m.nelem) if m.bc[e, 1] == 4 and m.bc[e, 2] == 1)
    for e in range(m.nelem):
        if m.bc[e, 1] == 4:
            m.bc[e, 1] = 0
    return m, n, pin


def run(wmom, cap=60):
    m, n, pin = build(); N = m.N
    st = SolverState(m, diff_matrix(N), nu=1.0/RE, dt=0.5, fac1=1.0,
                     w_mom=wmom, w_mass=0.0)
    inlet = lambda x, y, t: 6.0*((y-0.5)/0.5)*(1.0-(y-0.5)/0.5)
    nit = [0]
    def pcg(state, b, fu, fv, M, mw, pin_p=False, max_iter=5000, tol=1e-6,
            cgsfac=0.0, precond=None, **kw):
        pre = P.make('pmg2', state, fu, fv, M, pin_p, pc=max(2, N//2), deg=4, coarse_deg=10)
        x, it = _p(state, b, fu, fv, M, mw, pin_p=pin_p, max_iter=300000,
                   tol=1e-6, cgsfac=1e-3, precond=pre)
        nit[0] += it; return x, it
    S.pcg_solve = pcg
    U = np.load(START)['U'].copy()               # already-converged start
    hist = [U]; t0 = time.perf_counter(); status = 'cap'
    try:
        for s in range(cap):
            Up = hist[0].copy()
            U = S.step_bdf(st, hist, time=0.0, max_newton=1, newton_tol=1e-14,
                           newton_factor=0.0, custom_inlet=inlet, pin_p=pin,
                           cgsfac=1e-3, cg_max_iter=300000, verbose=False)
            if not np.all(np.isfinite(U)): status = 'NaN'; break
            um = np.abs(U[..., 0]).max()
            if um > 20.0: status = f'diverged({um:.0f})'; break
            if s > 1 and np.max(np.abs(U-Up)) < 1e-11: status = 'conv'; break
    finally:
        S.pcg_solve = _p
    wall = time.perf_counter()-t0
    if status not in ('conv', 'cap'):
        return dict(w=wmom, status=status, it=s+1, wall=wall)
    D = diff_matrix(N); w = lgl_weights(N)
    xn, yn, hy = m.xnod, m.ynod, m.hy
    ux = dUdx(np.ascontiguousarray(U[..., 0]), D, m.facx)
    vy = dUdy(np.ascontiguousarray(U[..., 1]), D, m.facy)
    fl = lambda e, i: np.sum(w*U[e, i, :, 0])*(hy[e]/2)
    xmin, xmax = xn.min(), xn.max()
    INL = [e for e in range(m.nelem) if abs(xn[e, 0]-xmin) < 1e-9 and yn[e, 0] > 0.4]
    OUT = [e for e in range(m.nelem) if abs(xn[e, -1]-xmax) < 1e-9]
    xs, tw = [], []
    for e in range(m.nelem):
        if yn[e, 0] > 0.01 or xn[e, 0] < -1e-9: continue
        for i in range(n):
            xs.append(xn[e, i]); tw.append(np.dot(D[0, :], U[e, i, :, 0])*(2.0/hy[e]))
    o = np.argsort(xs); xs, tw = np.array(xs)[o], np.array(tw)[o]; xr = np.nan
    for k in range(len(xs)-1):
        if tw[k] < 0 and tw[k+1] > 0 and xs[k] > 0.05:
            xr = xs[k]-tw[k]*(xs[k+1]-xs[k])/(tw[k+1]-tw[k]); break
    x2, g2 = top_shear(U, xn, yn); cr = crossings(x2, g2); sep = rea = np.nan
    for k, (xc, dd) in enumerate(cr):
        if dd == '+':
            sep = xc; rea = next((z for z, e2 in cr[k+1:] if e2 == '-'), np.nan); break
    bub = (rea-sep)/H if np.isfinite(sep) and np.isfinite(rea) else np.nan
    ue = np.array([U[e, -1, j, 0] for e in OUT for j in range(n)])
    pe = np.array([U[e, -1, j, 2] for e in OUT for j in range(n)])
    np.savez_compressed(f'{SC}/bfswm_{wmom:g}.npz', U=U, xnod=xn, ynod=yn, hy=hy)
    return dict(w=wmom, status=status, it=s+1, cg=nit[0], wall=wall,
                coef=ls_coeffs(st),
                q=float(sum(fl(e, -1) for e in OUT)/sum(fl(e, 0) for e in INL)),
                div=float(np.sqrt(((ux+vy)**2).mean())),
                umax=float(np.abs(U[..., 0]).max()), xr=float(xr/H), bub=float(bub),
                psp=float(pe.max()-pe.min()), rev=float(100*np.mean(ue < 0)))


print("BFS Chan Re=389, LONG domain, a_mass = 0 (pure steady), p-MG, loose solve")
print(f"restarted from the converged w_mom=1 field ({os.path.basename(START)})")
print("Fortran: x_r/h 8.154  bubble 2.404  p_sprd 0.113   |   Chan: x_r 8.11  bubble 1.82\n")
print(f"{'w_mom':>7}{'a_flux':>8}{'iters':>7}{'CG':>9}{'status':>14}{'Qout/Qin':>10}"
      f"{'rms div':>10}{'max|u|':>8}{'x_r/h':>8}{'bubble':>8}{'p_sprd':>8}{'rev':>7}{'wall':>7}")
for wm in (0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0):
    r = run(wm)
    if 'q' not in r:
        print(f"{wm:>7}{'':>8}{r['it']:>7}{'':>9}{r['status']:>14}{'':>59}{r['wall']:>7.0f}")
    else:
        print(f"{wm:>7}{r['coef'][1]:>8.2f}{r['it']:>7}{r['cg']:>9}{r['status']:>14}"
              f"{r['q']:>10.4f}{r['div']:>10.2e}{r['umax']:>8.3f}{r['xr']:>8.3f}"
              f"{r['bub']:>8.3f}{r['psp']:>8.3f}{r['rev']:>6.1f}%{r['wall']:>7.0f}")
    sys.stdout.flush()
