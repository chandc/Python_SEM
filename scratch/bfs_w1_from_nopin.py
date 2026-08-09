"""SHORT domain at w_mom = 1.0, started from the converged NO-PIN solution.

Context: in the w_mom sweep (bfs_wmom_short.py) w_mom = 1.0 never converged --
it hit the 60-iteration cap after 37,946 CG iterations, starting from the
spin-up field.  This asks whether a different, genuinely converged start gets
there.  The IC is the no-pin w_mom = 0.1 solution (bfsnp2_off_nopin.npz), which
converged in 4 Newton iterations.

Run with the pin OFF (matching the IC) and, as a control, with it ON -- the pin
was shown to be a mask on the increment, not a pressure datum, so the two should
agree except for solver noise.

Loose solve (cgsfac 1e-3, tol 1e-6), p-MG, w_mass = 0.
"""
import os, sys, time
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
sys.path.insert(0, SC)
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
from fgrid import load
from lssem2d.lgl import diff_matrix, lgl_weights
from lssem2d.lssem import SolverState, apply_L
from lssem2d.operators import dUdx, dUdy
import lssem2d.solver as S
from lssem2d import precond as P

RE, H, WMOM = 389.0, 0.5, 1.0
START = 'bfsnp2_off_nopin.npz'
_p = S.pcg_solve
CAP, WALL = 60, 1200.0


def build():
    m, _, _ = load('/Users/danielchan/Dropbox/F90_SEM/pmg_clean/cnos_short_grid.dat')
    n = m.N+1
    pin = next((e, n-1, 0) for e in range(m.nelem) if m.bc[e, 1] == 4 and m.bc[e, 2] == 1)
    for e in range(m.nelem):
        if m.bc[e, 1] == 4:
            m.bc[e, 1] = 0
    return m, n, pin


def merit(st, U):
    f = np.ascontiguousarray(U[..., 0]/2.0); g = np.ascontiguousarray(U[..., 1]/2.0)
    st.update_linearisation(f, g)
    r = apply_L(st, U, f, g)
    return float(np.sum(r*r/st.mesh.wq[..., None]))


def diag(U, m, pin):
    N = m.N; n = N+1
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
        if yn[e, 0] > 0.01 or xn[e, 0] < -1e-9:
            continue
        for i in range(n):
            xs.append(xn[e, i]); tw.append(np.dot(D[0, :], U[e, i, :, 0])*(2.0/hy[e]))
    o = np.argsort(xs); xs, tw = np.array(xs)[o], np.array(tw)[o]; xr = np.nan
    for k in range(len(xs)-1):
        if tw[k] < 0 and tw[k+1] > 0 and xs[k] > 0.05:
            xr = xs[k]-tw[k]*(xs[k+1]-xs[k])/(tw[k+1]-tw[k]); break
    ue = np.array([U[e, -1, j, 0] for e in OUT for j in range(n)])
    pe = np.array([U[e, -1, j, 2] for e in OUT for j in range(n)])
    pi = np.array([U[e, 0, j, 2] for e in INL for j in range(n)])
    return dict(q=float(sum(fl(e, -1) for e in OUT)/sum(fl(e, 0) for e in INL)),
                div=float(np.sqrt(((ux+vy)**2).mean())),
                umax=float(np.abs(U[..., 0]).max()), xr=float(xr/H),
                psp=float(pe.max()-pe.min()), dp=float(pi.mean()-pe.mean()),
                rev=float(100*np.mean(ue < 0)),
                pmean=float((U[..., 2]*m.wq).sum()/m.wq.sum()))


def run(U0, use_pin):
    m, n, pin = build(); N = m.N
    st = SolverState(m, diff_matrix(N), nu=1.0/RE, dt=0.5, fac1=1.0,
                     w_mom=WMOM, w_mass=0.0)
    inlet = lambda x, y, t: 6.0*((y-0.5)/0.5)*(1.0-(y-0.5)/0.5)
    nit = [0]; pp = pin if use_pin else None

    def pcg(state, b, fu, fv, M, mw, pin_p=False, max_iter=5000, tol=None,
            cgsfac=None, precond=None, **kw):
        pre = P.make('pmg2', state, fu, fv, M, pin_p,
                     pc=max(2, N//2), deg=4, coarse_deg=10)
        x, it = _p(state, b, fu, fv, M, mw, pin_p=pin_p, max_iter=300000,
                   tol=1e-6, cgsfac=1e-3, precond=pre)
        nit[0] += it; return x, it
    S.pcg_solve = pcg

    U = U0.copy(); hist = [U]; t0 = time.perf_counter(); status = 'cap'; s = 0
    J0 = merit(st, U); trace = []
    try:
        for s in range(CAP):
            Up = hist[0].copy()
            U = S.step_bdf(st, hist, time=0.0, max_newton=1, newton_tol=1e-14,
                           newton_factor=0.0, custom_inlet=inlet, pin_p=pp,
                           cgsfac=1e-3, cg_max_iter=300000, verbose=False)
            dU = np.max(np.abs(U-Up)); um = np.abs(U[..., 0]).max()
            if s < 8 or s % 10 == 0:
                trace.append((s+1, dU, um))
            if not np.all(np.isfinite(U)):
                status = 'NaN'; break
            if um > 20.0:
                status = f'DIVERGED({um:.1f})'; break
            if s > 1 and dU < 1e-11:
                status = 'conv'; break
            if time.perf_counter()-t0 > WALL:
                status = 'WALL'; break
    finally:
        S.pcg_solve = _p
    ok = np.all(np.isfinite(U)) and np.abs(U[..., 0]).max() < 20.0
    return dict(status=status, it=s+1, cg=nit[0], wall=time.perf_counter()-t0,
                J0=J0, J1=merit(st, U) if ok else np.nan,
                d=diag(U, m, pin) if ok else None, trace=trace, U=U, m=m)


m0, _, pin0 = build()
U0 = np.load(f'{SC}/{START}')['U']
st_ref = SolverState(m0, diff_matrix(m0.N), nu=1.0/RE, dt=0.5, fac1=1.0,
                     w_mom=WMOM, w_mass=0.0)
d0 = diag(U0, m0, pin0)

print("BFS Chan Re=389, SHORT domain, steady form w_mom = 1.0 (w_mass = 0), p-MG, loose")
print(f"IC = {START}  (the converged NO-PIN w_mom=0.1 solution, 4 Newton iterations)\n")
print("IC as it stands, measured at w_mom = 1.0:")
print(f"   J = {merit(st_ref, U0):.4e}   Qout/Qin {d0['q']:.4f}   div {d0['div']:.2e}   "
      f"max|u| {d0['umax']:.3f}   x_r/h {d0['xr']:.3f}   p_sprd {d0['psp']:.3f}   rev {d0['rev']:.1f}%")
print("\nfor comparison, the sweep's w_mom = 1.0 run (different start) NEVER converged:")
print("   CAP at 60 iters, 37,946 CG, Qout 0.9909, div 1.03e-01, max|u| 2.300, "
      "x_r/h 3.433, p_sprd 2.864, rev 29.5%\n")

hdr = (f"{'':<12}{'status':>12}{'it':>5}{'CG':>9}{'wall':>7}{'J start':>11}{'J end':>11}"
       f"{'Qout/Qin':>10}{'rms div':>10}{'max|u|':>8}{'x_r/h':>8}{'p_sprd':>8}{'dp':>8}{'rev':>7}")
print(hdr)
for tag, up in (('NO pin', False), ('with pin', True)):
    r = run(U0, up); d = r['d']
    if d is None:
        print(f"  {tag:<10}{r['status']:>12}{r['it']:>5}{r['cg']:>9}{r['wall']:>7.0f}"
              f"{r['J0']:>11.3e}")
    else:
        print(f"  {tag:<10}{r['status']:>12}{r['it']:>5}{r['cg']:>9}{r['wall']:>7.0f}"
              f"{r['J0']:>11.3e}{r['J1']:>11.3e}{d['q']:>10.4f}{d['div']:>10.2e}"
              f"{d['umax']:>8.3f}{d['xr']:>8.3f}{d['psp']:>8.3f}{d['dp']:>8.3f}{d['rev']:>6.1f}%")
        print(f"  {'':<10}mean p = {d['pmean']:+.6g}")
        np.savez_compressed(f"{SC}/bfsw1_{'pin' if up else 'nopin'}.npz", U=r['U'],
                            xnod=r['m'].xnod, ynod=r['m'].ynod, hy=r['m'].hy)
    print(f"  {'':<10}trace (it, max|dU|, max|u|): "
          + ", ".join(f"({a},{b:.2e},{c:.2f})" for a, b, c in r['trace'][:8]))
    sys.stdout.flush()
