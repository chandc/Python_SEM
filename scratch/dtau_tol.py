"""TIME-MARCHING short domain with the real time term ON, sweeping dtau.

w_mass = w_mom = 1.0 and dt = 0.1, so dt_eff = dt*w_mom/w_mass = 0.1 and the run
is time-accurate -- unlike every other short-domain run in this session, which
set w_mass = 0 and deleted the physical time derivative entirely.

  a_mass = fac1*w_mass/dt = 10*fac1     a_flux = w_mom = 1     kappa = 1/dtau

so dtau 0.1..0.5 gives kappa 10..2, i.e. 1/dt* = fac1/dt_eff + 1/dtau = 10 + 1/dtau.

Cold start from the developed profile.  J is reported as the STEADY functional
(evaluated with a w_mass=0 state) so it is comparable across runs and against
the steady-form results; the marching residual itself is not.
"""
import os, sys, time
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
sys.path.insert(0, SC)
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
from fgrid import load
from lssem2d.lgl import diff_matrix, lgl_weights
from lssem2d.lssem import SolverState, apply_L, ls_pseudo
from lssem2d.operators import dUdx, dUdy
import lssem2d.solver as S
from lssem2d import precond as P

RE, H = 389.0, 0.5
DT, WMOM, WMASS = 0.1, 1.0, 1.0
NST, WALL, TOL = 4000, 1500.0, 1e-10
NSUB = int(os.environ.get('NSUB', '5'))
_p = S.pcg_solve


def build():
    m, _, _ = load('/Users/danielchan/Dropbox/F90_SEM/pmg_clean/cnos_short_grid.dat')
    n = m.N+1
    for e in range(m.nelem):
        if m.bc[e, 1] == 4:
            m.bc[e, 1] = 0
    return m, n


def devc_ic(m):
    n = m.N+1
    U0 = np.zeros((m.nelem, n, n, 4)); Lb = 1.0
    u_dev = lambda y: 3.0*y*(1.0-y)
    du_dev = lambda y: 3.0 - 6.0*y

    def u_step(y):
        if y <= 0.5:
            return 0.0
        eta = 2.0*y - 1.0
        return 6.0*eta*(1.0-eta)

    def du_step(y):
        if y <= 0.5:
            return 0.0
        eta = 2.0*y - 1.0
        return 12.0*(1.0 - 2.0*eta)

    def G(y):
        I = 1.5*y*y - y**3
        if y > 0.5:
            eta = 2.0*y - 1.0
            I -= 0.5*(3.0*eta*eta - 2.0*eta**3)
        return I

    for e in range(m.nelem):
        for i in range(n):
            xx = m.xnod[e, i]
            if xx <= 0.0:
                t, sp, spp = 0.0, 0.0, 0.0
            else:
                t = min(xx/Lb, 1.0)
                sp = (6.0*t - 6.0*t*t)/Lb
                spp = (6.0 - 12.0*t)/Lb**2
                if xx >= Lb:
                    sp = spp = 0.0
            sv = 3.0*t*t - 2.0*t**3
            for j in range(n):
                yy = m.ynod[e, j]
                if xx < 0.0:
                    eta = (yy-0.5)/0.5
                    U0[e, i, j, 0] = 6.0*eta*(1.0-eta)
                    U0[e, i, j, 3] = -12.0*(1.0-2.0*eta)
                else:
                    U0[e, i, j, 0] = (1.0-sv)*u_step(yy) + sv*u_dev(yy)
                    U0[e, i, j, 1] = -sp*G(yy)
                    U0[e, i, j, 3] = (-spp*G(yy)
                                      - ((1.0-sv)*du_step(yy) + sv*du_dev(yy)))
    return U0


def steady_J(m, U):
    """The steady LS functional, so runs are comparable to the w_mass=0 results."""
    st = SolverState(m, diff_matrix(m.N), nu=1.0/RE, dt=DT, fac1=1.0,
                     w_mom=WMOM, w_mass=0.0)
    f = np.ascontiguousarray(U[..., 0]/2.0); g = np.ascontiguousarray(U[..., 1]/2.0)
    st.update_linearisation(f, g)
    r = apply_L(st, U, f, g).copy()
    return float(np.sum(r*r/m.wq[..., None]))


def diag(U, m):
    Nn = m.N; n = Nn+1
    D = diff_matrix(Nn); w = lgl_weights(Nn)
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
    return dict(q=float(sum(fl(e, -1) for e in OUT)/sum(fl(e, 0) for e in INL)),
                div=float(np.sqrt(((ux+vy)**2).mean())),
                umax=float(np.abs(U[..., 0]).max()), xr=float(xr/H),
                psp=float(pe.max()-pe.min()), rev=float(100*np.mean(ue < 0)))


def run(dtau):
    m, n = build(); Nn = m.N
    st = SolverState(m, diff_matrix(Nn), nu=1.0/RE, dt=DT, fac1=1.0,
                     w_mom=WMOM, w_mass=WMASS, dtau=dtau)
    inlet = lambda x, y, t: 6.0*((y-0.5)/0.5)*(1.0-(y-0.5)/0.5)
    nit = [0]

    def pcg(state, b, fu, fv, M, mw, pin_p=False, max_iter=5000, tol=None,
            cgsfac=None, precond=None, **kw):
        pre = P.make('pmg2', state, fu, fv, M, pin_p,
                     pc=max(2, Nn//2), deg=4, coarse_deg=10)
        x, it = _p(state, b, fu, fv, M, mw, pin_p=pin_p, max_iter=300000,
                   tol=float(os.environ.get('CGTOL','1e-6')), cgsfac=1e-3, precond=pre)
        nit[0] += it; return x, it
    S.pcg_solve = pcg

    U = devc_ic(m); hist = [U]; t0 = time.perf_counter(); status = 'cap'; s = 0
    try:
        for s in range(NST):
            Up = hist[0].copy()
            U = S.step_bdf(st, hist, time=s*DT, max_newton=NSUB, newton_tol=1e-10,
                           newton_factor=0.0, custom_inlet=inlet, pin_p=None,
                           cgsfac=1e-3, cg_max_iter=300000, verbose=False)
            if not np.all(np.isfinite(U)):
                status = 'NaN'; break
            um = np.abs(U[..., 0]).max()
            if um > 20.0:
                status = f'DIVERGED({um:.0f})'; break
            if s > 3 and np.max(np.abs(U-Up)) < TOL:
                status = 'steady'; break
            if time.perf_counter()-t0 > WALL:
                status = 'WALL'; break
    finally:
        S.pcg_solve = _p
    ok = np.all(np.isfinite(U)) and np.abs(U[..., 0]).max() < 20.0
    return dict(status=status, steps=s+1, t=(s+1)*DT, cg=nit[0],
                wall=time.perf_counter()-t0, kappa=ls_pseudo(st),
                J=steady_J(m, U) if ok else np.nan,
                d=diag(U, m) if ok else None, U=U, m=m)


_d = os.environ.get('DTAU', 'none')
DTAU = None if _d == 'none' else float(_d)
TAG = 'none' if DTAU is None else ('d' + _d.replace('.', 'p'))

print(f"TIME-MARCHING short domain, cold developed IC, w_mom = w_mass = 1.0, dt = {DT}")
print(f"nsub = {NSUB}   sub-iterations per time step")
print(f"dtau = {DTAU}   (kappa = {'0' if DTAU is None else 1.0/DTAU})")
print(f"{'dtau':>7}{'kappa':>8}{'status':>14}{'steps':>7}{'t':>8}{'CG':>10}{'wall':>7}"
      f"{'J':>12}{'Qout/Qin':>10}{'rms div':>10}{'max|u|':>8}{'x_r/h':>8}{'p_sprd':>8}{'rev':>7}")
r = run(DTAU); d = r['d']
lab = 'inf' if DTAU is None else f'{DTAU:g}'
if d is None:
    print(f"{lab:>7}{r['kappa']:>8.2f}{r['status']:>14}{r['steps']:>7}{r['t']:>8.1f}"
          f"{r['cg']:>10}{r['wall']:>7.0f}")
else:
    print(f"{lab:>7}{r['kappa']:>8.2f}{r['status']:>14}{r['steps']:>7}{r['t']:>8.1f}"
          f"{r['cg']:>10}{r['wall']:>7.0f}{r['J']:>12.4e}{d['q']:>10.4f}"
          f"{d['div']:>10.2e}{d['umax']:>8.3f}{d['xr']:>8.3f}{d['psp']:>8.3f}"
          f"{d['rev']:>6.1f}%")
    np.savez_compressed(f'{SC}/tolt_{TAG}.npz', U=r['U'], xnod=r['m'].xnod,
                        ynod=r['m'].ynod, hy=r['m'].hy)
