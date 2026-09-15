"""Does dtau do what the line search does?  (PSEUDO_TIME_DESIGN.md sec 6 step 5)

Two cases that need globalisation today, both run WITHOUT a line search so the
pseudo-time term is the only thing that can save them:

  A. short domain seeded from the interpolated LONG-domain solution.  Undamped
     this diverges at Newton step 2, max|u| 1.51 -> 115.
  B. short domain at w_mom = 1.0 from the converged no-pin field.  Undamped this
     stalls, and needs 53 iterations with the line search.

And the accuracy question (step 4): where the field converges, how far does dtau
MOVE it?  The yardstick is the 1.1% J gap that separates the two converged
states in STEADY_FORM_STUDY.md sec 8 -- a benign stabilisation must land well
inside that.
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
_p = S.pcg_solve
CAP, WALL = 80, 900.0


def build():
    m, _, _ = load('/Users/danielchan/Dropbox/F90_SEM/pmg_clean/cnos_short_grid.dat')
    n = m.N+1
    pin = next((e, n-1, 0) for e in range(m.nelem) if m.bc[e, 1] == 4 and m.bc[e, 2] == 1)
    for e in range(m.nelem):
        if m.bc[e, 1] == 4:
            m.bc[e, 1] = 0
    return m, n, pin


def merit(st, U):
    """True LS functional -- _drop_pseudo keeps this comparable across dtau."""
    f = np.ascontiguousarray(U[..., 0]/2.0); g = np.ascontiguousarray(U[..., 1]/2.0)
    st.update_linearisation(f, g)
    r = apply_L(st, U, f, g).copy()
    S._drop_pseudo(st, r, U)
    return float(np.sum(r*r/st.mesh.wq[..., None]))


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


def run(U0, wmom, dtau, ls=False):
    m, n, pin = build(); Nn = m.N
    st = SolverState(m, diff_matrix(Nn), nu=1.0/RE, dt=0.5, fac1=1.0,
                     w_mom=wmom, w_mass=0.0, dtau=dtau)
    inlet = lambda x, y, t: 6.0*((y-0.5)/0.5)*(1.0-(y-0.5)/0.5)
    nit = [0]

    def pcg(state, b, fu, fv, M, mw, pin_p=False, max_iter=5000, tol=None,
            cgsfac=None, precond=None, **kw):
        pre = P.make('pmg2', state, fu, fv, M, pin_p,
                     pc=max(2, Nn//2), deg=4, coarse_deg=10)
        x, it = _p(state, b, fu, fv, M, mw, pin_p=pin_p, max_iter=300000,
                   tol=1e-6, cgsfac=1e-3, precond=pre)
        nit[0] += it; return x, it
    S.pcg_solve = pcg

    U = U0.copy(); hist = [U]; t0 = time.perf_counter(); status = 'cap'; s = 0
    trace = []
    try:
        for s in range(CAP):
            Up = hist[0].copy()
            U = S.step_bdf(st, hist, time=0.0, max_newton=1, newton_tol=1e-14,
                           newton_factor=0.0, custom_inlet=inlet, pin_p=None,
                           cgsfac=1e-3, cg_max_iter=300000, verbose=False,
                           line_search=ls)
            dU = np.max(np.abs(U-Up)); um = np.abs(U[..., 0]).max()
            if s < 5:
                trace.append((s+1, dU, um))
            if not np.all(np.isfinite(U)):
                status = 'NaN'; break
            if um > 20.0:
                status = f'DIVERGED({um:.0f})'; break
            if s > 1 and dU < 1e-11:
                status = 'conv'; break
            if time.perf_counter()-t0 > WALL:
                status = 'WALL'; break
    finally:
        S.pcg_solve = _p
    ok = np.all(np.isfinite(U)) and np.abs(U[..., 0]).max() < 20.0
    return dict(status=status, it=s+1, cg=nit[0], wall=time.perf_counter()-t0,
                J=merit(st, U) if ok else np.nan, kappa=ls_pseudo(st),
                d=diag(U, m) if ok else None, trace=trace, U=U, m=m)


HDR = (f"{'dtau':>7}{'kappa':>8}{'status':>16}{'it':>5}{'CG':>9}{'wall':>7}"
       f"{'J':>12}{'Qout/Qin':>10}{'rms div':>10}{'max|u|':>8}{'x_r/h':>8}"
       f"{'p_sprd':>8}{'rev':>7}")


def show(tag, r):
    d = r['d']
    if d is None:
        print(f"{tag:>7}{r['kappa']:>8.3f}{r['status']:>16}{r['it']:>5}{r['cg']:>9}"
              f"{r['wall']:>7.0f}")
    else:
        print(f"{tag:>7}{r['kappa']:>8.3f}{r['status']:>16}{r['it']:>5}{r['cg']:>9}"
              f"{r['wall']:>7.0f}{r['J']:>12.4e}{d['q']:>10.4f}{d['div']:>10.2e}"
              f"{d['umax']:>8.3f}{d['xr']:>8.3f}{d['psp']:>8.3f}{d['rev']:>6.1f}%")
    if r['trace']:
        print(f"       trace: " + ", ".join(f"({a},{b:.1e},{c:.2f})" for a, b, c in r['trace']))
    sys.stdout.flush()


LBLEND = 1.0


def devc_ic(m):
    """COLD start: the continuous developed profile, divergence-free by
    construction.  Copied from bfs_dt.py IC == 'devc' -- blend from the
    step-exit profile to the fully developed one over LBLEND, v from
    continuity.  This is a genuine cold start: no prior solution of any kind."""
    n = m.N+1
    U0 = np.zeros((m.nelem, n, n, 4))
    Lb = LBLEND
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


print("=" * 108)
print("COLD START from the developed profile (no prior solution), steady form w_mom = 0.1")
print("=" * 108)
m0, _, _ = build()
U_cold = devc_ic(m0)
np.savez_compressed(f'{SC}/cold_IC.npz', U=U_cold, xnod=m0.xnod, ynod=m0.ynod, hy=m0.hy)
print(HDR)
for kap, tag in ((0.0, 'none'), (0.1, 'k0p1'), (0.3, 'k0p3'), (1.0, 'k1')):
    dtau = None if kap == 0.0 else 0.1/kap
    r = run(U_cold, 0.1, dtau)
    show('inf' if dtau is None else f'{dtau:.3g}', r)
    if r['d'] is not None:
        np.savez_compressed(f'{SC}/cold_{tag}.npz', U=r['U'], xnod=r['m'].xnod,
                            ynod=r['m'].ynod, hy=r['m'].hy)
