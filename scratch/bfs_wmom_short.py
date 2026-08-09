"""BFS SHORT domain, steady form (a_mass = 0): sweep w_mom from a converged flow.

Companion to bfs_wmom_sweep.py (long domain).  Same protocol:

  * w_mass = 0 so the momentum row is exactly  w_mom * N(U)  -- dt is dead input
  * every run starts from the SAME already-converged field, so Newton only has
    to move to the new minimiser rather than develop the flow
  * free outflow + SE-corner pressure pin, continuous developed IC, p-MG

Difference: L/h = 5 instead of 17.  This domain is MULTI-VALUED (several
converged states exist), and the outflow plane cuts through the recirculation,
so the exit diagnostics matter more here than the interior ones.

Spin-up: the legacy time-stepping field (dt=0.5, devc IC, SE pin) is first run
to convergence in the steady form at w_mom = 1; that becomes the common start.
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

# top_shear / crossings are copied verbatim from upper_wall.py rather than
# imported: that module plots at import time and its savefig is not safe to
# re-enter from a second process.


def top_shear(U, XP, YP):
    """du/dy at the TOP wall (y=1), sorted in x, interface duplicates averaged."""
    n = U.shape[1]; D = diff_matrix(n-1)
    xs, g = [], []
    for e in range(U.shape[0]):
        if YP[e, -1] < 0.999:
            continue
        hy = YP[e, -1] - YP[e, 0]
        for i in range(n):
            xs.append(XP[e, i])
            g.append(np.dot(D[-1, :], U[e, i, :, 0]) * (2.0 / hy))
    xs, g = np.array(xs), np.array(g)
    o = np.argsort(xs, kind='stable'); xs, g = xs[o], g[o]
    ux, ug = [], []
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j+1] - xs[i] < 1e-9:
            j += 1
        ux.append(xs[i]); ug.append(np.mean(g[i:j+1])); i = j + 1
    return np.array(ux), np.array(ug)


def crossings(x, g, xmin=0.2):
    """All sign changes of g, linearly interpolated. Returns (x, direction)."""
    out = []
    for k in range(len(x)-1):
        if x[k] < xmin or g[k] == 0.0:
            continue
        if g[k]*g[k+1] < 0:
            out.append((x[k] - g[k]*(x[k+1]-x[k])/(g[k+1]-g[k]),
                        '+' if g[k+1] > 0 else '-'))
    return out


RE, H = 389.0, 0.5
LEGACY = f'{SC}/dt_dt0p5_devc_short_state.npz'   # converged legacy time-stepping
START = f'{SC}/bfs_short_steady_w1.npz'          # written by the spin-up below
_p = S.pcg_solve


def build():
    m, _, _ = load('/Users/danielchan/Dropbox/F90_SEM/pmg_clean/cnos_short_grid.dat')
    n = m.N+1
    pin = next((e, n-1, 0) for e in range(m.nelem) if m.bc[e, 1] == 4 and m.bc[e, 2] == 1)
    for e in range(m.nelem):
        if m.bc[e, 1] == 4:
            m.bc[e, 1] = 0                       # free outflow
    return m, n, pin


def solve(wmom, U0, cap=60):
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
    U = U0.copy(); hist = [U]; t0 = time.perf_counter(); status = 'cap'; s = 0
    try:
        for s in range(cap):
            Up = hist[0].copy()
            U = S.step_bdf(st, hist, time=0.0, max_newton=1, newton_tol=1e-14,
                           newton_factor=0.0, custom_inlet=inlet, pin_p=pin,
                           cgsfac=1e-3, cg_max_iter=300000, verbose=False)
            if not np.all(np.isfinite(U)):
                status = 'NaN'; break
            um = np.abs(U[..., 0]).max()
            if um > 20.0:
                status = f'diverged({um:.0f})'; break
            if s > 1 and np.max(np.abs(U-Up)) < 1e-11:
                status = 'conv'; break
    finally:
        S.pcg_solve = _p
    return U, m, st, dict(status=status, it=s+1, cg=nit[0],
                          wall=time.perf_counter()-t0, coef=ls_coeffs(st))


def metrics(U, m):
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
    x2, g2 = top_shear(U, xn, yn); cr = crossings(x2, g2); sep = rea = np.nan
    for k, (xc, dd) in enumerate(cr):
        if dd == '+':
            sep = xc; rea = next((z for z, e2 in cr[k+1:] if e2 == '-'), np.nan); break
    bub = (rea-sep)/H if np.isfinite(sep) and np.isfinite(rea) else np.nan
    ue = np.array([U[e, -1, j, 0] for e in OUT for j in range(n)])
    pe = np.array([U[e, -1, j, 2] for e in OUT for j in range(n)])
    return dict(q=float(sum(fl(e, -1) for e in OUT)/sum(fl(e, 0) for e in INL)),
                div=float(np.sqrt(((ux+vy)**2).mean())),
                umax=float(np.abs(U[..., 0]).max()),
                xr=float(xr/H), bub=float(bub), sep=float(sep/H),
                psp=float(pe.max()-pe.min()), rev=float(100*np.mean(ue < 0)))


HDR = (f"{'w_mom':>7}{'a_flux':>8}{'iters':>7}{'CG':>9}{'status':>14}{'Qout/Qin':>10}"
       f"{'rms div':>10}{'max|u|':>8}{'x_r/h':>8}{'sep x/h':>9}{'bubble':>8}"
       f"{'p_sprd':>8}{'rev':>7}{'wall':>7}")


def show(tag, r, mm):
    print(f"{tag:>7}{r['coef'][1]:>8.2f}{r['it']:>7}{r['cg']:>9}{r['status']:>14}"
          f"{mm['q']:>10.4f}{mm['div']:>10.2e}{mm['umax']:>8.3f}{mm['xr']:>8.3f}"
          f"{mm['sep']:>9.3f}{mm['bub']:>8.3f}{mm['psp']:>8.3f}{mm['rev']:>6.1f}%"
          f"{r['wall']:>7.0f}")
    sys.stdout.flush()


print("BFS Chan Re=389, SHORT domain (L/h=5), a_mass = 0 (pure steady), p-MG, loose solve")
print("free outflow + SE-corner pin, continuous developed IC lineage")
print("NOTE: the outflow plane cuts the recirculation here -- this domain is multi-valued.\n")

d0 = np.load(LEGACY)
m0, _, _ = build()
mm0 = metrics(d0['U'], m0)
print("legacy time-stepping start (dt=0.5, devc IC):")
print(f"   Qout/Qin {mm0['q']:.4f}  div {mm0['div']:.2e}  max|u| {mm0['umax']:.3f}  "
      f"x_r/h {mm0['xr']:.3f}  sep {mm0['sep']:.3f}  bubble {mm0['bub']:.3f}  "
      f"p_sprd {mm0['psp']:.3f}  rev {mm0['rev']:.1f}%\n")

print("spin-up: steady form at w_mom = 1 from the legacy field")
print(HDR)
Uw1, m, st, r = solve(1.0, d0['U'])
mm = metrics(Uw1, m)
show('spinup', r, mm)
np.savez_compressed(START, U=Uw1, xnod=m.xnod, ynod=m.ynod, hy=m.hy)

print("\nsweep, every run restarted from that converged w_mom = 1 field")
print(HDR)
for wm in (0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0):
    U, m, st, r = solve(wm, Uw1)
    if r['status'] in ('conv', 'cap'):
        mm = metrics(U, m)
        show(f'{wm:g}', r, mm)
        np.savez_compressed(f'{SC}/bfswms_{wm:g}.npz', U=U, xnod=m.xnod,
                            ynod=m.ynod, hy=m.hy)
    else:
        print(f"{wm:>7}{r['coef'][1]:>8.2f}{r['it']:>7}{r['cg']:>9}{r['status']:>14}")
        sys.stdout.flush()
