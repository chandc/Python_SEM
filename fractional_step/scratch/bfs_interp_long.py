"""SHORT domain seeded by INTERPOLATING the converged LONG-domain solution.

The short domain's outflow plane (x = 2.5) sits in the INTERIOR of the long
domain (x up to 8.5), so the long-domain solution restricted to x <= 2.5 is the
"right" answer for that region -- it carries the downstream influence that the
truncated domain cannot represent.  Seeding with it and re-solving asks a sharp
question: does the short domain STAY at the correct field, or does it fall back
to its own outflow-corrupted state?

Both grids are order 10 with the same y extent, so the only work is a spatial
interpolation.  Done with matplotlib.tri on the long-domain node cloud, with the
solid step masked out so nothing interpolates across it.

Run: w_mom = 0.1, w_mass = 0, loose solve (cgsfac 1e-3, tol 1e-6), p-MG.
Both without the pin (matching the recent runs) and with it, as a control.

References for the answer:
  long   w_mom=0.1  (the source)          -- Qout 0.9997  div 3.53e-03  x_r/h 8.331
  short  w_mom=0.1  (own converged state) -- Qout 0.9997  div 4.28e-03  x_r/h 3.312
                                             max|u| 2.494  p_sprd 3.866  rev 27.3%
"""
import os, sys, time
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
sys.path.insert(0, SC)
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
from matplotlib.tri import Triangulation, LinearTriInterpolator
from scipy.interpolate import griddata
from fgrid import load
from lssem2d.lgl import diff_matrix, lgl_weights
from lssem2d.lssem import SolverState, apply_L
from lssem2d.operators import dUdx, dUdy
import lssem2d.solver as S
from lssem2d import precond as P

RE, H, WMOM = 389.0, 0.5, 0.1
LONG = 'bfswm_0.1.npz'                 # converged long-domain w_mom = 0.1
SHORT_OWN = 'bfsnp2_off_nopin.npz'     # converged short-domain w_mom = 0.1
_p = S.pcg_solve
CAP, WALL = 120, 1200.0


def build():
    m, _, _ = load('/Users/danielchan/Dropbox/F90_SEM/pmg_clean/cnos_short_grid.dat')
    n = m.N+1
    pin = next((e, n-1, 0) for e in range(m.nelem) if m.bc[e, 1] == 4 and m.bc[e, 2] == 1)
    for e in range(m.nelem):
        if m.bc[e, 1] == 4:
            m.bc[e, 1] = 0
    return m, n, pin


def interpolate(mshort):
    """Long-domain solution -> short-domain nodes, field by field."""
    d = np.load(f'{SC}/{LONG}')
    UL, xl, yl = d['U'], d['xnod'], d['ynod']
    nl = UL.shape[1]
    px, py = [], []
    for e in range(UL.shape[0]):
        for i in range(nl):
            for j in range(nl):
                px.append(xl[e, i]); py.append(yl[e, j])
    px, py = np.array(px), np.array(py)
    tri = Triangulation(px, py)
    cx = px[tri.triangles].mean(axis=1); cy = py[tri.triangles].mean(axis=1)
    tri.set_mask((cx < 0) & (cy < 0.5))          # never interpolate through the step

    n = mshort.N+1
    qx, qy = [], []
    for e in range(mshort.nelem):
        for i in range(n):
            for j in range(n):
                qx.append(mshort.xnod[e, i]); qy.append(mshort.ynod[e, j])
    qx, qy = np.array(qx), np.array(qy)

    U = np.zeros((mshort.nelem, n, n, 4))
    nfill = 0
    for c in range(4):
        vals = np.array([UL[e, i, j, c] for e in range(UL.shape[0])
                         for i in range(nl) for j in range(nl)])
        got = np.array(LinearTriInterpolator(tri, vals)(qx, qy).filled(np.nan))
        bad = ~np.isfinite(got)
        if bad.any():                            # boundary round-off only
            nfill = max(nfill, int(bad.sum()))
            got[bad] = griddata((px, py), vals, (qx[bad], qy[bad]), method='nearest')
        U[..., c] = got.reshape(mshort.nelem, n, n)
    return U, nfill


def merit(st, U):
    f = np.ascontiguousarray(U[..., 0]/2.0); g = np.ascontiguousarray(U[..., 1]/2.0)
    st.update_linearisation(f, g)
    r = apply_L(st, U, f, g)
    return float(np.sum(r*r/st.mesh.wq[..., None]))


def diag(U, m):
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
    return dict(q=float(sum(fl(e, -1) for e in OUT)/sum(fl(e, 0) for e in INL)),
                div=float(np.sqrt(((ux+vy)**2).mean())),
                umax=float(np.abs(U[..., 0]).max()), xr=float(xr/H),
                psp=float(pe.max()-pe.min()), rev=float(100*np.mean(ue < 0)),
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
            if s < 8 or s % 20 == 0:
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
                d=diag(U, m) if ok else None, trace=trace, U=U, m=m)


m0, _, _ = build()
U_int, nfill = interpolate(m0)
np.savez_compressed(f'{SC}/bfsint_IC.npz', U=U_int, xnod=m0.xnod, ynod=m0.ynod, hy=m0.hy)

st0 = SolverState(m0, diff_matrix(m0.N), nu=1.0/RE, dt=0.5, fac1=1.0,
                  w_mom=WMOM, w_mass=0.0)
d_ic = diag(U_int, m0)
d_own = diag(np.load(f'{SC}/{SHORT_OWN}')['U'], m0)

print("SHORT domain, w_mom = 0.1 (w_mass = 0), p-MG, LOOSE solve")
print(f"IC = converged LONG-domain w_mom=0.1 solution, interpolated onto the short grid")
print(f"    (linear interpolation on the long node cloud, step masked; "
      f"{nfill} nearest-neighbour fills at the boundary)\n")

hdr = (f"{'':<26}{'J':>12}{'Qout/Qin':>10}{'rms div':>10}{'max|u|':>8}"
       f"{'x_r/h':>8}{'p_sprd':>8}{'rev':>7}{'mean p':>10}")
print(hdr)
row = lambda t, d, J: print(f"{t:<26}{J:>12.4e}{d['q']:>10.4f}{d['div']:>10.2e}"
                            f"{d['umax']:>8.3f}{d['xr']:>8.3f}{d['psp']:>8.3f}"
                            f"{d['rev']:>6.1f}%{d['pmean']:>10.4f}")
row('interpolated IC', d_ic, merit(st0, U_int))
row('short own state (ref)', d_own, merit(st0, np.load(f'{SC}/{SHORT_OWN}')['U']))

print(f"\n{'':<26}{'status':>10}{'it':>5}{'CG':>9}{'wall':>7}"
      f"{'J end':>12}{'Qout/Qin':>10}{'rms div':>10}{'max|u|':>8}{'x_r/h':>8}"
      f"{'p_sprd':>8}{'rev':>7}")
out = {}
for tag, up in (('solved, NO pin', False), ('solved, with pin', True)):
    r = run(U_int, up); out[tag] = r; d = r['d']
    if d is None:
        print(f"{tag:<26}{r['status']:>10}{r['it']:>5}{r['cg']:>9}{r['wall']:>7.0f}")
    else:
        print(f"{tag:<26}{r['status']:>10}{r['it']:>5}{r['cg']:>9}{r['wall']:>7.0f}"
              f"{r['J1']:>12.4e}{d['q']:>10.4f}{d['div']:>10.2e}{d['umax']:>8.3f}"
              f"{d['xr']:>8.3f}{d['psp']:>8.3f}{d['rev']:>6.1f}%")
        np.savez_compressed(f"{SC}/bfsint_{'pin' if up else 'nopin'}.npz", U=r['U'],
                            xnod=r['m'].xnod, ynod=r['m'].ynod, hy=r['m'].hy)
    print(f"{'':<26}trace: " + ", ".join(f"({a},{b:.2e},{c:.2f})" for a, b, c in r['trace'][:8]))
    sys.stdout.flush()

# Did it stay near the long-domain field, or fall back to the short-domain state?
Uo = np.load(f'{SC}/{SHORT_OWN}')['U']
b = out['solved, NO pin']['U']
if np.all(np.isfinite(b)):
    print("\n=== where did it end up? max|difference| in velocity ===")
    for tag, ref in (('vs the interpolated long-domain IC', U_int),
                     ('vs the short domain own state', Uo)):
        print(f"  {tag:<38} max|du| {np.abs(b[...,0]-ref[...,0]).max():.4e}"
              f"   max|dv| {np.abs(b[...,1]-ref[...,1]).max():.4e}")
