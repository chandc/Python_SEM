"""SHORT domain seeded from the LONG-domain solution -- SPECTRAL interpolation.

The first attempt (bfs_interp_long.py) used linear interpolation on the node
cloud and diverged at Newton step 2 (max|u| 1.51 -> 115).  Two confounds had to
be separated:

  1. the grids genuinely differ downstream -- 61 of 122 unique short-grid x
     nodes have no counterpart in the long grid, so that was an interpolation,
     not a restriction, and LINEAR interpolation of an order-10 spectral field
     injects sub-element error;
  2. no globalisation -- the blow-up shape is what the line search exists for.

This script removes confound 1 by evaluating the long-domain polynomial exactly
(tensor-product barycentric Lagrange within the containing long element) and
tests confound 2 by running with and without the line search.

w_mom = 0.1, w_mass = 0, loose solve (cgsfac 1e-3, tol 1e-6), p-MG, no pin.
"""
import os, sys, time
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
sys.path.insert(0, SC)
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
from fgrid import load
from lssem2d.lgl import diff_matrix, lgl_weights, lgl_nodes
from lssem2d.lssem import SolverState, apply_L
from lssem2d.operators import dUdx, dUdy
import lssem2d.solver as S
from lssem2d import precond as P

RE, H, WMOM = 389.0, 0.5, 0.1
LONG = 'bfswm_0.1.npz'
SHORT_OWN = 'bfsnp2_off_nopin.npz'
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


def bary_weights(z):
    n = len(z); w = np.ones(n)
    for j in range(n):
        for k in range(n):
            if k != j:
                w[j] /= (z[j]-z[k])
    return w


def lagrange_row(z, w, xq):
    """Barycentric Lagrange basis values at xq for nodes z."""
    d = xq - z
    hit = np.where(np.abs(d) < 1e-13)[0]
    r = np.zeros(len(z))
    if hit.size:
        r[hit[0]] = 1.0
        return r
    r = w/d
    return r/r.sum()


def interpolate_spectral(mshort):
    """Evaluate the long-domain order-N polynomial at every short-grid node."""
    d = np.load(f'{SC}/{LONG}')
    UL, xl, yl = d['U'], d['xnod'], d['ynod']
    nl = UL.shape[1]
    z = lgl_nodes(nl-1); wb = bary_weights(z)
    # element bounding boxes of the long grid
    x0, x1 = xl[:, 0], xl[:, -1]
    y0, y1 = yl[:, 0], yl[:, -1]

    n = mshort.N+1
    U = np.zeros((mshort.nelem, n, n, 4))
    misses = 0
    for e in range(mshort.nelem):
        for i in range(n):
            xq = mshort.xnod[e, i]
            for j in range(n):
                yq = mshort.ynod[e, j]
                cand = np.where((xq >= x0-1e-9) & (xq <= x1+1e-9) &
                                (yq >= y0-1e-9) & (yq <= y1+1e-9))[0]
                if cand.size == 0:
                    misses += 1
                    continue
                E = cand[0]
                xi = 2.0*(xq-x0[E])/(x1[E]-x0[E]) - 1.0
                et = 2.0*(yq-y0[E])/(y1[E]-y0[E]) - 1.0
                lx = lagrange_row(z, wb, xi)
                ly = lagrange_row(z, wb, et)
                U[e, i, j, :] = np.einsum('a,b,abc->c', lx, ly, UL[E])
    return U, misses


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


def run(U0, ls):
    m, n, pin = build(); N = m.N
    st = SolverState(m, diff_matrix(N), nu=1.0/RE, dt=0.5, fac1=1.0,
                     w_mom=WMOM, w_mass=0.0)
    inlet = lambda x, y, t: 6.0*((y-0.5)/0.5)*(1.0-(y-0.5)/0.5)
    nit = [0]

    def pcg(state, b, fu, fv, M, mw, pin_p=False, max_iter=5000, tol=None,
            cgsfac=None, precond=None, **kw):
        pre = P.make('pmg2', state, fu, fv, M, pin_p,
                     pc=max(2, N//2), deg=4, coarse_deg=10)
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
                J1=merit(st, U) if ok else np.nan,
                d=diag(U, m) if ok else None, trace=trace, U=U, m=m)


m0, _, _ = build()
t0 = time.perf_counter()
U_sp, misses = interpolate_spectral(m0)
print(f"spectral interpolation: {time.perf_counter()-t0:.1f}s, {misses} nodes not located\n")
np.savez_compressed(f'{SC}/bfsint2_IC.npz', U=U_sp, xnod=m0.xnod, ynod=m0.ynod, hy=m0.hy)

st0 = SolverState(m0, diff_matrix(m0.N), nu=1.0/RE, dt=0.5, fac1=1.0,
                  w_mom=WMOM, w_mass=0.0)
U_lin = np.load(f'{SC}/bfsint_IC.npz')['U']
U_own = np.load(f'{SC}/{SHORT_OWN}')['U']

print("SHORT domain, w_mom = 0.1 (w_mass = 0), p-MG, LOOSE solve, NO pin")
print("IC = converged LONG-domain w_mom=0.1 solution\n")
hdr = (f"{'':<30}{'J':>12}{'Qout/Qin':>10}{'rms div':>10}{'max|u|':>8}"
       f"{'x_r/h':>8}{'p_sprd':>8}{'rev':>7}")
print(hdr)
for t, U in (('IC: SPECTRAL interpolation', U_sp),
             ('IC: linear interpolation', U_lin),
             ('short domain own state', U_own)):
    d = diag(U, m0)
    print(f"{t:<30}{merit(st0, U):>12.4e}{d['q']:>10.4f}{d['div']:>10.2e}"
          f"{d['umax']:>8.3f}{d['xr']:>8.3f}{d['psp']:>8.3f}{d['rev']:>6.1f}%")
print(f"\n   spectral vs linear IC: max|du| {np.abs(U_sp[...,0]-U_lin[...,0]).max():.3e}"
      f"   max|dom| {np.abs(U_sp[...,3]-U_lin[...,3]).max():.3e}")

print(f"\n{'':<30}{'status':>16}{'it':>5}{'CG':>9}{'wall':>7}{'J end':>12}"
      f"{'Qout/Qin':>10}{'rms div':>10}{'max|u|':>8}{'x_r/h':>8}{'p_sprd':>8}{'rev':>7}")
for tag, U0, ls in (('spectral IC, no LS', U_sp, False),
                    ('spectral IC, +LS', U_sp, True),
                    ('linear IC, +LS', U_lin, True)):
    r = run(U0, ls); d = r['d']
    if d is None:
        print(f"{tag:<30}{r['status']:>16}{r['it']:>5}{r['cg']:>9}{r['wall']:>7.0f}")
    else:
        print(f"{tag:<30}{r['status']:>16}{r['it']:>5}{r['cg']:>9}{r['wall']:>7.0f}"
              f"{r['J1']:>12.4e}{d['q']:>10.4f}{d['div']:>10.2e}{d['umax']:>8.3f}"
              f"{d['xr']:>8.3f}{d['psp']:>8.3f}{d['rev']:>6.1f}%")
        np.savez_compressed(f"{SC}/bfsint2_{tag.split(',')[0].replace(' ','')}"
                            f"{'_ls' if ls else ''}.npz", U=r['U'], xnod=r['m'].xnod,
                            ynod=r['m'].ynod, hy=r['m'].hy)
    print(f"{'':<30}trace: " + ", ".join(f"({a},{b:.2e},{c:.2f})" for a, b, c in r['trace'][:8]))
    sys.stdout.flush()
