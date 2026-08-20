"""Streamlines and pressure for the short-domain BFS converged under the
admissible outflow pair (p = 0 and d(omega)/dx = 0).

Free outflow cannot be plotted: it blows up on step 1 from every initial
condition tried (max|u| 3603 / 2890 / 398 against a physical 1.5), so there is
no field to show.  The two panels are therefore the SAME boundary condition from
two very different starts -- U = 0, and the local fully developed parabola
everywhere -- which converged to J = 4.451 and max|u| = 1.500 in both cases.
If the fields agree, the short domain has ONE state under a proper outflow
condition, not the two that STEADY_FORM_STUDY.md sec 8 reports.

Reattachment uses plot_short_tight.py's detector, which selects bottom-wall
elements GEOMETRICALLY (y0 ~ 0, x >= 0).  bfs_outflow_ic.py selected them by BC
code and returned nan; that is the suspected cause and this checks it.
"""
import os, sys, time
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
sys.path.insert(0, SC)
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, LinearTriInterpolator
import lssem2d
lssem2d.set_backend('numpy')
from fgrid import load
from lssem2d.lgl import diff_matrix, lgl_weights
from lssem2d.lssem import SolverState, apply_L
import lssem2d.solver as S
import lssem2d.bc as BC
from bfs_outflow_ic import ic_para, GRID, RE

H = 0.5
OB = BC.apply_bc


def solve(ic, cap=400, wall=700.0):
    m, _, _ = load(GRID); n = m.N+1; N = m.N
    pin = next((e, n-1, 0) for e in range(m.nelem)
               if m.bc[e, 1] == 4 and m.bc[e, 2] == 1)
    D = diff_matrix(N)
    xmax = m.xnod.max()
    out = [e for e in range(m.nelem) if abs(m.xnod[e, -1]-xmax) < 1e-9]

    def bc2(mesh, U, **kw):
        U = OB(mesh, U, **kw)
        for e in out:
            U[e, -1, :, 3] = -(D[-1, :-1] @ U[e, :-1, :, 3])/D[-1, -1]
        return U

    st = SolverState(m, D, nu=1.0/RE, dt=1.0, fac1=1.0, w_mom=1.0, w_mass=1.0)
    st.get_global_mask(pin_p=False)
    for e in out:
        st._global_mask[e, -1, :, 2] = 0.0
        st._global_mask[e, -1, :, 3] = 0.0
    S.apply_bc = bc2
    inlet = lambda x, y, t: 6.0*((y-0.5)/0.5)*(1.0-(y-0.5)/0.5)
    U = np.zeros((m.nelem, n, n, 4)) if ic == 'cold' else ic_para(m, n)
    t0 = time.perf_counter(); d = np.nan
    try:
        h = [U.copy()]
        for s in range(cap):
            prev = h[0].copy()
            U = S.step_bdf(st, h, time=s, max_newton=1, newton_tol=1e-12,
                           newton_factor=0.0, custom_inlet=inlet, pin_p=False,
                           cgsfac=1e-8, cg_tol=1e-10, cg_max_iter=300000)
            if not np.all(np.isfinite(U)):
                break
            d = float(np.abs(U-prev).max())
            if d < 1e-12 or time.perf_counter()-t0 > wall:
                break
    finally:
        S.apply_bc = OB
    f = np.ascontiguousarray(U[..., 0]/2.0); g = np.ascontiguousarray(U[..., 1]/2.0)
    st.update_linearisation(f, g)
    r = apply_L(st, U, f, g)
    J = float(np.sum(r*r/m.wq[..., None]))
    print(f"  {ic:>5}: {s+1} steps, |dU| = {d:.3e}, J = {J:.4e}, "
          f"max|u| = {np.abs(U[...,0]).max():.4f}", flush=True)
    return U, m, J, d


def reattach(U, xn, yn, hy):
    """plot_short_tight.py's detector: bottom-wall elements chosen GEOMETRICALLY."""
    n = U.shape[1]; D = diff_matrix(n-1)
    xs, tw = [], []
    for e in range(U.shape[0]):
        if yn[e, 0] > 0.01 or xn[e, 0] < -1e-9:
            continue
        for i in range(n):
            xs.append(xn[e, i])
            tw.append(np.dot(D[0, :], U[e, i, :, 0])*(2.0/hy[e]))
    o = np.argsort(xs); xs, tw = np.array(xs)[o], np.array(tw)[o]
    for k in range(len(xs)-1):
        if tw[k] < 0 and tw[k+1] > 0 and xs[k] > 0.05:
            return xs[k]-tw[k]*(xs[k+1]-xs[k])/(tw[k+1]-tw[k])
    return np.nan


def nodes(U, xn, yn, k):
    ne, n = U.shape[0], U.shape[1]
    px, py, pq = [], [], []
    for e in range(ne):
        for i in range(n):
            for j in range(n):
                px.append(xn[e, i]); py.append(yn[e, j]); pq.append(U[e, i, j, k])
    return map(np.array, (px, py, pq))


print("Short BFS, Re = 389, dt = 1, w_mom = w_mass = 1, outlet p = 0 and dw/dx = 0")
RES = {}
for ic in ('cold', 'para'):
    RES[ic] = solve(ic)

fig = plt.figure(figsize=(15.0, 8.6))
gs = fig.add_gridspec(3, 2, width_ratios=[2.3, 1.0], hspace=.42, wspace=.22)
axw = fig.add_subplot(gs[0, 1])     # bottom-wall pressure
axo = fig.add_subplot(gs[1, 1])     # outlet-plane pressure
axd = fig.add_subplot(gs[2, 1])     # cold vs para difference

for row, ic in enumerate(('cold', 'para')):
    U, m, J, d = RES[ic]
    xn, yn, hy, n = m.xnod, m.ynod, m.hy, m.N+1
    px, py, pu = nodes(U, xn, yn, 0)
    _, _, pv = nodes(U, xn, yn, 1)
    tri = Triangulation(px, py)
    cx = px[tri.triangles].mean(axis=1); cy = py[tri.triangles].mean(axis=1)
    tri.set_mask((cx < 0) & (cy < 0.5))
    fu = LinearTriInterpolator(tri, pu); fv = LinearTriInterpolator(tri, pv)
    gx = np.linspace(px.min(), px.max(), 760); gy = np.linspace(0, 1, 200)
    GX, GY = np.meshgrid(gx, gy)
    ui = np.array(fu(GX, GY).filled(np.nan)); vi = np.array(fv(GX, GY).filled(np.nan))

    ax = fig.add_subplot(gs[row, 0])
    ax.contourf(GX, GY, np.ma.masked_invalid(np.hypot(ui, vi)), levels=40,
                cmap='viridis', alpha=.85)
    rev = np.ma.masked_where(np.nan_to_num(ui) >= 0, np.ones_like(GX))
    ax.contourf(GX, GY, rev, levels=[.5, 1.5], colors=['red'], alpha=.30)
    ax.streamplot(gx, gy, np.nan_to_num(ui), np.nan_to_num(vi), density=2.2,
                  color='w', linewidth=.6, arrowsize=.65)
    ax.add_patch(plt.Rectangle((px.min(), 0), -px.min(), .5, fc='0.85',
                               ec='k', lw=1.1, zorder=5))
    xr = reattach(U, xn, yn, hy)
    if np.isfinite(xr):
        ax.plot([xr], [0], 'r^', ms=10, zorder=7, clip_on=False)
    ax.axvline(xn.max(), color='yellow', lw=3, zorder=6)
    ax.set_xlim(px.min(), px.max()); ax.set_ylim(0, 1)
    ax.set_title(f"IC = {ic}   |   J = {J:.4e},  max|u| = {np.abs(U[...,0]).max():.4f},  "
                 f"x_r/h = {xr/H:.3f}" if np.isfinite(xr) else
                 f"IC = {ic}   |   J = {J:.4e},  max|u| = {np.abs(U[...,0]).max():.4f},  "
                 f"x_r = none detected", fontsize=10)
    ax.set_ylabel('y')
    if row == 1:
        ax.set_xlabel('x')

    # bottom-wall pressure p(x, y=0)
    xs, ps = [], []
    for e in range(m.nelem):
        if yn[e, 0] > 0.01 or xn[e, 0] < -1e-9:
            continue
        for i in range(n):
            xs.append(xn[e, i]); ps.append(U[e, i, 0, 2])
    o = np.argsort(xs)
    axw.plot(np.array(xs)[o], np.array(ps)[o], lw=1.8, label=f'IC = {ic}')
    # outlet plane p(y)
    ys, po = [], []
    xmax = xn.max()
    for e in range(m.nelem):
        if abs(xn[e, -1]-xmax) < 1e-9:
            for j in range(n):
                ys.append(yn[e, j]); po.append(U[e, -1, j, 2])
    o = np.argsort(ys)
    axo.plot(np.array(po)[o], np.array(ys)[o], lw=1.8, label=f'IC = {ic}')

axw.set_title('bottom-wall pressure  p(x, y=0)', fontsize=10)
axw.set_xlabel('x'); axw.set_ylabel('p'); axw.grid(alpha=.3); axw.legend(fontsize=8)
axo.set_title('outlet-plane pressure  p(y)   [imposed = 0]', fontsize=10)
axo.set_xlabel('p'); axo.set_ylabel('y'); axo.grid(alpha=.3); axo.legend(fontsize=8)

du = np.abs(RES['cold'][0]-RES['para'][0])
axd.bar(range(4), [du[..., k].max() for k in range(4)],
        color=['tab:blue', 'tab:orange', 'tab:green', 'tab:red'])
axd.set_xticks(range(4)); axd.set_xticklabels(['u', 'v', 'p', 'ω'])
axd.set_yscale('log'); axd.grid(alpha=.3, axis='y')
axd.set_title('max |cold − para|  per field\n(same state ⇒ one basin)', fontsize=10)

fig.suptitle('Short-domain BFS, Re = 389, admissible outflow pair '
             '(p = 0, ∂ω/∂x = 0).  Free outflow blows up on step 1 and cannot '
             'be shown.', fontsize=11.5)
fig.savefig(f'{SC}/../figs/bfs_pz_streamlines.png', dpi=125, bbox_inches='tight')
print(f"\nmax |cold - para| per field: " +
      "  ".join(f"{k}={du[...,i].max():.3e}" for i, k in enumerate('uvpw')))
print("figs/bfs_pz_streamlines.png")
