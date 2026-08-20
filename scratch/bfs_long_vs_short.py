"""LONG-domain BFS (x to 8.5) against the SHORT one (x to 2.5), free vs P+Z.

The short domain ends at x = 2.5 while reattachment sits at x_r ~ 4.1
(x_r/h ~ 8.2, h = 0.5), so its recirculation runs straight off the end and the
outflow boundary sits in REVERSED flow -- fluid entering through the "outflow".
Free outflow blows up there on step 1 from every IC.

The long domain reaches 8.5, well past reattachment, so the exit flow is clean
and unidirectional.  If free outflow WORKS there while failing on the short
domain, that pins the failure to the outlet sitting in inflow rather than to the
missing conditions per se -- the deficiency is the same in both, but only bites
when the boundary has to carry reversed flow.

Four solves, all cold start, dt = 1, w_mom = w_mass = 1:
  long / free, long / P+Z, and the short pair for reference (short/P+Z is
  reloaded from bfs_pz_state.npz rather than re-solved).
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
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState
import lssem2d.solver as S
import lssem2d.bc as BC

RE, H = 389.0, 0.5
LONG = '/Users/danielchan/Dropbox/F90_SEM/pmg_clean/cnos_long_grid.dat'
OB = BC.apply_bc


def solve(grid, pz, cap=400, wall=1500.0):
    m, _, _ = load(grid); n = m.N+1; N = m.N
    D = diff_matrix(N)
    ipin = next((e, n-1, 0) for e in range(m.nelem)
                if m.bc[e, 1] == 4 and m.bc[e, 2] == 1)
    for e in range(m.nelem):
        if m.bc[e, 1] == 4 and not pz:
            m.bc[e, 1] = 0
    xmax = m.xnod.max()
    out = [e for e in range(m.nelem) if abs(m.xnod[e, -1]-xmax) < 1e-9]
    pin = False if pz else ipin

    def bc2(mesh, U, **kw):
        U = OB(mesh, U, **kw)
        for e in out:
            U[e, -1, :, 3] = -(D[-1, :-1] @ U[e, :-1, :, 3])/D[-1, -1]
        return U

    st = SolverState(m, D, nu=1.0/RE, dt=1.0, fac1=1.0, w_mom=1.0, w_mass=1.0)
    if pz:
        st.get_global_mask(pin_p=pin)
        for e in out:
            st._global_mask[e, -1, :, 2] = 0.0
            st._global_mask[e, -1, :, 3] = 0.0
        S.apply_bc = bc2
    inlet = lambda x, y, t: 6.0*((y-0.5)/0.5)*(1.0-(y-0.5)/0.5)
    U = np.zeros((m.nelem, n, n, 4)); h = [U.copy()]
    t0 = time.perf_counter(); status = 'CAP'; d = np.nan
    try:
        for s in range(cap):
            prev = h[0].copy()
            U = S.step_bdf(st, h, time=s, max_newton=1, newton_tol=1e-12,
                           newton_factor=0.0, custom_inlet=inlet, pin_p=pin,
                           cgsfac=1e-8, cg_tol=1e-10, cg_max_iter=300000)
            if not np.all(np.isfinite(U)):
                status = 'NaN'; break
            d = float(np.abs(U-prev).max())
            if np.abs(U[..., 0]).max() > 20.0:
                status = 'BLEWUP'; break
            if d < 1e-12:
                status = 'conv'; break
            if time.perf_counter()-t0 > wall:
                status = 'WALLCAP'; break
    finally:
        S.apply_bc = OB
    mu = float(np.abs(U[..., 0]).max()) if np.all(np.isfinite(U)) else np.nan
    print(f"  {'P+Z' if pz else 'free':>4}: {status:>8}  {s+1:>4} steps  "
          f"|dU| = {d:.3e}  max|u| = {mu:.4f}  {time.perf_counter()-t0:.0f}s",
          flush=True)
    # SAVE the field.  These solves are ~25 min each; not persisting them meant a
    # 50 min re-solve just to plot velocity profiles.  Learned the hard way.
    np.savez(f"{SC}/bfs_long_{'pz' if pz else 'free'}.npz",
             U=U, xnod=m.xnod, ynod=m.ynod, hy=m.hy, N=m.N, status=status)
    return U, m, status


def reattach(U, xn, yn, hy):
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


def panel(ax, U, xn, yn, hy, title, xlim):
    n = U.shape[1]
    px, py, pu, pv = [], [], [], []
    for e in range(U.shape[0]):
        for i in range(n):
            for j in range(n):
                px.append(xn[e, i]); py.append(yn[e, j])
                pu.append(U[e, i, j, 0]); pv.append(U[e, i, j, 1])
    px, py, pu, pv = map(np.array, (px, py, pu, pv))
    tri = Triangulation(px, py)
    cx = px[tri.triangles].mean(axis=1); cy = py[tri.triangles].mean(axis=1)
    tri.set_mask((cx < 0) & (cy < 0.5))
    fu = LinearTriInterpolator(tri, pu); fv = LinearTriInterpolator(tri, pv)
    gx = np.linspace(px.min(), px.max(), 1100); gy = np.linspace(0, 1, 240)
    GX, GY = np.meshgrid(gx, gy)
    ui = np.array(fu(GX, GY).filled(np.nan)); vi = np.array(fv(GX, GY).filled(np.nan))
    ax.contourf(GX, GY, np.ma.masked_invalid(np.hypot(ui, vi)), levels=40,
                cmap='viridis', vmin=0, vmax=1.6)
    rev = np.ma.masked_where(np.nan_to_num(ui) >= 0, np.ones_like(GX))
    ax.contourf(GX, GY, rev, levels=[.5, 1.5], colors=['red'], alpha=.28)
    ax.streamplot(gx, gy, np.nan_to_num(ui), np.nan_to_num(vi), density=2.6,
                  color='w', linewidth=.65, arrowsize=.7)
    ax.add_patch(plt.Rectangle((px.min(), 0), -px.min(), .5, fc='0.85',
                               ec='k', lw=1.2, zorder=5))
    xr = reattach(U, xn, yn, hy)
    if np.isfinite(xr):
        ax.plot([xr], [0], 'r^', ms=11, zorder=7, clip_on=False)
    ax.axvline(xn.max(), color='yellow', lw=3.5, zorder=6)
    ax.set_xlim(*xlim); ax.set_ylim(0, 1); ax.set_ylabel('y')
    xrs = f"x_r = {xr:.3f}  (x_r/h = {xr/H:.2f})" if np.isfinite(xr) else "x_r: none in domain"
    ax.set_title(f"{title}   |   {xrs}", fontsize=10)
    return xr


print("LONG domain BFS (x to 8.5), cold start, dt = 1, w_mom = w_mass = 1")
RL = {}
for pz in (False, True):
    RL[pz] = solve(LONG, pz)

d = np.load(f'{SC}/bfs_pz_state.npz')
Us, xs_, ys_, hys = d['U'], d['xnod'], d['ynod'], d['hy']

rows = [(Us, xs_, ys_, hys, 'SHORT domain, P+Z  (converged)', (-1, 8.5))]
for pz in (True, False):
    U, m, st = RL[pz]
    if st == 'conv':
        rows.append((U, m.xnod, m.ynod, m.hy,
                     f"LONG domain, {'P+Z' if pz else 'free outflow'}  (converged)", (-1, 8.5)))
    else:
        print(f"  long/{'P+Z' if pz else 'free'} did not converge ({st}) -- not plotted")

fig, axs = plt.subplots(len(rows), 1, figsize=(14.5, 3.3*len(rows)))
if len(rows) == 1:
    axs = [axs]
for ax, (U, xn, yn, hy, t, xl) in zip(axs, rows):
    panel(ax, U, xn, yn, hy, t, xl)
axs[-1].set_xlabel('x')
fig.suptitle('BFS Re = 389: short domain truncates the recirculation, long domain '
             'contains it.\nRed = reversed flow, yellow = outlet plane, '
             '▲ = reattachment.', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(f'{SC}/../figs/bfs_long_vs_short.png', dpi=125, bbox_inches='tight')
print('\nfigs/bfs_long_vs_short.png')
