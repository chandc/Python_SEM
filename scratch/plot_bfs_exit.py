"""Converged BFS steady solve: outlet-plane pressure and streamlines.

The pure steady form (w_mass=0, w_mom=1) converges only with an INEXACT linear
solve (cgsfac=1e-3, tol=1e-6).  Jacobi and p-MG agree there, which is the check
that the state is real rather than preconditioner-dependent.  The tight-tolerance
Jacobi run is shown too: it hit its 120-iteration cap after 40 minutes and is NOT
converged -- included so the failure is visible rather than described.

Reference: legacy time-stepping at dt=0.5 (same grid, IC and pin).
"""
import os, sys
import numpy as np
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from lssem2d.lgl import diff_matrix

H = 0.5
CASES = [
    ('steady, Jacobi  (converged, 13 it)', 'bfs_steady_jacobi_1e-06.npz', 'tab:blue',  '-'),
    ('steady, p-MG    (converged, 11 it)', 'bfs_steadyMG_pmg_1e-06.npz',  'tab:green', '--'),
    ('steady, tight tol (NOT converged)',  'bfs_steady_jacobi_1e-10.npz', 'tab:red',   ':'),
    ('legacy dt=0.5   (323 steps)',        'bfsw_A_W0.5.npz',             'k',         '-'),
]


def load(f):
    d = np.load(f'{SC}/{f}')
    return d['U'], d['xnod'], d['ynod'], d['hy']


def outlet(U, xn, yn):
    n = U.shape[1]; xmax = xn.max()
    ys, ps, us = [], [], []
    for e in range(U.shape[0]):
        if abs(xn[e, -1]-xmax) < 1e-9:
            for j in range(n):
                ys.append(yn[e, j]); ps.append(U[e, -1, j, 2]); us.append(U[e, -1, j, 0])
    o = np.argsort(ys)
    ys, ps, us = np.array(ys)[o], np.array(ps)[o], np.array(us)[o]
    k = np.concatenate(([True], np.diff(ys) > 1e-12))
    return ys[k], ps[k], us[k]


def reattach(U, xn, yn, hy):
    n = U.shape[1]; D = diff_matrix(n-1)
    xs, tw = [], []
    for e in range(U.shape[0]):
        if yn[e, 0] > 0.01 or xn[e, 0] < -1e-9: continue
        for i in range(n):
            xs.append(xn[e, i]); tw.append(np.dot(D[0, :], U[e, i, :, 0])*(2.0/hy[e]))
    o = np.argsort(xs); xs, tw = np.array(xs)[o], np.array(tw)[o]
    for k in range(len(xs)-1):
        if tw[k] < 0 and tw[k+1] > 0 and xs[k] > 0.05:
            return xs[k]-tw[k]*(xs[k+1]-xs[k])/(tw[k+1]-tw[k])
    return np.nan


fig = plt.figure(figsize=(15.5, 8.6))
gs = fig.add_gridspec(3, 3, width_ratios=[1.15, 1.15, 1.0], hspace=.52, wspace=.28)

# ---- streamlines: converged steady, legacy, and the failed tight run --------
for row, (lab, f, c, lsty) in enumerate([CASES[0], CASES[3], CASES[2]]):
    U, xn, yn, hy = load(f)
    ne, n = U.shape[0], U.shape[1]
    px, py, pu, pv = [], [], [], []
    for e in range(ne):
        for i in range(n):
            for j in range(n):
                px.append(xn[e, i]); py.append(yn[e, j])
                pu.append(U[e, i, j, 0]); pv.append(U[e, i, j, 1])
    px, py, pu, pv = map(np.array, (px, py, pu, pv))
    for col, (xlo, xhi, dens, nx) in enumerate([(-1, 8.5, 1.7, 900), (6.0, 8.5, 3.2, 520)]):
        ax = fig.add_subplot(gs[row, col])
        gx = np.linspace(xlo, xhi, nx); gy = np.linspace(0, 1, 175)
        GX, GY = np.meshgrid(gx, gy)
        ui = griddata((px, py), pu, (GX, GY), method='linear')
        vi = griddata((px, py), pv, (GX, GY), method='linear')
        solid = (GX < 0) & (GY < 0.5)
        ui[solid] = np.nan; vi[solid] = np.nan
        ax.contourf(GX, GY, np.ma.masked_invalid(np.hypot(ui, vi)), levels=40,
                    cmap='viridis', alpha=.82, vmin=0, vmax=1.6)
        rev = np.ma.masked_where(np.nan_to_num(ui) >= 0, np.ones_like(GX))
        ax.contourf(GX, GY, rev, levels=[.5, 1.5], colors=['red'], alpha=.26)
        ax.streamplot(gx, gy, np.nan_to_num(ui), np.nan_to_num(vi), density=dens,
                      color='w', linewidth=.65, arrowsize=.7)
        if xlo < 0:
            ax.add_patch(plt.Rectangle((xlo, 0), -xlo, .5, fc='0.85', ec='k', lw=1.1, zorder=5))
        xr = reattach(U, xn, yn, hy)
        if np.isfinite(xr) and xlo < xr < xhi:
            ax.plot([xr], [0], 'r^', ms=9, zorder=7, clip_on=False)
        ax.axvline(xn.max(), color='yellow', lw=3, zorder=6)
        ax.plot([xn.max()], [0.0], marker='o', ms=8, mfc='none', mec='yellow',
                mew=2.2, zorder=8, clip_on=False)
        ax.set_xlim(xlo, xhi); ax.set_ylim(0, 1); ax.set_ylabel('y', fontsize=8)
        t = lab if col == 0 else 'exit region'
        if col == 0 and np.isfinite(xr): t += f'   x_r/h = {xr/H:.3f}'
        ax.set_title(t, fontsize=9)
        if row == 2: ax.set_xlabel('x')

# ---- outlet pressure -------------------------------------------------------
ax = fig.add_subplot(gs[:, 2])
print(f"{'case':<38}{'p_out spread':>14}{'mean p_out':>12}{'exit rev':>10}")
for lab, f, c, lsty in CASES:
    U, xn, yn, hy = load(f)
    y, p, u = outlet(U, xn, yn)
    ax.plot(p, y, lsty, color=c, lw=2.2 if c == 'k' else 1.8, label=lab)
    print(f"{lab:<38}{p.max()-p.min():>14.4f}{p.mean():>12.4f}{100*np.mean(u<0):>9.1f}%")
ax.set_xlabel('pressure at the outlet plane'); ax.set_ylabel('y')
ax.set_title('outlet pressure — the soft mode\n(flat = well resolved)', fontsize=10)
ax.grid(alpha=.3); ax.legend(fontsize=7.5, loc='lower left'); ax.set_ylim(0, 1)

fig.suptitle('BFS Chan Re=389, long domain — pure steady form (w_mass=0, w_mom=1) vs legacy time-stepping.  '
             'Red = reversed flow; yellow = free-outflow plane, circle = pressure pin.', fontsize=11)
out = f'{SC}/bfs_exit.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f"\nsaved {out}")
