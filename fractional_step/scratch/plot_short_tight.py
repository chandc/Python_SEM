"""What tightening the linear solve does to a CONVERGED short-domain BFS field.

Left column: streamlines.  Right column: the outflow-plane pressure, which is
where the damage concentrates -- the constraint rows (Qout/Qin, div) are
bit-identical across all three rows, so everything that changed is momentum-side.
"""
import os, sys
import numpy as np
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, LinearTriInterpolator
from lssem2d.lgl import diff_matrix

H = 0.5
CASES = [
    ('loose  cgsfac 1e-3 / tol 1e-6  — converged, 0 CG, J = 3.69e-05',
     'bfsst_w0.1_t1e-06.npz', 'tab:blue', '-'),
    ('tighter  1e-5 / 1e-8  — WALL at 23 it, 130,617 CG, J = 4.02e-04',
     'bfsst_w0.1_t1e-08.npz', 'tab:orange', '--'),
    ('tight  1e-8 / 1e-10  — WALL at 16 it, 131,328 CG, J = 8.04e-04',
     'bfsst_w0.1_t1e-10.npz', 'tab:red', ':'),
]


def nodes(U, xn, yn):
    ne, n = U.shape[0], U.shape[1]
    px, py, pu, pv = [], [], [], []
    for e in range(ne):
        for i in range(n):
            for j in range(n):
                px.append(xn[e, i]); py.append(yn[e, j])
                pu.append(U[e, i, j, 0]); pv.append(U[e, i, j, 1])
    return map(np.array, (px, py, pu, pv))


def reattach(U, xn, yn, hy):
    n = U.shape[1]; D = diff_matrix(n-1)
    xs, tw = [], []
    for e in range(U.shape[0]):
        if yn[e, 0] > 0.01 or xn[e, 0] < -1e-9:
            continue
        for i in range(n):
            xs.append(xn[e, i]); tw.append(np.dot(D[0, :], U[e, i, :, 0])*(2.0/hy[e]))
    o = np.argsort(xs); xs, tw = np.array(xs)[o], np.array(tw)[o]
    for k in range(len(xs)-1):
        if tw[k] < 0 and tw[k+1] > 0 and xs[k] > 0.05:
            return xs[k]-tw[k]*(xs[k+1]-xs[k])/(tw[k+1]-tw[k])
    return np.nan


def outlet(U, xn, yn):
    n = U.shape[1]; xmax = xn.max()
    ys, ps = [], []
    for e in range(U.shape[0]):
        if abs(xn[e, -1]-xmax) < 1e-9:
            for j in range(n):
                ys.append(yn[e, j]); ps.append(U[e, -1, j, 2])
    o = np.argsort(ys); ys, ps = np.array(ys)[o], np.array(ps)[o]
    k = np.concatenate(([True], np.diff(ys) > 1e-12))
    return ys[k], ps[k]


fig = plt.figure(figsize=(15.6, 8.2))
gs = fig.add_gridspec(3, 2, width_ratios=[2.35, 1.0], hspace=.46, wspace=.20)

axp = fig.add_subplot(gs[:, 1])

for row, (lab, f, c, lsty) in enumerate(CASES):
    d = np.load(f'{SC}/{f}')
    U, xn, yn, hy = d['U'], d['xnod'], d['ynod'], d['hy']
    n = U.shape[1]
    px, py, pu, pv = nodes(U, xn, yn)

    tri = Triangulation(px, py)
    cx = px[tri.triangles].mean(axis=1); cy = py[tri.triangles].mean(axis=1)
    tri.set_mask((cx < 0) & (cy < 0.5))
    fu = LinearTriInterpolator(tri, pu); fv = LinearTriInterpolator(tri, pv)

    gx = np.linspace(px.min(), px.max(), 760); gy = np.linspace(0, 1, 200)
    GX, GY = np.meshgrid(gx, gy)
    ui = np.array(fu(GX, GY).filled(np.nan)); vi = np.array(fv(GX, GY).filled(np.nan))

    ax = fig.add_subplot(gs[row, 0])
    ax.contourf(GX, GY, np.ma.masked_invalid(np.hypot(ui, vi)), levels=40,
                cmap='viridis', alpha=.82, vmin=0, vmax=2.7)
    rev = np.ma.masked_where(np.nan_to_num(ui) >= 0, np.ones_like(GX))
    ax.contourf(GX, GY, rev, levels=[.5, 1.5], colors=['red'], alpha=.28)
    ax.streamplot(gx, gy, np.nan_to_num(ui), np.nan_to_num(vi), density=2.1,
                  color='w', linewidth=.6, arrowsize=.65)
    ax.add_patch(plt.Rectangle((px.min(), 0), -px.min(), .5, fc='0.85',
                               ec='k', lw=1.1, zorder=5))
    xr = reattach(U, xn, yn, hy)
    if np.isfinite(xr):
        ax.plot([xr], [0], 'r^', ms=9, zorder=7, clip_on=False)
    ax.axvline(xn.max(), color='yellow', lw=3, zorder=6)
    OUT = [e for e in range(U.shape[0]) if abs(xn[e, -1]-xn.max()) < 1e-9]
    ue = np.array([U[e, -1, j, 0] for e in OUT for j in range(n)])
    ax.set_title(f"{lab}\nx_r/h = {xr/H:.2f}    max|u| = {np.abs(U[...,0]).max():.2f}"
                 f"    exit rev = {100*np.mean(ue<0):.0f}%", fontsize=9)
    ax.set_xlim(px.min(), px.max()); ax.set_ylim(0, 1)
    ax.set_ylabel('y', fontsize=8); ax.tick_params(labelsize=7)
    if row == 2:
        ax.set_xlabel('x', fontsize=9)

    y, p = outlet(U, xn, yn)
    axp.plot(p, y, lsty, color=c, lw=2.0,
             label=f"{lab.split('—')[0].strip()}\n   spread {p.max()-p.min():.3f}")

axp.set_xlabel('pressure on the outflow plane')
axp.set_ylabel('y')
axp.set_title('outlet pressure — where the damage goes\n(flat = well resolved)', fontsize=10)
axp.grid(alpha=.3); axp.set_ylim(0, 1); axp.legend(fontsize=7.5, loc='lower right')

fig.suptitle('BFS Chan Re=389, SHORT domain, steady form w_mom=0.1 (w_mass=0), p-MG — restarting a CONVERGED field '
             'and tightening the linear solve.\nConstraint rows are unmoved throughout: Qout/Qin = 0.9997 and rms div = 4.26e-03 in ALL three.  '
             'Red = reversed flow, yellow = outflow plane.', fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, 0.90])
out = f'{SC}/short_tight_streamlines.png'
fig.savefig(out, dpi=145, bbox_inches='tight')
print('saved', out)
