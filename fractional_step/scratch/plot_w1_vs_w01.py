"""w_mom = 0.1 vs 1.0 on the SHORT domain, NO pressure pin.

Both are steady-form (w_mass = 0), p-MG, loose solve, no pin.  The w_mom = 0.1
field is the converged start; w_mom = 1.0 is what Newton reaches from it, shown
both as it capped (60 iters) and as it converges with the non-monotone line
search (53 iters) -- those two are the same state, which is the point.

The exit pressure is plotted twice: raw, and with the mean removed.  With no pin
the pressure level is arbitrary (an exact null mode), so only the DE-MEANED
curve is physically comparable between runs.
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
    ('w_mom = 0.1   no pin   (converged, 4 it)',      'bfsnp2_off_nopin.npz', 'tab:blue',  '-'),
    ('w_mom = 1.0   no pin   (CAP, 60 it)',           'bfsw1_nopin.npz',      'tab:orange', '--'),
    ('w_mom = 1.0   no pin + line search (conv, 53 it)', 'bfsw1ls_nopin.npz', 'tab:red',   ':'),
]


def load(f):
    d = np.load(f'{SC}/{f}')
    return d['U'], d['xnod'], d['ynod'], d['hy']


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
    ys, ps, us = [], [], []
    for e in range(U.shape[0]):
        if abs(xn[e, -1]-xmax) < 1e-9:
            for j in range(n):
                ys.append(yn[e, j]); ps.append(U[e, -1, j, 2]); us.append(U[e, -1, j, 0])
    o = np.argsort(ys)
    ys, ps, us = np.array(ys)[o], np.array(ps)[o], np.array(us)[o]
    k = np.concatenate(([True], np.diff(ys) > 1e-12))
    return ys[k], ps[k], us[k]


fig = plt.figure(figsize=(15.6, 8.4))
gs = fig.add_gridspec(3, 3, width_ratios=[2.25, 1.0, 1.0], hspace=.46, wspace=.30)

for row, (lab, f, c, lsty) in enumerate(CASES):
    U, xn, yn, hy = load(f)
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
                cmap='viridis', alpha=.82, vmin=0, vmax=2.6)
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
    y, p, ue = outlet(U, xn, yn)
    ax.set_title(f"{lab}\nx_r/h = {xr/H:.3f}    max|u| = {np.abs(U[...,0]).max():.3f}"
                 f"    exit rev = {100*np.mean(ue<0):.1f}%", fontsize=9)
    ax.set_xlim(px.min(), px.max()); ax.set_ylim(0, 1)
    ax.set_ylabel('y', fontsize=8); ax.tick_params(labelsize=7)
    if row == 2:
        ax.set_xlabel('x', fontsize=9)

axr = fig.add_subplot(gs[:, 1])
axd = fig.add_subplot(gs[:, 2])
print(f"{'case':<50}{'p spread':>10}{'mean p':>10}{'exit rev':>10}")
for lab, f, c, lsty in CASES:
    U, xn, yn, hy = load(f)
    y, p, ue = outlet(U, xn, yn)
    axr.plot(p, y, lsty, color=c, lw=2.2, label=f'{lab.split("(")[0].strip()}')
    axd.plot(p-p.mean(), y, lsty, color=c, lw=2.2,
             label=f'spread {p.max()-p.min():.3f}')
    print(f"{lab:<50}{p.max()-p.min():>10.4f}{p.mean():>10.4f}{100*np.mean(ue<0):>9.1f}%")

axr.set_xlabel('pressure on the outflow plane'); axr.set_ylabel('y')
axr.set_title('exit pressure, RAW\n(level is arbitrary with no pin)', fontsize=9.5)
axr.grid(alpha=.3); axr.set_ylim(0, 1); axr.legend(fontsize=7, loc='lower right')

axd.set_xlabel('p - mean(p) on the outflow plane'); axd.set_ylabel('y')
axd.set_title('exit pressure, DE-MEANED\nthe physically comparable one', fontsize=9.5)
axd.grid(alpha=.3); axd.set_ylim(0, 1); axd.legend(fontsize=7.5, loc='lower right')

fig.suptitle('BFS Chan Re=389, SHORT domain, steady form (w_mass = 0), p-MG, loose solve, NO pressure pin — w_mom 0.1 vs 1.0.\n'
             'Red = reversed flow, yellow = free-outflow plane, triangle = lower-wall reattachment.',
             fontsize=10.5, y=1.02)
fig.tight_layout(rect=[0, 0, 1, 0.965])
out = f'{SC}/w1_vs_w01_nopin.png'
fig.savefig(out, dpi=145, bbox_inches='tight')
print('\nsaved', out)
