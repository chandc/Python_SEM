"""The interpolated IC: what the LONG-domain solution looks like on the short grid.

Row 1  the long-domain source (x up to 8.5), with the short domain's outflow
       plane marked at x = 2.5 -- note it cuts through the recirculation, which
       does not reattach until x_r/h = 8.33 (x = 4.17).
Row 2  that solution interpolated spectrally onto the short grid: this is the
       PHYSICALLY CORRECT field for this region.
Row 3  the short domain's own converged state at the same w_mom, for contrast.

The exit-plane pressure of row 2 IS the long-domain pressure at x = 2.5, since
the spectral interpolation is exact.  Levels are arbitrary (no pin), so the
comparison panel is de-meaned; the IC's excursion is ~160x smaller and gets its
own zoomed axis.
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
XCUT = 2.5
PANELS = [
    ('LONG domain, w_mom = 0.1 (the source)   x_r/h = 8.33', 'bfswm_0.1.npz'),
    ('interpolated onto the SHORT grid — the physically correct field', 'bfsint2_IC.npz'),
    ('SHORT domain own converged state, same w_mom', 'bfsnp2_off_nopin.npz'),
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


fig = plt.figure(figsize=(16.0, 8.0))
gs = fig.add_gridspec(3, 3, width_ratios=[2.5, 0.95, 0.95], hspace=.50, wspace=.32)

for row, (lab, f) in enumerate(PANELS):
    U, xn, yn, hy = load(f)
    n = U.shape[1]
    px, py, pu, pv = nodes(U, xn, yn)
    tri = Triangulation(px, py)
    cx = px[tri.triangles].mean(axis=1); cy = py[tri.triangles].mean(axis=1)
    tri.set_mask((cx < 0) & (cy < 0.5))
    fu = LinearTriInterpolator(tri, pu); fv = LinearTriInterpolator(tri, pv)
    gx = np.linspace(px.min(), px.max(), 1100); gy = np.linspace(0, 1, 200)
    GX, GY = np.meshgrid(gx, gy)
    ui = np.array(fu(GX, GY).filled(np.nan)); vi = np.array(fv(GX, GY).filled(np.nan))

    ax = fig.add_subplot(gs[row, 0])
    ax.contourf(GX, GY, np.ma.masked_invalid(np.hypot(ui, vi)), levels=40,
                cmap='viridis', alpha=.82, vmin=0, vmax=2.5)
    rev = np.ma.masked_where(np.nan_to_num(ui) >= 0, np.ones_like(GX))
    ax.contourf(GX, GY, rev, levels=[.5, 1.5], colors=['red'], alpha=.28)
    ax.streamplot(gx, gy, np.nan_to_num(ui), np.nan_to_num(vi), density=1.9,
                  color='w', linewidth=.6, arrowsize=.65)
    ax.add_patch(plt.Rectangle((px.min(), 0), -px.min(), .5, fc='0.85',
                               ec='k', lw=1.1, zorder=5))
    if row == 0:
        ax.axvline(XCUT, color='cyan', lw=2.4, ls='--', zorder=7)
        ax.text(XCUT+0.08, 0.88, 'short domain ends here', color='cyan',
                fontsize=8.5, zorder=8)
        ax.plot([8.331*H], [0], 'r^', ms=9, zorder=8, clip_on=False)
    else:
        ax.axvline(xn.max(), color='yellow', lw=3, zorder=6)
    ax.set_xlim(px.min(), px.max()); ax.set_ylim(0, 1)
    ax.set_ylabel('y', fontsize=8); ax.tick_params(labelsize=7)
    ax.set_title(f'{lab}    max|u| = {np.abs(U[...,0]).max():.3f}', fontsize=9.5)
    if row == 2:
        ax.set_xlabel('x', fontsize=9)

# ---- exit-plane pressure at x = 2.5 -----------------------------------------
Uic, xi, yi, _ = load('bfsint2_IC.npz')
Uo, xo, yo, _ = load('bfsnp2_off_nopin.npz')
y1, p1 = outlet(Uic, xi, yi)
y2, p2 = outlet(Uo, xo, yo)

axb = fig.add_subplot(gs[:, 1])
axb.plot(p1-p1.mean(), y1, '-', color='tab:green', lw=2.6,
         label=f'interpolated IC\n(= long domain at x=2.5)\nspread {p1.max()-p1.min():.4f}')
axb.plot(p2-p2.mean(), y2, '--', color='tab:red', lw=2.0,
         label=f'short domain own state\nspread {p2.max()-p2.min():.4f}')
axb.set_xlabel('p - mean(p)  at x = 2.5'); axb.set_ylabel('y')
axb.set_title('exit-plane pressure\nsame axis — the IC is nearly flat', fontsize=9.5)
axb.grid(alpha=.3); axb.set_ylim(0, 1); axb.legend(fontsize=7, loc='lower right')

axz = fig.add_subplot(gs[:, 2])
axz.plot(p1-p1.mean(), y1, '-', color='tab:green', lw=2.6)
axz.set_xlabel('p - mean(p)  at x = 2.5'); axz.set_ylabel('y')
axz.set_title(f'the IC alone, zoomed\nspread {p1.max()-p1.min():.4f}  '
              f'({(p2.max()-p2.min())/(p1.max()-p1.min()):.0f}x smaller)', fontsize=9.5)
axz.grid(alpha=.3); axz.set_ylim(0, 1)
axz.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

fig.suptitle('BFS Chan Re=389 — seeding the SHORT domain with the converged LONG-domain solution (w_mom = 0.1, w_mass = 0).\n'
             'Red = reversed flow.  The correct field has max|u| = 1.513 (physical peak 1.5) and a flat exit pressure; '
             "the short domain's own state has 2.494 and a spread of 3.87.",
             fontsize=10.5, y=1.045)
fig.tight_layout(rect=[0, 0, 1, 0.965])
out = f'{SC}/interp_ic.png'
fig.savefig(out, dpi=145, bbox_inches='tight')
print('saved', out)
print(f"IC   exit p: spread {p1.max()-p1.min():.5f}  mean {p1.mean():+.4f}")
print(f"own  exit p: spread {p2.max()-p2.min():.5f}  mean {p2.mean():+.4f}")
