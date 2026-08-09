"""Streamlines across the SHORT-domain w_mom sweep (steady form, w_mass = 0).

Interpolation is matplotlib.tri rather than scipy.griddata -- scipy's fitpack
import is broken in this environment.

Every panel is the same converged-start protocol; the legacy time-stepping
field that seeded the spin-up is shown first for reference.
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
PANELS = [
    ('legacy dt=0.5 (time-stepping start)', 'dt_dt0p5_devc_short_state.npz'),
    ('spin-up: steady w_mom=1 (CAP, 60 it)', 'bfs_short_steady_w1.npz'),
    ('w_mom = 0.1   (conv, 5 it)',  'bfswms_0.1.npz'),
    ('w_mom = 0.2   (conv, 5 it)',  'bfswms_0.2.npz'),
    ('w_mom = 0.3   (conv, 11 it)', 'bfswms_0.3.npz'),
    ('w_mom = 0.5   (conv, 47 it)', 'bfswms_0.5.npz'),
    ('w_mom = 0.7   (conv, 42 it)', 'bfswms_0.7.npz'),
    ('w_mom = 1.0   (CAP, 60 it)',  'bfswms_1.npz'),
    ('w_mom = 1.5   (CAP, 60 it)',  'bfswms_1.5.npz'),
    ('w_mom = 2.0   (CAP, 60 it)',  'bfswms_2.npz'),
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


fig, axs = plt.subplots(5, 2, figsize=(15.0, 11.4))
axs = axs.ravel()

for ax, (lab, f) in zip(axs, PANELS):
    d = np.load(f'{SC}/{f}')
    U, xn, yn, hy = d['U'], d['xnod'], d['ynod'], d['hy']
    n = U.shape[1]
    px, py, pu, pv = nodes(U, xn, yn)

    # mask the solid step block out of the triangulation
    tri = Triangulation(px, py)
    cx = px[tri.triangles].mean(axis=1); cy = py[tri.triangles].mean(axis=1)
    tri.set_mask((cx < 0) & (cy < 0.5))
    fu = LinearTriInterpolator(tri, pu)
    fv = LinearTriInterpolator(tri, pv)

    gx = np.linspace(px.min(), px.max(), 700)
    gy = np.linspace(0, 1, 190)
    GX, GY = np.meshgrid(gx, gy)
    ui = np.array(fu(GX, GY).filled(np.nan))
    vi = np.array(fv(GX, GY).filled(np.nan))

    ax.contourf(GX, GY, np.ma.masked_invalid(np.hypot(ui, vi)), levels=40,
                cmap='viridis', alpha=.82, vmin=0, vmax=1.8)
    rev = np.ma.masked_where(np.nan_to_num(ui) >= 0, np.ones_like(GX))
    ax.contourf(GX, GY, rev, levels=[.5, 1.5], colors=['red'], alpha=.28)
    ax.streamplot(gx, gy, np.nan_to_num(ui), np.nan_to_num(vi), density=2.0,
                  color='w', linewidth=.6, arrowsize=.65)
    ax.add_patch(plt.Rectangle((px.min(), 0), -px.min(), .5, fc='0.85',
                               ec='k', lw=1.1, zorder=5))

    xr = reattach(U, xn, yn, hy)
    if np.isfinite(xr):
        ax.plot([xr], [0], 'r^', ms=9, zorder=7, clip_on=False)
    ax.axvline(xn.max(), color='yellow', lw=3, zorder=6)

    # outflow diagnostics for the title
    OUT = [e for e in range(U.shape[0]) if abs(xn[e, -1]-xn.max()) < 1e-9]
    ue = np.array([U[e, -1, j, 0] for e in OUT for j in range(n)])
    t = lab
    if np.isfinite(xr):
        t += f'    x_r/h = {xr/H:.2f}'
    t += f'    max|u| = {np.abs(U[..., 0]).max():.2f}    rev = {100*np.mean(ue<0):.0f}%'
    ax.set_title(t, fontsize=9.5)
    ax.set_xlim(px.min(), px.max()); ax.set_ylim(0, 1)
    ax.set_ylabel('y', fontsize=8)
    ax.tick_params(labelsize=7)

for ax in axs[-2:]:
    ax.set_xlabel('x', fontsize=9)

fig.suptitle('BFS Chan Re=389, SHORT domain (L/h = 5) — steady form (w_mass = 0), p-MG, loose solve.\n'
             'Red = reversed flow;  yellow = free-outflow plane;  red triangle = lower-wall reattachment.  '
             'Physical inlet peak is max|u| = 1.5.', fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.945])
out = f'{SC}/wmom_short_streamlines.png'
fig.savefig(out, dpi=140, bbox_inches='tight')
print('saved', out)
