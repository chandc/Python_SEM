"""Armaly-specification BFS: streamlines, u/v/p profiles, and the one experimental
comparison that is actually available.

WHAT CAN BE COMPARED WITH EXPERIMENT, AND WHAT CANNOT.
Armaly (JFM 127, 1983) measured, in the laminar range:
  * reattachment length x1/S vs Re  (his figure 4) -- YES, we can compare, and it
    is the quantity the whole case is anchored on.
  * velocity profiles -- only at Re = 1095 and Re = 1290 (figures 5, 6), BOTH above
    his own Re < 400 limit for two-dimensionality (p.474).  There is NO experimental
    velocity profile at Re ~ 389.  The u/v/p panels below are therefore code-to-code
    (P+Z vs free, long vs short), not code-to-experiment.

Setup: grids/armaly_er194_{short,long}_grid.dat -- expansion ratio 1.94, no-slip
top AND bottom (the F90 armaly_* grids have a SYMMETRY top, which gives x1/S = 18),
nu = 2h/Re = 5.141388e-03 for Armaly's D = 2h convention.
"""
import os, sys
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
sys.path.insert(0, SC)
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, LinearTriInterpolator
from lssem2d.lgl import diff_matrix, lgl_weights

S_STEP, H_TOT = 0.94, 1.94
STATIONS_S = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]          # in x/S
CASES = [('armaly_long_pz.npz',   'LONG / P+Z',   'tab:green', dict(ls='-', lw=2.2)),
         ('armaly_short_pz.npz',  'SHORT / P+Z',  'tab:blue',
          dict(ls='none', marker='o', ms=4.6, mfc='none', mew=1.3)),
         ('armaly_short_free.npz', 'SHORT / free', 'tab:red',
          dict(ls='none', marker='s', ms=4.2, mfc='none', mew=1.1))]
# long/free BLEW UP (max|u| = 21.4) -- nothing to plot


def pack(f, lab, col, sty):
    d = np.load(f'{SC}/{f}')
    U, xn, yn, hy = d['U'].copy(), d['xnod'], d['ynod'], d['hy']
    n = U.shape[1]; wq = lgl_weights(n-1); xmin = xn.min()
    tot = a = 0.0
    for e in range(U.shape[0]):
        if abs(xn[e, 0]-xmin) < 1e-9:
            tot += np.sum(wq*U[e, 0, :, 2])*(hy[e]/2); a += hy[e]
    U[..., 2] -= tot/a                                   # common datum
    px, py, q = [], [], [[], [], []]
    for e in range(U.shape[0]):
        for i in range(n):
            for j in range(n):
                px.append(xn[e, i]); py.append(yn[e, j])
                for k in range(3):
                    q[k].append(U[e, i, j, k])
    px, py = np.array(px), np.array(py)
    tri = Triangulation(px, py)
    cx = px[tri.triangles].mean(1); cy = py[tri.triangles].mean(1)
    tri.set_mask((cx < 0) & (cy < S_STEP))
    return dict(lab=lab, col=col, sty=sty, U=U, xn=xn, yn=yn, hy=hy, n=n,
                xmin=px.min(), xmax=px.max(), status=str(d['status']),
                f=[LinearTriInterpolator(tri, np.array(q[k])) for k in range(3)])


def reatt(c):
    n = c['n']; D = diff_matrix(n-1); xs, tw = [], []
    for e in range(c['U'].shape[0]):
        if c['yn'][e, 0] > 0.01 or c['xn'][e, 0] < -1e-9:
            continue
        for i in range(n):
            xs.append(c['xn'][e, i])
            tw.append(np.dot(D[0, :], c['U'][e, i, :, 0])*(2.0/c['hy'][e]))
    o = np.argsort(xs); xs, tw = np.array(xs)[o], np.array(tw)[o]
    for k in range(len(xs)-1):
        if tw[k] < 0 and tw[k+1] > 0 and xs[k] > 0.05:
            return xs[k]-tw[k]*(xs[k+1]-xs[k])/(tw[k+1]-tw[k])
    return float('nan')


C = [pack(*c) for c in CASES]

# ---------- 1. streamlines ----------
fig, axs = plt.subplots(len(C), 1, figsize=(14.5, 2.9*len(C)))
for ax, c in zip(axs, C):
    gx = np.linspace(c['xmin'], c['xmax'], 1200); gy = np.linspace(0, H_TOT, 260)
    GX, GY = np.meshgrid(gx, gy)
    ui = np.array(c['f'][0](GX, GY).filled(np.nan))
    vi = np.array(c['f'][1](GX, GY).filled(np.nan))
    ax.contourf(GX, GY, np.ma.masked_invalid(np.hypot(ui, vi)), levels=40,
                cmap='viridis', vmin=0, vmax=1.6)
    rev = np.ma.masked_where(np.nan_to_num(ui) >= 0, np.ones_like(GX))
    ax.contourf(GX, GY, rev, levels=[.5, 1.5], colors=['red'], alpha=.28)
    ax.streamplot(gx, gy, np.nan_to_num(ui), np.nan_to_num(vi), density=2.4,
                  color='w', linewidth=.6, arrowsize=.65)
    ax.add_patch(plt.Rectangle((c['xmin'], 0), -c['xmin'], S_STEP, fc='0.85',
                               ec='k', lw=1.1, zorder=5))
    xr = reatt(c)
    if np.isfinite(xr):
        ax.plot([xr], [0], 'r^', ms=11, zorder=7, clip_on=False)
    # Armaly's measured reattachment, with its digitisation band
    ax.axvspan((8.05-0.7)*S_STEP, (8.05+0.7)*S_STEP, color='gold', alpha=.22, zorder=1)
    ax.axvline(8.05*S_STEP, color='goldenrod', lw=2.0, ls='--', zorder=6)
    ax.axvline(c['xmax'], color='yellow', lw=3, zorder=6)
    ax.set_xlim(-2, 17.2); ax.set_ylim(0, H_TOT); ax.set_ylabel('y')
    s = f"x_r/S = {xr/S_STEP:.3f}" if np.isfinite(xr) else "x_r: none in domain"
    ax.set_title(f"{c['lab']} ({c['status']})   |   max|u| = "
                 f"{np.abs(c['U'][...,0]).max():.4f}   |   {s}", fontsize=10)
axs[-1].set_xlabel('x')
fig.suptitle('BFS at ARMALY specification (ER 1.94, no-slip top, Re = 389 via D = 2h).\n'
             'Gold dashed = Armaly measured reattachment x_r/S = 8.05 (band = digitisation '
             '+/- 0.7);  red triangle = ours;  yellow = outlet.\n'
             'LONG / free is absent: it blew up (max|u| = 21.4).', fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig(f'{SC}/../figs/armaly_streamlines.png', dpi=120, bbox_inches='tight')
print('figs/armaly_streamlines.png')

# ---------- 2. u, v, p profiles ----------
yy = np.linspace(0.002, H_TOT-0.002, 420)
NM = ['u  (axial)', 'v  (vertical)', 'p - p_inlet']
fig, axs = plt.subplots(3, len(STATIONS_S), figsize=(3.0*len(STATIONS_S), 10.6),
                        sharey=True)
for r in range(3):
    for k, xs_ in enumerate(STATIONS_S):
        x = xs_*S_STEP
        ax = axs[r, k]
        for c in C:
            if x > c['xmax']+1e-9:
                continue
            v = np.array(c['f'][r](np.full_like(yy, x), yy).filled(np.nan))
            if c['sty'].get('marker'):
                ax.plot(v[::18], yy[::18], color=c['col'], label=c['lab'], **c['sty'])
            else:
                ax.plot(v, yy, color=c['col'], label=c['lab'], **c['sty'])
        ax.axvline(0, color='k', lw=.7, ls=':')
        ax.axhline(S_STEP, color='0.6', lw=.7, ls=':')
        ax.grid(alpha=.3)
        if r == 0:
            ax.set_title(f'x/S = {xs_:g}   (x = {x:.2f})', fontsize=10)
        if r == 2:
            ax.set_xlabel('value')
        if k == 0:
            ax.set_ylabel(f'{NM[r]}\n\ny')
axs[0, 0].legend(fontsize=8, loc='upper left')
fig.suptitle('Armaly specification, Re = 389 -- u, v, p at stations in STEP heights.  '
             'Dotted line = step height y = 0.94.\n'
             'NO experimental profiles exist at this Re (Armaly figs 5,6 are at '
             'Re = 1095 and 1290, above his 2-D limit) -- these are code-to-code.',
             fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(f'{SC}/../figs/armaly_profiles.png', dpi=120, bbox_inches='tight')
print('figs/armaly_profiles.png')

# ---------- 3. reattachment vs Armaly's curve ----------
meas = np.loadtxt(f'{SC}/../reference/armaly_fig4_x1_measured.csv',
                  delimiter=',', skiprows=6)
pred = np.loadtxt(f'{SC}/../reference/armaly_fig13a_x1_predicted.csv',
                  delimiter=',', skiprows=6)
fig, ax = plt.subplots(figsize=(8.4, 5.6))
ax.plot(meas[:, 0], meas[:, 1], '-', color='k', lw=2.2,
        label='Armaly measured (fig 4, extracted)')
ax.plot(pred[:, 0], pred[:, 1], '--', color='0.45', lw=1.8,
        label='Armaly computed (fig 13a, extracted)')
ax.fill_between(meas[:, 0], meas[:, 1]-0.7, meas[:, 1]+0.7, color='gold', alpha=.25,
                label='digitisation band +/- 0.7')
pts = [('ours: ER 1.94, no-slip, long, P+Z', 389, 8.145, 'tab:green', 'o'),
       ('ours: cnos ER 2.0, long, P+Z', 389, 8.200, 'tab:blue', 's'),
       ('Fortran: cnos ER 2.0, long, free', 389, 8.154, 'tab:orange', '^')]
for lab, re, v, col, mk in pts:
    ax.plot([re], [v], mk, color=col, ms=11, mec='k', mew=.8, label=lab, zorder=5)
ax.axvline(400, color='crimson', lw=1.2, ls=':',
           label='Re = 400: Armaly 2-D limit')
ax.set_xlim(0, 520); ax.set_ylim(0, 11)
ax.set_xlabel('Re  (= V*2h/nu, Armaly convention)'); ax.set_ylabel('$x_r/S$')
ax.grid(alpha=.3); ax.legend(fontsize=8.5, loc='upper left')
ax.set_title('Primary reattachment vs Armaly (1983), laminar branch', fontsize=11)
fig.tight_layout()
fig.savefig(f'{SC}/../figs/armaly_reattachment.png', dpi=125, bbox_inches='tight')
print('figs/armaly_reattachment.png')

print(f"\n{'case':>34}{'x_r/S':>9}{'vs measured 8.05':>19}")
for lab, re, v, _, _ in pts:
    print(f"{lab:>34}{v:>9.3f}{(v-8.05)/8.05*100:>18.1f}%")
for c in C:
    xr = reatt(c)
    if np.isfinite(xr):
        print(f"{c['lab']+' (this figure)':>34}{xr/S_STEP:>9.3f}"
              f"{(xr/S_STEP-8.05)/8.05*100:>18.1f}%")
