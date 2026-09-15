"""Converged steady Gartling Re=800 solutions: streamlines and u, v, p, omega.

    uv run --quiet python scratch/gartling_plot.py

Reads the four converged steady fields (gartling_steady_nx*_N*.npz) and produces

    figs/gartling_steady_streamlines.png   streamlines + reversed-flow shading
    figs/gartling_steady_profiles.png      u, v, p, omega at x = 7 and x = 15

The u, v and omega panels carry Gartling's benchmark as a black line, extracted
from Chan & Mittal fig. 3 by reference/gartling_digitize.py.  There is NO
benchmark for pressure -- Chan's fig. 3 does not plot it -- so those two panels
are code-to-code only and are labelled as such.

Pressure needs no datum shift: the P+Z outlet fixes p = 0 on the outflow plane,
so p is absolute and directly comparable between the four runs.
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

# Two mesh families, both 11x4:
#   UNIFORM  dx = 1.5455 throughout -- our original reading of Chan's text, which
#            gives element counts and says nothing about grading.
#   GRADED   x = 0,1,2,3,4,5,7,9,11,13,15,17 -- measured off Chan's own grid
#            skeleton (top panel of his fig. 5): five unit-width elements over the
#            recirculation, six double-width downstream, graded 2:1.
# They converge to the reattachment length from OPPOSITE sides (uniform 6.868 ->
# 6.100 from above, graded 5.155 -> 6.187 from below), bracketing Chan's 6.1.
CASES = [('gartling_steady_nx11_N5.npz',  'uniform N=5', 'tab:orange',
          dict(ls='none', marker='o', ms=4.2, mfc='none', mew=1.0)),
         ('gartling_steady_nx11_N6.npz',  'uniform N=6', 'tab:blue',
          dict(ls='none', marker='^', ms=4.4, mfc='none', mew=1.0)),
         ('gartling_steady_nx11_N7.npz',  'uniform N=7', 'tab:red',
          dict(ls='-', lw=2.0)),
         ('gartling_steady_nx11g_N5.npz', 'graded  N=5', 'tab:orange',
          dict(ls=':', lw=1.5)),
         ('gartling_steady_nx11g_N6.npz', 'graded  N=6', 'tab:blue',
          dict(ls=':', lw=1.5)),
         ('gartling_steady_nx11g_N7.npz', 'graded  N=7', 'tab:red',
          dict(ls='--', lw=1.9))]
CHAN = dict(lo=6.1, usep=4.8, ure=10.5)
STATIONS = [7.0, 15.0]
QTY = [('u', 0, 'u  (axial velocity)'), ('v', 1, 'v  (vertical velocity)'),
       ('p', 2, 'p  (pressure, p=0 at outlet)'), ('omega', 3, r'$\omega$  (vorticity)')]


def pack(f, lab, col, sty):
    d = np.load(f'{SC}/{f}', allow_pickle=True)
    U, xn, yn = d['U'], d['xnod'], d['ynod']
    n = U.shape[1]
    px, py, q = [], [], [[], [], [], []]
    for e in range(U.shape[0]):
        for i in range(n):
            for j in range(n):
                px.append(xn[e, i]); py.append(yn[e, j])
                for k in range(4):
                    q[k].append(U[e, i, j, k])
    px, py = np.array(px), np.array(py)
    tri = Triangulation(px, py)
    return dict(lab=lab, col=col, sty=sty, U=U, xn=xn, yn=yn,
                status=str(d['status']), it=int(d['iters']),
                lo=float(d['lo_reatt']), usep=float(d['up_sep']),
                ure=float(d['up_reatt']),
                f=[LinearTriInterpolator(tri, np.array(q[k])) for k in range(4)])


C = [pack(*c) for c in CASES]

# ---------------- 1. streamlines ----------------
fig, axs = plt.subplots(len(C), 1, figsize=(15.5, 2.35*len(C)))
gx = np.linspace(0, 17, 1500); gy = np.linspace(-0.5, 0.5, 200)
GX, GY = np.meshgrid(gx, gy)
for ax, c in zip(axs, C):
    ui = np.array(c['f'][0](GX, GY).filled(np.nan))
    vi = np.array(c['f'][1](GX, GY).filled(np.nan))
    ax.contourf(GX, GY, np.ma.masked_invalid(np.hypot(ui, vi)), levels=40,
                cmap='viridis', vmin=0, vmax=1.5)
    rev = np.ma.masked_where(np.nan_to_num(ui) >= 0, np.ones_like(GX))
    ax.contourf(GX, GY, rev, levels=[.5, 1.5], colors=['red'], alpha=.25)
    ax.streamplot(gx, gy, np.nan_to_num(ui), np.nan_to_num(vi), density=2.6,
                  color='w', linewidth=.6, arrowsize=.6)
    for xv, cl in ((c['lo'], 'red'), (c['usep'], 'darkorange'), (c['ure'], 'red')):
        if np.isfinite(xv):
            ax.plot([xv], [-0.5 if cl == 'red' and xv == c['lo'] else 0.5],
                    marker='^' if xv == c['lo'] else 'v', color=cl, ms=10,
                    zorder=7, clip_on=False)
    for xv in (CHAN['lo'], CHAN['usep'], CHAN['ure']):
        ax.axvline(xv, color='gold', lw=1.4, ls=':', zorder=5)
    for xs in STATIONS:
        ax.axvline(xs, color='k', lw=1.0, ls='--', alpha=.55, zorder=6)
    ax.set_xlim(0, 17); ax.set_ylim(-0.5, 0.5); ax.set_ylabel('y')
    ax.set_title(f"{c['lab']}   ({c['status']}, {c['it']} it)   |   "
                 f"lower reattach {c['lo']:.3f} (Chan 6.1)   |   "
                 f"upper sep {c['usep']:.3f} (4.8)   reattach {c['ure']:.3f} (10.5)",
                 fontsize=10)
axs[-1].set_xlabel('x')
fig.suptitle('Gartling BFS, Re = 800 (nu = 1/800), steady form w_mass = 0, P+Z outlet.  '
             'Uniform and Chan-graded 11x4 meshes.\n'
             'Red shading = reversed flow;  gold dotted = Chan & Mittal\'s reported '
             'separation/reattachment;  black dashed = the x = 7 and x = 15 profile stations.',
             fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig('figs/gartling_steady_streamlines.png', dpi=120, bbox_inches='tight')
print('figs/gartling_steady_streamlines.png')

# ---------------- 2. profiles ----------------
yy = np.linspace(-0.4995, 0.4995, 400)
fig, axs = plt.subplots(4, 2, figsize=(11.5, 15.0), sharey=True)
for r, (key, k, nm) in enumerate(QTY):
    for cidx, xs in enumerate(STATIONS):
        ax = axs[r, cidx]
        # Gartling benchmark, where one exists
        bf = f'reference/gartling_re800_x{xs:g}_{key}.csv'
        if os.path.exists(bf):
            b = np.loadtxt(bf, delimiter=',', skiprows=9)
            ax.plot(b[:, 1], b[:, 0], '-', color='k', lw=2.6, alpha=.75,
                    label='Gartling benchmark', zorder=1)
        for c in C:
            vals = np.array(c['f'][k](np.full_like(yy, xs), yy).filled(np.nan))
            if c['sty'].get('marker'):
                ax.plot(vals[::16], yy[::16], color=c['col'], label=c['lab'], **c['sty'])
            else:
                ax.plot(vals, yy, color=c['col'], label=c['lab'], **c['sty'])
        ax.axvline(0, color='k', lw=.7, ls=':')
        ax.axhline(0, color='0.6', lw=.7, ls=':')
        ax.grid(alpha=.3)
        if r == 0:
            ax.set_title(f'x = {xs:g}', fontsize=12)
        if cidx == 0:
            ax.set_ylabel(f'{nm}\n\ny')
        if key == 'p':
            ax.text(.03, .04, 'no benchmark:\nChan fig.3 omits p', transform=ax.transAxes,
                    fontsize=8, color='0.35')
axs[0, 0].legend(fontsize=8.5, loc='upper left')
fig.suptitle('Gartling BFS, Re = 800 -- converged steady profiles at x = 7 and x = 15.\n'
             'Black = Gartling benchmark digitised from Chan & Mittal fig. 3 '
             '(mass gate 0.7% at x=7, 1.0% at x=15).  Solid/markers = UNIFORM 11x4 mesh, '
             'dotted/dashed = Chan\'s GRADED 11x4 mesh.\n'
             'The two meshes bracket the reattachment (uniform 6.100, graded 6.187, Chan 6.1). '
             'Symbols are OURS, not Chan\'s.', fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.945])
fig.savefig('figs/gartling_steady_profiles.png', dpi=120, bbox_inches='tight')
print('figs/gartling_steady_profiles.png')

# ---------------- numbers ----------------
print(f"\n{'case':>12}{'status':>9}{'lo_reatt':>10}{'up_sep':>9}{'up_re':>8}"
      f"{'max|u|':>9}{'int u dy @7':>13}{'@15':>9}")
for c in C:
    ints = []
    for xs in STATIONS:
        uu = np.array(c['f'][0](np.full_like(yy, xs), yy).filled(np.nan))
        ints.append(np.trapezoid(uu, yy))
    print(f"{c['lab']:>12}{c['status']:>9}{c['lo']:>10.3f}{c['usep']:>9.3f}"
          f"{c['ure']:>8.3f}{np.abs(c['U'][...,0]).max():>9.4f}"
          f"{ints[0]:>13.5f}{ints[1]:>9.5f}")
print(f"{'Chan/Gartling':>12}{'--':>9}{6.1:>10.3f}{4.8:>9.3f}{10.5:>8.3f}"
      f"{1.5:>9.4f}{0.5:>13.5f}{0.5:>9.5f}")
