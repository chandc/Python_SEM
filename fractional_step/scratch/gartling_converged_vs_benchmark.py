"""Converged Gartling solutions vs the benchmark: streamlines and u, v, p, omega.

    uv run --quiet python scratch/gartling_converged_vs_benchmark.py

WHICH RUNS QUALIFY.  Of the runs that stayed stable, most were stopped at t = 10
or t = 30 -- early transient, with the bubble only 20-25% of its steady length.
Comparing those to Gartling's STEADY benchmark would be meaningless, so they are
excluded.  What is compared:

  steady solver, uniform 11x4, N=5/6/7   w_mass = 0 -- no time derivative at all
  steady solver, Chan-graded 11x4, N=7   ditto, on Chan's own measured mesh
  unsteady 11x4 t=140 (steady restart)   held its fixed point, lo_reatt 6.184
  unsteady 18x4 t=140 (from rest)        reached 94% of steady, lo_reatt 5.819

The 18x4 run is included with that caveat attached: it is the only from-rest
unsteady run that got close to steady, and it is still short of it.

    figs/gartling_converged_streamlines.png
    figs/gartling_converged_profiles.png
"""
import os, sys
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, LinearTriInterpolator

CASES = [
    ('gartling_steady_nx11_N5.npz',  'steady solver, uniform N=5', 'tab:orange',
     dict(ls='none', marker='o', ms=4.2, mfc='none', mew=1.0)),
    ('gartling_steady_nx11_N6.npz',  'steady solver, uniform N=6', 'tab:blue',
     dict(ls='none', marker='^', ms=4.4, mfc='none', mew=1.0)),
    ('gartling_steady_nx11_N7.npz',  'steady solver, uniform N=7', 'tab:red',
     dict(ls='-', lw=2.2)),
    ('gartling_steady_nx11g_N7.npz', 'steady solver, graded  N=7', 'tab:purple',
     dict(ls='--', lw=1.8)),
    ('gartling_unsteady_nx11_N6_dt0.1_nsub3_pz_steady_wm0.1_ws0.1.npz',
     'unsteady 11x4 t=140 (restart)', 'tab:green',
     dict(ls='-.', lw=1.8)),
    ('gartling_unsteady_nx18_N6_dt0.1_nsub3_pz_stagnant_wm0.1_ws0.1.npz',
     'unsteady 18x4 t=140 (from rest, 94% of steady)', 'tab:brown',
     dict(ls=':', lw=2.0)),
]
STATIONS = [7.0, 15.0]
QTY = [('u', 0, 'u  (axial velocity)'), ('v', 1, 'v  (vertical velocity)'),
       ('p', 2, 'p  (pressure, p = 0 at outlet)'), ('omega', 3, r'$\omega$  (vorticity)')]


def pack(f, lab, col, sty):
    d = np.load(f'{SC}/{f}', allow_pickle=True)
    k = set(d.keys())
    U, xn, yn = d['U'], d['xnod'], d['ynod']
    n = U.shape[1]
    px, py, q = [], [], [[], [], [], []]
    for e in range(U.shape[0]):
        for i in range(n):
            for j in range(n):
                px.append(xn[e, i]); py.append(yn[e, j])
                for c in range(4):
                    q[c].append(U[e, i, j, c])
    tri = Triangulation(np.array(px), np.array(py))
    lo = float(d['lo_reatt']) if 'lo_reatt' in k else (
        float(d['hist'][-1, 3]) if 'hist' in k else np.nan)
    return dict(lab=lab, col=col, sty=sty, U=U, lo=lo,
                f=[LinearTriInterpolator(tri, np.array(q[c])) for c in range(4)])


C = [pack(*c) for c in CASES]

# ---------------- streamlines ----------------
gx = np.linspace(0, 17, 1400); gy = np.linspace(-0.5, 0.5, 190)
GX, GY = np.meshgrid(gx, gy)
fig, axs = plt.subplots(len(C), 1, figsize=(15.0, 2.3*len(C)))
for ax, c in zip(axs, C):
    ui = np.array(c['f'][0](GX, GY).filled(np.nan))
    vi = np.array(c['f'][1](GX, GY).filled(np.nan))
    ax.contourf(GX, GY, np.ma.masked_invalid(np.hypot(ui, vi)), levels=40,
                cmap='viridis', vmin=0, vmax=1.5)
    rev = np.ma.masked_where(np.nan_to_num(ui) >= 0, np.ones_like(GX))
    ax.contourf(GX, GY, rev, levels=[.5, 1.5], colors=['red'], alpha=.25)
    ax.streamplot(gx, gy, np.nan_to_num(ui), np.nan_to_num(vi), density=2.6,
                  color='w', linewidth=.6, arrowsize=.6)
    for xv in (6.1, 4.8, 10.5):
        ax.axvline(xv, color='gold', lw=1.3, ls=':', zorder=6)
    for xs in STATIONS:
        ax.axvline(xs, color='k', lw=1.0, ls='--', alpha=.5, zorder=6)
    ax.set_xlim(0, 17); ax.set_ylim(-0.5, 0.5); ax.set_ylabel('y')
    ax.set_title(f"{c['lab']}   |   lower reattach {c['lo']:.3f}  (Gartling 6.1)   |   "
                 f"max|u| = {np.abs(c['U'][..., 0]).max():.4f}", fontsize=10)
axs[-1].set_xlabel('x')
fig.suptitle('Gartling BFS Re = 800 -- CONVERGED solutions only.  '
             'Gold dotted = Gartling 6.1 / 4.8 / 10.5;  black dashed = profile stations.\n'
             'Runs stopped at t = 10-30 are excluded: their bubble is only ~20% of the '
             'steady length, so they are transients, not converged states.', fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig('figs/gartling_converged_streamlines.png', dpi=118, bbox_inches='tight')
print('figs/gartling_converged_streamlines.png')

# ---------------- profiles ----------------
yy = np.linspace(-0.4995, 0.4995, 400)
fig, axs = plt.subplots(4, 2, figsize=(12.0, 15.5), sharey=True)
err = {}
for r, (key, kk, nm) in enumerate(QTY):
    for ci, xs in enumerate(STATIONS):
        ax = axs[r, ci]
        bf = f'reference/gartling_re800_x{xs:g}_{key}.csv'
        bench = None
        if os.path.exists(bf):
            b = np.loadtxt(bf, delimiter=',', skiprows=9)
            ax.plot(b[:, 1], b[:, 0], '-', color='k', lw=2.8, alpha=.8,
                    label='Gartling benchmark', zorder=1)
            bench = b
        for c in C:
            v = np.array(c['f'][kk](np.full_like(yy, xs), yy).filled(np.nan))
            if c['sty'].get('marker'):
                ax.plot(v[::18], yy[::18], color=c['col'], label=c['lab'], **c['sty'])
            else:
                ax.plot(v, yy, color=c['col'], label=c['lab'], **c['sty'])
            if bench is not None:
                bv = np.interp(yy, bench[:, 0], bench[:, 1])
                rng = bench[:, 1].max()-bench[:, 1].min()
                err[(c['lab'], key, xs)] = np.nanmax(np.abs(v-bv))/rng*100
        ax.axvline(0, color='k', lw=.7, ls=':'); ax.axhline(0, color='0.6', lw=.7, ls=':')
        ax.grid(alpha=.3)
        if r == 0:
            ax.set_title(f'x = {xs:g}', fontsize=12)
        if ci == 0:
            ax.set_ylabel(f'{nm}\n\ny')
        if key == 'p':
            ax.text(.03, .04, 'no benchmark:\nChan fig.3 omits p',
                    transform=ax.transAxes, fontsize=8, color='0.35')
axs[0, 0].legend(fontsize=7.6, loc='upper left')
fig.suptitle('Converged Gartling solutions vs benchmark, x = 7 and x = 15.\n'
             'Black = Gartling benchmark (digitised from Chan & Mittal fig. 3).  '
             'Pressure has no benchmark -- Chan fig. 3 omits it.', fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.945])
fig.savefig('figs/gartling_converged_profiles.png', dpi=118, bbox_inches='tight')
print('figs/gartling_converged_profiles.png')

print(f"\nmax |error| vs benchmark, as % of the benchmark's own range")
print(f"{'case':>48}{'u@7':>8}{'v@7':>8}{'w@7':>8}{'u@15':>8}{'v@15':>8}{'w@15':>8}")
for c in C:
    row = [err.get((c['lab'], q, x), np.nan)
           for x in STATIONS for q in ('u', 'v', 'omega')]
    row = [row[0], row[1], row[2], row[3], row[4], row[5]]
    print(f"{c['lab']:>48}" + "".join(f'{v:>8.1f}' for v in row))
