"""Pressure convergence vs cost, AC off and on -- cavity Re = 1000.

    uv run --quiet python scratch/cavity_ac_pconv_plot.py

Reads the per-step histories written by scratch/cavity_ac_pconv.py.  Plots
max|dp| per step against CUMULATIVE CG ITERATIONS and against CUMULATIVE WALL
TIME, one column per dt.

Cost on the x-axis, not step number.  Against steps every curve looks similar --
AC changes what a step COSTS far more than how many steps are needed, so a
per-step plot hides the entire effect.

    figs/cavity_ac_pressure_convergence.png
"""
import os, sys, glob
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

FILES = sorted(glob.glob(f'{SC}/cavity_ac_pconv_dt*_k*.npz'))
if not FILES:
    raise SystemExit('no cavity_ac_pconv_dt*.npz -- run cavity_ac_pconv.py first')
RUNS = []
for f in FILES:
    d = np.load(f, allow_pickle=True)
    RUNS.append(dict(dt=float(d['dt']), a_mass=float(d['a_mass']),
                     kap=float(d['kappa_p']), kfrac=float(d['kfrac']),
                     H=d['hist']))
DTS = sorted({r['dt'] for r in RUNS}, reverse=True)
COL = {0.0: 'tab:red', 0.25: 'tab:purple', 0.5: 'tab:blue', 1.0: 'tab:green',
       2.0: 'tab:olive'}


def label(r):
    return ('AC off' if r['kfrac'] == 0 else
            rf"$\kappa_p$ = {r['kfrac']:g}$a_{{mass}}$ = {r['kap']:g}")


fig, axs = plt.subplots(2, len(DTS), figsize=(7.0*len(DTS), 10.0), squeeze=False)
for c, dt in enumerate(DTS):
    rs = sorted([r for r in RUNS if r['dt'] == dt], key=lambda r: r['kfrac'])
    a_mass = rs[0]['a_mass']
    for row, (xcol, xlab) in enumerate(((1, 'cumulative CG iterations'),
                                        (2, 'cumulative wall time  (s)'))):
        ax = axs[row][c]
        for r in rs:
            H = r['H']
            ax.plot(H[:, xcol], H[:, 3], lw=2.0,
                    color=COL.get(r['kfrac'], 'k'), label=label(r))
        # BOTH axes log.  The AC-off run costs up to 30x more than the AC ones,
        # so on a linear cost axis every AC curve collapses into the left edge
        # and the interesting part -- how they differ from each other -- is
        # unreadable.
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel(xlab); ax.set_ylabel(r'max$|\Delta p|$ per step')
        ax.grid(alpha=.3, which='both')
        ax.set_title(f'dt = {dt:g}   ($a_{{mass}}$ = {a_mass:g})   —  '
                     f'{"CG cost" if row == 0 else "wall time"}', fontsize=11.5)
        if row == 0 and c == 0:
            ax.legend(fontsize=9)
        for tol in (1e-4, 1e-6):
            ax.axhline(tol, color='0.6', lw=.9, ls=':')
            ax.annotate(f'{tol:g}', xy=(ax.get_xlim()[0], tol), fontsize=8,
                        color='0.45', va='bottom', ha='left')

fig.suptitle('Pressure convergence per unit cost — lid-driven cavity Re = 1000, '
             '6x6 elements N = 10, 300 steps from rest.\n'
             r'AC acts directly on the continuity row ($\kappa_p(p-p_{prev})$), '
             'so pressure is the field it should move most.  Lower-left is better.',
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.945])
fig.savefig('figs/cavity_ac_pressure_convergence.png', dpi=120,
            bbox_inches='tight')
print('figs/cavity_ac_pressure_convergence.png\n')

hdr = (f"{'dt':>6}{'a_mass':>8}{'kappa_p':>9}{'CG its':>10}{'wall':>9}"
       f"{'|dp| end':>11}{'CG<1e-4':>10}{'CG<1e-6':>10}{'s<1e-6':>9}")
print(hdr); print('-'*len(hdr))
for dt in DTS:
    base = None
    for r in sorted([r for r in RUNS if r['dt'] == dt], key=lambda r: r['kfrac']):
        H = r['H']

        def first(col, tol):
            i = np.where(H[:, 3] < tol)[0]
            return np.nan if len(i) == 0 else H[i[0], col]
        c4, c6, w6 = first(1, 1e-4), first(1, 1e-6), first(2, 1e-6)
        if r['kfrac'] == 0:
            base = c6
        sp = '' if (base is None or not np.isfinite(c6) or not np.isfinite(base)
                    or r['kfrac'] == 0) else f'  ({base/c6:.1f}x)'
        print(f"{r['dt']:>6g}{r['a_mass']:>8g}{r['kap']:>9g}{int(H[-1,1]):>10d}"
              f"{H[-1,2]:>8.0f}s{H[-1,3]:>11.2e}"
              f"{'--' if not np.isfinite(c4) else f'{c4:.0f}':>10}"
              f"{'--' if not np.isfinite(c6) else f'{c6:.0f}':>10}"
              f"{'--' if not np.isfinite(w6) else f'{w6:.0f}':>9}{sp}")
    print()
