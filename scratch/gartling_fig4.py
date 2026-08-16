"""Chan & Mittal fig. 4: wall vorticity on the upper and lower walls, Re = 800.

    uv run --quiet python scratch/gartling_fig4.py

Fig. 4 caption (p.353): "Predicted wall vorticity distribution for a
backward-facing step with Re = 800, o 5th order, /\\ 6th order, [] 7th order,
(a) upper wall and (b) lower wall."  The text adds: "Figure 4 shows the vorticity
distribution, which is proportional to shear stress, along the bottom and top
boundaries.  By examining these plots, one can determine both the separation and
reattachment points.  Along the lower wall, UniFlo predicts a reattachment length
of 6.1, whereas along the upper wall, it predicts a separation at the streamwise
location of 4.8 and a reattachment at the streamwise location of 10.5."

omega is a SOLVED variable in this VVP formulation, so the wall values are read
straight off the solution -- no differentiation of u is involved.  On a wall
v = 0 identically, so dv/dx = 0 there and omega = -du/dy exactly; a zero crossing
is therefore a separation or reattachment point.

    figs/gartling_fig4_wall_vorticity.png
"""
import os, sys
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
sys.path.insert(0, SC)
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

CASES = [('gartling_steady_nx11_N5.npz', '11x4, N=5 (5th order)', 'tab:orange',
          dict(ls='none', marker='o', ms=4.5, mfc='none', mew=1.1)),
         ('gartling_steady_nx11_N6.npz', '11x4, N=6 (6th order)', 'tab:blue',
          dict(ls='none', marker='^', ms=4.8, mfc='none', mew=1.1)),
         ('gartling_steady_nx11_N7.npz', '11x4, N=7 (7th order)', 'tab:red',
          dict(ls='none', marker='s', ms=4.4, mfc='none', mew=1.1)),
         ('gartling_steady_nx18_N6.npz', '18x4, N=6', 'tab:green',
          dict(ls='-', lw=1.5))]
CHAN = {'lower reattach': 6.1, 'upper separate': 4.8, 'upper reattach': 10.5}


def wall_omega(f):
    """(x, omega) along the bottom wall and the top wall, read from the solution."""
    d = np.load(f'{SC}/{f}', allow_pickle=True)
    U, xn, yn = d['U'], d['xnod'], d['ynod']
    n = U.shape[1]
    lo, up = [], []
    ymin, ymax = yn.min(), yn.max()
    for e in range(U.shape[0]):
        if abs(yn[e, 0]-ymin) < 1e-9:                  # bottom wall row, j = 0
            for i in range(n):
                lo.append((xn[e, i], U[e, i, 0, 3]))
        if abs(yn[e, -1]-ymax) < 1e-9:                 # top wall row, j = N
            for i in range(n):
                up.append((xn[e, i], U[e, i, -1, 3]))
    lo = np.array(sorted(lo)); up = np.array(sorted(up))
    return lo, up, d


def zeros(a, xmin=0.05):
    out = []
    for k in range(len(a)-1):
        if a[k, 0] < xmin:
            continue
        y0, y1 = a[k, 1], a[k+1, 1]
        if (y0 < 0 <= y1) or (y0 > 0 >= y1):
            out.append(a[k, 0] - y0*(a[k+1, 0]-a[k, 0])/(y1-y0))
    return out


fig, axs = plt.subplots(2, 1, figsize=(12.5, 9.5))
store = {}
for f, lab, col, sty in CASES:
    lo, up, d = wall_omega(f)
    store[lab] = (lo, up)
    step = 1 if sty.get('ls') == '-' else 3
    axs[0].plot(up[::step, 0], up[::step, 1], color=col, label=lab, **sty)
    axs[1].plot(lo[::step, 0], lo[::step, 1], color=col, label=lab, **sty)

for ax, ttl, marks in ((axs[0], '(a) upper wall',
                        [('upper separate', 4.8), ('upper reattach', 10.5)]),
                       (axs[1], '(b) lower wall', [('lower reattach', 6.1)])):
    ax.axhline(0, color='k', lw=1.0)
    for nm, xv in marks:
        ax.axvline(xv, color='gold', lw=1.8, ls=':', zorder=1)
        ax.annotate(f'Chan {nm.split()[1]} {xv}', xy=(xv, ax.get_ylim()[1]),
                    xytext=(xv+0.15, 0.88), textcoords=('data', 'axes fraction'),
                    fontsize=8.5, color='darkgoldenrod')
    ax.set_xlim(0, 17); ax.grid(alpha=.3)
    ax.set_ylabel(r'wall vorticity  $\omega$'); ax.set_title(ttl, fontsize=11)
axs[1].set_xlabel('x')
axs[0].legend(fontsize=8.5, loc='upper right')
fig.suptitle('Gartling BFS Re = 800 -- wall vorticity (Chan & Mittal fig. 4).\n'
             r'$\omega$ is a solved variable here, so these are read directly off the '
             'solution;  gold dotted = Chan\'s reported separation/reattachment.',
             fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig('figs/gartling_fig4_wall_vorticity.png', dpi=125, bbox_inches='tight')
print('figs/gartling_fig4_wall_vorticity.png')

print(f"\n{'case':>22}{'lower zeros':>28}{'upper zeros':>34}")
for lab, (lo, up) in store.items():
    zl = ', '.join(f'{z:.3f}' for z in zeros(lo))
    zu = ', '.join(f'{z:.3f}' for z in zeros(up))
    print(f'{lab:>22}{zl:>28}{zu:>34}')
print(f"{'Chan & Mittal':>22}{'6.1':>28}{'4.8, 10.5':>34}")
