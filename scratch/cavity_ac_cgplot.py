"""CG iterations with and without artificial compressibility -- cavity Re = 1000.

    uv run --quiet python scratch/cavity_ac_cgplot.py

DATA below is the measured output of scratch/cavity_ac_cgiters.py -- 40 steps
from rest, 6x6 elements N = 10, nsub = 5, cg_tol = 1e-8, cgsfac = 1e-3.  Re-run
that script and update DATA if anything in the solver changes.
Documented in ARTIFICIAL_COMPRESSIBILITY.md sec 5.2.

    figs/cavity_ac_cg_iterations.png
"""
import os, sys
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

# (dt, a_mass, kappa_p, its_per_call, wall_s)
DATA = [(0.25, 6, 0.0, 690.5, 77.6),
        (0.25, 6, 3.0, 336.7, 40.1),
        (0.25, 6, 6.0, 227.5, 28.3),
        (0.05, 30, 0.0, 1379.0, 150.2),
        (0.05, 30, 15.0, 74.4, 11.5),
        (0.05, 30, 30.0, 50.1, 8.6)]

fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.6))

# --- (a) iterations per CG call vs a_mass, AC off / half / match ---
ax = axs[0]
am = sorted({d[1] for d in DATA})
for tag, sel, col, mk in (('AC off', lambda d: d[2] == 0, 'tab:red', 'o'),
                          (r'AC on, $\kappa_p=a_{mass}/2$',
                           lambda d: d[2] != 0 and abs(d[2]-d[1]/2) < 1e-9,
                           'tab:blue', 's'),
                          (r'AC on, $\kappa_p=a_{mass}$',
                           lambda d: d[2] != 0 and abs(d[2]-d[1]) < 1e-9,
                           'tab:green', '^')):
    pts = sorted([(d[1], d[3]) for d in DATA if sel(d)])
    ax.plot([p[0] for p in pts], [p[1] for p in pts], mk+'-', color=col, ms=10,
            lw=2.0, label=tag)
    for x, y in pts:
        ax.annotate(f'{y:.0f}', (x, y), textcoords='offset points',
                    xytext=(0, 9), ha='center', fontsize=9, color=col)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(4.8, 78); ax.set_ylim(28, 3300)
ax.set_xticks(am); ax.set_xticklabels([f'{a:g}' for a in am])
ax.set_xlabel(r'$a_{mass} = w_{mass}\,fac_1/dt$   (6 at dt=0.25,  30 at dt=0.05)')
ax.set_ylabel('CG iterations per solve')
ax.set_title('Conditioning: AC reverses the slope', fontsize=12)
ax.grid(alpha=.3, which='both'); ax.legend(fontsize=9.5, loc='lower left')
ax.annotate('without AC, refining dt\nmakes the system HARDER',
            xy=(30, 1379), xytext=(33, 1500), fontsize=9, color='tab:red',
            arrowprops=dict(arrowstyle='->', color='tab:red', lw=1.2))
ax.annotate('with AC, refining dt\nmakes it EASIER',
            xy=(30, 50.1), xytext=(33, 60), fontsize=9, color='tab:green',
            arrowprops=dict(arrowstyle='->', color='tab:green', lw=1.2))

# --- (b) speed-up factor ---
ax = axs[1]
w = 0.35
labels, s_half, s_match = [], [], []
for a in am:
    off = [d[3] for d in DATA if d[1] == a and d[2] == 0][0]
    h = [d[3] for d in DATA if d[1] == a and abs(d[2]-a/2) < 1e-9][0]
    m = [d[3] for d in DATA if d[1] == a and abs(d[2]-a) < 1e-9][0]
    ws = {d[2]: d[4] for d in DATA if d[1] == a}
    labels.append(f'$a_{{mass}}$ = {a:g}   (dt = {[d[0] for d in DATA if d[1]==a][0]:g})\n'
                  f'wall  {ws[0.0]:.0f}s $\\rightarrow$ {ws[a/2]:.0f}s '
                  f'$\\rightarrow$ {ws[a]:.0f}s')
    s_half.append(off/h); s_match.append(off/m)
x = np.arange(len(am))
b1 = ax.bar(x-w/2, s_half, w, color='tab:blue', label=r'$\kappa_p=a_{mass}/2$')
b2 = ax.bar(x+w/2, s_match, w, color='tab:green', label=r'$\kappa_p=a_{mass}$')
for bars in (b1, b2):
    for b in bars:
        ax.annotate(f'{b.get_height():.1f}x', (b.get_x()+b.get_width()/2, b.get_height()),
                    textcoords='offset points', xytext=(0, 3), ha='center', fontsize=10)
ax.axhline(1.0, color='k', lw=1.0, ls='--')
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylim(0, max(s_match)*1.22)
ax.set_ylabel('CG-iteration reduction vs AC off')
ax.set_title('Benefit grows with $a_{mass}$', fontsize=12)
ax.grid(alpha=.3, axis='y'); ax.legend(fontsize=9.5, loc='upper left')

fig.suptitle('Artificial compressibility and linear-solver cost -- lid-driven cavity '
             'Re = 1000, 6x6 elements N = 10\n'
             '40 steps from rest, nsub = 5, cg_tol = 1e-8, Jacobi-preconditioned CG',
             fontsize=11.5)
fig.text(0.5, 0.005, 'Mechanism: pressure enters only the momentum rows, so without AC the '
         r'Jacobi preconditioner has no pressure diagonal ($a_{33}=0$).  AC supplies '
         r'$a_{33}=\kappa_p P$ — exactly where the conditioning is worst.',
         ha='center', fontsize=10)
fig.tight_layout(rect=[0, 0.035, 1, 0.91])
fig.savefig('figs/cavity_ac_cg_iterations.png', dpi=125, bbox_inches='tight')
print('figs/cavity_ac_cg_iterations.png')
for d in DATA:
    print(f'  dt={d[0]:<6g} a_mass={d[1]:<4g} kappa_p={d[2]:<6g} '
          f'its/call={d[3]:>7.1f}  wall={d[4]:>6.1f}s')
