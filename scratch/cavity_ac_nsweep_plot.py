"""Does AC's benefit scale with GLL order N?  Cavity Re = 1000, 6x6 elements.

    uv run --quiet python scratch/cavity_ac_nsweep_plot.py

Reads scratch/nsweep_N*_*.npz written by scratch/cavity_ac_nsweep.py.

    figs/cavity_ac_n_sweep.png
"""
import os, sys, glob
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = {}
for f in glob.glob(f'{SC}/nsweep_N*_*.npz'):
    d = np.load(f, allow_pickle=True)
    R[(int(d['N']), str(d['kspec']))] = dict(
        its=float(d['its_per_call']), dofs=int(d['dofs']),
        kap=float(d['kappa_p']), wall=float(d['wall_s']))
NS = sorted({k[0] for k in R})
TAGS = [('off', 'AC off', 'tab:red', 'o'),
        ('half', r'AC on, $\kappa_p=a_{mass}/2$ = 15', 'tab:blue', 's'),
        ('match', r'AC on, $\kappa_p=a_{mass}$ = 30', 'tab:green', '^')]
have = [n for n in NS if all((n, t) in R for t, _, _, _ in TAGS)]
if len(have) < len(NS):
    print(f'note: incomplete N = {sorted(set(NS)-set(have))} dropped',
          file=sys.stderr)
NS = have

fig, axs = plt.subplots(1, 2, figsize=(14.0, 5.8))

# ---- (a) iterations per solve vs N ----
ax = axs[0]
for tag, lab, col, mk in TAGS:
    y = [R[(n, tag)]['its'] for n in NS]
    ax.plot(NS, y, mk+'-', color=col, ms=9, lw=2.0, label=lab)
    for n, v in zip(NS, y):
        ax.annotate(f'{v:.0f}', (n, v), textcoords='offset points',
                    xytext=(0, 9 if tag != 'match' else -15), ha='center',
                    fontsize=8.5, color=col)
ax.set_yscale('log')
ax.set_xticks(NS)
ax.set_xlabel('GLL polynomial order  N     (6x6 elements; dof = '
              + ', '.join(f'{R[(n,"off")]["dofs"]//1000}k' for n in NS) + ')')
ax.set_ylabel('CG iterations per solve')
ax.set_title('Both curves grow ~linearly in N', fontsize=12)
ax.grid(alpha=.3, which='both'); ax.legend(fontsize=9.5, loc='upper left')
ax.set_ylim(8, 9000)

# ---- (b) speed-up vs N: the actual question ----
ax = axs[1]
for tag, lab, col, mk in TAGS[1:]:
    sp = [R[(n, 'off')]['its']/R[(n, tag)]['its'] for n in NS]
    ax.plot(NS, sp, mk+'-', color=col, ms=9, lw=2.2, label=lab)
    for n, v in zip(NS, sp):
        ax.annotate(f'{v:.1f}x', (n, v), textcoords='offset points',
                    xytext=(0, 10), ha='center', fontsize=9, color=col)
    ax.axhline(np.mean(sp), color=col, lw=1.0, ls=':', alpha=.7)
ax.set_xticks(NS); ax.set_ylim(0, 36)
ax.set_xlabel('GLL polynomial order  N')
ax.set_ylabel('CG-iteration reduction vs AC off')
ax.set_title('...so the benefit is essentially N-INDEPENDENT', fontsize=12)
ax.grid(alpha=.3); ax.legend(fontsize=9.5, loc='lower left')
for tag, _, col, _ in TAGS[1:]:
    sp = [R[(n, 'off')]['its']/R[(n, tag)]['its'] for n in NS]
    ax.annotate(f'mean {np.mean(sp):.1f}x,  spread {min(sp):.1f}–{max(sp):.1f}x',
                xy=(NS[0], np.mean(sp)), xytext=(4, -14),
                textcoords='offset points', fontsize=8.5, color=col, ha='left')

fig.suptitle('Artificial compressibility vs polynomial order — lid-driven cavity '
             'Re = 1000, 6x6 elements, dt = 0.05 ($a_{mass}$ = 30)\n'
             '40 steps from rest, nsub = 5, cg_tol = 1e-8 (200 solves per case).  '
             'Mesh fixed, so N is the only variable.', fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig('figs/cavity_ac_n_sweep.png', dpi=125, bbox_inches='tight')
print('figs/cavity_ac_n_sweep.png\n')

hdr = (f"{'N':>4}{'dof':>8}{'AC off':>10}{'k=a/2':>9}{'k=a':>8}"
       f"{'a/2 gain':>10}{'a gain':>9}")
print(hdr); print('-'*len(hdr))
for n in NS:
    o, h, m = (R[(n, t)]['its'] for t in ('off', 'half', 'match'))
    print(f'{n:>4}{R[(n,"off")]["dofs"]:>8}{o:>10.1f}{h:>9.1f}{m:>8.1f}'
          f'{o/h:>9.1f}x{o/m:>8.1f}x')
