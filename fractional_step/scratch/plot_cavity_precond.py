"""Ghia cavity Re=1000: preconditioner scaling. Measured values from cavity_scaling.py."""
import os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

SC = os.path.dirname(os.path.abspath(__file__))
dof = np.array([5184, 17424, 43264, 67600])
lab = ['4x4\np=8', '6x6\np=10', '8x8\np=12', '10x10\np=12']
S = {  # name: (iters, wall s, colour, marker)
    'Jacobi':            ([687, 1471, 2443, 3012], [0.143, 0.688, 2.353, 4.210], 'tab:blue',   'o'),
    'Chebyshev4 d=6':    ([152,  318,  529,  655], [0.208, 0.980, 3.288, 5.924], 'tab:orange', 's'),
    'Chebyshev4 d=10':   ([100,  207,  346,  428], [0.213, 0.996, 3.325, 6.053], 'tab:green',  '^'),
    'p-MG (pc=p/2, d=4)':([ 89,  122,  175,  188], [0.281, 0.825, 2.453, 3.862], 'tab:red',    'D'),
}
fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.6))

for name, (it, tw, c, mk) in S.items():
    axs[0].loglog(dof, it, mk+'-', color=c, lw=1.9, ms=6, label=name)
axs[0].set_title('CG iterations to cgsfac=1e-3', fontsize=10)
axs[0].set_ylabel('CG iterations'); axs[0].legend(fontsize=8)

for name, (it, tw, c, mk) in S.items():
    axs[1].loglog(dof, tw, mk+'-', color=c, lw=1.9, ms=6, label=name)
axs[1].set_title('wall-clock time per Newton solve', fontsize=10)
axs[1].set_ylabel('seconds')

jt = np.array(S['Jacobi'][1])
for name, (it, tw, c, mk) in S.items():
    axs[2].semilogx(dof, jt/np.array(tw), mk+'-', color=c, lw=1.9, ms=6, label=name)
axs[2].axhline(1.0, color='k', lw=1.4, ls='--')
axs[2].annotate('Jacobi baseline', (dof[0], 1.02), fontsize=8, va='bottom')
axs[2].set_title('speed-up vs Jacobi  (>1 = faster than Jacobi)', fontsize=10)
axs[2].set_ylabel('speed-up'); axs[2].set_ylim(0.3, 1.25)

for ax in axs:
    ax.set_xlabel('degrees of freedom'); ax.grid(alpha=.3, which='both')
    ax.set_xticks(dof); ax.set_xticklabels(lab, fontsize=8)
    ax.xaxis.set_minor_formatter(plt.NullFormatter())

fig.suptitle('Ghia lid-driven cavity, Re=1000 — p-multigrid overtakes Jacobi only above ~50k DOF',
             fontsize=12)
fig.tight_layout()
out = f'{SC}/cavity_precond.png'
fig.savefig(out, dpi=150, bbox_inches='tight')

print(f"{'mesh':<12}{'DOF':>8}{'Jac it':>8}{'pMG it':>8}{'it ratio':>10}{'pMG speedup':>13}")
for k in range(len(dof)):
    ji, pi = S['Jacobi'][0][k], S['p-MG (pc=p/2, d=4)'][0][k]
    sp = S['Jacobi'][1][k]/S['p-MG (pc=p/2, d=4)'][1][k]
    print(f"{lab[k].replace(chr(10),' '):<12}{dof[k]:>8}{ji:>8}{pi:>8}{ji/pi:>9.1f}x{sp:>12.2f}x")
print(f"\nsaved {out}")
