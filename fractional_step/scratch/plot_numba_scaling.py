"""Plot numba speedup vs grid resolution, separating p- and h-refinement."""
import os
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SC = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(f'{SC}/bench_numba_scaling.json'))
P, H = R['p'], R['h']

C = {'sL': ('tab:blue', 'o', 'apply_L'),
     'sLT': ('tab:orange', 's', 'apply_LT'),
     'sA': ('tab:red', 'D', 'apply_A  (full matvec)')}

fig, axs = plt.subplots(1, 3, figsize=(16, 4.9))

# ---- panel 1: p-refinement -------------------------------------------------
x = [r['N'] for r in P]
for k, (c, mk, lab) in C.items():
    axs[0].plot(x, [r[k] for r in P], mk+'-', color=c, lw=2, ms=6,
                label=lab, mew=1.2 if k == 'sA' else 1.0)
axs[0].set_xlabel('polynomial order  p')
axs[0].set_title('p-refinement  (36 elements fixed)\nblocks grow: BLAS gets efficient',
                 fontsize=10)
axs[0].set_xticks(x); axs[0].tick_params(axis='x', labelsize=8)

# ---- panel 2: h-refinement -------------------------------------------------
x2 = [r['nelem'] for r in H]
for k, (c, mk, lab) in C.items():
    axs[1].plot(x2, [r[k] for r in H], mk+'-', color=c, lw=2, ms=6, label=lab)
axs[1].set_xscale('log')
axs[1].set_xlabel('number of elements')
axs[1].set_title('h-refinement  (order 8 fixed)\nblock size unchanged: advantage persists',
                 fontsize=10)
axs[1].set_xticks(x2)
axs[1].set_xticklabels([str(v) for v in x2], fontsize=8)
axs[1].xaxis.set_minor_formatter(plt.NullFormatter())
axs[1].xaxis.set_minor_locator(plt.NullLocator())

for ax in axs[:2]:
    ax.axhline(1.0, color='k', ls='--', lw=1.3)
    ax.set_ylabel('speed-up vs NumPy')
    ax.set_ylim(0, 10.5)
    ax.grid(alpha=.3)
    ax.legend(fontsize=8, loc='upper right', framealpha=.92)

# ---- panel 3: the point -- DOF alone does NOT predict the speedup ----------
axs[2].plot([r['ndof'] for r in P], [r['sA'] for r in P], 'o-',
            color='tab:purple', lw=2.2, ms=7, label='p-refinement (36 elem, p=3..16)')
axs[2].plot([r['ndof'] for r in H], [r['sA'] for r in H], 's-',
            color='tab:green', lw=2.2, ms=7, label='h-refinement (p=8, 4..196 elem)')
axs[2].axhline(1.0, color='k', ls='--', lw=1.3)
axs[2].set_xscale('log')
axs[2].set_xlabel('degrees of freedom')
axs[2].set_ylabel('apply_A speed-up vs NumPy')
axs[2].set_title('same DOF, different speed-up\npolynomial order is what matters',
                 fontsize=10)
axs[2].set_ylim(0, 6.5)
axs[2].grid(alpha=.3, which='both')
axs[2].legend(fontsize=8.5, loc='upper right', framealpha=.92)

# default log minor-tick labels collide into an unreadable smear here; use a
# handful of explicit round ticks instead.
ticks = [1000, 2000, 5000, 10000, 20000, 50000]
axs[2].xaxis.set_major_locator(plt.FixedLocator(ticks))
axs[2].xaxis.set_minor_locator(plt.NullLocator())
axs[2].xaxis.set_minor_formatter(plt.NullFormatter())
axs[2].set_xticklabels([f'{t//1000}k' for t in ticks], fontsize=8)
axs[2].set_xlim(1050, 75000)

# annotate the two ~32k DOF points: same problem size, very different speed-up.
# purple goes to the LEFT (empty band below both curves), green ABOVE.
pa = min(P, key=lambda r: abs(r['ndof']-32400))
ha = min(H, key=lambda r: abs(r['ndof']-32400))
axs[2].annotate(f"p={pa['N']}, {pa['nelem']} elem\n{pa['sA']:.2f}x",
                (pa['ndof'], pa['sA']), textcoords='offset points',
                xytext=(-82, -4), fontsize=7.5, color='tab:purple',
                ha='left', arrowprops=dict(arrowstyle='->', color='tab:purple', lw=1.1))
axs[2].annotate(f"p={ha['N']}, {ha['nelem']} elem\n{ha['sA']:.2f}x",
                (ha['ndof'], ha['sA']), textcoords='offset points',
                xytext=(-34, 30), fontsize=7.5, color='tab:green',
                ha='center', arrowprops=dict(arrowstyle='->', color='tab:green', lw=1.1))

fig.suptitle('numba backend speed-up vs grid resolution — Apple M3 Max, NumPy 2.4.6 (Accelerate), numba 0.66',
             fontsize=12)
fig.tight_layout()
out = f'{SC}/numba_scaling.png'
fig.savefig(out, dpi=150, bbox_inches='tight')

print(f"{'sweep':<16}{'range':<24}{'apply_A speed-up':>20}")
print(f"{'p-refinement':<16}{'p=3 -> p=16 (36 elem)':<24}"
      f"{f'{P[0][chr(115)+chr(65)]:.2f}x -> {P[-1][chr(115)+chr(65)]:.2f}x':>20}")
print(f"{'h-refinement':<16}{'4 -> 196 elem (p=8)':<24}"
      f"{f'{H[0][chr(115)+chr(65)]:.2f}x -> {H[-1][chr(115)+chr(65)]:.2f}x':>20}")
print(f"\nat ~32k DOF:  p={pa['N']}/{pa['nelem']}elem = {pa['sA']:.2f}x   "
      f"vs   p={ha['N']}/{ha['nelem']}elem = {ha['sA']:.2f}x")
print(f"saved {out}")
