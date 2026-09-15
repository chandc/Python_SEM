"""Poiseuille Re=100: impact of dt on profile accuracy and pressure drop."""
import os, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, NullLocator, FixedLocator

SC = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(f'{SC}/poiseuille_dt.json'))
DP_EXACT = 1.2
STY = {'control': ('tab:blue', 'o', 'control  (parabolic inlet, p=8)'),
       'develop': ('tab:orange', 's', 'developing  (uniform inlet, p=8)'),
       'coarse':  ('tab:red', '^', 'coarse  (uniform inlet, p=4)')}

# p-block / u-block diagonal ratio of L^T L, measured separately: ratio = a_flux^2
ratio = lambda dt: dt**2

fig, axs = plt.subplots(1, 3, figsize=(16, 4.9))

for k, (c, mk, lab) in STY.items():
    d = sorted([r for r in R if r['name'] == k and r['ok'] and r['dt'] > 0],
               key=lambda r: r['dt'])
    dts = [r['dt'] for r in d]
    axs[0].loglog(dts, [r['prof_err'] for r in d], mk+'-', color=c, lw=2, ms=6.5, label=lab)
    axs[1].semilogx(dts, [r['dp'] for r in d], mk+'-', color=c, lw=2, ms=6.5, label=lab)
    # dt=0 (pure steady form) shown as an open marker at the left edge
    z = [r for r in R if r['name'] == k and r['ok'] and r['dt'] == 0]
    if z:
        axs[0].loglog([0.035], [z[0]['prof_err']], mk, color=c, mfc='none', mew=1.8, ms=8)
        axs[1].semilogx([0.035], [z[0]['dp']], mk, color=c, mfc='none', mew=1.8, ms=8)

axs[0].axvline(1.0, color='k', ls='--', lw=1.4)
axs[0].annotate('equal weight\n(a_flux = 1)', (1.0, 3e-1), textcoords='offset points',
                xytext=(7, 0), fontsize=8.5)
axs[0].set_ylabel('$|u-u_{exact}|_{rms}\\, /\\, U_{max}$')
axs[0].set_title('velocity profile error at the outlet', fontsize=10)
axs[0].legend(fontsize=8, loc='lower left')

axs[1].axhline(DP_EXACT, color='k', ls='-', lw=1.6)
axs[1].annotate('exact $\\Delta p = 1.2$', (0.05, DP_EXACT), textcoords='offset points',
                xytext=(2, -14), fontsize=8.5)
axs[1].axvline(1.0, color='k', ls='--', lw=1.4)
axs[1].set_ylim(1.0, 3.4)
axs[1].set_ylabel('pressure drop over $L=10$')
axs[1].set_title('pressure drop  (open marker = pure steady form)', fontsize=10)
axs[1].legend(fontsize=8, loc='upper right')

# --- why: the pressure block of L^T L is weighted a_flux^2 -------------------
dd = np.logspace(-1.5, 0.8, 100)
axs[2].loglog(dd, ratio(dd), '-', color='tab:purple', lw=2.2, label='$p$-block / $u$-block of $L^TL$')
axs[2].axhline(1.0, color='k', ls='-', lw=1.5)
axs[2].axvline(1.0, color='k', ls='--', lw=1.4)
ctrl = sorted([r for r in R if r['name'] == 'control' and r['ok'] and r['dt'] > 0],
              key=lambda r: r['dt'])
ax2 = axs[2].twinx()
ax2.loglog([r['dt'] for r in ctrl], [r['prof_err'] for r in ctrl], 'o--',
           color='tab:blue', lw=1.6, ms=6, alpha=.75, label='control profile error')
ax2.set_ylabel('profile error', color='tab:blue', fontsize=9)
ax2.tick_params(axis='y', colors='tab:blue', labelsize=8)
axs[2].annotate('pressure under-weighted\n$L^TL$ near-singular in $p$', (0.06, 4e-3),
                fontsize=8, color='tab:purple')
axs[2].set_ylabel('diagonal ratio')
axs[2].set_title('pressure enters ONLY the momentum rows', fontsize=10)
axs[2].legend(fontsize=8, loc='upper left')

for ax in axs:
    ax.set_xlabel('dt')
    ax.grid(alpha=.3, which='both')
    ax.xaxis.set_major_locator(FixedLocator([0.05, 0.1, 0.5, 1, 2, 5]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticklabels(['0.05', '0.1', '0.5', '1', '2', '5'], fontsize=8.5)

fig.suptitle('Plane Poiseuille, Re = $U_{mean}H/\\nu$ = 100 — the least-squares weight is dt, '
             'and it decides whether pressure is constrained at all', fontsize=11.5)
fig.tight_layout()
out = f'{SC}/poiseuille_dt.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f"saved {out}")
for k in STY:
    d = sorted([r for r in R if r['name'] == k and r['ok'] and r['dt'] > 0], key=lambda r: r['dt'])
    p = np.array([r['prof_err'] for r in d])
    best = d[int(np.argmin(p))]
    print(f"  {k:<9} best at dt={best['dt']:<5} profile {best['prof_err']:.2e}  "
          f"dp {best['dp']:.5f}   spread over dt {p.max()/p.min():8.0f}x")
