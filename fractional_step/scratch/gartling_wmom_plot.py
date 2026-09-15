"""Reattachment length as a function of w_mom, Gartling Re = 800, 18x4 N=6.

    uv run --quiet python scratch/gartling_wmom_plot.py

Every unsteady point is a CONTINUATION from the converged w_mom = 0.1 field with
one parameter changed, run to t = 400, and every one reached a genuine fixed
point (max|v| peak-to-peak 2e-11 to 2e-10 over the last 20 time units).  Held
fixed: 18x4 grid, N = 6, dt = 0.1, w_mass = 0.1, nsub = 3, P+Z outlet, nu=1/800.
Since w_mass and dt are fixed, a_mass = w_mass*fac1/dt = 1.5 throughout -- well
inside the stable band -- so nothing here is contaminated by the a_mass
instability that kills runs at a_mass >= 12.

The two STEADY-FORM points (w_mass = 0, a_mass = 0) are overlaid for contrast.
At a fixed point the BDF2 mass term cancels identically, so the two formulations
ought to share a solution; they do not, and the gap grows as w_mom falls.

    figs/gartling_reattach_vs_wmom.png
"""
import os, sys
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
import glob
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

GART = dict(lo=6.1, sep=4.8, re=10.5)

rows = []
for f in glob.glob(f'{SC}/gartling_unsteady_nx18_N6_dt0.1_T400_nsub3_pz_restart_wm*.npz'):
    d = np.load(f, allow_pickle=True); h = d['hist']
    t = h[:, 0]; m = t > t[-1]-20
    rows.append((float(d['wmom']), h[-1, 3], h[-1, 4], h[-1, 5],
                 h[m, 2].max()-h[m, 2].min()))
d = np.load(f'{SC}/gartling_unsteady_nx18_N6_dt0.1_T400_nsub3_pz_stagnant_wm0.1_ws0.1.npz',
            allow_pickle=True)
h = d['hist']; t = h[:, 0]; m = t > t[-1]-20
rows.append((0.1, h[-1, 3], h[-1, 4], h[-1, 5], h[m, 2].max()-h[m, 2].min()))
rows.sort()
w = np.array([r[0] for r in rows]); lo = np.array([r[1] for r in rows])
sep = np.array([r[2] for r in rows]); ur = np.array([r[3] for r in rows])

# steady-form points (w_mass = 0), from the run logs
STEADY = [(0.1, 5.392, 4.102, 10.211), (1.0, 6.158, 4.916, 10.466)]
sw = np.array([s[0] for s in STEADY]); slo = np.array([s[1] for s in STEADY])

fig, axs = plt.subplots(1, 2, figsize=(14.0, 5.6))

ax = axs[0]
ax.axhline(GART['lo'], color='goldenrod', lw=2.0, ls='--', zorder=1,
           label="Gartling / Chan  $x_r$ = 6.1")
ax.axhspan(GART['lo']*0.99, GART['lo']*1.01, color='gold', alpha=.18, zorder=0,
           label='+/- 1%')
ax.plot(w, lo, 'o-', color='tab:blue', ms=8, lw=2.0,
        label=r'unsteady, $w_{mass}$=0.1 ($a_{mass}$=1.5)')
ax.plot(sw, slo, 's--', color='tab:red', ms=9, lw=1.6,
        label=r'steady form, $w_{mass}$=0 ($a_{mass}$=0)')
for x, y in zip(w, lo):
    ax.annotate(f'{y:.3f}', (x, y), textcoords='offset points', xytext=(0, 9),
                ha='center', fontsize=8.5, color='tab:blue')
for x, y in zip(sw, slo):
    ax.annotate(f'{y:.3f}', (x, y), textcoords='offset points', xytext=(0, -15),
                ha='center', fontsize=8.5, color='tab:red')
ax.set_xscale('log'); ax.set_xlabel(r'$w_{mom}$  ( = $a_{flux}$ )')
ax.set_ylabel('lower-wall reattachment  $x_r$')
ax.set_title('Reattachment vs momentum weight', fontsize=12)
ax.grid(alpha=.3, which='both'); ax.legend(fontsize=9, loc='lower right')

ax = axs[1]
for arr, nm, col, tgt in ((lo, 'lower reattach', 'tab:blue', GART['lo']),
                          (sep, 'upper separation', 'tab:green', GART['sep']),
                          (ur, 'upper reattach', 'tab:purple', GART['re'])):
    ax.plot(w, arr/tgt, 'o-', color=col, ms=7, lw=1.8, label=f'{nm} / {tgt}')
ax.axhline(1.0, color='goldenrod', lw=2.0, ls='--', label='Gartling')
ax.axhspan(0.99, 1.01, color='gold', alpha=.18)
ax.set_xscale('log'); ax.set_xlabel(r'$w_{mom}$')
ax.set_ylabel('our value / Gartling value')
ax.set_title('All three features, normalised', fontsize=12)
ax.grid(alpha=.3, which='both'); ax.legend(fontsize=9)

fig.suptitle('Gartling BFS Re = 800, 18x4 N=6 -- converged reattachment vs $w_{mom}$.\n'
             'Continuations from the converged $w_{mom}$=0.1 field; every point a true '
             'fixed point (max|v| p2p 2e-11 .. 2e-10).  $w_{mass}$=0.1, dt=0.1 fixed, '
             'so $a_{mass}$=1.5 throughout.', fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig('figs/gartling_reattach_vs_wmom.png', dpi=125, bbox_inches='tight')
print('figs/gartling_reattach_vs_wmom.png')

print(f"\n{'w_mom':>7}{'lo_reatt':>10}{'vs 6.1':>9}{'up_sep':>9}{'vs 4.8':>9}"
      f"{'up_re':>8}{'vs 10.5':>9}")
for r in rows:
    print(f"{r[0]:>7g}{r[1]:>10.4f}{(r[1]/GART['lo']-1)*100:>8.1f}%"
          f"{r[2]:>9.3f}{(r[2]/GART['sep']-1)*100:>8.1f}%"
          f"{r[3]:>8.3f}{(r[3]/GART['re']-1)*100:>8.1f}%")
print(f"\nsteady form (w_mass = 0):")
for s in STEADY:
    print(f"{s[0]:>7g}{s[1]:>10.4f}{(s[1]/GART['lo']-1)*100:>8.1f}%"
          f"{s[2]:>9.3f}{(s[2]/GART['sep']-1)*100:>8.1f}%"
          f"{s[3]:>8.3f}{(s[3]/GART['re']-1)*100:>8.1f}%")
