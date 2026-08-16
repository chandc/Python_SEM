"""Permissible region in the (dt, w_mom) plane -- Gartling Re = 800, LSSEM VVP.

    uv run --quiet python scratch/gartling_permissible.py

TWO CONSTRAINTS BOUND THE USABLE REGION, AND THEY CLOSE ON EACH OTHER.

1. STABILITY.  The continuity and vorticity rows of the least-squares functional
   carry weight exactly 1 (hard-coded in apply_L), so

       a_mass = w_mass*fac1/dt

   measures the time-derivative term against incompressibility.  Eliminating dt
   and w_mass through dt_eff = dt*w_mom/w_mass gives the identity

       a_mass = fac1 * a_flux / dt_eff          (verified exactly on 20 runs)

   Measured on this problem: every run with a_mass <= 6.05 stayed bounded, every
   run with a_mass >= 12.1 diverged, 25 runs with no crossover.  In terms of the
   plotted variables that is

       w_mom <= (a_crit/fac1) * dt_eff,   a_crit in [6.05, 12.1]

   i.e. slope between 4.03 and 8.07.  The band between is untested.

2. ACCURACY.  Pressure appears ONLY in the momentum rows, always multiplied by
   a_flux = w_mom, so the pressure block of L^T L scales as a_flux^2.  Sweeping
   w_mom on the converged 18x4 solution gave reattachment within +/-1.2% of
   Gartling for w_mom in [0.25, 2.0] and -4.6% at w_mom = 0.1 (-11.6% for the
   steady form).  So w_mom >~ 0.25 is required for the answer to be right.

The two together leave a WEDGE that closes below dt_eff ~ 0.06: there is a
smallest usable physical time step, and no choice of weights gets under it.

AXES.  Plotted against dt_eff (the step the scheme actually takes) and
a_flux = w_mom.  For a time-accurate run w_mass = w_mom, so dt_eff = dt and the
axes are literally (dt, w_mom).  Runs with w_mass != w_mom are placed at their
dt_eff, which is where they actually live.

    figs/gartling_permissible_region.png
"""
import os, sys, glob
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

FAC1 = 1.5
A_LO, A_HI = 6.05, 12.1          # stability bracket, measured
W_ACC = 0.25                     # accuracy floor on w_mom, measured

rows = []
for f in glob.glob(f'{SC}/gartling_unsteady_nx1*_N6_*.npz'):
    d = np.load(f, allow_pickle=True); k = set(d.keys())
    g = lambda n, dv: (float(d[n]) if n in k else dv)
    wm = g('wmom', 1.0); ws = g('wmass', 1.0); dt = g('dt', 0.1)
    if not ('dt_eff' in k or wm == ws):
        continue                                  # untrustworthy time axis
    st = str(d['status']); h = d['hist']
    ok = (not st.startswith(('BLEWUP', 'NaN'))) and len(h) and h[:, 1].max() < 2.0
    rows.append((dt*wm/ws, wm, ok, float(h[-1, 0]) if len(h) else 0.0))

fig, ax = plt.subplots(figsize=(10.5, 7.6))
x = np.logspace(-3.1, 0.55, 400)

# --- unstable region: w_mom above the stability line ---
ax.fill_between(x, A_LO/FAC1*x, 1e3, color='tab:red', alpha=.13, zorder=0)
ax.fill_between(x, A_LO/FAC1*x, A_HI/FAC1*x, color='tab:red', alpha=.16, zorder=0)
ax.plot(x, A_LO/FAC1*x, '-', color='tab:red', lw=2.0,
        label=r'stability: $a_{mass}$ = 6.05  ($w_{mom}$ = 4.03 $dt$)')
ax.plot(x, A_HI/FAC1*x, '--', color='tab:red', lw=1.6,
        label=r'stability: $a_{mass}$ = 12.1  ($w_{mom}$ = 8.07 $dt$)')

# --- inaccurate region: w_mom below the accuracy floor ---
ax.axhspan(1e-3, W_ACC, color='tab:blue', alpha=.13, zorder=0)
ax.axhline(W_ACC, color='tab:blue', lw=2.0,
           label=r'accuracy: $w_{mom}$ = 0.25 (pressure block $\propto w_{mom}^2$)')

# --- the permissible wedge ---
xw = x[(A_LO/FAC1*x) > W_ACC]
if len(xw):
    ax.fill_between(xw, W_ACC, A_LO/FAC1*xw, color='tab:green', alpha=.22,
                    zorder=1, label='PERMISSIBLE')
    dt_min = W_ACC*FAC1/A_LO
    ax.axvline(dt_min, color='tab:green', lw=1.6, ls=':')
    ax.annotate(f'wedge closes at\n$dt$ = {dt_min:.3f}', (dt_min, 0.30),
                xytext=(dt_min*1.25, 0.30), fontsize=9, color='darkgreen',
                arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.2))

# --- the runs ---
sx = [r[0] for r in rows if r[2]]; sy = [r[1] for r in rows if r[2]]
bx = [r[0] for r in rows if not r[2]]; by = [r[1] for r in rows if not r[2]]
ax.plot(sx, sy, 'o', ms=10, mfc='none', mec='darkgreen', mew=2.0,
        label=f'STABLE  ({len(sx)} runs)', zorder=5)
ax.plot(bx, by, 'x', ms=11, color='darkred', mew=2.4,
        label=f'DIVERGED  ({len(bx)} runs)', zorder=5)

ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(1e-3, 3.0); ax.set_ylim(2e-2, 5.0)
ax.set_xlabel(r'$dt_{eff}$   ( = $dt$ when $w_{mass}$ = $w_{mom}$, i.e. time-accurate )',
              fontsize=11)
ax.set_ylabel(r'$w_{mom}$   ( = $a_{flux}$ )', fontsize=11)
ax.grid(alpha=.3, which='both')
ax.legend(fontsize=9, loc='upper left')
ax.set_title('Permissible ($dt$, $w_{mom}$) region -- Gartling BFS Re = 800, 11x4 / 18x4, N = 6\n'
             r'$a_{mass} = fac_1 \, a_{flux} / dt_{eff}$ : refining $dt$ at fixed $w_{mom}$ '
             'walks you into the unstable region;\nlowering $w_{mom}$ to escape walks you into '
             'the inaccurate one.', fontsize=11.5)
fig.tight_layout()
fig.savefig('figs/gartling_permissible_region.png', dpi=130, bbox_inches='tight')
print('figs/gartling_permissible_region.png')

print(f"\n{'dt_eff':>9}{'w_mom':>8}{'a_mass':>8}{'ran to':>8}  verdict")
for r in sorted(set(rows)):
    print(f'{r[0]:>9g}{r[1]:>8g}{FAC1*r[1]/r[0]:>8.3g}{r[3]:>8.1f}  '
          f'{"STABLE" if r[2] else "DIVERGED"}')
print(f'\nwedge closes at dt_eff = {W_ACC*FAC1/A_LO:.4f} (conservative, a_crit=6.05)')
print(f'                        {W_ACC*FAC1/A_HI:.4f} (optimistic,  a_crit=12.1)')
