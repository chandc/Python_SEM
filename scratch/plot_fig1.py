"""Chan (1996) Figure 1 reproduced: Stokes decay + temporal accuracy."""
import os, sys
import numpy as np
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

SIGMA_EXACT = 9.3137398539     # our eigenproblem
SIGMA_CHAN = 9.313316          # Chan's reported value
DTS = (0.0025, 0.00125, 0.000625)          # Chan's three, left panel
ACC = (0.1, 0.05, 0.02, 0.01, 0.005, 0.0025, 0.00125, 0.000625)
FIT = (0.02, 0.01, 0.005, 0.0025)          # the clean slope-2 region

d = np.load(f'{SC}/stokes_traces.npz')

fig, axs = plt.subplots(1, 2, figsize=(13.2, 5.0))

# ---- left: ln(E/E0) vs t, the three time steps against the analytic line ----
ax = axs[0]
tt = np.linspace(0, 0.1, 200)
ax.plot(tt, -2*SIGMA_EXACT*tt, 'k-', lw=2.2, zorder=1,
        label=f'analytical   $\\sigma$ = {SIGMA_EXACT:.6f}')
marks = ['o', 's', '^']
for k, dt in enumerate(DTS):
    t = d[f'dt{dt:g}_t']; e = d[f'dt{dt:g}_e']
    step = max(1, len(t)//22)
    ax.plot(t[::step], np.log(e[::step]), marks[k], ms=6.5, mfc='none',
            mew=1.5, zorder=3, label=f'dt = {dt:g}')
ax.set_xlabel('Time'); ax.set_ylabel('Natural Log of Total Kinetic Energy  $E/E_0$')
ax.set_title('Stokes decay in a periodic channel\n'
             '2$\\times$4 elements, order 6, Re$^{-1}$ = 1', fontsize=10)
ax.grid(alpha=.3); ax.legend(fontsize=9, loc='lower left')
ax.set_xlim(0, 0.1)

# ---- right: temporal accuracy, Chan's full dt span ----
ax = axs[1]
dts = np.array(ACC)
errs = np.array([abs(float(d[f'dt{dt:g}_s'][0]) - SIGMA_EXACT)/SIGMA_EXACT
                 for dt in ACC])
ax.loglog(dts, errs, 'o-', ms=8, mfc='none', mew=1.8, lw=1.6, color='tab:blue',
          label='LSSEM')
# highlight the three time steps used in the left panel
mask = np.array([dt in DTS for dt in ACC])
ax.loglog(dts[mask], errs[mask], 'o', ms=8, color='tab:green',
          label="Chan's three dt (left panel)")

fi = np.array([dt in FIT for dt in ACC])
p = np.polyfit(np.log(dts[fi]), np.log(errs[fi]), 1)[0]
ref = errs[fi][0]*(dts/dts[fi][0])**2
ax.loglog(dts, ref, 'k--', lw=1.4, label='slope 2 reference')

ax.axvspan(0.0025, 0.02, color='0.85', zorder=0)
ax.text(0.006, errs.min()*1.5, f'slope-2 region\nfitted {p:.2f}',
        fontsize=8.5, ha='center')
ax.annotate('spatial error floor\n(2$\\times$4 elem, order 6)',
            xy=(6.25e-4, errs[-1]), xytext=(1.2e-3, errs[-1]*0.35),
            fontsize=8, ha='center',
            arrowprops=dict(arrowstyle='->', lw=1.0))
ax.annotate('large-dt saturation', xy=(0.07, errs[0]), xytext=(0.02, errs[0]*2.2),
            fontsize=8, ha='center', arrowprops=dict(arrowstyle='->', lw=1.0))
ax.set_xlabel('Time Step Size'); ax.set_ylabel('Relative error in $\\sigma$')
ax.set_title('Temporal accuracy', fontsize=10)
ax.grid(alpha=.3, which='both'); ax.legend(fontsize=8, loc='lower right')

fig.suptitle('Chan (1996) Figure 1 reproduced — '
             f'at dt = 6.25e-4 we get $\\sigma$ = {float(d["dt0.000625_s"][0]):.6f}, '
             f'Chan reports {SIGMA_CHAN}', fontsize=11.5, y=1.02)
fig.tight_layout()
out = f'{SC}/chan_fig1.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print('saved', out)
for dt in ACC:
    s = float(d[f'dt{dt:g}_s'][0])
    print(f"  dt = {dt:<9g} sigma = {s:.7f}   err vs exact {abs(s-SIGMA_EXACT)/SIGMA_EXACT:.3e}"
          f"   err vs Chan {abs(s-SIGMA_CHAN)/SIGMA_CHAN:.3e}")
print(f"  fitted convergence slope = {p:.3f}")
