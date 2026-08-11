"""Chan (1996) Figure 1 reproduced: Stokes decay + temporal accuracy."""
import os, sys
import numpy as np
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

SIGMA_EXACT = 9.3137398539     # our eigenproblem
SIGMA_CHAN = 9.313316          # Chan's reported value
DTS = (0.0025, 0.00125, 0.000625)

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

# ---- right: temporal accuracy ----
ax = axs[1]
errs = np.array([abs(float(d[f'dt{dt:g}_s'][0]) - SIGMA_EXACT)/SIGMA_EXACT
                 for dt in DTS])
dts = np.array(DTS)
ax.loglog(dts, errs, 'o-', ms=8, mfc='none', mew=1.8, lw=1.6, color='tab:blue',
          label='LSSEM')
# slope-2 reference through the coarsest point
ref = errs[0]*(dts/dts[0])**2
ax.loglog(dts, ref, 'k--', lw=1.4, label='2nd order (slope 2)')
p = np.polyfit(np.log(dts), np.log(errs), 1)[0]
ax.set_xlabel('Time Step Size'); ax.set_ylabel('Relative error in $\\sigma$')
ax.set_title(f'Temporal accuracy\nfitted slope = {p:.2f}', fontsize=10)
ax.grid(alpha=.3, which='both'); ax.legend(fontsize=9)

fig.suptitle('Chan (1996) Figure 1 reproduced — '
             f'at dt = 6.25e-4 we get $\\sigma$ = {float(d["dt0.000625_s"][0]):.6f}, '
             f'Chan reports {SIGMA_CHAN}', fontsize=11.5, y=1.02)
fig.tight_layout()
out = f'{SC}/chan_fig1.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print('saved', out)
for dt in DTS:
    s = float(d[f'dt{dt:g}_s'][0])
    print(f"  dt = {dt:<9g} sigma = {s:.7f}   err vs exact {abs(s-SIGMA_EXACT)/SIGMA_EXACT:.3e}"
          f"   err vs Chan {abs(s-SIGMA_CHAN)/SIGMA_CHAN:.3e}")
print(f"  fitted convergence slope = {p:.3f}")
