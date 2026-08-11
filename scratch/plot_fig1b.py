"""Chan (1996) Fig. 1 with the p-refinement study folded into the right panel."""
import os, sys
import numpy as np
SC = os.path.dirname(os.path.abspath(__file__))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

SIGMA_EXACT = 9.3137398539
SIGMA_CHAN = 9.313316
DTS = (0.0025, 0.00125, 0.000625)
NS = (6, 8, 10, 14)

pre = np.load(f'{SC}/stokes_pref.npz')
dts = pre['dts']
tr6 = np.load(f'{SC}/stokes_traces_N6.npz')

fig, axs = plt.subplots(1, 2, figsize=(13.6, 5.2))

# ---- left: ln(E/E0) vs t ----
ax = axs[0]
tt = np.linspace(0, 0.1, 200)
ax.plot(tt, -2*SIGMA_EXACT*tt, 'k-', lw=2.2, zorder=1,
        label=f'analytical   $\\sigma$ = {SIGMA_EXACT:.6f}')
for k, (dt, mk) in enumerate(zip(DTS, ['o', 's', '^'])):
    t = tr6[f'dt{dt:g}_t']; e = tr6[f'dt{dt:g}_e']
    st = max(1, len(t)//22)
    ax.plot(t[::st], np.log(e[::st]), mk, ms=6.5, mfc='none', mew=1.5,
            zorder=3, label=f'dt = {dt:g}')
ax.set_xlabel('Time'); ax.set_ylabel('Natural Log of Total Kinetic Energy  $E/E_0$')
ax.set_title('Stokes decay, periodic channel\n2$\\times$4 elements, order 6', fontsize=10)
ax.grid(alpha=.3); ax.legend(fontsize=9, loc='lower left'); ax.set_xlim(0, 0.1)

# ---- right: temporal accuracy at four polynomial orders ----
ax = axs[1]
cols = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
mks = ['o', 's', '^', 'D']
m = (dts <= 0.02) & (dts >= 0.00125)
for n, c, mk in zip(NS, cols, mks):
    e = pre[f'N{n}']
    p = np.polyfit(np.log(dts[m]), np.log(e[m]), 1)[0]
    ax.loglog(dts, e, mk+'-', ms=7, mfc='none', mew=1.6, lw=1.4, color=c,
              label=f'N = {n}   (slope {p:.2f})')
ref = pre['N14'][m][0]*(dts/dts[m][0])**2
ax.loglog(dts, ref, 'k--', lw=1.4, label='slope 2 reference')

ax.axvspan(0.00125, 0.02, color='0.9', zorder=0)
ax.text(0.012, 2.2e-5, 'temporal regime\n(all orders coincide)',
        fontsize=8.5, ha='center')
ax.annotate('spatial floor drops 14$\\times$\nas N goes 6 $\\rightarrow$ 10',
            xy=(6.6e-4, pre['N10'][-1]), xytext=(2.6e-3, 3.0e-6),
            fontsize=8, ha='center', arrowprops=dict(arrowstyle='->', lw=1.0))
ax.annotate('large-dt saturation', xy=(0.08, pre['N6'][0]),
            xytext=(0.03, pre['N6'][0]*2.4), fontsize=8, ha='center',
            arrowprops=dict(arrowstyle='->', lw=1.0))
ax.set_xlabel('Time Step Size'); ax.set_ylabel('Relative error in $\\sigma$')
ax.set_title('Temporal accuracy vs polynomial order', fontsize=10)
ax.set_ylim(5e-7, 4e-1)
ax.grid(alpha=.3, which='both'); ax.legend(fontsize=8, loc='lower right')

fig.suptitle('Chan (1996) Figure 1 reproduced, with p-refinement.  '
             f'At dt = 6.25e-4, N = 6: $\\sigma$ = {float(tr6["dt0.000625_s"][0]):.6f} '
             f'vs Chan {SIGMA_CHAN}.\n'
             'Error is identical across all orders for dt $\\geq$ 0.005 (purely temporal); '
             'raising N only lowers the floor at the finest dt, which lifts the fitted slope to 1.99.',
             fontsize=10.5, y=1.04)
fig.tight_layout()
out = f'{SC}/chan_fig1_pref.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print('saved', out)
