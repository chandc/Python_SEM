"""Chan (1996) Figure 2 reproduced: Orr-Sommerfeld growth at three orders."""
import os, numpy as np
SC = os.path.dirname(os.path.abspath(__file__))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

SIG = 0.00223497
d = np.load(f'{SC}/os_traces_w0p15.npz')     # 0.3/1.4/0.3 wall elements
dbad = np.load(f'{SC}/os_traces.npz')        # 0.6/0.8/0.6, the wrong reading
NS = (8, 10, 14)
cols = {8: 'tab:red', 10: 'tab:blue', 14: 'tab:green'}
mks = {8: 's', 10: 'o', 14: '^'}
tt = np.linspace(0, 100, 200)

fig, axs = plt.subplots(1, 3, figsize=(17.4, 4.9))

# (a) the growth curves
ax = axs[0]
ax.plot(tt, 2*SIG*tt, 'k-', lw=2.2, zorder=1,
        label=f'Linear Theory   $\\sigma$ = {SIG}')
for N in NS:
    t = d[f'N{N}_t']; e = d[f'N{N}_e']
    st = max(1, len(t)//26)
    ax.plot(t[::st], np.log(e[::st]), mks[N], ms=6.5, mfc='none', mew=1.5,
            color=cols[N], zorder=3,
            label=f'{N}th Order  ($\\sigma$ = {float(d[f"N{N}_s"][0]):.5f})')
ax.set_xlabel('Time'); ax.set_ylabel('Natural Log of Energy Ratio')
ax.set_xlim(0, 100); ax.set_ylim(-0.02, 0.50)
ax.set_title('Predicted Growth Rate, Three Orders\nRe = 7500, 1$\\times$3 elements, dt = 0.1',
             fontsize=10)
ax.grid(alpha=.3); ax.legend(fontsize=8.5, loc='upper left')

# (b) what the wall-element size does at N = 8
ax = axs[1]
ax.plot(tt, 2*SIG*tt, 'k-', lw=2.0, label='Linear Theory')
ax.plot(dbad['N8_t'], np.log(dbad['N8_e']), ls=':', lw=2.0, color='tab:red',
        label='wall elem 0.6  (30% of full width)')
ax.plot(d['N8_t'], np.log(d['N8_e']), ls='-', lw=2.0, color='tab:red',
        label='wall elem 0.3  (30% of half-width)')
# mark the departure with a rule rather than an arrow across the data
ax.axvline(18, color='0.45', ls='--', lw=1.1, zorder=0)
ax.text(20, 1.33, 'spurious mode takes over', fontsize=8.5,
        ha='left', va='top', color='0.25')
ax.set_xlabel('Time'); ax.set_ylabel('Natural Log of Energy Ratio')
ax.set_xlim(0, 100); ax.set_ylim(-0.05, 1.4)
ax.set_title('N = 8: the wall-element reading\n(over- vs under-predicting)', fontsize=10)
ax.grid(alpha=.3); ax.legend(fontsize=8, loc='lower right')

# (c) p-refinement
ax = axs[2]
err = np.array([abs(float(d[f'N{N}_s'][0])-SIG)/SIG for N in NS])
errb = np.array([abs(float(dbad[f'N{N}_s'][0])-SIG)/SIG for N in NS])
ax.semilogy(NS, errb, 's--', ms=7, mfc='none', mew=1.4, lw=1.2, color='0.6',
            label='wall elem 0.6 (wrong reading)')
ax.semilogy(NS, err, 'o-', ms=9, mfc='none', mew=1.8, lw=1.6, color='tab:purple',
            label='wall elem 0.3')
ax.axhline(0.0076, color='k', ls='--', lw=1.4, label="Chan's N=14: 0.76%")
for N, e in zip(NS, err):
    lab = f'{e:.1%}' if e >= 0.01 else f'{e*100:.3f}%'
    ax.annotate(lab, xy=(N, e), xytext=(14, 2), textcoords='offset points',
                ha='left', fontsize=8.5, color='tab:purple')
ax.set_xlabel('Polynomial order N'); ax.set_ylabel('Relative error in growth rate')
ax.set_xticks(NS); ax.set_xlim(7.4, 15.4); ax.set_ylim(5e-5, 12)
ax.set_title('p-refinement of the growth rate', fontsize=10)
ax.grid(alpha=.3, which='both'); ax.legend(fontsize=8, loc='lower left')

fig.suptitle('Chan (1996) Figure 2 reproduced.  Wall elements span 30% of the HALF-width (0.3/1.4/0.3), not of the full width.\n'
             'Growth rate 22.8% $\\rightarrow$ 0.136% $\\rightarrow$ 0.014% for N = 8, 10, 14 — Chan reports 0.76% at N = 14.',
             fontsize=11, y=1.03)
fig.tight_layout()
out = f'{SC}/chan_fig2.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print('saved', out)
