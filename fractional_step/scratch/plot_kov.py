"""Kovasznay figures: h-refinement, spectral convergence, accuracy-vs-cost."""
import os, sys, json
import numpy as np
SC = os.path.dirname(os.path.abspath(__file__))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

d = json.load(open(f'{SC}/kov_sweep.json'))
LX, LY = 1.5, 1.0
# element size: sqrt(area/nelem).  L_x/N_x is wrong here because the
# meshes do not scale N_y with N_x (4x2 -> 8x5 -> 15x10).
hof = lambda ne: np.sqrt(LX*LY/np.asarray(ne, float))

# Chan's tables.  Total work = his Mflops * his seconds -- hardware-independent,
# unlike wall time, and the only fair cost axis across 30 years of hardware.
CH_P = [(4, 6.44e-2, 8.7, 25.7), (9, 1.56e-6, 30.3, 59.7), (14, 9.22e-13, 353, 96.0)]
CH_H = [(15, 5.49e-2, 31.6, 52.4), (30, 1.07e-2, 258, 59.7), (60, 1.56e-3, 1916, 60.5)]
ch_p_gf = [(N, e, t*m/1e3) for N, e, t, m in CH_P]
ch_h_gf = [(x, e, t*m/1e3) for x, e, t, m in CH_H]

g = lambda k, f: np.array([r[f] for r in d[k]])
fig = plt.figure(figsize=(15.0, 11.2))
gs = fig.add_gridspec(2, 2, hspace=.30, wspace=.26)

# ---------------- (a) h-refinement ----------------
ax = fig.add_subplot(gs[0, 0])
for key, N, c, mk in (('h2', 2, 'tab:blue', 'o'), ('h4', 4, 'tab:green', 's')):
    h = hof(g(key, 'nelem')); e = g(key, 'eu')
    ax.loglog(h, e, mk+'-', color=c, ms=8, mfc='none', mew=1.8, lw=1.7,
              label=f'LSSEM, N = {N}')
    s = np.polyfit(np.log(h), np.log(e), 1)[0]
    ax.text(h[len(h)//2]*1.15, e[len(e)//2]*1.6, f'slope {s:.2f}',
            color=c, fontsize=10, fontweight='bold')
    ref = e[0]*(h/h[0])**(N+1)
    ax.loglog(h, ref, ':', color=c, lw=1.2, alpha=.75)
hc = hof([150, 600, 2400]); ec = np.array([e for _, e, _ in ch_h_gf])
ax.loglog(hc, ec, '^--', color='tab:red', ms=9, mfc='none', mew=1.8, lw=1.6,
          label='Chan (1996), N = 2')
sc = np.polyfit(np.log(hc), np.log(ec), 1)[0]
ax.text(hc[1]*1.15, ec[1]*1.9, f'slope {sc:.2f}', color='tab:red',
        fontsize=10, fontweight='bold')
ax.set_xlabel(r'element size  $h=\sqrt{A/N_{elem}}$'); ax.set_ylabel(r'$\epsilon_u$  (rms)')
ax.set_title('(a) h-refinement — algebraic convergence\n'
             'dotted = theoretical slope $N{+}1$', fontsize=10.5)
ax.grid(alpha=.3, which='both'); ax.legend(fontsize=8.5)

# ---------------- (b) spectral convergence ----------------
ax = fig.add_subplot(gs[0, 1])
for key, lab, c, mk in (('p_rel', 'relative CG guard (true minimiser)', 'tab:green', 'o'),
                        ('p_abs', 'absolute CG guard (as shipped)', 'tab:blue', 's')):
    ax.semilogy(g(key, 'N'), g(key, 'eu'), mk+'-', color=c, ms=8, mfc='none',
                mew=1.8, lw=1.7, label=lab)
ax.semilogy([N for N, _, _ in ch_p_gf], [e for _, e, _ in ch_p_gf], '^--',
            color='tab:red', ms=9, mfc='none', mew=1.8, lw=1.6, label='Chan (1996)')
Nr = g('p_rel', 'N'); er = g('p_rel', 'eu')
k = (Nr >= 4) & (er > 1e-14)
if k.sum() > 2:
    b = np.polyfit(Nr[k], np.log(er[k]), 1)
    ax.semilogy(Nr, np.exp(np.polyval(b, Nr)), 'k:', lw=1.4,
                label=f'$\\propto e^{{{b[0]:.2f}N}}$  (straight = spectral)')
ax.axhspan(1e-16, 1e-13, color='0.85', zorder=0)
ax.text(2.4, 2.5e-15, 'double-precision floor', fontsize=8, color='0.35')
ax.set_xlabel('polynomial order N'); ax.set_ylabel(r'$\epsilon_u$  (rms)')
ax.set_title('(b) spectral convergence — 8 elements, fixed mesh\n'
             'straight line on log-linear axes = exponential', fontsize=10.5)
ax.grid(alpha=.3, which='both'); ax.legend(fontsize=8.5, loc='lower left')

# ---------------- (c) accuracy vs cost ----------------
ax = fig.add_subplot(gs[1, 0])
ax.loglog(g('p_rel', 'gflop'), g('p_rel', 'eu'), 'o-', color='tab:green', ms=8,
          mfc='none', mew=1.8, lw=1.7, label='p-refinement (N = 2..14, 8 elem)')
ax.loglog(g('h2', 'gflop'), g('h2', 'eu'), 's-', color='tab:blue', ms=8,
          mfc='none', mew=1.8, lw=1.7, label='h-refinement, N = 2')
ax.loglog(g('h4', 'gflop'), g('h4', 'eu'), 'd-', color='tab:purple', ms=8,
          mfc='none', mew=1.8, lw=1.7, label='h-refinement, N = 4')
ax.loglog([f for _, _, f in ch_p_gf], [e for _, e, _ in ch_p_gf], '^--',
          color='tab:red', ms=9, mfc='none', mew=1.8, lw=1.6, label='Chan p-refinement')
ax.loglog([f for _, _, f in ch_h_gf], [e for _, e, _ in ch_h_gf], 'v--',
          color='tab:orange', ms=9, mfc='none', mew=1.8, lw=1.6, label='Chan h-refinement')
ax.set_xlabel('total work  (Gflop, modelled — hardware independent)')
ax.set_ylabel(r'$\epsilon_u$  (rms)')
ax.set_title('(c) accuracy per unit work\n'
             'lower-left is better; p-refinement dominates for a smooth solution',
             fontsize=10.5)
ax.grid(alpha=.3, which='both'); ax.legend(fontsize=8, loc='lower left')

# ---------------- (d) achieved rate ----------------
ax = fig.add_subplot(gs[1, 1])
ax.plot(g('p_abs', 'N'), g('p_abs', 'mflops'), 'o-', color='tab:blue', ms=8,
        mfc='none', mew=1.8, lw=1.7, label='ours, 8 elem (NumPy backend)')
ax.plot(g('h2', 'N')*0 + 2, g('h2', 'mflops'), 's', color='tab:cyan', ms=7,
        mfc='none', mew=1.6, label='ours, N = 2 h-sweep')
ax2 = ax.twinx()
ax2.plot([r[0] for r in CH_P], [r[3] for r in CH_P],
         '^--', color='tab:red', ms=9, mfc='none', mew=1.8, lw=1.6,
         label='Chan (IBM SP2, 1995)')
ax2.set_ylabel('Chan Mflops (IBM SP2, 1995)', color='tab:red', fontsize=9)
ax2.tick_params(axis='y', labelcolor='tab:red')
ax.set_xlabel('polynomial order N'); ax.set_ylabel('achieved Mflops (ours)')
ax.set_title('(d) achieved flop rate rises with order\n'
             'Chan: 25.7 -> 96 (3.7x).  Ours: see left axis.  '
             'SEPARATE AXES — 1995 vs 2026 hardware', fontsize=10.5)
ax.grid(alpha=.3); ax.legend(fontsize=8.5, loc='upper left')

fig.suptitle('Kovasznay flow, Re = 40, domain $[-0.5,1.0]\\times[-0.5,0.5]$ — '
             'Chan (1996) replication\n'
             'Steady LSSEM ($w_{mass}$ = 0), Jacobi-CG, velocity Dirichlet from '
             'the exact solution, pressure pinned.  '
             r'$\epsilon_u$ is the rms error over unique global nodes.',
             fontsize=12, y=0.985)
fig.tight_layout(rect=[0, 0, 1, 0.955])
out = f'{SC}/kovasznay.png'
fig.savefig(out, dpi=145, bbox_inches='tight')
print('saved', out)

for key in ('h2', 'h4'):
    h = hof(g(key, 'nelem')); e = g(key, 'eu')
    print(f"{key}: fitted h-slope {np.polyfit(np.log(h), np.log(e), 1)[0]:.3f}  "
          f"(theory {d[key][0]['N']+1})")
print(f"Chan N=2 h-slope {sc:.3f}")
