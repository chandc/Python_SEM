"""Transient energy and enstrophy of the TGV Re = 100 run -> figs/tgv_re100_transient.png.

    uv run --quiet python scratch/tgv_re100_transient_plot.py

Left: E(t)/E0 and Omega(t)/Omega0 with the stretching peak and the initial
enstrophy dip annotated.  Right: the literature-comparable curve -- volumetric
dissipation eps(t) = 2 nu Omega / V in the standard TGV normalisation
(E(0)/V = 1/8) -- with the exact initial value 2 nu Omega0/V = 0.0075 marked.
Data: scratch/tgv_diag_re100.npz (per-step series saved by tgv3d.py).
"""
import os, sys
SC = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.dirname(SC))
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

d = np.load('scratch/tgv_diag_re100.npz')
t, E, Om, nu = d['t'], d['E'], d['Om'], float(d['nu'])
V = (2*np.pi)**3
eps = 2*nu*Om/V
ip = int(np.argmax(Om))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4))

ax1.plot(t, E/E[0], 'C0-', lw=2, label='$E/E_0$  (kinetic energy)')
ax1.plot(t, Om/Om[0], 'C3-', lw=2, label='$\\Omega/\\Omega_0$  (enstrophy)')
ax1.plot(t[ip], Om[ip]/Om[0], 'k*', ms=14)
ax1.annotate(f'stretching peak\n$\\Omega/\\Omega_0$ = {Om[ip]/Om[0]:.2f} at '
             f't = {t[ip]:.2f}', xy=(t[ip], Om[ip]/Om[0]),
             xytext=(6.6, 1.62), fontsize=10,
             arrowprops=dict(arrowstyle='->', lw=1))
idip = int(np.argmin(Om[:ip]))
ax1.annotate('initial dip: viscosity beats\nstretching until t $\\approx$ '
             f'{t[idip]:.1f}', xy=(t[idip], Om[idip]/Om[0]),
             xytext=(1.3, 0.72), fontsize=9,
             arrowprops=dict(arrowstyle='->', lw=0.9))
ax1.set_xlabel('t'); ax1.set_ylabel('normalised')
ax1.set_title('Taylor–Green vortex, Re = 100: energy decay and\n'
              'enstrophy growth (vortex stretching, impossible in 2D)')
ax1.legend(loc='center right'); ax1.grid(alpha=0.3)

ax2.plot(t, eps, 'C0-', lw=2, label='$\\varepsilon = 2\\nu\\Omega/V$')
epsn = -np.gradient(E, t)/V
ax2.plot(t, epsn, 'k--', lw=1.2, label='$-\\,d(E/V)/dt$')
ax2.plot(0, 2*nu*Om[0]/V, 'C3o', ms=8, mfc='none', mew=2,
         label=f'exact $\\varepsilon(0) = 2\\nu\\Omega_0/V$ = {2*nu*Om[0]/V:.4f}')
ax2.plot(t[ip], eps[ip], 'k*', ms=14)
ax2.annotate(f'$\\varepsilon_{{max}}$ = {eps[ip]:.5f}\nat t = {t[ip]:.2f}',
             xy=(t[ip], eps[ip]), xytext=(7.2, 0.0122), fontsize=10,
             arrowprops=dict(arrowstyle='->', lw=1))
ax2.set_xlabel('t'); ax2.set_ylabel('$\\varepsilon$  (per unit volume)')
ax2.set_title('Volumetric dissipation — the Brachet et al. (1983)\n'
              'comparison curve (standard normalisation, $E_0/V$ = 1/8)')
ax2.legend(loc='lower center', fontsize=9); ax2.grid(alpha=0.3)

fig.suptitle('TGV Re = 100  —  24$^3$ (3$\\times$3 elems N = 8, $N_z$ = 24), '
             'dt = 0.02, RKW3/CN, legacy row weights, AC off.  '
             'Energy balance holds to 0.7% worst-case.', fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.92])
out = 'figs/tgv_re100_transient.png'
fig.savefig(out, dpi=150)
print('wrote', out)
