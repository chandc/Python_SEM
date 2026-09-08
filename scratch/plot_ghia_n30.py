"""Ghia Re=1000 centreline profiles vs the converged N=30 4x4 solution
(Jacobi and ladder runs of pmg_ghia_cavity.py -- same steady state)."""
import os, sys
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, 'scratch')); os.chdir(_R)
os.environ.setdefault('CAV_RE', '1000')
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from lssem2d.mesh import build_channel
from pmg_ghia_cavity import centreline_u, centreline_v, GHIA_X, GHIA_V

N, EX = 30, 4
m = build_channel(1.0, 1.0, EX, EX, N, bcs=(1, 1, 1, 2)); m.compute_global_indices()
gh = np.load('cavity_re1000_data.npz'); gy, gu = gh['ghia_y'], gh['ghia_u']
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
for f, lab, c in (('scratch/pmg_ghia_cavity_jac_N30_r1.npz', 'FOSLS N=30 4x4, Jacobi', 'k'),
                  ('scratch/pmg_ghia_cavity_lad_N30_r1.npz', 'FOSLS N=30 4x4, PMG ladder', 'r')):
    z = np.load(f, allow_pickle=True); U = z['U']
    y, u = centreline_u(U, m, N); x, v = centreline_v(U, m, N)
    ax[0].plot(u, y, '-', color=c, lw=1.2 if c == 'k' else 0.8, ls='-' if c == 'k' else '--', label=f'{lab} (rms {float(z["rms_u"]):.2e})')
    ax[1].plot(x, v, '-', color=c, lw=1.2 if c == 'k' else 0.8, ls='-' if c == 'k' else '--', label=f'{lab} (rms {float(z["rms_v"]):.2e})')
    print(f'{lab}: rms_u {float(z["rms_u"]):.3e} rms_v {float(z["rms_v"]):.3e} steps {int(z["steps"])} CG total {int(z["cg"])} wall {float(z["wall"]):.0f}s')
ax[0].plot(gu, gy, 'o', mfc='none', ms=7, color='b', label='Ghia et al. 1982, Table I'); ax[0].set(xlabel='u', ylabel='y', title='u on x = 0.5'); ax[0].grid(alpha=.3); ax[0].legend(fontsize=8)
ax[1].plot(GHIA_X, GHIA_V, 'o', mfc='none', ms=7, color='b', label='Ghia et al. 1982, Table II'); ax[1].set(xlabel='x', ylabel='v', title='v on y = 0.5'); ax[1].grid(alpha=.3); ax[1].legend(fontsize=8)
fig.suptitle('Lid-driven cavity Re=1000, FOSLS 4x4 elements N=30 (converged, dt=1) vs Ghia, Ghia & Shin (1982)')
fig.tight_layout(); fig.savefig('scratch/ghia_n30.png', dpi=130); print('wrote scratch/ghia_n30.png')
