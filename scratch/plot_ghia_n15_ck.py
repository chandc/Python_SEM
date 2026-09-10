"""Centreline profiles at every checkpoint of the N=15 run vs Ghia Re=1000."""
import os, sys, glob
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, 'scratch')); os.chdir(_R)
os.environ.setdefault('CAV_RE', '1000')
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from lssem2d.mesh import build_channel
from pmg_ghia_cavity import centreline_u, centreline_v, GHIA_X, GHIA_V

D = sys.argv[1] if len(sys.argv) > 1 else 'scratch/ghia_n15'
cks = sorted(glob.glob(f'{D}/checkpoint_*.npz'))
z0 = np.load(cks[-1]); N = int(z0['N']); EX = int(z0['EX'])
m = build_channel(1.0, 1.0, EX, EX, N, bcs=(1, 1, 1, 2)); m.compute_global_indices()
gh = np.load('cavity_re1000_data.npz'); gy, gu = gh['ghia_y'], gh['ghia_u']; o = np.argsort(GHIA_X)
steady = np.load('scratch/pmg_ghia_cavity_lad_N30_r1.npz', allow_pickle=True)['U']   # converged N=30 (dt=1) reference
m30 = build_channel(1.0, 1.0, 4, 4, 30, bcs=(1, 1, 1, 2)); m30.compute_global_indices()
fig, ax = plt.subplots(1, 3, figsize=(17, 5.2)); cols = plt.cm.viridis(np.linspace(0.1, 0.95, len(cks)))
rows = []
for c, f in zip(cols, cks):
    z = np.load(f); t = float(z['t']); y, u = centreline_u(z['U0'], m, N); x, v = centreline_v(z['U0'], m, N)
    ru = np.sqrt(np.mean((np.interp(gy, y, u) - gu)**2)); rv = np.sqrt(np.mean((np.interp(GHIA_X[o], x, v) - GHIA_V[o])**2))
    rows.append((int(z['step']), t, ru, rv, np.mean(z['cg_it'][-500:]) if len(z['cg_it']) else np.nan))
    ax[0].plot(u, y, '-', color=c, lw=1, label=f't={t:.1f}  rms {ru:.3f}'); ax[1].plot(x, v, '-', color=c, lw=1, label=f't={t:.1f}  rms {rv:.3f}')
ys, us = centreline_u(steady, m30, 30); xs, vs = centreline_v(steady, m30, 30)
ax[0].plot(us, ys, 'k--', lw=1, label='steady, N=30 (dt=1 solver)'); ax[1].plot(vs, xs, 'k--', lw=1) if False else ax[1].plot(xs, vs, 'k--', lw=1, label='steady, N=30 (dt=1 solver)')
ax[0].plot(gu, gy, 'o', mfc='none', ms=7, color='r', label='Ghia 1982'); ax[1].plot(GHIA_X, GHIA_V, 'o', mfc='none', ms=7, color='r', label='Ghia 1982')
ax[0].set(xlabel='u', ylabel='y', title='u on x = 0.5'); ax[1].set(xlabel='x', ylabel='v', title='v on y = 0.5')
for a in ax[:2]: a.grid(alpha=.3); a.legend(fontsize=7)
r = np.array(rows); ax[2].semilogy(r[:, 1], r[:, 2], 'o-', label='rms u vs Ghia'); ax[2].semilogy(r[:, 1], r[:, 3], 's-', label='rms v vs Ghia')
ax[2].set(xlabel='t', ylabel='rms', title='distance to Ghia vs time'); ax[2].grid(alpha=.3); ax[2].legend()
fig.suptitle(f'Re=1000 cavity, {EX}x{EX} elements N={N}, dt={float(z0["dt"]):g}, BDF2 from rest, condensed vertex-patch Schwarz: checkpoints vs Ghia')
fig.tight_layout(); out = f'{D}/vs_ghia.png'; fig.savefig(out, dpi=130)
print(f'{"step":>5} {"t":>4} {"rms u":>7} {"rms v":>7} {"CG it/step":>10}')
for s, t, ru, rv, it in rows: print(f'{s:5d} {t:4.1f} {ru:7.4f} {rv:7.4f} {it:10.0f}')
print('converged N=30 reference rms: u 3.21e-3, v 6.66e-3 ; wrote', out)
