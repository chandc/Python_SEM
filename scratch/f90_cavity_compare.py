"""Fortran LSSEM (SEM_2D_PMG_CLEAN, legacy weighting) vs the Python port on the
same 4x4 N=15 Re=1000 cavity at the same dt and time: centreline u on x=0.5,
the under-lid zigzag metric, and the field difference.
    python scratch/f90_cavity_compare.py <fortran restart .dat> <python checkpoint .npz> [png]"""
import os, sys
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, 'scratch')); os.chdir(_R); os.environ.setdefault('CAV_RE', '1000')
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from fsol import load_solution
from lssem2d.mesh import build_channel
from pmg_ghia_cavity import centreline_u, centreline_v, GHIA_X, GHIA_V

fs = load_solution(sys.argv[1]); zp = np.load(sys.argv[2]); Up = zp['U0']; N = fs['N']
m = build_channel(1, 1, 4, 4, N, bcs=(1, 1, 1, 2)); m.compute_global_indices()
print(f'Fortran: t={fs["time"]:.3f} Re={fs["re"]:.0f} N={N} nelem={fs["nelem"]};  Python: t={float(zp["t"]):.3f} dt={float(zp["dt"]):g}')
# element numbering differs between the grid writer and build_channel: match by corner coordinates
perm = np.array([np.argmin(np.hypot(m.xnod[:, 0] - fs['xnod'][e, 0], m.ynod[:, 0] - fs['ynod'][e, 0])) for e in range(fs['nelem'])])
Uf = np.zeros_like(Up); Uf[perm] = fs['U']
xf = np.zeros_like(m.xnod); xf[perm] = fs['xnod']; yf = np.zeros_like(m.ynod); yf[perm] = fs['ynod']
print('grid coordinates agree (after element matching) to', max(np.abs(xf - m.xnod).max(), np.abs(yf - m.ynod).max()))
# orientation check: the lid (u = 1) must be the j = N row in Python's U[e, i, j, f]
top = [e for e in range(m.nelem) if m.bc[e, 3] == 2]
print('lid check  Python u[top, :, N] mean', Up[top, :, N, 0].mean().round(4), '| Fortran u[top, :, N]', Uf[top, :, N, 0].mean().round(4), ' u[top, N, :]', Uf[top, N, :, 0].mean().round(4))
if abs(Uf[top, N, :, 0].mean() - 1) < abs(Uf[top, :, N, 0].mean() - 1):
    Uf = np.transpose(Uf, (0, 2, 1, 3)); print('  -> Fortran array is (j,i); transposed')
for f, nm in enumerate(('u', 'v', 'p', 'om')):
    print(f'  max|{nm}_F - {nm}_P| = {np.abs(Uf[..., f] - Up[..., f]).max():.3e}   (max|{nm}_P| = {np.abs(Up[..., f]).max():.3e})')
def zz(U):
    y, u = centreline_u(U, m, N); sel = (y > 0.75) & (y < 0.95); d = np.diff(u[sel])
    return y, u, np.abs(d).mean(), np.abs(d).max(), int((np.sign(d[1:]) != np.sign(d[:-1])).sum()), d.size - 1
gh = np.load('cavity_re1000_data.npz'); gy, gu = gh['ghia_y'], gh['ghia_u']
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
for U, lab, c in ((Uf, 'Fortran SEM_2D_PMG_CLEAN', 'k'), (Up, 'Python lssem2d', 'r')):
    y, u, a, mx, sc, nn = zz(U); x, v = centreline_v(U, m, N)
    rms_u = np.sqrt(np.mean((np.interp(gy, y, u) - gu)**2)); o = np.argsort(GHIA_X); rms_v = np.sqrt(np.mean((np.interp(GHIA_X[o], x, v) - GHIA_V[o])**2))
    print(f'  {lab:26s}: under-lid zigzag mean|du| {a:.4f} max {mx:.4f} sign changes {sc}/{nn} | rms vs Ghia u {rms_u:.4f} v {rms_v:.4f}')
    ax[0].plot(u, y, '-', color=c, lw=1.2 if c == 'k' else 0.8, ls='-' if c == 'k' else '--', label=lab); ax[1].plot(x, v, '-', color=c, lw=1.2 if c == 'k' else 0.8, ls='-' if c == 'k' else '--', label=lab)
ax[0].plot(gu, gy, 'o', mfc='none', color='b', label='Ghia (steady)'); ax[1].plot(GHIA_X, GHIA_V, 'o', mfc='none', color='b')
ax[0].set(xlabel='u', ylabel='y', title='u on x = 0.5'); ax[1].set(xlabel='x', ylabel='v', title='v on y = 0.5')
for a_ in ax: a_.grid(alpha=.3); a_.legend(fontsize=8)
fig.suptitle(f'Re=1000 cavity, 4x4 elements N={N}, dt={float(zp["dt"]):g}, t={fs["time"]:.1f}: Fortran vs Python, legacy weighting')
fig.tight_layout(); out = sys.argv[3] if len(sys.argv) > 3 else 'scratch/f90_cavity_compare.png'; fig.savefig(out, dpi=130); print('wrote', out)
