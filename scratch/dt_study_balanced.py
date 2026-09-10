"""dt-dependence of the balanced-weighting (w=sqrt(dt)) cavity solution at N=15.

For each run directory: latest checkpoint (or final.npz) -> rms/max error vs
Ghia on the two centrelines, the under-lid zigzag metric (consecutive-GLL-node
du on x=0.5, 0.75<y<0.95: mean |du|, sign changes), and the drift from the
dt=1e-2 steady field (max |U-U_ref| over u,v).
"""
import os, sys, glob
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, 'scratch')); os.chdir(_R)
os.environ.setdefault('CAV_RE', '1000')
import numpy as np
from lssem2d.mesh import build_channel
from pmg_ghia_cavity import centreline_u, centreline_v, GHIA_X, GHIA_V

RUNS = [('0.1  (from rest)',            'scratch/ghia_n15_wsqrt_dt1e-1'),
        ('1e-2 (from rest, reference)', 'scratch/ghia_n15_w0.1_steady'),
        ('1e-3 (from rest)',            'scratch/ghia_n15_wsqrt_dt1e-3'),
        ('1e-3 (restart from 1e-2 steady)', 'scratch/ghia_n15_wsqrt_dt1e-3_from_steady'),
        ('1e-4 (restart from 1e-2 steady)', 'scratch/ghia_n15_wsqrt_dt1e-4_from_steady'),
        ('legacy 1e-2 (plateau)',       'scratch/ghia_n15_dt1e-2'),
        ('legacy 1   (t=223)',          'scratch/ghia_n15_dt1')]


def latest(d):
    f = f'{d}/final.npz'
    cks = sorted(glob.glob(f'{d}/checkpoint_*.npz'))
    if os.path.exists(f) and (not cks or os.path.getmtime(f) >= os.path.getmtime(cks[-1])): return f
    return cks[-1] if cks else None


def zigzag(y, u, lo=0.75, hi=0.95):
    k = (y > lo) & (y < hi); du = np.diff(u[k])
    return np.mean(np.abs(du)), int(np.sum(np.sign(du[1:]) != np.sign(du[:-1]))), len(du) - 1


if __name__ == '__main__':
    N = 15; m = build_channel(1.0, 1.0, 4, 4, N, bcs=(1, 1, 1, 2)); m.compute_global_indices()
    gh = np.load('cavity_re1000_data.npz'); gy, gu = gh['ghia_y'], gh['ghia_u']; o = np.argsort(GHIA_X)
    Uref = np.load('scratch/ghia_n15_w0.1_steady/final.npz')['U0']
    print(f'{"dt":34s} {"t":>7} {"rms u":>7} {"rms v":>7} {"max|du|":>8} {"zigzag mean|du|":>15} {"sign chg":>9} {"max|U-Uref|":>12} {"CG/step":>8}')
    for lab, d in RUNS:
        f = latest(d)
        if f is None: print(f'{lab:34s}  (no data yet)'); continue
        z = np.load(f); U = z['U0']; t = float(z['t'])
        y, u = centreline_u(U, m, N); x, v = centreline_v(U, m, N)
        ru = np.sqrt(np.mean((np.interp(gy, y, u) - gu)**2)); rv = np.sqrt(np.mean((np.interp(GHIA_X[o], x, v) - GHIA_V[o])**2))
        mu = np.max(np.abs(np.interp(gy, y, u) - gu))
        zm, zs, zn = zigzag(y, u)
        drift = np.abs(U[..., :2] - Uref[..., :2]).max()
        it = np.mean(z['cg_it'][-500:]) if len(z['cg_it']) else np.nan
        print(f'{lab:34s} {t:7.2f} {ru:7.4f} {rv:7.4f} {mu:8.4f} {zm:15.4f} {zs:5d}/{zn:<3d} {drift:12.2e} {it:8.0f}')
