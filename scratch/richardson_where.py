"""Where do the dt-differences of the cavity transient live?  Richardson
differences split by region (corner boxes vs the rest) and along the centrelines."""
import os, sys
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, 'scratch')); os.chdir(_R)
os.environ.setdefault('CAV_RE', '1000')
import numpy as np
from lssem2d.mesh import build_channel
from pmg_ghia_cavity import centreline_u, centreline_v

if __name__ == '__main__':
    N = 15; m = build_channel(1.0, 1.0, 4, 4, N, bcs=(1, 1, 1, 2)); m.compute_global_indices()
    X = np.broadcast_to(m.xnod[:, :, None], (m.nelem, N+1, N+1)); Y = np.broadcast_to(m.ynod[:, None, :], (m.nelem, N+1, N+1))
    corner = ((X < 0.1) | (X > 0.9)) & (Y > 0.9)          # the two lid corners
    lid = Y > 0.9
    prefix = os.environ.get('PREFIX', 'rich2')
    for tag, dts in (('balanced', (0.02, 0.01, 0.005, 0.0025, 0.00125)), ('legacy', (0.01, 0.005, 0.0025))):
        F = {}
        for dt in dts:
            f = f'scratch/{prefix}_{"bal" if tag == "balanced" else "leg"}_dt{dt}/final.npz'
            if os.path.exists(f): F[dt] = np.load(f)['U0']
        print(f'== {tag}')
        print(f'   {"dt":>7} {"dt/2":>7} | {"L2 u all":>9} {"L2 u no-corner":>14} {"L2 u y<0.9":>10} | {"max u":>8} {"at (x,y)":>14} | {"centreline rms du":>17} {"dv":>8}')
        prev = None
        for a, b in zip(dts[:-1], dts[1:]):
            if a not in F or b not in F: continue
            d = F[a] - F[b]; du = np.sqrt(d[..., 0]**2 + d[..., 1]**2)
            def l2(msk): return np.sqrt(np.sum(du[msk]**2*m.wq[msk])/np.sum(m.wq[msk]))
            i = np.unravel_index(du.argmax(), du.shape)
            ya, ua = centreline_u(F[a], m, N); yb, ub = centreline_u(F[b], m, N)
            xa, va = centreline_v(F[a], m, N); xb, vb = centreline_v(F[b], m, N)
            row = (l2(np.ones_like(corner)), l2(~corner), l2(~lid), du.max(), np.sqrt(np.mean((ua-ub)**2)), np.sqrt(np.mean((va-vb)**2)))
            ordr = '' if prev is None else '  orders: ' + ' '.join(f'{np.log2(p/r):5.2f}' for p, r in zip(prev, row))
            print(f'   {a:7g} {b:7g} | {row[0]:9.2e} {row[1]:14.2e} {row[2]:10.2e} | {row[3]:8.2e} ({X[i]:.3f},{Y[i]:.3f}) | {row[4]:17.2e} {row[5]:8.2e}{ordr}')
            prev = row
