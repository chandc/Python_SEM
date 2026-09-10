"""Richardson self-convergence of the cavity start-up at t = 1 (N = 15, 4x4,
from rest): ||U(dt) - U(dt/2)|| for successive dt, legacy vs balanced.
The observed order is log2 of the ratio of successive differences."""
import os, sys, glob
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, 'scratch')); os.chdir(_R)
os.environ.setdefault('CAV_RE', '1000')
import numpy as np
from lssem2d.mesh import build_channel

if __name__ == '__main__':
    N = 15; m = build_channel(1.0, 1.0, 4, 4, N, bcs=(1, 1, 1, 2)); m.compute_global_indices()
    w = m.wq[..., None]
    for tag, dts in (('balanced', (0.02, 0.01, 0.005, 0.0025, 0.00125)), ('legacy', (0.01, 0.005, 0.0025))):
        fields = {}
        for dt in dts:
            f = f'scratch/{os.environ.get("PREFIX", "rich")}_{"bal" if tag == "balanced" else "leg"}_dt{dt}/final.npz'
            if os.path.exists(f):
                z = np.load(f); fields[dt] = (z['U0'], float(z['t']))
        print(f'== {tag}: available dt = {sorted(fields)}')
        diffs = []
        for a, b in zip(dts[:-1], dts[1:]):
            if a in fields and b in fields:
                Ua, Ub = fields[a][0], fields[b][0]
                d = Ua - Ub
                eu = np.sqrt(np.sum(d[..., :2]**2*w)/np.sum(w*np.ones_like(d[..., :2])))
                em = np.abs(d[..., :2]).max()
                ep = np.sqrt(np.sum(d[..., 2]**2*m.wq)/np.sum(m.wq))
                diffs.append((a, b, eu, em, ep))
        print(f'   {"dt":>8} {"dt/2":>8} {"L2 |du|":>10} {"max |du|":>10} {"L2 |dp|":>10} {"order (L2 u)":>13} {"order (max)":>12}')
        for i, (a, b, eu, em, ep) in enumerate(diffs):
            o = f'{np.log2(diffs[i-1][2]/eu):13.2f}' if i > 0 else f'{"":>13}'
            om = f'{np.log2(diffs[i-1][3]/em):12.2f}' if i > 0 else f'{"":>12}'
            print(f'   {a:8g} {b:8g} {eu:10.3e} {em:10.3e} {ep:10.3e} {o} {om}')
