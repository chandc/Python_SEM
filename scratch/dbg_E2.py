import os, sys, time
for v in ('OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ[v]='12'
sys.path.insert(0,'.'); sys.path.insert(0,'scratch')
sys.argv = ['x', '--consistent']
import numpy as np, importlib
sw = importlib.import_module('fs_minchan_sweep')
s = sw.build()
from lssem3d import project as PJ, epmg, helmholtz as HHx
import numpy as np
d = np.load('scratch/_minchan_fs/final_s20.npz')
Uc = d['U']
# the RHS the stage-0 pressure solve actually sees: use uhat ~ Uc directly
dtc = 3.5e-4*0.23125
dj = HHx.jacobi_diagonal_analytic(s['m'], sw.N, s['m'].wq, s['kz']**2, 1.0, 2,
                                  s['nk'], mask=None)
ji = HHx.jacobi_inverse(dj, s['mask_p'])
cfgs = [('jacobi', lambda: (lambda r: r*ji))]
for deg in (2, 3, 6):
    cfgs.append((f'epmg-deg{deg}', lambda deg=deg: epmg.ConsistentPMG(
        s['m'], sw.N, s['kz'], s['nk'], sw.NZ, deg=deg, like=s['mask_p'])))
for orders in ((8,6,4,2),):
    cfgs.append((f'epmg-4lvl-deg3', lambda orders=orders: epmg.ConsistentPMG(
        s['m'], sw.N, s['kz'], s['nk'], sw.NZ, orders=orders, deg=3,
        like=s['mask_p'])))
for name, mk in cfgs:
    t0 = time.time(); s['Mp'] = mk(); su = time.time()-t0
    t0 = time.time()
    u2, phi, it, res = PJ.project_consistent(s, Uc.copy(), dtc)
    dtb = time.time()-t0
    print(f'{name:>15}: setup {su:5.1f}s  CG {it:>4}  res {res:.1e}  '
          f'solve {dtb:6.1f}s', flush=True)
