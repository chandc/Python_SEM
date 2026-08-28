import os, sys, time
for v in ('OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ[v]='12'
sys.path.insert(0,'.'); sys.path.insert(0,'scratch')
sys.argv = ['x', '--consistent']
import numpy as np, importlib
sw = importlib.import_module('fs_minchan_sweep')
s = sw.build()
from lssem3d import project as PJ, convect as CV, timestep as T, helmholtz as HH, epmg
d = np.load('scratch/_minchan_fs/final_s20.npz')
U0, p0 = d['U'], d['p']
DT = 3.5e-4
pre0 = HH.fdm_preconditioner(s['m'], sw.N, T.implicit_coeff(DT,0)+sw.NU*(s['kz']**2),
                             sw.NU, s['mask_u'], 6, s['nk'], like=s['mask_u'])
dj = HH.jacobi_diagonal_analytic(s['m'], sw.N, s['m'].wq, s['kz']**2, 1.0, 2,
                                 s['nk'], mask=None)
ji = HH.jacobi_inverse(dj, s['mask_p'])
out = open('scratch/_minchan_fs/epmg_bench.txt', 'w')
def bench(name, mk):
    t0 = time.time(); Mp = mk(); su = time.time()-t0
    s['Mp'] = Mp; s['Mu'] = pre0
    Uc, pc = U0.copy(), p0.copy()
    Nk = -CV.convective(Uc, s['Dg'], s['fxg'], s['fyg'], s['kzg'], sw.NZ,
                        skew=True) + s['Fm']
    t0 = time.time()
    U1, pc, inf = PJ.substage(s, Uc, pc, Nk, np.zeros_like(Uc), 0, DT)
    dtb = time.time()-t0
    ok = np.isfinite(np.abs(U1).max())
    line = (f'{name:>15}: setup {su:5.1f}s  it_p {inf[2]:>4}  res {inf[3]:.1e}'
            f'  stage {dtb:6.1f}s  finite={ok}')
    print(line, flush=True); out.write(line+'\n'); out.flush()
bench('jacobi', lambda: (lambda r: r*ji))
bench('epmg-deg2', lambda: epmg.ConsistentPMG(s['m'], sw.N, s['kz'], s['nk'],
                                              sw.NZ, deg=2, like=s['mask_p']))
bench('epmg-deg6', lambda: epmg.ConsistentPMG(s['m'], sw.N, s['kz'], s['nk'],
                                              sw.NZ, deg=6, like=s['mask_p']))
out.close()
