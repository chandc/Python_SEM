import os, sys, time
for v in ('OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ[v]='12'
sys.path.insert(0,'.'); sys.path.insert(0,'scratch')
sys.argv = ['x']
import numpy as np, importlib
sw = importlib.import_module('fs_minchan_sweep')
s = sw.build()
from lssem3d import project as PJ, convect as CV, timestep as T, helmholtz as HH
d = np.load('scratch/_minchan_fs/final_s20.npz')
DT = 3.5e-4
pre = [HH.fdm_preconditioner(s['m'], sw.N, T.implicit_coeff(DT,k)+sw.NU*(s['kz']**2),
       sw.NU, s['mask_u'], 6, s['nk'], like=s['mask_u']) for k in range(T.NSTAGE)]
def run5():
    Uc, pc = d['U'].copy(), d['p'].copy()
    s['ubc'] = None; Nprev = np.zeros_like(Uc)
    t0 = time.time(); its = 0
    for i in range(5):
        for k in range(T.NSTAGE):
            s['Mu'] = pre[k]
            Nk = -CV.convective(Uc, s['Dg'], s['fxg'], s['fyg'], s['kzg'],
                                sw.NZ, skew=True) + s['Fm']
            Uc, pc, inf = PJ.substage(s, Uc, pc, Nk, Nprev, k, DT)
            Nprev = Nk; its += inf[0] + inf[2]
    return Uc, (time.time()-t0)/5, its/5
U1, per, its = run5()
print(f'with freezing:  {per:.2f} s/step  CG {its:.0f}/step  '
      f'|U| {float(np.sqrt((np.abs(U1)**2).sum())):.8e}')
