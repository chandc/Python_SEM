import os, sys
for v in ('OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ[v]='12'
sys.path.insert(0,'.'); sys.path.insert(0,'scratch')
import numpy as np
sys.argv = ['x', '--consistent']
import importlib
sw = importlib.import_module('fs_minchan_sweep')
s = sw.build()
# E-multigrid
from lssem3d import epmg
import time as _t; _t0=_t.time()
s['Mp'] = epmg.ConsistentPMG(s['m'], sw.N, s['kz'], s['nk'], sw.NZ,
                             like=s['mask_p'])
print(f'E-PMG setup {_t.time()-_t0:.1f}s', flush=True)
from lssem3d import project as PJ, convect as CV, timestep as T, helmholtz as HH
import numpy as np
d = np.load('scratch/_minchan_fs/final_s20.npz')
Uc, pc = d['U'], d['p']
DT = 3.5e-4
pre = [HH.fdm_preconditioner(s['m'], sw.N, T.implicit_coeff(DT,k)+sw.NU*(s['kz']**2),
       sw.NU, s['mask_u'], 6, s['nk'], like=s['mask_u']) for k in range(T.NSTAGE)]
nrm = lambda a: float(np.sqrt((np.abs(a)**2).sum()))
print(f'|U0|={nrm(Uc):.4e}  |pc0|={nrm(pc):.4e}')
Nprev = np.zeros_like(Uc)
for k in range(1):
    s['Mu'] = pre[k]
    Nk = -CV.convective(Uc, s['Dg'], s['fxg'], s['fyg'], s['kzg'], sw.NZ,
                        skew=True) + s['Fm']
    U1, pc, inf = PJ.substage(s, Uc, pc, Nk, Nprev, k, DT)
    print(f'stage {k}: it_u={inf[0]} res_u={inf[1]:.1e} it_p={inf[2]} '
          f'res_p={inf[3]:.1e}  |U|={nrm(U1):.4e}  |pc|={nrm(pc):.4e}')
    Uc = U1; Nprev = Nk
