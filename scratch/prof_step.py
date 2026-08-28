import os, sys
for v in ('OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ[v]='12'
sys.path.insert(0,'.'); sys.path.insert(0,'scratch')
sys.argv = ['x']
import numpy as np, importlib, cProfile, pstats, io
sw = importlib.import_module('fs_minchan_sweep')
s = sw.build()
from lssem3d import project as PJ, convect as CV, timestep as T, helmholtz as HH
d = np.load('scratch/_minchan_fs/final_s20.npz')
Uc, pc = d['U'], d['p']
DT = 3.5e-4
pre = [HH.fdm_preconditioner(s['m'], sw.N, T.implicit_coeff(DT,k)+sw.NU*(s['kz']**2),
       sw.NU, s['mask_u'], 6, s['nk'], like=s['mask_u']) for k in range(T.NSTAGE)]
s['ubc'] = None
Nprev = np.zeros_like(Uc)
def step():
    global Uc, pc, Nprev
    for k in range(T.NSTAGE):
        s['Mu'] = pre[k]
        Nk = -CV.convective(Uc, s['Dg'], s['fxg'], s['fyg'], s['kzg'], sw.NZ,
                            skew=True) + s['Fm']
        u, p2, _ = PJ.substage(s, Uc, pc, Nk, Nprev, k, DT)
        Uc, pc, Nprev = u, p2, Nk
step()   # warm
pr = cProfile.Profile(); pr.enable()
for _ in range(3): step()
pr.disable()
st = io.StringIO()
ps = pstats.Stats(pr, stream=st).sort_stats('cumulative')
ps.print_stats(18)
print('\n'.join(l for l in st.getvalue().splitlines()
                if 'lssem' in l or 'einsum' in l or 'fft' in l or 'ncalls' in l
                or 'tottime' in l or l.strip().startswith(('3 ', '9 '))))
st2 = io.StringIO()
ps2 = pstats.Stats(pr, stream=st2).sort_stats('tottime')
ps2.print_stats(14)
print("==== by tottime ====")
print('\n'.join(st2.getvalue().splitlines()[4:22]))
