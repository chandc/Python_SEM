import os, sys, time
for v in ('OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ[v]='12'
sys.path.insert(0,'.'); sys.path.insert(0,'scratch')
sys.argv = ['x', '--consistent']
import numpy as np, importlib
sw = importlib.import_module('fs_minchan_sweep')
s = sw.build()
from lssem3d import project as PJ, convect as CV, timestep as T, helmholtz as HH, epmg
s['Mp'] = epmg.ConsistentPMG(s['m'], sw.N, s['kz'], s['nk'], sw.NZ, deg=6,
                             like=s['mask_p'])
d = np.load('scratch/_minchan_fs/final_s20.npz')
Uc, pc, t = d['U'], d['p'], float(d['t'])
DT = 3.5e-4
pre = [HH.fdm_preconditioner(s['m'], sw.N, T.implicit_coeff(DT,k)+sw.NU*(s['kz']**2),
       sw.NU, s['mask_u'], 6, s['nk'], like=s['mask_u']) for k in range(T.NSTAGE)]
s['ubc'] = None
s['tol_p'] = float(os.environ.get('TOLP', '1e-5'))
print(f"tol_p = {s['tol_p']:g}", flush=True)
Nprev = np.zeros_like(Uc)
nrm = lambda a: float(np.sqrt((np.abs(a)**2).sum()))
w0 = time.time()
for i in range(10):
    tot = 0
    for k in range(T.NSTAGE):
        s['Mu'] = pre[k]
        Nk = -CV.convective(Uc, s['Dg'], s['fxg'], s['fyg'], s['kzg'], sw.NZ,
                            skew=True) + s['Fm']
        Uc, pc, inf = PJ.substage(s, Uc, pc, Nk, Nprev, k, DT)
        Nprev = Nk; tot += inf[0] + inf[2]
    t += DT
    if i % 5 == 4:
        sd = PJ.divergence(Uc, s['Dg'], s['fxg'], s['fyg'], s['kzg'])
        # weak divergence: what E actually controls
        from lssem3d import deriv as DV, solver3d as S3
        wd = S3.gs(s['m'], PJ._split(
            DV.ddxT(s['wq3']*Uc[...,0:1,:], s['Dg'], s['fxg'])
            + DV.ddyT(s['wq3']*Uc[...,1:2,:], s['Dg'], s['fyg'])
            - 1j*s['kzg']*(s['wq3']*Uc[...,2:3,:])))*s['mask_p']
        print(f'step {i+1}: t={t:.4f} |U|={nrm(Uc):.4e} CG={tot} '
              f'weakdiv={nrm(wd)/nrm(Uc):.1e} strongdiv={nrm(sd)/nrm(Uc):.1e} '
              f'[{time.time()-w0:.0f}s]', flush=True)
print(f's/step = {(time.time()-w0)/10:.2f}', flush=True)
