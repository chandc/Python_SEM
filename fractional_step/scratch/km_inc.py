"""Does the incremental pressure term change the SOLUTION at all?"""
import sys; sys.path.insert(0,'.'); sys.path.insert(0,'scratch')
import numpy as np, lssem3d; lssem3d.set_backend('cupy')
import cupy as xp
from lssem3d import project as PJ, helmholtz as HH, hpmg
import fs_phase2 as F2
NU,NE,N,NZ = 1/800.,11,8,88; dt = 0.00567493
s = F2.build(N=N, ne=NE, nz=NZ, nu=NU, tol=1e-8, backend='cupy')
s['Mu'] = HH.fdm_preconditioner(s['m'],N,2.0/dt+NU*(s['kz']**2),NU,s['mask_u'],6,s['nk'],like=s['mask_u'])
s['Mp'] = hpmg.HelmholtzPMG(s['m'],N,s['kz']**2,1.0,1,s['nk'],NZ,wall=False,pin_kz0=True,deg=6,like=s['mask_p'])
D,fx,fy,kz = s['Dg'],s['fxg'],s['fyg'],s['kzg']
nrm=lambda a: float(xp.sqrt((xp.abs(a)**2).sum()))
res={}
for tag, inc in (('free',False),('inc',True)):
    Uc = F2.ic_tgv(s)
    pc  = xp.zeros((s['m'].nelem,N+1,N+1,1,s['nk']),dtype=complex) if inc else None
    phi = xp.zeros((s['m'].nelem,N+1,N+1,1,s['nk']),dtype=complex)
    tr=[]
    for i in range(1,41):
        Uc, phi, inf, pc = PJ.step_kim_moin(s,Uc,phi,dt,pc=pc,skew=True)
        if inc and i in (1,5,10,20,40):
            tr.append(f'{i}:|pc|={nrm(pc):.3e},|phi|={nrm(phi):.2e}')
    res[tag]=Uc
    if tr: print('  pc trajectory (c=2): '+'  '.join(tr), flush=True)
print(f'|u_inc - u_free| / |u| after 40 steps = '
      f'{nrm(res["inc"]-res["free"])/nrm(res["free"]):.4e}', flush=True)
# what does grad(pc) weigh against the rest of the RHS?
Uc = F2.ic_tgv(s)
up = Uc
print(f'|wq3*(2/dt)*up| = {nrm(s["wq3"]*(2.0/dt)*up):.4e}', flush=True)
