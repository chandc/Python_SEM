import sys; sys.path.insert(0,'.'); sys.path.insert(0,'scratch')
import numpy as np, lssem3d; lssem3d.set_backend('cupy')
import cupy as xp
from lssem3d import project as PJ, helmholtz as HH, hpmg
import fs_phase2 as F2
NU,NE,N,NZ = 1/800.,11,8,88
dt = 0.00567493
s = F2.build(N=N, ne=NE, nz=NZ, nu=NU, tol=1e-6, backend='cupy')
s['Mu'] = HH.fdm_preconditioner(s['m'],N,2.0/dt+NU*(s['kz']**2),NU,s['mask_u'],6,s['nk'],like=s['mask_u'])
s['Mp'] = hpmg.HelmholtzPMG(s['m'],N,s['kz']**2,1.0,1,s['nk'],NZ,wall=False,pin_kz0=True,deg=6,like=s['mask_p'])
nrm = lambda a: float(xp.sqrt((xp.abs(a)**2).sum()))
Uc = F2.ic_tgv(s)
pc  = xp.zeros((s['m'].nelem,N+1,N+1,1,s['nk']),dtype=complex)
phi = xp.zeros((s['m'].nelem,N+1,N+1,1,s['nk']),dtype=complex)
print(f'{"step":>4} {"|pc|":>12} {"|phi|":>12} {"|grad pc|":>12} {"|div u|":>12} {"|dt grad phi|":>13} {"E":>13}')
for i in range(1,21):
    Uc, phi, inf, pc = PJ.step_kim_moin(s,Uc,phi,dt,pc=pc,skew=True)
    if i in (1,2,3,5,10,20):
        g  = PJ.gradient(pc, s['Dg'], s['fxg'], s['fyg'], s['kzg'])
        dv = PJ.divergence(Uc, s['Dg'], s['fxg'], s['fyg'], s['kzg'])
        gp = PJ.gradient(phi, s['Dg'], s['fxg'], s['fyg'], s['kzg'])
        E,Om = F2.diagnostics(s,Uc)
        print(f'{i:>4} {nrm(pc):>12.4e} {nrm(phi):>12.4e} {nrm(g):>12.4e} '
              f'{nrm(dv):>12.4e} {dt*nrm(gp):>13.4e} {E:>13.6f}')
