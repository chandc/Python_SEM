import sys; sys.path.insert(0,'.'); sys.path.insert(0,'scratch')
import numpy as np, lssem3d; lssem3d.set_backend('cupy')
import cupy as xp
from lssem3d import project as PJ, helmholtz as HH, hpmg
import fs_phase2 as F2
NU,NE,N,NZ = 1/800.,11,8,88
dt = 0.00567493
s = F2.build(N=N, ne=NE, nz=NZ, nu=NU, tol=1e-8, backend='cupy')
s['Mu'] = HH.fdm_preconditioner(s['m'],N,2.0/dt+NU*(s['kz']**2),NU,s['mask_u'],6,s['nk'],like=s['mask_u'])
s['Mp'] = hpmg.HelmholtzPMG(s['m'],N,s['kz']**2,1.0,1,s['nk'],NZ,wall=False,pin_kz0=True,deg=6,like=s['mask_p'])
D,fx,fy,kz = s['Dg'],s['fxg'],s['fyg'],s['kzg']
nrm=lambda a: float(xp.sqrt((xp.abs(a)**2).sum()))
for tag, inc in (('pressure-free  ',False), ('incremental x2 ',True)):
    Uc = F2.ic_tgv(s)
    pc  = xp.zeros((s['m'].nelem,N+1,N+1,1,s['nk']),dtype=complex) if inc else None
    phi = xp.zeros((s['m'].nelem,N+1,N+1,1,s['nk']),dtype=complex)
    E,Om = F2.diagnostics(s,Uc); out=[]
    for i in range(1,301):
        pE,pOm = E,Om
        Uc, phi, inf, pc = PJ.step_kim_moin(s,Uc,phi,dt,pc=pc,skew=True)
        E,Om = F2.diagnostics(s,Uc)
        if not np.isfinite(E): out.append(f'step{i}: BLEW UP'); break
        if i in (10,50,100,200,300):
            bal=(-(E-pE)/dt)/(2*NU*0.5*(Om+pOm))
            out.append(f'step{i}: bal={bal:.4f} |div|/|u|={nrm(PJ.divergence(Uc,D,fx,fy,kz))/nrm(Uc):.2e}')
    print(tag+'  '+'  '.join(out), flush=True)
