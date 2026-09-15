"""Is the 23.6% a bug or an O(dt) splitting error?  Halve dt, hold t fixed."""
import sys; sys.path.insert(0,'.'); sys.path.insert(0,'scratch')
import numpy as np, lssem3d; lssem3d.set_backend('cupy')
import cupy as xp
from lssem3d import project as PJ, helmholtz as HH, hpmg
import fs_phase2 as F2
NU,NE,N,NZ = 1/800.,11,8,88
s = F2.build(N=N, ne=NE, nz=NZ, nu=NU, tol=1e-9, backend='cupy')
s['Mp'] = hpmg.HelmholtzPMG(s['m'],N,s['kz']**2,1.0,1,s['nk'],NZ,wall=False,pin_kz0=True,deg=6,like=s['mask_p'])
TEND = 0.5675
print(f'{"dt":>10} {"nstep":>6} {"balance":>9} {"excess":>10} {"ratio":>7}   (pressure-free / incremental)')
for inc in (False, True):
    prev=None
    for f in (1,2,4):
        dt = 0.00567493/f; n = int(round(TEND/dt))
        s['Mu'] = HH.fdm_preconditioner(s['m'],N,2.0/dt+NU*(s['kz']**2),NU,s['mask_u'],6,s['nk'],like=s['mask_u'])
        Uc = F2.ic_tgv(s)
        pc  = xp.zeros((s['m'].nelem,N+1,N+1,1,s['nk']),dtype=complex) if inc else None
        phi = xp.zeros((s['m'].nelem,N+1,N+1,1,s['nk']),dtype=complex)
        E,Om = F2.diagnostics(s,Uc)
        for i in range(n):
            pE,pOm = E,Om
            Uc, phi, _, pc = PJ.step_kim_moin(s,Uc,phi,dt,pc=pc,skew=True)
            E,Om = F2.diagnostics(s,Uc)
        bal=(-(E-pE)/dt)/(2*NU*0.5*(Om+pOm)); ex=bal-1.0
        r = f'{prev/ex:6.2f}x' if prev else '   --'
        print(f'{dt:>10.6f} {n:>6} {bal:>9.4f} {ex:>10.5f} {r:>7}   inc={inc}', flush=True)
        prev=ex
