"""Does skew / incremental-p actually change the Kim-Moin step? 20 steps each."""
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
import inspect
print('signature:', inspect.signature(PJ.step_kim_moin))
print('skew in source:', 'skew=skew' in inspect.getsource(PJ.step_kim_moin))
for skew in (False, True):
    for inc in (False, True):
        Uc = F2.ic_tgv(s)
        pc  = xp.zeros((s['m'].nelem,N+1,N+1,1,s['nk']),dtype=complex) if inc else None
        phi = xp.zeros((s['m'].nelem,N+1,N+1,1,s['nk']),dtype=complex)
        E0,Om0 = F2.diagnostics(s,Uc); prevE=E0
        for i in range(20):
            Uc, phi, inf, pc = PJ.step_kim_moin(s,Uc,phi,dt,pc=pc,skew=skew)
        E,Om = F2.diagnostics(s,Uc)
        bal = (-(E-prevE)/(20*dt))/(2*NU*0.5*(Om+Om0))
        print(f'skew={skew!s:5} incremental={inc!s:5}  E={E:.9f}  Om={Om:.6f}  bal={bal:.4f}')
