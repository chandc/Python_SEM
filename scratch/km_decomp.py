"""Isolate the 23.6% excess dissipation: viscous alone, convection alone, full."""
import sys; sys.path.insert(0,'.'); sys.path.insert(0,'scratch')
import numpy as np, lssem3d; lssem3d.set_backend('cupy')
import cupy as xp
from lssem3d import project as PJ, helmholtz as HH, hpmg, convect as CV
from lssem3d import solver3d as S3
import fs_phase2 as F2
NU,NE,N,NZ = 1/800.,11,8,88
dt = 0.00567493; NSTEP = 20
s = F2.build(N=N, ne=NE, nz=NZ, nu=NU, tol=1e-10, backend='cupy')
lam_h = 2.0/dt + NU*(s['kz']**2)      # host, for the FDM build
s['Mu'] = HH.fdm_preconditioner(s['m'],N,lam_h,NU,s['mask_u'],6,s['nk'],like=s['mask_u'])
s['Mp'] = hpmg.HelmholtzPMG(s['m'],N,s['kz']**2,1.0,1,s['nk'],NZ,wall=False,pin_kz0=True,deg=6,like=s['mask_p'])
D,fx,fy,wq,kz = s['Dg'],s['fxg'],s['fyg'],s['wqg'],s['kzg']
lam = 2.0/dt + NU*(kz**2)             # device, for the solves

# --- A: PURE DIFFUSION.  TGV IC has |k|^2 = 3, so E(t) = E0 exp(-2*nu*3*t) exactly
Uc = F2.ic_tgv(s); E0,_ = F2.diagnostics(s,Uc)
for i in range(NSTEP):
    r = s['wq3']*((2.0/dt)*Uc) - PJ.visc_weak(Uc,D,fx,fy,wq,kz,NU)
    b = (S3.gs(s['m'], PJ._split(r)))*s['mask_u']
    u,_,_ = HH.solve(b,D,fx,fy,wq,lam,NU,s['m'],s['mask_u'],s['Mu'],tol=1e-12)
    Uc = PJ._join(u)
E,_ = F2.diagnostics(s,Uc)
exact = E0*np.exp(-2*NU*3*NSTEP*dt)
print(f'A  diffusion only : E={E:.9f}  exact={exact:.9f}  rel err={(E-exact)/exact:+.3e}')

# --- B: CONVECTION ONLY, Jameson RK4, skew.  Energy should be conserved.
for skew in (True, False):
    Uc = F2.ic_tgv(s); E0,_ = F2.diagnostics(s,Uc)
    for i in range(NSTEP):
        un = Uc; u = un
        for a in PJ.JAMESON:
            H = -CV.convective(u,D,fx,fy,kz,s['nz'],skew=skew)
            u = un + (dt*a)*H
        Uc = u
    E,_ = F2.diagnostics(s,Uc)
    print(f'B  convection only (skew={skew!s:5}): dE/E0 = {(E-E0)/E0:+.3e}')

# --- C: CONVECTION + PROJECTION, no viscosity
Uc = F2.ic_tgv(s); E0,_ = F2.diagnostics(s,Uc)
phi = xp.zeros((s['m'].nelem,N+1,N+1,1,s['nk']),dtype=complex)
for i in range(NSTEP):
    un = Uc; u = un
    for a in PJ.JAMESON:
        H = -CV.convective(u,D,fx,fy,kz,s['nz'],skew=True)
        u = un + (dt*a)*H
    div = PJ.divergence(u,D,fx,fy,kz)
    bp = -S3.gs(s['m'], s['wq1']*PJ._split(div/dt))*s['mask_p']
    v = s.get('null_kz0')
    if v is not None:
        num = float((bp[...,0:1,0:1]*v*s['mw1']).sum()); bp = bp.copy()
        bp[...,0:1,0:1] -= (num/s['null_norm'])*v
    ph,_,_ = HH.solve(bp,D,fx,fy,wq,kz**2,1.0,s['m'],s['mask_p'],s['Mp'],tol=1e-10)
    phi = PJ._join(ph)
    Uc = u - dt*PJ.gradient(phi,D,fx,fy,kz)
E,_ = F2.diagnostics(s,Uc)
print(f'C  convection + projection, no viscosity: dE/E0 = {(E-E0)/E0:+.3e}'
      f'   (viscous loss over the same span would be {-(1-np.exp(-2*NU*3*NSTEP*dt)):+.3e})')
