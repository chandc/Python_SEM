"""Why did the line search stop the loose p-MG BFS case from converging?
Same configuration both ways; print the accepted alpha and the step size."""
import os,sys,time
SC=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,'/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo'); sys.path.insert(0,SC)
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
from fgrid import load
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState
import lssem2d.solver as S
from lssem2d import precond as P
_p=S.pcg_solve; LB=1.0
def build():
    m,_,_=load('/Users/danielchan/Dropbox/F90_SEM/pmg_clean/cnos_long_grid.dat'); n=m.N+1
    pin=next((e,n-1,0) for e in range(m.nelem) if m.bc[e,1]==4 and m.bc[e,2]==1)
    for e in range(m.nelem):
        if m.bc[e,1]==4: m.bc[e,1]=0
    return m,n,pin
def devc(m,n):
    U=np.zeros((m.nelem,n,n,4))
    ud=lambda y:3.0*y*(1.0-y); dud=lambda y:3.0-6.0*y
    def us(y):
        if y<=0.5: return 0.0
        e=2.0*y-1.0; return 6.0*e*(1.0-e)
    def dus(y):
        if y<=0.5: return 0.0
        e=2.0*y-1.0; return 12.0*(1.0-2.0*e)
    def G(y):
        I=1.5*y*y-y**3
        if y>0.5:
            e=2.0*y-1.0; I-=0.5*(3.0*e*e-2.0*e**3)
        return I
    for e in range(m.nelem):
        for i in range(n):
            x=m.xnod[e,i]
            if x<=0.0: t=sp=spp=0.0
            else:
                t=min(x/LB,1.0); sp=(6.0*t-6.0*t*t)/LB; spp=(6.0-12.0*t)/LB**2
                if x>=LB: sp=spp=0.0
            sv=3.0*t*t-2.0*t**3
            for j in range(n):
                y=m.ynod[e,j]
                if x<0.0:
                    eta=(y-0.5)/0.5
                    U[e,i,j,0]=6.0*eta*(1.0-eta); U[e,i,j,3]=-12.0*(1.0-2.0*eta)
                else:
                    U[e,i,j,0]=(1.0-sv)*us(y)+sv*ud(y); U[e,i,j,1]=-sp*G(y)
                    U[e,i,j,3]=-spp*G(y)-((1.0-sv)*dus(y)+sv*dud(y))
    return U
def go(ls,nit=30,mem=10,cgsfac=1e-3,tol=1e-6):
    m,n,pin=build(); N=m.N
    st=SolverState(m,diff_matrix(N),nu=1/389.,dt=0.5,fac1=1.0,w_mom=1.0,w_mass=0.0)
    inlet=lambda x,y,t:6.0*((y-0.5)/0.5)*(1.0-(y-0.5)/0.5)
    def pcg(state,b,fu,fv,M,mw,pin_p=False,max_iter=5000,t=1e-6,cf=0.0,precond=None,**kw):
        pre=P.make('pmg2',state,fu,fv,M,pin_p,pc=max(2,N//2),deg=4,coarse_deg=10)
        return _p(state,b,fu,fv,M,mw,pin_p=pin_p,max_iter=300000,tol=tol,cgsfac=cgsfac,precond=pre)
    S.pcg_solve=pcg
    U=devc(m,n); hist=[U]; rows=[]
    try:
        for s in range(nit):
            Up=hist[0].copy()
            U=S.step_bdf(st,hist,time=0.0,max_newton=1,newton_tol=1e-14,newton_factor=0.0,
                         custom_inlet=inlet,pin_p=pin,cgsfac=cgsfac,cg_max_iter=300000,
                         verbose=False,line_search=ls,ls_memory=mem)
            um=float(np.abs(U[...,0]).max())
            rows.append((s+1,getattr(st,'_last_alpha',1.0),float(np.max(np.abs(U-Up))),um))
            if not np.isfinite(um) or um>50: rows.append(('DIVERGED',0,0,um)); break
            if s>2 and rows[-1][2]<1e-11: break
    finally:
        S.pcg_solve=_p
    return rows

def summary(tag,rows):
    last=rows[-1]
    if last[0]=='DIVERGED': return f"{tag:<34} DIVERGED (max|u|={last[3]:.1f}) at iter {len(rows)-1}"
    conv = last[2]<1e-11
    return (f"{tag:<34} {'converged' if conv else 'CAP':<10} {len(rows):>3} iters   "
            f"final |dU| {last[2]:.2e}   max|u| {last[3]:.3f}   "
            f"min alpha {min(r[1] for r in rows if r[0]!='DIVERGED'):g}")

print("BFS steady form, p-MG.  Reference: no line search, loose tol -> 11 iterations.\n")
print("--- LOOSE tolerance (cgsfac=1e-3, tol=1e-6) ---")
print("  "+summary("no line search",            go(False)))
print("  "+summary("monotone LS   (ls_memory=1)", go(True,mem=1)))
print("  "+summary("NON-MONOTONE  (ls_memory=10)",go(True,mem=10)))
print("\n--- TIGHT tolerance (cgsfac=1e-8, tol=1e-10) : diverged without LS ---")
print("  "+summary("no line search",            go(False,nit=12,cgsfac=1e-8,tol=1e-10)))
print("  "+summary("NON-MONOTONE  (ls_memory=10)",go(True,nit=30,mem=10,cgsfac=1e-8,tol=1e-10)))
