"""BFS Chan Re=389, long domain: the PURE steady form (w_mass = 0, w_mom = 1).

The momentum row becomes exactly N(U) with no time-derivative term, so each
step_bdf call is one Newton iteration on the steady problem -- no time-stepping
damping at all.  Poiseuille handled this easily, but the BFS is a far harsher
test: the convective term does NOT vanish, the step corner is singular, and the
free outflow carries modes ~8300x softer than generic.

Run at both linear-solve tolerances, because the pure steady form on Poiseuille
turned out to be entirely limited by the default absolute floor tol=1e-6.

Reference (legacy dt=0.5, same grid/IC/pin):
    x_r/h = 8.190,  Qout/Qin = 0.9925,  rms div = 4.83e-02,  exit p spread 0.247
"""
import os, sys, time
SC=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,'/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
sys.path.insert(0,SC)
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
from fgrid import load
from lssem2d.lgl import diff_matrix, lgl_weights
from lssem2d.lssem import SolverState, ls_coeffs
from lssem2d.operators import dUdx, dUdy
import lssem2d.solver as S
from lssem2d import precond as P
RE,H,LB = 389.0,0.5,1.0
GRID='/Users/danielchan/Dropbox/F90_SEM/pmg_clean/cnos_long_grid.dat'
_p=S.pcg_solve

def build():
    m,_,_=load(GRID); n=m.N+1
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
                    U[e,i,j,0]=(1.0-sv)*us(y)+sv*ud(y)
                    U[e,i,j,1]=-sp*G(y)
                    U[e,i,j,3]=-spp*G(y)-((1.0-sv)*dus(y)+sv*dud(y))
    return U

def run(kind,cgsfac,tol,cap=120):
    m,n,pin=build(); N=m.N
    st=SolverState(m,diff_matrix(N),nu=1.0/RE,dt=0.5,fac1=1.0,w_mom=1.0,w_mass=0.0)
    inlet=lambda x,y,t:6.0*((y-0.5)/0.5)*(1.0-(y-0.5)/0.5)
    D=diff_matrix(N); w=lgl_weights(N)
    INL=[e for e in range(m.nelem) if m.bc[e,0]==3]
    OUT=[e for e in range(m.nelem) if abs(m.xnod[e,-1]-m.xnod.max())<1e-9]
    fl=lambda U,e,i: np.sum(w*U[e,i,:,0])*(m.hy[e]/2)
    nit=[0]
    def pcg(state,b,fu,fv,M,mw,pin_p=False,max_iter=5000,t=1e-6,cf=0.0,precond=None,**kw):
        pre=P.make('pmg2',state,fu,fv,M,pin_p,pc=max(2,N//2),deg=4,coarse_deg=10) if kind=='pmg' else None
        x,it=_p(state,b,fu,fv,M,mw,pin_p=pin_p,max_iter=300000,tol=tol,cgsfac=cgsfac,precond=pre)
        nit[0]+=it; return x,it
    S.pcg_solve=pcg
    U=devc(m,n); hist=[U]; t0=time.perf_counter(); status='cap'; tr=[]
    try:
        for s in range(cap):
            Up=hist[0].copy()
            U=S.step_bdf(st,hist,time=0.0,max_newton=1,newton_tol=1e-14,newton_factor=0.0,
                         custom_inlet=inlet,pin_p=pin,cgsfac=cgsfac,cg_max_iter=300000,
                         verbose=False)
            if not np.all(np.isfinite(U)): status='NaN'; break
            um=np.abs(U[...,0]).max(); du=float(np.max(np.abs(U-Up)))
            tr.append((s+1,du,um))
            if um>20.0: status=f'diverged max|u|={um:.1f}'; break
            if s>2 and du<1e-11: status='converged'; break
    finally:
        S.pcg_solve=_p
    wall=time.perf_counter()-t0
    out=dict(kind=kind,cgsfac=cgsfac,tol=tol,status=status,it=s+1,cg=nit[0],wall=wall,tr=tr)
    if status not in ('converged','cap'): return out
    ux=dUdx(np.ascontiguousarray(U[...,0]),D,m.facx); vy=dUdy(np.ascontiguousarray(U[...,1]),D,m.facy)
    xs,tw=[],[]
    for e in range(m.nelem):
        if m.ynod[e,0]>0.01 or m.xnod[e,0]<-1e-9: continue
        for i in range(n):
            xs.append(m.xnod[e,i]); tw.append(np.dot(D[0,:],U[e,i,:,0])*(2.0/m.hy[e]))
    o=np.argsort(xs); xs,tw=np.array(xs)[o],np.array(tw)[o]; xr=np.nan
    for k in range(len(xs)-1):
        if tw[k]<0 and tw[k+1]>0 and xs[k]>0.05:
            xr=xs[k]-tw[k]*(xs[k+1]-xs[k])/(tw[k+1]-tw[k]); break
    ue=np.array([U[e,-1,j,0] for e in OUT for j in range(n)])
    pe=np.array([U[e,-1,j,2] for e in OUT for j in range(n)])
    np.savez_compressed(f'{SC}/bfs_steadyMG_{kind}_{tol:g}.npz',U=U,xnod=m.xnod,ynod=m.ynod,hy=m.hy)
    out.update(q=float(sum(fl(U,e,-1) for e in OUT)/sum(fl(U,e,0) for e in INL)),
               div=float(np.sqrt(((ux+vy)**2).mean())), umax=float(np.abs(U[...,0]).max()),
               xr=float(xr/H), pspread=float(pe.max()-pe.min()), rev=float(100*np.mean(ue<0)))
    return out

print("BFS Chan Re=389, LONG domain, PURE STEADY form (w_mass=0, w_mom=1)")
print("each iteration is one Newton step -- no time-stepping damping\n")
print("reference legacy dt=0.5:  x_r/h 8.190   Q 0.9925   div 4.83e-02   p_sprd 0.247\n")
print(f"{'precond':>8}{'cgsfac':>9}{'tol':>8}{'iters':>7}{'CG tot':>9}{'status':>22}"
      f"{'Qout/Qin':>10}{'rms div':>11}{'max|u|':>8}{'x_r/h':>8}{'p_sprd':>9}{'rev':>7}{'wall':>7}")
for kind,cf,tl in (('pmg',1e-8,1e-10),('pmg',1e-3,1e-6)):
    r=run(kind,cf,tl)
    if 'q' not in r:
        print(f"{r['kind']:>8}{cf:>9.0e}{tl:>8.0e}{r['it']:>7}{r['cg']:>9}{r['status']:>22}"
              f"{'':>53}{r['wall']:>7.0f}")
        print(f"         trace (iter, max|dU|, max|u|): "
              +", ".join(f"({a},{b:.1e},{c:.2f})" for a,b,c in r['tr'][:8]))
    else:
        print(f"{r['kind']:>8}{cf:>9.0e}{tl:>8.0e}{r['it']:>7}{r['cg']:>9}{r['status']:>22}"
              f"{r['q']:>10.4f}{r['div']:>11.3e}{r['umax']:>8.3f}{r['xr']:>8.3f}"
              f"{r['pspread']:>9.3f}{r['rev']:>6.1f}%{r['wall']:>7.0f}")
    sys.stdout.flush()
