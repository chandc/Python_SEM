"""Does the headline Poiseuille dt sensitivity survive a TIGHT linear solve?

The 1875x spread and the 98% error at dt=0.05 were all measured with the default
absolute CG tolerance tol=1e-6.  The pure steady form's apparent weight
sensitivity turned out to be entirely a solver artifact, so the time-stepping
result has to be re-checked the same way: tol=1e-10, cgsfac=1e-8, and run to a
rate-based steady state with a generous cap.
"""
import os, sys, time
SC=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,'/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix, lgl_weights
from lssem2d.lssem import SolverState
import lssem2d.solver as S
NU=0.01; N,EX,EY=8,10,2; DP=1.2
u_ex=lambda y:6.0*y*(1.0-y); _p=S.pcg_solve
def run(dt,cgsfac,tol,tmin=300.,cap=9000):
    m=build_channel(10.,1.,EX,EY,N,bcs=(3,4,1,1)); n=N+1
    pin=next((e,0,0) for e in range(m.nelem) if m.bc[e,0]==3 and m.bc[e,2]==1)
    for e in range(m.nelem):
        if m.bc[e,1]==4: m.bc[e,1]=0
    st=SolverState(m,diff_matrix(N),nu=NU,dt=dt,fac1=1.0)
    inlet=lambda x,y,t:u_ex(y)
    def pcg(state,b,fu,fv,M,mw,pin_p=False,max_iter=5000,t=1e-6,cf=0.0,precond=None,**kw):
        return _p(state,b,fu,fv,M,mw,pin_p=pin_p,max_iter=300000,tol=tol,cgsfac=cgsfac)
    S.pcg_solve=pcg
    U=np.zeros((m.nelem,n,n,4)); hist=[U]; t0=time.perf_counter(); conv=False
    try:
        for s in range(cap):
            Up=hist[0].copy()
            U=S.step_bdf(st,hist,time=s*dt,max_newton=1,newton_tol=1e-14,newton_factor=0.0,
                         custom_inlet=inlet,pin_p=pin,cgsfac=cgsfac,cg_max_iter=300000,verbose=False)
            if not np.all(np.isfinite(U)): return None
            if (s+1)*dt>=tmin and np.max(np.abs(U-Up))/dt<1e-9: conv=True; break
    finally:
        S.pcg_solve=_p
    w=lgl_weights(N); xn,yn,hy=m.xnod,m.ynod,m.hy; xmax=xn.max()
    ys,ps,us=[],[],[]
    for e in range(m.nelem):
        if abs(xn[e,-1]-xmax)<1e-9:
            for j in range(n): ys.append(yn[e,j]); ps.append(U[e,-1,j,2]); us.append(U[e,-1,j,0])
    o=np.argsort(ys); ys,ps,us=np.array(ys)[o],np.array(ps)[o],np.array(us)[o]
    k=np.concatenate(([True],np.diff(ys)>1e-12)); ys,ps,us=ys[k],ps[k],us[k]
    def pbar(edge):
        tot=a=0.0
        for e in range(m.nelem):
            xe=xn[e,0] if edge=='in' else xn[e,-1]; ref=xn.min() if edge=='in' else xmax
            if abs(xe-ref)<1e-9:
                i=0 if edge=='in' else -1
                tot+=np.sum(w*U[e,i,:,2])*(hy[e]/2); a+=hy[e]
        return tot/a
    return dict(steps=s+1,conv=conv,prof=float(np.sqrt(np.mean((us-u_ex(ys))**2))/1.5),
                dp=float(pbar('in')-pbar('out')),spread=float(ps.max()-ps.min()),
                wall=time.perf_counter()-t0)
print("Legacy time-stepping (mass term present), run to steady state\n")
print(f"{'dt':>6} | {'tol=1e-6 cgsfac=1e-3 (as published)':^40} | {'tol=1e-10 cgsfac=1e-8':^40}")
print(f"{'':>6} | {'steps':>7}{'conv':>6}{'prof err':>12}{'dp':>11}{'spread':>7} |"
      f"{'steps':>7}{'conv':>6}{'prof err':>12}{'dp':>11}{'spread':>7}")
L=[];T=[]
for dt in (0.05,0.1,0.5,1.0,2.0):
    a=run(dt,1e-3,1e-6); b=run(dt,1e-8,1e-10)
    if a: L.append(a['prof'])
    if b: T.append(b['prof'])
    fa=(f"{a['steps']:>7}{str(a['conv']):>6}{a['prof']:>12.3e}{a['dp']:>11.6f}{a['spread']:>7.0e}"
        if a else f"{'DIVERGED':>43}")
    fb=(f"{b['steps']:>7}{str(b['conv']):>6}{b['prof']:>12.3e}{b['dp']:>11.6f}{b['spread']:>7.0e}"
        if b else f"{'DIVERGED':>43}")
    print(f"{dt:>6} | {fa} |{fb}")
    sys.stdout.flush()
print(f"\n  spread over dt, LOOSE solve : {max(L)/min(L):9.1f}x   (published: 1875x)")
print(f"  spread over dt, TIGHT solve : {max(T)/min(T):9.1f}x")
