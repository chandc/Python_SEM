"""Is the lid-driven cavity as dt/weight-sensitive as Poiseuille?

Hypothesis: no, because the cavity is LID-driven (forcing is a velocity BC) while
Poiseuille and the BFS are PRESSURE-driven.  Pressure enters only the momentum
rows, so under-weighting them corrupts the driving force in a pressure-driven
flow but only the passive pressure field in a lid-driven one.
"""
import os, sys, time
SC=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,'/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix, lgl_weights
from lssem2d.lssem import SolverState
from lssem2d.operators import dUdx, dUdy
import lssem2d.solver as S

RE=1000.0; NU=1.0/RE; N,EX=8,4
RATE_TOL,T_MIN,CAP = 1e-9, 150.0, 6000
gh=np.load('cavity_re1000_data.npz'); GU,GY=gh['ghia_u'],gh['ghia_y']

def lag(xn,xq):
    n=len(xn); w=np.ones(n)
    for i in range(n):
        for j in range(n):
            if i!=j: w[i]/=(xn[i]-xn[j])
    d=xq-xn
    if np.any(np.abs(d)<1e-13):
        L=np.zeros(n); L[np.argmin(np.abs(d))]=1.0; return L
    num=w/d; return num/num.sum()

def run(dt):
    m=build_channel(1.,1.,EX,EX,N,bcs=(1,1,1,2)); n=N+1
    st=SolverState(m,diff_matrix(N),nu=NU,dt=dt,fac1=1.0)
    U=np.zeros((m.nelem,n,n,4)); hist=[U]; t0=time.perf_counter()
    conv=False
    for s in range(min(int(np.ceil(T_MIN/dt))*4,CAP)):
        Up=hist[0].copy()
        U=S.step_bdf(st,hist,time=s*dt,max_newton=1,newton_tol=1e-12,
                     newton_factor=0.0,pin_p=True,cgsfac=1e-3,
                     cg_max_iter=40000,verbose=False)
        if not np.all(np.isfinite(U)): return None
        if (s+1)*dt>=T_MIN and np.max(np.abs(U-Up))/dt<RATE_TOL: conv=True; break
    D=diff_matrix(N)
    ys,us=[],[]
    for e in range(m.nelem):
        xs=m.xnod[e]
        if xs[0]-1e-9<=0.5<=xs[-1]+1e-9:
            L=lag(xs,0.5)
            for j in range(n): ys.append(m.ynod[e,j]); us.append(np.dot(L,U[e,:,j,0]))
    o=np.argsort(ys); ys,us=np.array(ys)[o],np.array(us)[o]
    k=np.concatenate(([True],np.diff(ys)>1e-12)); ys,us=ys[k],us[k]
    rms=float(np.sqrt(np.mean((np.interp(GY,ys,us)-GU)**2)))
    ux=dUdx(np.ascontiguousarray(U[...,0]),D,m.facx)
    vy=dUdy(np.ascontiguousarray(U[...,1]),D,m.facy)
    return dict(dt=dt,steps=s+1,conv=conv,rms=rms,
                div=float(np.sqrt(((ux+vy)**2).mean())),
                pspread=float(U[...,2].max()-U[...,2].min()),
                umax=float(np.abs(U[...,0]).max()),
                wall=time.perf_counter()-t0)

print(f"Lid-driven cavity Re={RE:g}, {EX}x{EX} order {N}, legacy weighting (w = dt)")
print(f"accuracy metric: centreline u vs Ghia 1982, RMS\n")
print(f"{'dt':>6}{'a_flux':>8}{'steps':>7}{'conv':>6}{'RMS vs Ghia':>13}"
      f"{'% of umax':>11}{'rms div':>11}{'p spread':>10}{'wall s':>8}")
res=[]
for dt in (0.05,0.1,0.5,1.0,2.0,5.0):
    r=run(dt)
    if r is None: print(f"{dt:>6}   DIVERGED"); continue
    res.append(r)
    print(f"{dt:>6}{dt:>8.2f}{r['steps']:>7}{str(r['conv']):>6}{r['rms']:>13.4e}"
          f"{100*r['rms']/max(abs(GU.max()),abs(GU.min())):>10.2f}%{r['div']:>11.3e}"
          f"{r['pspread']:>10.4f}{r['wall']:>8.1f}")
if len(res)>1:
    a=np.array([r['rms'] for r in res])
    print(f"\n  CAVITY   accuracy spread over dt = {a.max()/a.min():.1f}x")
    print(f"  POISEUILLE (measured earlier)      = 1875x")
