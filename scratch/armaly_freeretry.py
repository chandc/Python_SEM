"""Can long/free be made to converge on the Armaly domain?

It blew up at dt=1, w_mom=w_mass=1, cgsfac=1e-3/tol=1e-6 (max|u| = 21.4 by step 25),
while P+Z converged to x_r/S = 8.145.  Three things are known to matter:

  dt / weighting  the Fortran reference runs legacy weighting at dt = 0.5, i.e.
                  a_flux = dt = 0.5 -- HALF the momentum weight of w = 1.
  CG effort       OUTFLOW_BC_STUDY.md sec 7c: on a truncated free outflow the system
                  is near-singular and OVER-solving amplifies the near-null mode.
                  The Fortran study says the same ("nitcgs=40000 -> NaN,
                  nitcgs=500 -> clean convergence").  Looser = more regularised.

Cycles both, free outflow throughout, so the long/free hole in the 2x2 gets filled
or is shown to be unfillable.
"""
import os, sys, time
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo'); sys.path.insert(0, SC)
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np, lssem2d
lssem2d.set_backend('numpy')
from fgrid import load
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState, ls_coeffs
import lssem2d.solver as S
GRID='grids/armaly_er194_long_grid.dat'
RE=389.0; H_IN=1.0; Y_STEP=0.94; NU=2.0*H_IN/RE; S_STEP=Y_STEP

def inlet_profile(y):
    eta=(np.asarray(y)-Y_STEP)/H_IN
    return np.where((eta>=0.0)&(eta<=1.0),6.0*eta*(1.0-eta),0.0)

def reattach(U,xn,yn,hy,N):
    D=diff_matrix(N); n=N+1; xs,tw=[],[]
    for e in range(U.shape[0]):
        if yn[e,0]>0.01 or xn[e,0]<-1e-9: continue
        for i in range(n):
            xs.append(xn[e,i]); tw.append(np.dot(D[0,:],U[e,i,:,0])*(2.0/hy[e]))
    o=np.argsort(xs); xs,tw=np.array(xs)[o],np.array(tw)[o]
    for k in range(len(xs)-1):
        if tw[k]<0 and tw[k+1]>0 and xs[k]>0.05:
            return xs[k]-tw[k]*(xs[k+1]-xs[k])/(tw[k+1]-tw[k])
    return float('nan')

def run(dt,wm,cgsfac,tol,nitmax,cap=2000,wall=1500.0):
    m,_,_=load(GRID); N=m.N; n=N+1
    pin=next((e,n-1,0) for e in range(m.nelem) if m.bc[e,1]==4 and m.bc[e,2]==1)
    for e in range(m.nelem):
        if m.bc[e,1]==4: m.bc[e,1]=0            # FREE outflow
    st=SolverState(m,diff_matrix(N),nu=NU,dt=dt,fac1=1.0,w_mom=wm,w_mass=wm)
    am,af,_=ls_coeffs(st)
    inl=lambda x,y,t: inlet_profile(y)
    U=np.zeros((m.nelem,n,n,4)); h=[U.copy()]; t0=time.perf_counter(); status='CAP'; d=np.nan
    for s in range(cap):
        prev=h[0].copy()
        U=S.step_bdf(st,h,time=s*dt,max_newton=1,newton_tol=1e-12,newton_factor=0.0,
                     custom_inlet=inl,pin_p=pin,cgsfac=cgsfac,cg_tol=tol,cg_max_iter=nitmax)
        if not np.all(np.isfinite(U)): status='NaN'; break
        d=float(np.abs(U-prev).max())
        if np.abs(U[...,0]).max()>20.0: status='BLEWUP'; break
        if d<1e-11: status='conv'; break
        if time.perf_counter()-t0>wall: status='WALLCAP'; break
    ok=np.all(np.isfinite(U))
    xr=reattach(U,m.xnod,m.ynod,m.hy,N) if ok else np.nan
    tag=f"dt{dt:g}_{'leg' if wm is None else 'w1'}_nit{nitmax}"
    np.savez(f'{SC}/armaly_long_free_{tag}.npz',U=U,xnod=m.xnod,ynod=m.ynod,
             hy=m.hy,N=N,dt=dt,status=status)
    print(f"{dt:>5g}{('legacy' if wm is None else 'w=1'):>8}{af:>8.2f}{nitmax:>8}"
          f"{status:>9}{s+1:>7}{d:>11.3e}{(np.abs(U[...,0]).max() if ok else np.nan):>9.3f}"
          f"{(xr/S_STEP if np.isfinite(xr) else np.nan):>9.3f}{time.perf_counter()-t0:>7.0f}",flush=True)

# one case per process:  argv = dt  legacy|w1  nitcgs
if __name__ == '__main__':
    dt = float(sys.argv[1]); wm = None if sys.argv[2] == 'legacy' else 1.0
    nit = int(sys.argv[3])
    print(f"{'dt':>5}{'weights':>8}{'a_flux':>8}{'nitcgs':>8}{'status':>9}{'steps':>7}"
          f"{'|dU|':>11}{'max|u|':>9}{'x_r/S':>9}{'s':>7}", flush=True)
    run(dt, wm, 1e-3, 1e-6, nit)
