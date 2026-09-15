"""Poiseuille Re=100 with w_mass = 0: the PURE steady least-squares problem.

w_mass = 0 gives a_mass = 0 and a_hist = 0, so the momentum row is simply

    w_mom * N(U)

with no time-derivative term at all.  dt becomes irrelevant.  This is the
cleanest possible isolation of the momentum weighting: nothing competes with the
constraints except N(U) itself, so the pressure/velocity block ratio is exactly
w_mom^2 and the balanced point should be w_mom = 1.

It also sidesteps the dt=0-through-step_bdf bug: there a_hist stayed at 1 while
a_mass went to 0, so the fixed point solved N(u) = 1.5u.  Here a_hist is
genuinely 0.

Each "step" is then a Newton/Picard iteration on the steady problem -- there is
no time integration to damp it, so convergence is judged on max|dU| directly.
"""
import os, sys, time
SC=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,'/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix, lgl_weights
from lssem2d.lssem import SolverState, ls_coeffs
import lssem2d.solver as S

LX,LY,RE=10.,1.,100.; NU=1.0*LY/RE; DP=12.0*NU*1.0/LY**2*LX
N,EX,EY,DT=8,10,2,0.5
TOL,CAP,TRIP=1e-11,600,50.0
u_ex=lambda y:6.0*y*(1.0-y)

def run(wmom):
    m=build_channel(LX,LY,EX,EY,N,bcs=(3,4,1,1)); n=N+1
    pin=next((e,0,0) for e in range(m.nelem) if m.bc[e,0]==3 and m.bc[e,2]==1)
    for e in range(m.nelem):
        if m.bc[e,1]==4: m.bc[e,1]=0
    st=SolverState(m,diff_matrix(N),nu=NU,dt=DT,fac1=1.0,w_mom=wmom,w_mass=0.0)
    inlet=lambda x,y,t:u_ex(y)
    U=np.zeros((m.nelem,n,n,4)); hist=[U]; t0=time.perf_counter(); status='cap'
    for s in range(CAP):
        Up=hist[0].copy()
        U=S.step_bdf(st,hist,time=0.0,max_newton=1,newton_tol=1e-12,
                     newton_factor=0.0,custom_inlet=inlet,pin_p=pin,
                     cgsfac=1e-3,cg_max_iter=40000,verbose=False)
        if not np.all(np.isfinite(U)): status='NaN'; break
        if np.abs(U[...,0]).max()>TRIP: status=f'diverged({np.abs(U[...,0]).max():.0f})'; break
        if s>2 and np.max(np.abs(U-Up))<TOL: status='converged'; break
    wall=time.perf_counter()-t0
    if status!='converged':
        return dict(w=wmom,status=status,steps=s+1,wall=wall)
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
    return dict(w=wmom,status=status,steps=s+1,wall=wall,coef=ls_coeffs(st),
                prof=float(np.sqrt(np.mean((us-u_ex(ys))**2))/1.5),
                dp=float(pbar('in')-pbar('out')),spread=float(ps.max()-ps.min()))

print("Poiseuille Re=100, order 8, 10x2, Jacobi, free outflow, inlet pin")
print(f"w_mass = 0  ->  momentum row is exactly  w_mom * N(U),  no time derivative")
print(f"exact: dp = {DP},  outlet pressure constant across the channel\n")
print(f"{'w_mom':>7}{'a_mass':>8}{'a_flux':>8}{'a_hist':>8}{'iters':>7}{'status':>12}"
      f"{'prof err':>11}{'dp':>11}{'dp err':>10}{'p_out spread':>14}{'wall':>7}")
res=[]
for wm in (0.1,0.2,0.3,0.5,0.7,0.9,1.0):
    r=run(wm)
    if r['status']!='converged':
        print(f"{wm:>7}{'':>24}{r['steps']:>7}{r['status']:>12}{'':>46}{r['wall']:>7.0f}")
        continue
    res.append(r); c=r['coef']
    print(f"{wm:>7}{c[0]:>8.1f}{c[1]:>8.2f}{c[2]:>8.1f}{r['steps']:>7}{r['status']:>12}"
          f"{r['prof']:>11.3e}{r['dp']:>11.5f}{abs(r['dp']-DP)/DP:>10.2e}"
          f"{r['spread']:>14.3e}{r['wall']:>7.0f}")
if len(res)>1:
    a=np.array([r['prof'] for r in res])
    b=[r for r in res if abs(r['w']-1.0)<1e-9]
    print(f"\n  spread over w_mom = {a.max()/a.min():.1f}x")
    if b: print(f"  at w_mom = 1.0:  dp = {b[0]['dp']:.6f} (exact {DP}), "
                f"outlet spread = {b[0]['spread']:.3e}")
