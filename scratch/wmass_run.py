import numpy as np, time, sys, os
sys.path.insert(0,'/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix, lgl_weights
from lssem2d.lssem import SolverState, ls_coeffs
import lssem2d.solver as S
NU=0.01; DP=1.2; N,EX,EY=8,10,2
u_ex=lambda y:6.0*y*(1.0-y)
def run(dt,wf,wm,tmin=300.0,cap=1200):
    m=build_channel(10.,1.,EX,EY,N,bcs=(3,4,1,1)); n=N+1
    pin=next((e,0,0) for e in range(m.nelem) if m.bc[e,0]==3 and m.bc[e,2]==1)
    for e in range(m.nelem):
        if m.bc[e,1]==4: m.bc[e,1]=0
    st=SolverState(m,diff_matrix(N),nu=NU,dt=dt,fac1=1.0,w_mom=wf,w_mass=wm)
    inlet=lambda x,y,t:u_ex(y)
    U=np.zeros((m.nelem,n,n,4)); hist=[U]; t0=time.perf_counter(); conv=False
    dte = dt*(wf/wm) if (wf is not None and wm is not None) else dt
    nstep=max(int(np.ceil(tmin/dte)),50)
    for s in range(min(nstep,cap)):
        Up=hist[0].copy()
        U=S.step_bdf(st,hist,time=s*dt,max_newton=1,newton_tol=1e-12,
                     newton_factor=0.0,custom_inlet=inlet,pin_p=pin,
                     cgsfac=1e-3,cg_max_iter=40000,verbose=False)
        if not np.all(np.isfinite(U)): return None
        rate=np.max(np.abs(U-Up))/dte
        if s>10 and rate<1e-9: conv=True; break
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
    return dict(coef=ls_coeffs(st),dte=dte,steps=s+1,conv=conv,rate=rate,
                wall=time.perf_counter()-t0,
                prof=float(np.sqrt(np.mean((us-u_ex(ys))**2))/1.5),
                dp=float(pbar('in')-pbar('out')),spread=float(ps.max()-ps.min()))
print(f"Poiseuille Re=100, Jacobi, free outflow, inlet pin.  exact dp = {DP}\n")
print(f"{'config':<36}{'a_mass':>8}{'a_flux':>8}{'dt_eff':>8}{'steps':>7}{'conv':>6}"
      f"{'prof err':>11}{'dp':>10}{'spread':>10}")
for lab,dt,wf,wm in (
    ('legacy dt=1.0 (reference)',        1.0,None,None),
    ('dt=0.5 w_mass=0.5 w_mom=1',        0.5, 1.0, 0.5),
    ('dt=0.1 w_mass=0.1 w_mom=1',        0.1, 1.0, 0.1),
    ('dt=0.5 w_mass=1.0 w_mom=1',        0.5, 1.0, 1.0),
    ('legacy dt=0.5',                    0.5,None,None)):
    r=run(dt,wf,wm)
    if r is None: print(f"{lab:<36}   DIVERGED"); continue
    print(f"{lab:<36}{r['coef'][0]:>8.3f}{r['coef'][1]:>8.3f}{r['dte']:>8.3f}"
          f"{r['steps']:>7}{str(r['conv']):>6}{r['prof']:>11.3e}{r['dp']:>10.5f}"
          f"{r['spread']:>10.2e}")
