"""Does the Python short-BFS blow-up survive LEGACY weighting at the Fortran's dt?

The Fortran (SEM_2D_BFS_FREEOUT, run_chan389_short) converges on this exact grid
with FREE outflow -- |dU| = 9.3e-07 at t = 350, dt = 0.5, legacy weighting
(a_mass = fac1 = 1.5, a_flux = dt = 0.5), nsub = 1, p-MG, tol = 1e-6.

Our Python free-outflow run blew up on step 1 (max|u| = 3603) at dt = 1 with
w_mom = w_mass = 1, i.e. a_flux = 1 -- TWICE the Fortran momentum weight.

So the blow-up may be the WEIGHTING, not the boundary condition.  This sweeps
both, free outflow throughout, to separate them.
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
GRID='/Users/danielchan/Dropbox/F90_SEM/pmg_clean/cnos_short_grid.dat'
RE=389.0
def run(dt, wm, cap=700, wall=600.0, cgsfac=1e-3, tol=1e-6):
    m,_,_=load(GRID); n=m.N+1; N=m.N
    pin=next((e,n-1,0) for e in range(m.nelem) if m.bc[e,1]==4 and m.bc[e,2]==1)
    for e in range(m.nelem):
        if m.bc[e,1]==4: m.bc[e,1]=0            # FREE outflow, as the Fortran
    st=SolverState(m,diff_matrix(N),nu=1.0/RE,dt=dt,fac1=1.0,w_mom=wm,w_mass=wm)
    am,af,_=ls_coeffs(st)
    inlet=lambda x,y,t:6.0*((y-0.5)/0.5)*(1.0-(y-0.5)/0.5)
    U=np.zeros((m.nelem,n,n,4)); h=[U.copy()]; t0=time.perf_counter(); status='CAP'; d=np.nan
    for s in range(cap):
        prev=h[0].copy()
        U=S.step_bdf(st,h,time=s*dt,max_newton=1,newton_tol=1e-12,newton_factor=0.0,
                     custom_inlet=inlet,pin_p=pin,cgsfac=cgsfac,cg_tol=tol,cg_max_iter=100000)
        if not np.all(np.isfinite(U)): status='NaN'; break
        d=float(np.abs(U-prev).max())
        if np.abs(U[...,0]).max()>20.0: status='BLEWUP'; break
        if d<1e-11: status='conv'; break
        if time.perf_counter()-t0>wall: status='WALLCAP'; break
    mu=float(np.abs(U[...,0]).max()) if np.all(np.isfinite(U)) else np.nan
    np.savez(f'{SC}/bfs_short_legacy_dt{dt:g}_w{"leg" if wm is None else wm:}.npz',
             U=U,xnod=m.xnod,ynod=m.ynod,hy=m.hy,N=N,dt=dt,status=status)
    return status,s+1,d,mu,am,af,time.perf_counter()-t0
print("SHORT BFS, FREE outflow throughout.  Fortran converges here with legacy/dt=0.5.\n")
print(f"{'dt':>6}{'weights':>10}{'a_mass':>8}{'a_flux':>8}{'status':>9}{'steps':>7}{'|dU|':>11}{'max|u|':>10}{'s':>7}")
for dt,wm in ((0.5,None),(1.0,None),(0.5,1.0),(1.0,1.0)):
    r=run(dt,wm)
    print(f"{dt:>6g}{('legacy' if wm is None else f'w={wm:g}'):>10}{r[4]:>8.2f}{r[5]:>8.2f}"
          f"{r[0]:>9}{r[1]:>7}{r[2]:>11.3e}{r[3]:>10.4f}{r[6]:>7.0f}",flush=True)
