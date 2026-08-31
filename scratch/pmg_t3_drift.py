"""Is T3 sitting on a period-2 orbit?  |U(k)-U(k-1)| vs |U(k)-U(k-2)|."""
import os, sys
for v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(v,'1')
R='/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo'
sys.path.insert(0,R); sys.path.insert(0,R+'/scratch'); os.chdir(R)
import numpy as np, importlib.util
spec = importlib.util.spec_from_file_location('t3', R+'/scratch/pmg_t3_gartling.py')
t3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(t3)
import lssem2d.solver as S
from gartling_run import inlet_profile

def probe(pre, ic, nstep=12):
    m, st = t3.build()
    U = np.load(f'{R}/scratch/pmg_t3_gartling_{pre}_{ic}.npz')['U'].copy()
    _PCG = S.pcg_solve
    from lssem2d import precond as P
    def wrapped(state,b,fu,fv,M_inv,mw,pin_p=False,max_iter=5000,tol=1e-6,cgsfac=0.0,precond=None):
        if pre!='jacobi' and precond is None:
            kw=dict(pc=2,deg=4); kw['coarse_solver']='direct' if pre=='direct' else 'chebyshev'
            if pre!='direct': kw['coarse_deg']=10
            precond=P.make('pmg2',state,fu,fv,M_inv,pin_p,**kw)
        return _PCG(state,b,fu,fv,M_inv,mw,pin_p=pin_p,max_iter=max_iter,tol=tol,
                    cgsfac=cgsfac,precond=precond)
    S.pcg_solve = wrapped
    inl = lambda x,y,t: inlet_profile(y)
    hist=[U.copy()]; h=[U.copy()]
    try:
        for _ in range(nstep):
            U = S.step_bdf(st,h,time=0.0,max_newton=1,newton_tol=1e-12,newton_factor=0.0,
                           custom_inlet=inl,pin_p=False,cgsfac=1e-3,cg_tol=1e-10,
                           cg_max_iter=200000,line_search=True)
            hist.append(U.copy())
    finally:
        S.pcg_solve=_PCG
    d1=[float(np.abs(hist[i]-hist[i-1]).max()) for i in range(1,len(hist))]
    d2=[float(np.abs(hist[i]-hist[i-2]).max()) for i in range(2,len(hist))]
    print(f'{pre:7s} {ic:8s}  |U(k)-U(k-1)| = {np.mean(d1[-5:]):.3e}   '
          f'|U(k)-U(k-2)| = {np.mean(d2[-5:]):.3e}   ratio {np.mean(d1[-5:])/max(np.mean(d2[-5:]),1e-300):.1f}x')

if __name__=='__main__':
    print('If |U(k)-U(k-2)| << |U(k)-U(k-1)|, the iterate is on a PERIOD-2 ORBIT.\n')
    for pre in ('jacobi','direct'):
        probe(pre,'zero')
