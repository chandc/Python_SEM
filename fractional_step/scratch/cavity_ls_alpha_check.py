"""Does the CONVERGED transient run also exhaust the line search?  Yes.

    uv run --quiet python scratch/cavity_ls_alpha_check.py

Evidence for the caveat in ARTIFICIAL_COMPRESSIBILITY.md sec 5.3: exhausting the
backtracking is NOT by itself a failure signal.

Restarted on its own converged field, the transient dt = 0.05 AC-on cavity gives
alpha = 2.98e-08 (= 0.5**25, the max_backtrack floor) and _ls_exhausted = True on
every step -- yet that field matches Ghia to RMS u = 1.57e-02.

The reason is benign.  At a minimum of the merit no step can satisfy

    J(U + alpha*dU) <= (1 - 1e-4*alpha) * J_ref

because J(U + alpha*dU) ~ J_ref while the target sits strictly below it, so the
line search is correctly declining to move.  The flag fires at genuine
convergence and at a genuine stall alike; the RESIDUAL, not alpha, separates
them.
"""
import os, sys
for v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(v,'1')
sys.path.insert(0,'/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np, lssem2d
lssem2d.set_backend('numpy')
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState
import lssem2d.solver as S

RE,EX,N=1000.0,6,10; n=N+1
mesh=build_channel(1.0,1.0,EX,EX,N,bcs=(1,1,1,2))
U0=np.load('scratch/cavity_ac_dt0.05_match.npz',allow_pickle=True)['U'].copy()

# Continue the TRANSIENT run from its own converged field, watching alpha.
st=SolverState(mesh,diff_matrix(N),nu=1.0/RE,dt=0.05,fac1=1.0,w_mom=1.0,w_mass=1.0)
st.dtau_p=1.0/30.0
U=U0.copy(); hist=[U.copy()]
print('TRANSIENT dt=0.05, AC on, restarted on its own converged field:')
print(f"{'step':>5}{'alpha':>11}{'ls_exhausted':>14}{'|dU|':>12}")
for s in range(8):
    Up=hist[0].copy()
    U=S.step_bdf(st,hist,time=(s+1)*0.05,max_newton=5,newton_tol=1e-13,
                 newton_factor=1e-6,pin_p=True,cgsfac=1e-3,cg_tol=1e-8,
                 cg_max_iter=60000,line_search=True)
    print(f'{s+1:>5}{getattr(st,"_last_alpha",np.nan):>11.3e}'
          f'{str(getattr(st,"_ls_exhausted",None)):>14}'
          f'{float(np.abs(U-Up).max()):>12.3e}',flush=True)
print(f'  total LS exhaustions: {getattr(st,"_ls_exhausted_count",0)}')
