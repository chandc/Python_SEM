"""Symmetry and SPD of A(w_con) in the ASSEMBLED space, using the F0 assembler."""
import os, sys
for v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(v,'1')
R='/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo'
sys.path.insert(0,R); sys.path.insert(0,R+'/scratch'); os.chdir(R)
import numpy as np, lssem2d
lssem2d.set_backend('numpy')
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState
from lssem2d.mesh import build_channel
import fosls_assemble as FA

def build(w_con, N=6, ex=3, ey=3):
    m = build_channel(2.0, 1.0, ex, ey, N, bcs=(1,1,1,2)); m.compute_global_indices()
    st = SolverState(m, diff_matrix(N), nu=1/100., dt=1e4, fac1=1.0, w_mom=1.0,
                     w_con=w_con)
    n=N+1; z=np.zeros((m.nelem,n,n)); st.update_linearisation(z, z.copy())
    A, free, _ = FA.assemble(st, z, z.copy(), pin_p=True)
    return A[free][:,free].tocsr()

for w in (None, 1.0, 2.0, 5.0, 10.0):
    A = build(w).toarray()
    asym = np.abs(A-A.T).max()/max(np.abs(A).max(),1e-300)
    ev = np.linalg.eigvalsh(0.5*(A+A.T))
    print(f'w_con={str(w):5s}  asym={asym:.2e}  lam_min={ev.min():+.4e}  '
          f'lam_max={ev.max():.4e}  cond={ev.max()/max(ev.min(),1e-300):.3e}  SPD={ev.min()>0}')
