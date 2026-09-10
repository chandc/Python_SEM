"""Does smoothed-aggregation AMG (the F2 configuration: per-field near-null
space, energy prolongation) handle the LARGE-c operator?  F2 measured it only
at c ~ 1 (steady, dt = 1).  AMG on the assembled SEM operator is the strongest
form of option F (LOR would hand AMG a sparser but spectrally-equivalent
matrix); if the divergence-free kernel defeats it here, LOR cannot rescue it."""
import os, sys, time
for _v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '4')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, 'scratch')); os.chdir(_R)
import numpy as np, scipy.sparse as sp, pyamg
from adn_block_precond import pcg_ritz
from adn_hiptmair_2d import setup

print(f'{"c":>6} {"N":>3} {"ndof":>6} | {"jacobi":>14} | {"SA-AMG V(1,1), per-field B":>26} {"levels":>6} {"op.cx":>6} {"setup":>6} | {"SA-AMG + Jacobi-free?":>10}')
for DT in (1.85e-4, 1.0):
    for N in (8, 12):
        m, st, fu, fv, A, free, g = setup(N, DT)
        Af = A[free][:, free].tocsr(); nd = Af.shape[0]; fld = (np.arange(free.size) % 4)[free]
        B = np.zeros((nd, 4))
        for v in range(4): B[fld == v, v] = 1.0
        b = Af@np.random.default_rng(0).standard_normal(nd); b /= np.linalg.norm(b)
        rJ = pcg_ritz(Af, lambda r: r/Af.diagonal(), b, maxit=6000)
        t0 = time.time(); ml = pyamg.smoothed_aggregation_solver(Af, B=B, max_coarse=20, smooth='energy'); ts = time.time()-t0
        M = ml.aspreconditioner(cycle='V')
        rA = pcg_ritz(Af, lambda r: M.matvec(r), b, maxit=6000)
        print(f'{1/DT:6.0f} {N:3d} {nd:6d} | {rJ[0]:5d} {rJ[1]:8.1e} | {rA[0]:15d} {rA[1]:10.1e} {len(ml.levels):6d} {ml.operator_complexity():6.2f} {ts:5.1f}s |', flush=True)
