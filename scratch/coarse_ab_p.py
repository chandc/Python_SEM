"""The coarse-solver A/B as a function of p.

At p=8 cheb and direct tied (109 vs 110) -- but a halving ladder from 8 has only
three rungs and the coarse grid has almost nothing to do.  The 2D study
(PMG_ALGORITHM sec 6.9) found the coarse SOLVE decisive, and it ran to N=12-20.
If the coarse solver matters at all in 3D it has to show up as p grows, where the
fine operator is more ill-conditioned and the ladder is deeper.

Halving ladders: 8->4->2, 12->6->3, 16->8->4->2, 20->10->5.
"""
import os, sys, time
for _v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '8')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); os.chdir(_R)
os.environ.setdefault('LSSEM3D_BACKEND', 'numba')
import numpy as np

LADDER = {8: (8,4,2), 12: (12,6,3), 16: (16,8,4,2), 20: (20,10,5)}

def main():
    import lssem3d; lssem3d.set_backend(os.environ['LSSEM3D_BACKEND'])
    from lssem2d.mesh import build_channel
    from lssem2d.lgl import diff_matrix
    from lssem3d import bc as BC, operator as OP, precond as P3, solver3d as S3

    ex, ey, nz = 3, 3, 4
    nk = nz//2 + 1
    for cc in (5405.4, 525.0):
        print(f'\n================  c = {cc:g}  ================', flush=True)
        print(f'{"p":>3} {"ladder":>16} {"free dof":>9} | '
              f'{"cheb":>18} | {"DirectCoarse ref":>22} | {"DirectCoarseE":>20}',
              flush=True)
        for N in (8, 12, 16, 20):
            m = build_channel(2.0*np.pi, 2.0, ex, ey, N, bcs=(0, 0, 1, 1))
            m.periodic_x = 2.0*np.pi; m.compute_global_indices()
            kz = np.arange(nk)*2.0
            mask = BC.build_mask(m, nk, pin_p=False, nz=nz)
            BC.pin_dof(m, mask, OP.P_, 0)
            nu, D = 1/180., diff_matrix(N)
            rw = OP.momentum_row_weights(cc)
            shape = (m.nelem, N+1, N+1, OP.NVAR_R, nk)
            A = lambda x: S3.normal_op(x, D, m.facx, m.facy, kz, nu, cc, m, mask,
                                       m.wq, 0.0, rw)
            rng = np.random.default_rng(0)
            b = A(rng.standard_normal(shape)*mask); b /= np.linalg.norm(b)
            cells = []
            for dc in (False, True, 'element'):
                try:
                    t0 = time.perf_counter()
                    M = P3.PMG(m, nk, nz, nu, cc, kz, kap=0.0, rw=rw,
                               orders=LADDER[N], deg=6, coarse_deg=10,
                               pin_p=True, direct_coarse=dc, mask=mask)
                    tb = time.perf_counter()-t0
                    _, it, rt = S3.pcg(b, D, m.facx, m.facy, kz, nu, cc, mesh=m,
                                       mask=mask, M_inv=M, tol=1e-10,
                                       max_iter=6000, wq=m.wq, kap=0.0, rw=rw)
                    cells.append(f'CG {it:5d} bld {tb:6.2f}s')
                except Exception as e:
                    cells.append(f'{type(e).__name__}: {str(e)[:14]}')
            print(f'{N:3d} {str(LADDER[N]):>16} {int(mask.sum()):9d} | '
                  f'{cells[0]:>18} | {cells[1]:>22} | {cells[2]:>20}', flush=True)

if __name__ == '__main__':
    main()
