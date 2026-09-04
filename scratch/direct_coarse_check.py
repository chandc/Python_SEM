"""DirectCoarseE must be the SAME operator as DirectCoarse, just assembled
differently.  Small mesh so the O(global dof) probing path is affordable."""
import os, sys, time
for _v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '4')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); os.chdir(_R)
import numpy as np
from lssem2d.mesh import build_channel
from lssem3d import backend, bc as BC, operator as OP, precond as P3

def main():
    backend.set_backend('numpy')
    for (ex, ey, N, nz) in ((3, 3, 2, 4), (2, 4, 2, 8)):
        nk = nz//2 + 1
        m = build_channel(2.0*np.pi, 2.0, ex, ey, N, bcs=(0, 0, 1, 1))
        m.periodic_x = 2.0*np.pi
        m.compute_global_indices()
        kz = np.arange(nk)*1.7
        mask = BC.build_mask(m, nk, pin_p=False, nz=nz)
        BC.pin_dof(m, mask, OP.P_, 0)
        rw = OP.momentum_row_weights(525.0)
        lv = P3._Level(m, nk, nz, 1/180., 525.0, kz, 0.0, rw, False, mask=mask)
        t0 = time.perf_counter(); a = P3.DirectCoarse(lv);  ta = time.perf_counter()-t0
        t0 = time.perf_counter(); b = P3.DirectCoarseE(lv); tb = time.perf_counter()-t0
        # r must be a legitimate ASSEMBLED residual: every element-local copy of
        # a global node carries the same value, and r lies in range(A).  A raw
        # random vector satisfies neither, and both coarse solvers then "fail"
        # at 0.70 for reasons that have nothing to do with either of them.
        rng = np.random.default_rng(0)
        x = rng.standard_normal(lv.shape)*mask
        r = lv.A(x)
        za, zb = a(r), b(r)
        den = max(np.abs(za).max(), 1e-300)
        print(f'{ex}x{ey} N={N} nz={nz}  dofs/mode~{lv.shape[0]*(N+1)**2*14//1}  '
              f'probe {ta:6.2f}s  element {tb:6.2f}s ({ta/max(tb,1e-9):5.1f}x)  '
              f'max|dz|/max|z| = {np.abs(za-zb).max()/den:.3e}')
        # and the real test: does it actually SOLVE?  A z should return r.
        Az = lv.A(zb)
        print(f'      residual of the coarse solve  ||A z - r||/||r|| = '
              f'{np.linalg.norm(Az-r)/np.linalg.norm(r):.3e}   '
              f'(DirectCoarse: {np.linalg.norm(lv.A(za)-r)/np.linalg.norm(r):.3e})')

if __name__ == '__main__':
    main()
