"""Does PMG cross over as N grows?  Elements FIXED, polynomial order swept.

    uv run --quiet python scratch/pmg_Nscan.py

The degree sweep answered the wrong question.  Every (deg, coarse_deg,
hierarchy) at N=8 lands between 0.24x and 0.66x -- the surface is flat and
entirely losing, so no tuning rescues it THERE.  But the wall-time ratio is not
a constant:

    Jacobi iterations GROW with polynomial order (conditioning degrades),
    PMG iterations are RESOLUTION-INDEPENDENT (~320-425, measured),
    and the coarse levels get relatively cheaper as the fine level grows.

So the ratio should improve with N, and the question is whether it crosses 1
before the resolutions M7 needs.  Elements are held fixed at 3x3 so that N is
the only thing moving.
"""
import os, sys, time, json
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem3d import operator as OP, solver3d as S3, bc as BC, fourier as FR
from lssem3d import precond as PC

NU, C, TOL, EX, NZ = 1/180., 525., 1e-6, 3, 8


def hierarchy(N):
    """Descending p-hierarchy down to 2, halving."""
    h, p = [N], N
    while p > 4:
        p = max(2, p//2)
        h.append(p)
    if h[-1] != 2:
        h.append(2)
    return tuple(h)


if __name__ == '__main__':
    print(f'{EX}x{EX} elements FIXED, nz={NZ}, tol={TOL:g}, deg=4, coarse_deg=4')
    print(f"{'N':>4}{'hierarchy':>16}{'jac its':>9}{'jac s':>8}"
          f"{'PMG its':>9}{'PMG s':>8}{'it ratio':>10}{'TIME':>8}")
    out = []
    for N in (6, 8, 10, 12, 14, 16):
        m = build_channel(1., 1., EX, EX, N, bcs=(1, 1, 1, 2))
        D = diff_matrix(N)
        nk = NZ//2 + 1
        kz = FR.wavenumbers(NZ, 2*np.pi)
        mask = BC.build_mask(m, nk, pin_p=True, nz=NZ)
        shape = (m.nelem, N+1, N+1, OP.NVAR_R, nk)
        rw = OP.momentum_row_weights(C)
        kw = dict(mesh=m, mask=mask, wq=m.wq, kap=0.0, rw=rw)
        xs = S3.make_continuous(m, np.random.default_rng(0).standard_normal(shape))*mask
        b = S3.normal_op(xs, D, m.facx, m.facy, kz, NU, C, **kw)
        Mi = S3.jacobi_inverse(S3.jacobi_diagonal_analytic(
            shape, D, m.facx, m.facy, kz, NU, C, **kw), mask)
        t0 = time.perf_counter()
        _, itj, _ = S3.pcg(b, D, m.facx, m.facy, kz, NU, C, m, mask, Mi, TOL,
                           40000, None, m.wq, 0., rw)
        tj = time.perf_counter()-t0
        h = hierarchy(N)
        t0 = time.perf_counter()
        P = PC.PMG(m, nk, NZ, NU, C, kz, kap=0., rw=rw, orders=h, deg=4,
                   coarse_deg=4)
        tb = time.perf_counter()-t0
        t0 = time.perf_counter()
        _, itp, _ = S3.pcg(b, D, m.facx, m.facy, kz, NU, C, m, mask, P, TOL,
                           40000, None, m.wq, 0., rw)
        tp = time.perf_counter()-t0 + tb
        out.append(dict(N=N, jac_its=int(itj), jac_s=tj, pmg_its=int(itp),
                        pmg_s=tp, gain=tj/tp))
        print(f'{N:>4}{str(h):>16}{itj:>9}{tj:>8.1f}{itp:>9}{tp:>8.1f}'
              f'{itj/max(itp,1):>9.1f}x{tj/tp:>7.2f}x', flush=True)
    json.dump(out, open('scratch/pmg_Nscan.json', 'w'), indent=1)
