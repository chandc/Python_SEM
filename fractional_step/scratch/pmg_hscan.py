"""Crossover in 3D: ELEMENTS swept at fixed p=12, matching the 2D figure.

The 2D result (figs/cavity_precond.png) crosses over at ~50k DOF -- 10x10 at
p=12 gives 1.09x -- and its Jacobi iteration count GROWS steadily with size
(700 -> 3200).  The 3D p-scan at fixed 3x3 elements found Jacobi FLAT at ~2700
from N=6 to N=14, so there was nothing for PMG to overtake: at a_mass = 525 the
conditioning is a_mass-dominated, not resolution-dominated (3D_STATUS sec 6).

What did move Jacobi in 3D was the ELEMENT count (3x3 -> 4x4: 3447 -> 3904), so
this sweeps h at fixed p = 12, the order at which 2D crosses over.
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

NU, C, TOL, N, NZ = 1/180., 525., 1e-6, 12, 8

if __name__ == '__main__':
    print(f'p={N} FIXED (the 2D crossover order), nz={NZ}, tol={TOL:g}, '
          f'deg=4, coarse_deg=4')
    print(f"{'elems':>7}{'plane DOF':>11}{'jac its':>9}{'jac s':>8}"
          f"{'PMG its':>9}{'PMG s':>8}{'it ratio':>10}{'TIME':>8}")
    out = []
    for ex in (2, 3, 4, 6):
        m = build_channel(1., 1., ex, ex, N, bcs=(1, 1, 1, 2))
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
        t0 = time.perf_counter()
        P = PC.PMG(m, nk, NZ, NU, C, kz, kap=0., rw=rw, orders=(12, 6, 3, 2),
                   deg=4, coarse_deg=4)
        tb = time.perf_counter()-t0
        t0 = time.perf_counter()
        _, itp, _ = S3.pcg(b, D, m.facx, m.facy, kz, NU, C, m, mask, P, TOL,
                           40000, None, m.wq, 0., rw)
        tp = time.perf_counter()-t0 + tb
        dof = m.nelem*(N+1)**2*OP.NVAR_R
        out.append(dict(ex=ex, dof=dof, jac_its=int(itj), jac_s=tj,
                        pmg_its=int(itp), pmg_s=tp, gain=tj/tp))
        print(f'{ex}x{ex:<4}{dof:>11,}{itj:>9}{tj:>8.1f}{itp:>9}{tp:>8.1f}'
              f'{itj/max(itp,1):>9.1f}x{tj/tp:>7.2f}x', flush=True)
    json.dump(out, open('scratch/pmg_hscan.json', 'w'), indent=1)
