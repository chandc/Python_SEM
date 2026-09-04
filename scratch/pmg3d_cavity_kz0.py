"""Does p-multigrid go flat on the CAVITY in the 3D code, as it does in 2D?

    uv run --quiet python -u scratch/pmg3d_cavity_kz0.py

THE DISCRIMINATION THIS RUNS.  Two results sit in tension:

  PMG_ALGORITHM sec 6.9   2D cavity: the halving p-ladder is FLAT -- 1.05x over
                          N = 8..24 against Jacobi's 4.00x.
  3D_STATUS sec 7K        3D channel: the ratio is PINNED at ~7.4x, the V-cycle
                          growing nearly in lockstep with Jacobi.  sec 7K.2
                          diagnoses why -- the softest modes are ROUGH (pressure
                          roughness ~1300, omega 2300-9000), inverting
                          multigrid's slow=smooth premise -- and concludes "no
                          polynomial-coarsening multigrid can be N-independent
                          here, at any implementation quality."

Those differ in TWO variables at once: the PROBLEM (cavity vs channel) and the
FORMULATION (4 fields / 4 rows vs 14 real fields / 8 rows with row weights).
Running the 3D code on the SAME cavity at k_z = 0 changes only the formulation,
so it separates them:

  flat here    -> sec 7K's pinning is a property of the CHANNEL PROBLEM, and the
                  2D p-multigrid result should transfer to 3D cavity-like flows.
  pinned here  -> it is a property of the FOSLS FORMULATION, sec 7K.2 generalises,
                  and the 2D result does not transfer at all.

An OPERATOR study with a manufactured RHS (b = A x_rand), the same protocol
scratch/pmg_sweep.py uses -- iteration counts measure conditioning, which is the
question, and no flow needs to converge for that to be meaningful.

k_z = 0 only (nk = 1): the spanwise-invariant mode IS the 2D cavity.
"""
import os
import sys
import time

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R)
os.chdir(_R)

import numpy as np

from lssem2d.lgl import diff_matrix
from lssem2d.mesh import build_channel
from lssem3d import backend, bc as BC, operator as OP, precond as P3, solver3d as S3

RE, EX = 1000.0, 4
NU = 1.0/RE
# c = 1/(beta*dt).  525 is the PRODUCTION value -- mass-dominated, the regime F1
# showed is different in kind.  The 2D cavity ran STEADY.  Sweeping c separates
# "the FOSLS formulation pins the ratio" from "the mass-dominated regime does".
C = float(os.environ.get('CAV3D_C', 525.0))
TOL, MAXIT = 1e-8, 40000
ORDERS = tuple(int(v) for v in os.environ.get('CAV3D_N', '6,8,12,16,20').split(','))
OUT = os.path.join(_R, 'scratch', f'pmg3d_cavity_kz0_c{int(float(os.environ.get("CAV3D_C", 525)))}.npz')


def ladder(N):
    seq, p = [N], N
    while p > 2:
        p = max(2, p//2)
        seq.append(p)
    return tuple(seq)


def case(N):
    m = build_channel(1.0, 1.0, EX, EX, N, bcs=(1, 1, 1, 2))
    D = diff_matrix(N)
    kz = np.zeros(1)                       # k_z = 0 only -- the 2D cavity
    mask = BC.build_mask(m, 1, pin_p=True)
    shape = (m.nelem, N+1, N+1, OP.NVAR_R, 1)
    rw = OP.momentum_row_weights(C)
    kw = dict(mesh=m, mask=mask, wq=m.wq, kap=0.0, rw=rw)
    xs = S3.make_continuous(m, np.random.default_rng(0).standard_normal(shape))*mask
    b = S3.normal_op(xs, D, m.facx, m.facy, kz, NU, C, **kw)
    Mi = S3.jacobi_inverse(S3.jacobi_diagonal_analytic(shape, D, m.facx, m.facy,
                                                       kz, NU, C, **kw), mask)
    return m, D, kz, mask, rw, b, Mi, shape


def main():
    backend.set_backend('numba')
    print(f'3D code, Ghia cavity at k_z=0, Re={RE:.0f}, {EX}x{EX} elements, '
          f'c={C:g}, tol {TOL:g}, numba backend\n')
    print(f'{"N":>3}{"gDOF":>9}{"ladder":>16}{"jacobi":>9}{"pmg":>7}'
          f'{"ratio":>8}{"jac s":>9}{"pmg s":>9}{"vs jac":>8}')
    rows = []
    for N in ORDERS:
        m, D, kz, mask, rw, b, Mi, shape = case(N)
        t0 = time.perf_counter()
        _, itj, _ = S3.pcg(b, D, m.facx, m.facy, kz, NU, C, m, mask, Mi, TOL,
                           MAXIT, None, m.wq, 0.0, rw)
        tj = time.perf_counter() - t0
        itj = int(np.max(itj)) if np.ndim(itj) else int(itj)

        orders = ladder(N)
        try:
            pre = P3.PMG(m, 1, 1, NU, C, kz, kap=0.0, rw=rw, orders=orders,
                         deg=6, coarse_deg=10, pin_p=True,
                         direct_coarse=os.environ.get('CAV3D_DC', 'element'),
                         mask=mask)
            t0 = time.perf_counter()
            _, itp, _ = S3.pcg(b, D, m.facx, m.facy, kz, NU, C, m, mask, pre,
                               TOL, MAXIT, None, m.wq, 0.0, rw)
            tp = time.perf_counter() - t0
            itp = int(np.max(itp)) if np.ndim(itp) else int(itp)
        except Exception as e:
            print(f'{N:3d}  PMG failed: {type(e).__name__}: {str(e)[:60]}')
            continue
        g = int(mask.sum())
        print(f'{N:3d}{g:9d}{str(orders):>16}{itj:9d}{itp:7d}{itj/max(itp,1):7.1f}x'
              f'{tj:8.1f}s{tp:8.1f}s{tj/max(tp,1e-9):7.2f}x', flush=True)
        rows.append((N, g, itj, itp, tj, tp))
        np.savez_compressed(OUT, rows=np.array(rows, dtype=float),
                            cols=['N', 'freeDOF', 'jacobi', 'pmg', 't_jac', 't_pmg'])
    if len(rows) >= 2:
        r = np.array(rows, dtype=float)
        print(f'\n  growth N={int(r[0,0])}->{int(r[-1,0])}:  '
              f'jacobi {r[-1,2]/r[0,2]:5.2f}x   pmg {r[-1,3]/r[0,3]:5.2f}x')
        print(f'  iteration ratio: {r[0,2]/r[0,3]:.1f}x -> {r[-1,2]/r[-1,3]:.1f}x')
        print('\n  2D cavity for contrast (PMG_ALGORITHM sec 6.9): jacobi 4.00x, '
              'ladder 1.05x')
        print('  3D channel  (3D_STATUS sec 7K):  ratio PINNED at 7.3-7.4x')


if __name__ == '__main__':
    main()
