"""Tune the PMG V-cycle: which (deg, coarse_deg, hierarchy) actually pays?

    uv run --quiet python scratch/pmg_sweep.py [N] [ex] [nz]

PMG gives 10-12x fewer CG iterations but 0.6-0.7x wall time, because a V-cycle
costs ~22 operator applications with the 2D default degrees (6 smoother, 10
coarse).  Those defaults were copied verbatim and never tuned for this operator.

The knobs pull against each other, which is why this is a sweep and not a guess:

  deg         the pre/post smoother -- 2*deg fine-level applications per cycle,
              the dominant cost.  Lower deg -> cheaper cycle, more cycles.
  coarse_deg  runs on the COARSE mesh, so each application is far cheaper.
              Raising it may be nearly free and may replace fine-level work.
  orders      the p-hierarchy.  A deeper hierarchy makes the coarse level
              cheaper still; a two-level cycle avoids one transfer.

Reported per configuration: CG iterations, wall time, and -- the number that
decides adoption -- speed relative to Jacobi on the SAME problem.
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

NU, C, TOL = 1/180., 525., 1e-6


def main(N=8, ex=3, nz=16):
    m = build_channel(1., 1., ex, ex, N, bcs=(1, 1, 1, 2))
    D = diff_matrix(N)
    nk = nz//2 + 1
    kz = FR.wavenumbers(nz, 2*np.pi)
    mask = BC.build_mask(m, nk, pin_p=True, nz=nz)
    shape = (m.nelem, N+1, N+1, OP.NVAR_R, nk)
    rw = OP.momentum_row_weights(C)
    kw = dict(mesh=m, mask=mask, wq=m.wq, kap=0.0, rw=rw)
    xs = S3.make_continuous(m, np.random.default_rng(0).standard_normal(shape))*mask
    b = S3.normal_op(xs, D, m.facx, m.facy, kz, NU, C, **kw)

    Mi = S3.jacobi_inverse(S3.jacobi_diagonal_analytic(
        shape, D, m.facx, m.facy, kz, NU, C, **kw), mask)
    t0 = time.perf_counter()
    _, itj, _ = S3.pcg(b, D, m.facx, m.facy, kz, NU, C, m, mask, Mi, TOL,
                       20000, None, m.wq, 0., rw)
    tj = time.perf_counter()-t0
    print(f'N={N} {ex}x{ex} nz={nz}   Jacobi baseline: {itj} its, {tj:.1f}s\n')
    print(f"{'orders':>12}{'deg':>5}{'cdeg':>6}{'build':>7}{'its':>7}"
          f"{'solve s':>9}{'total':>8}{'vs jacobi':>11}")

    hier = [(N, N//2, 2)] + ([(N, 2)] if N > 2 else [])
    if N >= 8:
        hier.append((N, N//2, N//4, 2) if N//4 > 2 else (N, N//2, 2))
    out, best = [], None
    seen = set()
    for orders in hier:
        if orders in seen:
            continue
        seen.add(orders)
        for deg in (1, 2, 3, 4, 6):
            for cdeg in (4, 10, 20):
                t0 = time.perf_counter()
                try:
                    P = PC.PMG(m, nk, nz, NU, C, kz, kap=0., rw=rw,
                               orders=orders, deg=deg, coarse_deg=cdeg)
                except Exception as e:
                    print(f'{str(orders):>12}{deg:>5}{cdeg:>6}   skipped: {e}')
                    continue
                tb = time.perf_counter()-t0
                t0 = time.perf_counter()
                _, itp, _ = S3.pcg(b, D, m.facx, m.facy, kz, NU, C, m, mask, P,
                                   TOL, 20000, None, m.wq, 0., rw)
                tp = time.perf_counter()-t0
                tot = tb + tp
                gain = tj/tot
                out.append(dict(orders=list(orders), deg=deg, cdeg=cdeg,
                                its=int(itp), solve=tp, total=tot, gain=gain))
                if best is None or gain > best['gain']:
                    best = out[-1]
                print(f'{str(orders):>12}{deg:>5}{cdeg:>6}{tb:>7.2f}{itp:>7}'
                      f'{tp:>9.1f}{tot:>8.1f}{gain:>10.2f}x', flush=True)
    print(f"\n  BEST: orders={best['orders']} deg={best['deg']} "
          f"coarse_deg={best['cdeg']} -> {best['gain']:.2f}x vs Jacobi "
          f"({best['its']} its vs {itj})")
    with open(f'scratch/pmg_sweep_N{N}_nz{nz}.json', 'w') as f:
        json.dump(dict(jacobi_its=int(itj), jacobi_s=tj, runs=out), f, indent=1)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8,
         int(sys.argv[2]) if len(sys.argv) > 2 else 3,
         int(sys.argv[3]) if len(sys.argv) > 3 else 16)
