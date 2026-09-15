"""Can the per-mode solve be parallelised across cores?  Threads vs processes.

    uv run --quiet python scratch/prof3d_modes.py [nz] [maxworkers]

prof3d.py established that ONE thing matters: normal_op is 99.4% of a step, and
BLAS threading inside it buys nothing (95.51 -> 94.84 ms from 1 to 8 threads).
So the parallel axis is across k_z MODES, which are independent in the solve --
no communication at all until the FFT for convection, itself 0.6% of the step.

Two mechanisms, and the choice is not obvious a priori:
  THREADS   -- free sharing, no pickling; works only if numpy releases the GIL
               during the contractions (einsum/matmul do; small ops may not).
  PROCESSES -- immune to the GIL, but each call ships ~8 MB of state.

Measured here rather than assumed.  Correctness is checked too: chunking the
mode axis must give BITWISE the same answer as the monolithic call, since the
modes never interact.
"""
import os, sys, time
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '1'          # one BLAS thread per worker; we parallelise modes
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem3d import operator as OP, solver3d as S3, bc as BC, fourier as FR

RE, EX, N = 1000.0, 6, 10
NU, C, KAP = 1.0/RE, 2500.0, 2500.0


def chunks(nk, w):
    """Contiguous mode slices, as even as possible."""
    edges = np.linspace(0, nk, w+1).round().astype(int)
    return [slice(a, b) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def main(nz=32, maxw=16):
    n = N+1
    mesh = build_channel(1.0, 1.0, EX, EX, N, bcs=(1, 1, 1, 2))
    D = diff_matrix(N)
    nk = nz//2 + 1
    kz = FR.wavenumbers(nz, 2*np.pi)
    mask = BC.build_mask(mesh, nk, pin_p=True)
    U = np.random.default_rng(0).standard_normal(
        (mesh.nelem, n, n, OP.NVAR_R, nk))*mask

    def op(sl):
        return S3.normal_op(U[..., sl], D, mesh.facx, mesh.facy, kz[sl], NU, C,
                            mesh, mask[..., sl], mesh.wq, KAP)

    ref = op(slice(None))
    t0 = time.perf_counter(); op(slice(None)); t_ser = time.perf_counter()-t0

    print(f'Nz={nz}  modes={nk}  ({EX}x{EX} elements, N={N})')
    print(f'  serial (all modes in one call): {t_ser*1e3:7.1f} ms\n')
    print(f"{'workers':>8}{'chunks':>8}{'thread ms':>11}{'speedup':>9}"
          f"{'bitwise':>9}")
    for w in [x for x in (2, 4, 6, 8, 12, 16) if x <= maxw and x <= nk]:
        cs = chunks(nk, w)
        with ThreadPoolExecutor(max_workers=w) as ex:
            list(ex.map(op, cs))                       # warm
            t0 = time.perf_counter()
            out = list(ex.map(op, cs))
            t_par = time.perf_counter()-t0
        got = np.concatenate(out, axis=-1)
        exact = np.array_equal(got, ref)
        print(f'{w:>8}{len(cs):>8}{t_par*1e3:>11.1f}{t_ser/t_par:>8.2f}x'
              f'{("yes" if exact else "NO"):>9}')
    print('\n  bitwise = chunked result identical to the monolithic one;')
    print('  it must be, since modes never interact in the solve.')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 32,
         int(sys.argv[2]) if len(sys.argv) > 2 else 16)
