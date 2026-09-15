"""Threads plateau at ~3.9x.  Is that the GIL, or memory bandwidth?

    uv run --quiet python scratch/prof3d_procs.py [nz]

prof3d_modes.py found thread-parallel modes bitwise-correct but saturating near
6 workers on a 16-core machine.  Two candidate causes, with OPPOSITE remedies:

  GIL       -- normal_op is a chain of many SMALL numpy ops (masks, scales,
               adds).  Big einsums release the GIL; small ops hold it, so the
               serial fraction is interpreter time.  Processes would fix it.
  BANDWIDTH -- the kernel is memory-bound, and more workers just contend for
               the same DRAM.  Processes would NOT help at all.

Distinguishing them decides the whole parallel architecture, so measure both:
  * processes at the same widths (fork, so workers inherit arrays -- no pickling)
  * a larger Nz, giving each worker more modes per chunk; if the plateau is
    per-call interpreter overhead it should move, if bandwidth it should not.
"""
import os, sys, time
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '1'
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem3d import operator as OP, solver3d as S3, bc as BC, fourier as FR

RE, EX, N = 1000.0, 6, 10
NU, C, KAP = 1.0/RE, 2500.0, 2500.0
G = {}                       # inherited by forked workers; never pickled


def chunks(nk, w):
    e = np.linspace(0, nk, w+1).round().astype(int)
    return [slice(a, b) for a, b in zip(e[:-1], e[1:]) if b > a]


def op(sl):
    g = G
    return S3.normal_op(g['U'][..., sl], g['D'], g['fx'], g['fy'], g['kz'][sl],
                        NU, C, g['mesh'], g['mask'][..., sl], g['wq'], KAP)


def setup(nz):
    n = N+1
    mesh = build_channel(1.0, 1.0, EX, EX, N, bcs=(1, 1, 1, 2))
    nk = nz//2 + 1
    mask = BC.build_mask(mesh, nk, pin_p=True)
    G.update(mesh=mesh, D=diff_matrix(N), fx=mesh.facx, fy=mesh.facy,
             kz=FR.wavenumbers(nz, 2*np.pi), wq=mesh.wq, mask=mask,
             U=np.random.default_rng(0).standard_normal(
                 (mesh.nelem, n, n, OP.NVAR_R, nk))*mask)
    return nk


def run(nz):
    nk = setup(nz)
    ref = op(slice(None))
    t0 = time.perf_counter(); op(slice(None)); ser = time.perf_counter()-t0
    print(f'\nNz={nz}  modes={nk}   serial {ser*1e3:.1f} ms   '
          f'({EX}x{EX} elements, N={N})')
    print(f"{'workers':>8}{'modes/wk':>10}{'threads':>10}{'speedup':>9}"
          f"{'procs':>10}{'speedup':>9}{'ok':>5}")
    widths = [w for w in (2, 4, 6, 8, 12, 16) if w <= nk]
    for w in widths:
        cs = chunks(nk, w)
        with ThreadPoolExecutor(max_workers=w) as ex:
            list(ex.map(op, cs))
            t0 = time.perf_counter(); out = list(ex.map(op, cs))
            th = time.perf_counter()-t0
        ok = np.array_equal(np.concatenate(out, axis=-1), ref)
        ctx = mp.get_context('fork')          # inherit G, no pickling of inputs
        with ctx.Pool(w) as pool:
            pool.map(op, cs)
            t0 = time.perf_counter(); out2 = pool.map(op, cs)
            pr = time.perf_counter()-t0
        ok = ok and np.array_equal(np.concatenate(out2, axis=-1), ref)
        print(f'{w:>8}{nk/w:>10.1f}{th*1e3:>9.1f}m{ser/th:>8.2f}x'
              f'{pr*1e3:>9.1f}m{ser/pr:>8.2f}x{("y" if ok else "NO"):>5}')


if __name__ == '__main__':
    for nz in ([int(sys.argv[1])] if len(sys.argv) > 1 else [32, 64]):
        run(nz)
    print('\n  processes >> threads  => the plateau was the GIL')
    print('  processes ~= threads  => memory bandwidth; more cores will not help')
