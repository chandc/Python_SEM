"""Are the CG reductions the serialization point under numba?

THE HYPOTHESIS.  `solver3d._dot` is `np.sum`, and NumPy reductions do NOT
release the GIL.  `parallel.pcg` splits the z-modes over a ThreadPoolExecutor,
so with a NumPy matvec the GIL-releasing einsum let the threads overlap and the
GIL-bound reductions hid inside it.  Make the matvec 8x faster with a nogil
numba kernel and the reductions stop hiding: they become the bottleneck and the
threads contend on them.

That would explain the gap 3D_STATUS.md sec 7M leaves open -- numba worth 8.4x on
the bare kernels, 5.0x predicted per serial CG iteration, but only 2.4x measured
on the threaded channel run.

THE DISCRIMINATOR.  Run the same case at workers=1 and at workers=auto.

  * If the hypothesis holds, numba's gain should be MUCH better at workers=1
    (near the 5.0x the serial accounting predicts) than threaded -- i.e. threading
    HURTS numba relative to numpy, and numba's thread scaling should be visibly
    worse than numpy's.
  * If it fails, numba's gain should be roughly the same at both, and the
    missing factor is somewhere else entirely (convection, RHS assembly).

Either way this decides whether fusing `_dot` is worth building.
"""
import os, sys, json, time
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
SC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SC)
sys.path.insert(0, ROOT); sys.path.insert(0, SC); os.chdir(ROOT)
import numpy as np
from lssem3d import backend, parallel as PAR
import channel3d_stage5 as S5

NSTEP = 15
_pcg = PAR.pcg


def run(name, workers):
    # run_case does not thread `workers` through, so pin it at the one place it
    # is consumed rather than editing the rig.
    PAR.pcg = lambda *a, _w=workers, **kw: _pcg(*a, **{**kw, 'workers': _w})
    backend.set_backend(name)
    t0 = time.perf_counter()
    r = S5.run_case(0.01, 'perturbed', False, nstep=NSTEP, verbose=False,
                    rowweight=True)
    r['wall'] = time.perf_counter() - t0
    PAR.pcg = _pcg
    backend.set_backend('numpy')
    assert r['status'] == 'OK', r['status']
    return r


def main():
    out = {}
    for w in (1, None):
        for name in ('numpy', 'numba'):
            r = out[(name, w)] = run(name, w)
            print(f'  {name:>6}  workers={str(w):>4}  {r["wall"]:7.1f}s  '
                  f'{r["cg_per_step"]:7.1f} CG/step  '
                  f'E/E0={r["e_end"]/r["e0"]:.6f}', flush=True)

    s1n, s1b = out[('numpy', 1)]['wall'], out[('numba', 1)]['wall']
    sAn, sAb = out[('numpy', None)]['wall'], out[('numba', None)]['wall']
    print(f'\n  numba gain, workers=1    : {s1n/s1b:.2f}x   '
          f'(serial per-iteration accounting predicted ~5.0x)')
    print(f'  numba gain, workers=auto : {sAn/sAb:.2f}x')
    print(f'  thread scaling, numpy    : {s1n/sAn:.2f}x')
    print(f'  thread scaling, numba    : {s1b/sAb:.2f}x')
    verdict = ('CONFIRMED: threading costs numba its advantage -- the GIL-bound '
               'reductions are the bottleneck, fuse _dot'
               if s1n/s1b > 1.35*(sAn/sAb) else
               'REFUTED: numba gains about the same threaded and serial -- the '
               'missing factor is NOT the reductions')
    print(f'\n  {verdict}')
    json.dump({f'{k[0]}_w{k[1]}': {kk: vv for kk, vv in v.items()
                                   if not isinstance(vv, (list, np.ndarray))}
               for k, v in out.items()},
              open(f'{SC}/numba_gil_test.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
