"""Re-measure today's net-net with every leg in ONE process, back to back.

WHY THIS RE-RUN.  netnet_today.py took its numpy reference from
validate_row7_stage5.json -- a number measured in a DIFFERENT process, at a
different time, after ~45 minutes of thermal load from the w7=1 leg that ran
before it.  Comparing across that is exactly the kind of confound this project
keeps getting caught by, and numba_gil_test.py exposed it: a fresh 15-step run
put threaded numba at 3.64x, against the 2.36x that stored comparison implied.

So: all four legs, one process, same session, same thermal state.  Nothing is
carried in from a file.

Also measured: workers=1 vs the thread pool.  numba_gil_test.py found the pool
LOSING at this problem size (0.90x numpy, 0.77x numba) -- the documented 6.7x
mode-parallel speedup was at Nz=128 (65 modes); this case has 9.
"""
import os, sys, json, time
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
SC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SC)
sys.path.insert(0, ROOT); sys.path.insert(0, SC); os.chdir(ROOT)
import numpy as np
from lssem3d import backend, operator as OP, parallel as PAR
import channel3d_stage5 as S5

NSTEP = 15
_pcg = PAR.pcg
_rw = OP.momentum_row_weights


def run(name, workers, w7):
    # Patch the FUNCTION, not OP.ROW7_WEIGHT -- the default binds at definition
    # time, so reassigning the constant would silently compare a config to itself.
    OP.momentum_row_weights = lambda c, _f=_rw, _w=w7: _f(c, w7=_w)
    PAR.pcg = lambda *a, _w=workers, **kw: _pcg(*a, **{**kw, 'workers': _w})
    backend.set_backend(name)
    t0 = time.perf_counter()
    r = S5.run_case(0.01, 'perturbed', False, nstep=NSTEP, verbose=False,
                    rowweight=True)
    r['wall'] = time.perf_counter() - t0
    OP.momentum_row_weights, PAR.pcg = _rw, _pcg
    backend.set_backend('numpy')
    assert r['status'] == 'OK', r['status']
    return r


def main():
    cases = [('numpy', None, 1.0,    'w7=1     numpy  threaded  <- this morning'),
             ('numpy', None, 1e-4,   'w7=1e-4  numpy  threaded'),
             ('numba', None, 1e-4,   'w7=1e-4  numba  threaded'),
             ('numba', 1,    1e-4,   'w7=1e-4  numba  serial    <- best'),
             ('numpy', 1,    1.0,    'w7=1     numpy  serial')]
    out = {}
    for name, w, w7, label in cases:
        r = out[label] = run(name, w, w7)
        print(f'  {label:<42} {r["wall"]:7.1f}s  {r["cg_per_step"]:7.1f} CG/step  '
              f'E/E0={r["e_end"]/r["e0"]:.6f}', flush=True)
    base = out['w7=1     numpy  threaded  <- this morning']['wall']
    print()
    for label, r in out.items():
        print(f'  vs this morning: {base/r["wall"]:6.2f}x   {label}')
    json.dump({k: {kk: vv for kk, vv in v.items()
                   if not isinstance(vv, (list, np.ndarray))}
               for k, v in out.items()},
              open(f'{SC}/netnet_clean.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
