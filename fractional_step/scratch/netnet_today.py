"""The composed speedup from today's two changes, on ONE case.

Multiplying 5.4x (row-7) by 4.5x (numba) assumes they are independent.  They
should be -- one cuts ITERATIONS, the other cuts COST PER ITERATION -- but that
is an argument, and this session has been burned by arguments.  So the third leg
is measured on exactly the rig the first two were measured on
(scratch/validate_row7_stage5.json, same machine, same day):

    w7 = 1,    numpy   2277.5 s / 60 steps   3577.0 CG/step
    w7 = 1e-4, numpy    421.3 s / 60 steps    649.2 CG/step
    w7 = 1e-4, numba    <- this script

The physics is checked too: E/E0 and the mean error must match the numpy legs,
or the speedup is being bought with a different answer.
"""
import os, sys, json, time
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
SC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SC)
sys.path.insert(0, ROOT); sys.path.insert(0, SC); os.chdir(ROOT)
import numpy as np
from lssem3d import backend
import channel3d_stage5 as S5

REF = json.load(open(f'{SC}/validate_row7_stage5.json'))


def main(nstep=60):
    backend.set_backend('numba')
    t0 = time.perf_counter()
    r = S5.run_case(0.01, 'perturbed', False, nstep=nstep, verbose=False,
                    rowweight=True)
    r['wall'] = time.perf_counter() - t0
    backend.set_backend('numpy')

    base, mid = REF['1.0'], REF['0.0001']
    print(f"{'config':>24} {'wall':>9} {'CG/step':>9} {'E/E0':>10} {'meanerr':>10}")
    for name, d in (('w7=1     numpy', base), ('w7=1e-4  numpy', mid),
                    ('w7=1e-4  numba', r)):
        print(f'{name:>24} {d["wall"]:8.1f}s {d["cg_per_step"]:9.1f} '
              f'{d["e_end"]/d["e0"]:10.6f} {d["meanerr_end"]:10.3e}')
    print(f'\n  row-7 alone   : {base["wall"]/mid["wall"]:.2f}x')
    print(f'  numba alone   : {mid["wall"]/r["wall"]:.2f}x')
    print(f'  COMPOSED      : {base["wall"]/r["wall"]:.2f}x'
          f'   (product of the two: '
          f'{(base["wall"]/mid["wall"])*(mid["wall"]/r["wall"]):.2f}x)')
    d = abs(r['e_end']/r['e0'] - mid['e_end']/mid['e0'])/(mid['e_end']/mid['e0'])
    print(f'  physics vs numpy leg: E/E0 differs {d:.2e}')
    json.dump({k: v for k, v in r.items() if not isinstance(v, (list, np.ndarray))},
              open(f'{SC}/netnet_today_numba.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
