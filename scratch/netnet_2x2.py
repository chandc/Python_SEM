"""The missing cell: w7 = 1 WITH numba, completing the 2x2.

netnet_today.py measured numba at 2.36x on this case, against 4.5x on the Stokes
benchmark.  Two stories fit:

  (a) size/order effect -- N=6 here vs N=8 there;
  (b) AMDAHL -- the row-7 fix already cut iterations 5.5x, so the matvec is a
      much smaller fraction of the step than it used to be, and numba has less
      left to attack.  The two changes would then COMPETE for the same time,
      not compose.

They are distinguishable: under (b) numba should be worth much MORE at w7 = 1,
where the run is almost pure matvec (normal_op was profiled at 99.4% of a step
in that regime).  Under (a) it should be worth the same at both.

This runs the missing cell so the 2x2 can be read off instead of argued about.
"""
import os, sys, json, time
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
SC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SC)
sys.path.insert(0, ROOT); sys.path.insert(0, SC); os.chdir(ROOT)
import numpy as np
from lssem3d import backend, operator as OP
import channel3d_stage5 as S5


def main(nstep=60):
    # Patch the FUNCTION, not OP.ROW7_WEIGHT: `def f(c, w7=ROW7_WEIGHT)` binds
    # the default at DEFINITION time, so reassigning the constant changes
    # nothing and the A/B silently compares a configuration against itself.
    orig = OP.momentum_row_weights
    OP.momentum_row_weights = lambda c, _f=orig: _f(c, w7=1.0)
    backend.set_backend('numba')
    t0 = time.perf_counter()
    r = S5.run_case(0.01, 'perturbed', False, nstep=nstep, verbose=False,
                    rowweight=True)
    r['wall'] = time.perf_counter() - t0
    OP.momentum_row_weights = orig
    backend.set_backend('numpy')
    print(f"w7=1 numba: {r['wall']:.1f}s  {r['cg_per_step']:.1f} CG/step  "
          f"E/E0={r['e_end']/r['e0']:.6f}")
    json.dump({k: v for k, v in r.items() if not isinstance(v, (list, np.ndarray))},
              open(f'{SC}/netnet_2x2_w1_numba.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
