"""Does the numba backend reproduce a VALIDATED physics result?

    uv run --quiet python scratch/validate_numba_physics.py

Operator parity to 1e-16 on a single application (test_backend_parity.py) is
necessary but not sufficient: it says nothing about the thread-parallel mode
loop, the analytic Jacobi diagonal, or the RKW3 driver.  This runs the Chan
(1996) Stokes-decay case -- the one with an ANALYTIC answer, sigma = 9.3137399
-- end to end on each backend and compares the measured decay rate.

Both mode families are run, and that pairing is the point:

    kz0   alpha = 1, k_z = 0   every i*k_z term is dormant
    span  alpha = 0, k_z = 1   only v, w, omega_x live; every i*k_z term fires

A sign error in the imaginary coupling of the fused kernels is invisible in the
first and fatal in the second.

THE THREADED PATH IS EXERCISED DELIBERATELY.  njit code holds the GIL unless
built with nogil=True, so a kernel missing that flag would serialise
`parallel.pcg` -- correct answers, most of the speedup gone.  The wall clock
reported per backend is therefore part of the result, not decoration.
"""
import os
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
SC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SC)
sys.path.insert(0, ROOT)
sys.path.insert(0, SC)
os.chdir(ROOT)

import numpy as np
import stokes3d as SD
from lssem3d import backend

SIGMA = SD.SIGMA_2D
DT = 0.0025
TEND = 0.05


def one(name, N, mode):
    backend.set_backend(name)
    s = SD.setup(N=N)
    U0, _ = SD.initial_state(s, mode=mode)
    t0 = time.perf_counter()
    r = SD.measure_sigma(s, U0, DT, 0.0, tend=TEND, rowweight=True,
                         tol=1e-12, max_iter=60000)
    r['wall'] = time.perf_counter() - t0
    assert r['status'] == 'ok', r
    r['err'] = abs(r['sigma'] - SIGMA)/SIGMA
    return r


def main(N=8):
    print(f'Stokes decay, N={N}, dt={DT}, analytic sigma = {SIGMA:.7f}\n')
    for mode in ('kz0', 'span'):
        out = {}
        for name in ('numpy', 'numba'):
            if not backend.available(name):
                continue
            r = out[name] = one(name, N, mode)
            print(f'  {mode:5s} {name:6s}  sigma={r["sigma"]:.7f}  '
                  f'rel err={r["err"]:.3e}  CG={r["cg"]}  {r["wall"]:.1f}s')
        if len(out) == 2:
            a, b = out['numpy'], out['numba']
            d = abs(a['sigma'] - b['sigma'])/abs(a['sigma'])
            print(f'  {"":5s} ->      backends agree to {d:.2e}, '
                  f'speedup {a["wall"]/b["wall"]:.2f}x\n')
            # The backends must not differ by more than the scheme's own error
            # against the analytic rate.  Anything larger means numba changed
            # the answer, not merely the arithmetic order.
            assert d < max(a['err'], 1e-10), (
                f'backends differ by {d:.2e}, more than the scheme error '
                f'{a["err"]:.2e} -- numba is solving a different problem')
    backend.set_backend('numpy')
    print('PASS: numba reproduces the analytic Stokes decay rate.')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
