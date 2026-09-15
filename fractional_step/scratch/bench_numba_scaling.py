"""numba vs NumPy speedup as a function of grid resolution.

Resolution moves two independent ways in a spectral element method, and they do
NOT have the same effect:

  p-refinement  raises the polynomial order   -> larger (n x n) contraction blocks
  h-refinement  adds elements                 -> more blocks, same block size

The fused kernel's advantage comes from avoiding BLAS call overhead on tiny
blocks and eliminating temporaries, so it should fade with p (BLAS gets
efficient) and be roughly flat in h (block size unchanged).  Sweeping only DOF
confounds the two.  This benchmark sweeps them separately.

Timing is best-of-`REPEAT` median-free: the minimum over repeats is used, which
is the standard choice for microbenchmarks since noise is one-sided.
"""
import os
import json
import time

import numpy as np

from lssem2d import backend
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState, apply_L, apply_LT
from lssem2d.mesh import build_channel
from lssem2d.solver import apply_A
from lssem2d import kernels_numba as K

SC = os.path.dirname(os.path.abspath(__file__))
REPEAT = 5


def _time(fn, reps):
    best = float('inf')
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        best = min(best, (time.perf_counter() - t0)/reps)
    return best*1e6            # microseconds


def bench(N, EX):
    mesh = build_channel(1.0, 1.0, EX, EX, N, bcs=(1, 1, 1, 2))
    st = SolverState(mesh, diff_matrix(N), nu=1e-3, dt=0.1, fac1=1.5)
    rng = np.random.default_rng(0)
    shp = (mesh.nelem, N+1, N+1)
    U = rng.standard_normal(shp + (4,))
    fu = np.ascontiguousarray(rng.standard_normal(shp))
    fv = np.ascontiguousarray(rng.standard_normal(shp))
    st.update_linearisation(fu, fv)
    ndof = mesh.nelem*(N+1)**2*4
    reps = max(3, min(300, int(3e6/ndof)))

    out = {}
    for be in ('numpy', 'numba'):
        backend.set_backend(be)
        if be == 'numba':
            K.warmup(st)
            st.update_linearisation(fu, fv)
        su = apply_L(st, U, fu, fv)
        for _ in range(5):
            apply_A(st, U, fu, fv, pin_p=False)
        out[be] = dict(
            L=_time(lambda: apply_L(st, U, fu, fv), reps),
            LT=_time(lambda: apply_LT(st, su, fu, fv), reps),
            A=_time(lambda: apply_A(st, U, fu, fv, pin_p=False), reps),
        )
    backend.set_backend('numpy')
    return ndof, mesh.nelem, out


P_SWEEP = [(p, 6) for p in (3, 4, 5, 6, 7, 8, 10, 12, 14, 16)]   # fixed 36 elements
H_SWEEP = [(8, ex) for ex in (2, 3, 4, 6, 8, 10, 12, 14)]        # fixed order 8

results = {'p': [], 'h': []}
for tag, sweep, label in (('p', P_SWEEP, 'p-refinement (36 elements fixed)'),
                          ('h', H_SWEEP, 'h-refinement (order 8 fixed)')):
    print(f"\n=== {label} ===")
    print(f"{'order':>6}{'elems':>7}{'DOF':>9}"
          f"{'numpy A':>10}{'numba A':>10}{'L':>8}{'LT':>8}{'A':>8}")
    for N, EX in sweep:
        ndof, ne, r = bench(N, EX)
        row = dict(N=N, EX=EX, nelem=ne, ndof=ndof,
                   sL=r['numpy']['L']/r['numba']['L'],
                   sLT=r['numpy']['LT']/r['numba']['LT'],
                   sA=r['numpy']['A']/r['numba']['A'],
                   tA_np=r['numpy']['A'], tA_nb=r['numba']['A'])
        results[tag].append(row)
        print(f"{N:>6}{ne:>7}{ndof:>9}{r['numpy']['A']:>9.1f}u{r['numba']['A']:>9.1f}u"
              f"{row['sL']:>7.2f}x{row['sLT']:>7.2f}x{row['sA']:>7.2f}x")

with open(f'{SC}/bench_numba_scaling.json', 'w') as fh:
    json.dump(results, fh, indent=1)
print(f"\nsaved {SC}/bench_numba_scaling.json")
