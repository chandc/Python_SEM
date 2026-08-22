"""cuPyNumeric vs PyTorch on the LSSEM matvec, FP64, same machine and image.

    docker run --rm --gpus all --ipc=host -v cpn_matvec.py:/m.py:ro \
        cpn-spark:latest /opt/cpn/bin/python /m.py

THE QUESTION.  cuPyNumeric is a drop-in NumPy replacement, and that is exactly
what makes it interesting AND what makes it suspect here.  3D_STATUS.md sec 7M
measured the win as coming from FUSION -- collapsing ~30 passes over the state
into one -- because the operator is memory-bandwidth bound:

    apply_L + apply_LT at 88^3, Mac:   numpy 450.1 ms  ->  numba 59.6 ms  (7.6x)

A drop-in replacement preserves the ~30-pass structure and runs each pass on the
GPU.  If Legate's lazy evaluation fuses them, cuPyNumeric wins on effort by a
mile.  If it does not, it is fast-per-pass and still thirty passes.  Nobody
should guess this -- the whole point of running it.

TWO MEASUREMENTS, deliberately:

  A. THE SAME 4-EINSUM KERNEL the torch benchmark timed, so the numbers are
     directly comparable against 1.6 ms (48^3) and 10.1 ms (88^3).

  B. THE UNFUSED, PER-FIELD FORM -- 14 separate einsums plus row assembly, i.e.
     what operator.py's NumPy path actually executes.  This is the real drop-in
     question.  The GAP BETWEEN A AND B is the fusion penalty, and it is the
     number that decides between the two libraries.

dtypes are asserted before timing (lesson L15: MLX silently downcast float64 to
float32 and inverted a conclusion).
"""
import os
import time

import numpy as _np          # data generation only -- see note in main()

BACKEND = os.environ.get('CPN_BACKEND', 'cupynumeric')
if BACKEND == 'cupynumeric':
    import cupynumeric as np
else:
    import numpy as np

SHAPES = [('48^3 (ref)', 36, 9, 25), ('88^3', 121, 9, 45)]
TORCH_MS = {'48^3 (ref)': 1.6, '88^3': 10.1}       # measured, same image/machine
NUMBA_MAC_MS = {'48^3 (ref)': 9.4, '88^3': 59.6}   # measured, M3 Max, FUSED


def _sync():
    """Legate is lazy; force completion or the timings are meaningless."""
    if BACKEND == 'cupynumeric':
        try:
            import legate.core as lg
            lg.get_legate_runtime().issue_execution_fence(block=True)
        except Exception:
            pass


def _time(f, reps):
    r = f(); _sync()
    t = time.perf_counter()
    for _ in range(reps):
        r = f()
    _sync()
    return (time.perf_counter() - t)/reps, r


def batched(U, D, reps=5):
    """A: all 14 fields in 4 batched contractions -- the torch formulation."""
    def once():
        ux = np.einsum('pi,eijvk->epjvk', D, U)
        uy = np.einsum('qj,eijvk->eiqvk', D, U)
        R = ux + uy
        return (np.einsum('pi,epjvk->eijvk', D, R)
                + np.einsum('qj,eiqvk->eijvk', D, R))
    return _time(once, reps)[0]


def unfused(U, D, reps=5):
    """B: per-field derivatives + row assembly -- operator.py's actual shape."""
    nv = U.shape[3]

    def once():
        d = []
        for f in range(nv):                       # 14 fields x 2 directions
            Uf = U[:, :, :, f, :]
            d.append(np.einsum('pi,eijk->epjk', D, Uf))
            d.append(np.einsum('qj,eijk->eiqk', D, Uf))
        rows = [d[2*f] + d[2*f + 1] for f in range(nv)]   # 8-row assembly stand-in
        out = rows[0]
        for r in rows[1:]:
            out = out + r
        return out
    return _time(once, reps)[0]


def main():
    print(f'backend: {BACKEND}   (module {np.__name__})')
    print(f'{"shape":>12} {"dof":>8} {"A batched":>11} {"B unfused":>11} '
          f'{"B/A":>6} {"torch A":>9} {"Mac numba":>10}')
    # Arrays are generated with STOCK numpy and converted.  Not incidental:
    # cupynumeric's Generator has no standard_normal --
    #   AttributeError: 'Generator' object has no attribute 'standard_normal'
    # -- which is itself a data point about how complete "drop-in" is.  Building
    # the data identically on both backends also removes it as a variable.
    for tag, nel, n, nk in SHAPES:
        U = np.asarray(_np.random.default_rng(0).standard_normal((nel, n, n, 14, nk)))
        D = np.asarray(_np.random.default_rng(1).standard_normal((n, n)))
        assert U.dtype == np.float64 and D.dtype == np.float64, U.dtype
        ta = batched(U, D)
        tb = unfused(U, D)
        print(f'{tag:>12} {nel*n*n*14*nk/1e6:6.2f}M {ta*1e3:10.1f}ms '
              f'{tb*1e3:10.1f}ms {tb/ta:5.1f}x {TORCH_MS[tag]:8.1f}ms '
              f'{NUMBA_MAC_MS[tag]:9.1f}ms')
    print('\nA = 4 batched einsums (what a torch port would write)')
    print('B = 14 per-field einsums + assembly (what operator.py actually runs)')
    print('B/A is the fusion penalty a drop-in replacement would inherit.')


if __name__ == '__main__':
    main()
