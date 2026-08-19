"""Mode-parallel solve: the one place this algorithm is embarrassingly parallel.

Fourier modes do not talk to each other inside the implicit solve.  Every scalar
in `solver3d.pcg` -- alpha, beta, rz -- comes from `_dot`, which reduces over
SPATIAL and over elements and keeps the mode axis, so each mode already runs its
own independent CG recurrence.  The modes are coupled ONLY by the convective
term, which needs physical space, and that is 0.6% of a step (scratch/prof3d.py).

WHAT THE MEASUREMENTS SAID  (scratch/prof3d*.py, 12P+4E cores)

  * `normal_op` is 99.4% of a step; FFT and gather-scatter are ~0.6 ms each
    against a 95 ms matvec.  Only the matvec is worth parallelising.
  * BLAS threading inside the matvec buys NOTHING: 95.51 -> 94.84 ms going from
    1 to 8 threads.  The contractions are too small.  Hence one BLAS thread per
    worker, and parallelism across modes instead.
  * threads == processes (6.7x vs 6.5x at Nz=128).  Since processes are immune
    to the GIL and did not beat threads, the ceiling is MEMORY BANDWIDTH, not
    the interpreter.  Threads win on simplicity: no pickling, no fork, shared
    arrays.  Do not "fix" this with multiprocessing -- it was tried and tied.
  * the ceiling RISES with problem size, because each worker gets more modes
    per chunk: 3.8x at 17 modes, 5.7x at 33, 6.7x at 65.  Expect better than
    6.7x at production Nz, and little from tiny mode counts.

WHY PARALLELISE THE WHOLE SOLVE, NOT EACH MATVEC.  Two reasons, both real:
  1. one thread dispatch per solve instead of one per CG iteration (~45x less
     dispatch overhead);
  2. `pcg` exits on `np.all(rn < target)`, so a mode that converged long ago
     keeps iterating until the WORST mode catches up.  Chunked, each chunk exits
     on its own modes.  High-k modes are strongly damped and converge fast, so
     this is not a rounding-level effect.

Consequence of (2): the result is NOT bitwise identical to the serial `pcg`,
because converged modes take a different number of extra iterations.  Each mode
still meets the same per-mode tolerance -- that is what `test_parallel.py`
asserts, rather than asserting bitwise equality it cannot have.  A single
matvec IS bitwise identical when chunked, and that is tested too.
"""
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from . import solver3d as S3

_POOL = None
_POOL_W = 0


def n_workers(nmode=None, cap=None):
    """Threads to use.  Defaults to the performance cores, never more than the
    number of modes -- a chunk with no modes in it is pure overhead.

    12 on this machine (12 P-cores + 4 E-cores); measured best at 12 for 65
    modes.  Override with LSSEM3D_WORKERS.
    """
    if cap is None:
        cap = int(os.environ.get('LSSEM3D_WORKERS', 0)) or _perf_cores()
    return max(1, min(cap, nmode if nmode else cap))


def _perf_cores():
    try:                                        # Apple silicon: skip E-cores
        import subprocess
        out = subprocess.run(['sysctl', '-n', 'hw.perflevel0.logicalcpu'],
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            return int(out.stdout.strip())
    except Exception:
        pass
    return max(1, (os.cpu_count() or 2) - 2)


def pool(workers):
    """One persistent pool.  Creating a ThreadPoolExecutor per solve would pay
    thread-spawn cost on every stage of every step."""
    global _POOL, _POOL_W
    if _POOL is None or _POOL_W != workers:
        if _POOL is not None:
            _POOL.shutdown(wait=True)
        _POOL = ThreadPoolExecutor(max_workers=workers)
        _POOL_W = workers
    return _POOL


def shutdown():
    global _POOL, _POOL_W
    if _POOL is not None:
        _POOL.shutdown(wait=True)
    _POOL, _POOL_W = None, 0


def mode_chunks(nmode, workers):
    """Contiguous, near-equal slices of the mode axis.

    Contiguous rather than strided so each chunk is a view with unit stride on
    the fastest axis; strided chunks would force a copy in every matvec.
    """
    e = np.linspace(0, nmode, workers+1).round().astype(int)
    return [slice(a, b) for a, b in zip(e[:-1], e[1:]) if b > a]


def _sl(a, s):
    """Slice the mode axis of anything that has one; pass through None."""
    return None if a is None else a[..., s]


def apply_op(Ur, D, facx, facy, kz, nu, c, mesh=None, mask=None, wq=None,
             kap=0.0, workers=None):
    """Thread-parallel `solver3d.normal_op`.  Bitwise identical to the serial
    call -- the mode axis carries no cross-mode work, so splitting it is exact.
    """
    nk = Ur.shape[-1]
    w = n_workers(nk) if workers is None else min(workers, nk)
    if w <= 1:
        return S3.normal_op(Ur, D, facx, facy, kz, nu, c, mesh, mask, wq, kap)
    cs = mode_chunks(nk, w)
    fn = lambda s: S3.normal_op(Ur[..., s], D, facx, facy, kz[s], nu, c,
                                mesh, _sl(mask, s), wq, kap)
    return np.concatenate(list(pool(w).map(fn, cs)), axis=-1)


def pcg(b, D, facx, facy, kz, nu, c, mesh=None, mask=None, M_inv=None,
        tol=1e-10, max_iter=2000, x0=None, wq=None, kap=0.0, workers=None):
    """Thread-parallel `solver3d.pcg`, same signature plus `workers`.

    Returns (x, iters, resid) exactly as the serial version, with `iters` the
    WORST chunk's count -- that is the wall-clock-relevant number, and it keeps
    the value comparable to the serial one for reporting.

    Not bitwise-equal to serial (see the module docstring); per-mode tolerance
    is met identically, which is the property that matters.
    """
    nk = b.shape[-1]
    w = n_workers(nk) if workers is None else min(workers, nk)
    if w <= 1:
        return S3.pcg(b, D, facx, facy, kz, nu, c, mesh, mask, M_inv, tol,
                      max_iter, x0, wq, kap)
    cs = mode_chunks(nk, w)

    def solve(s):
        return S3.pcg(b[..., s], D, facx, facy, kz[s], nu, c, mesh,
                      _sl(mask, s), _sl(M_inv, s), tol, max_iter,
                      _sl(x0, s), wq, kap)

    out = list(pool(w).map(solve, cs))
    x = np.concatenate([o[0] for o in out], axis=-1)
    return x, max(o[1] for o in out), np.concatenate([o[2] for o in out])
