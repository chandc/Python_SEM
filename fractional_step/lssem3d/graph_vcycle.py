"""CUDA-graph replay of the consistent-projection V-cycle.

WHY THIS AND NOT THE CG LOOP.  Measured on an A100 (README_VENDORED.md), the
pressure solve costs ~75 ms per CG iteration, of which the V-cycle preconditioner
is 72.6 -- and the same operator E costs 2.371, 2.404 and 2.384 ms at polynomial
orders 8, 4 and 2, a ninefold range in degrees of freedom.  Flat to 1.4 %: the
launches are what is being paid for, not the arithmetic.

`cupy_graph.py` in this tree captures a BATCH of CG iterations, which it has to
do because CG's convergence test reads the residual to the host and a host
synchronisation is illegal inside a capture.  That is not necessary here.
`ConsistentPMG._v` has no host reads at all -- fixed recursion depth, fixed
Chebyshev degree, a dense coarse solve -- so the V-cycle alone can be captured,
and it is 97 % of the iteration.  No over-solving at the tail, no change to the
convergence criterion, and the CG loop is untouched.

THE TWO RULES THAT MAKE CAPTURE CORRECT RATHER THAN MERELY FAST:

 1. **Fixed pointers.**  Replay re-executes the recorded kernels against the
    recorded addresses.  The input is therefore copied into one pre-allocated
    buffer on every call, and the result is read from the array the capture pass
    produced -- whose reference is held for the lifetime of this object, so the
    pool cannot hand its memory to anything else.

 2. **A private memory pool, warmed before capture.**  `cudaMalloc` is illegal
    during capture, but a pool hit is not a CUDA call at all, so the V-cycle is
    run once beforehand to populate the pool.  The pool is private to this
    object because `_v` allocates its own temporaries: on the shared pool those
    blocks would be freed back after capture and could be handed to unrelated
    code, which would then be overwritten by every replay.  That failure is
    silent and produces wrong numbers, which is why it gets its own pool.

CORRECTNESS IS CHECKED, NOT ASSUMED.  `colab/fs_graph_check.py` compares this
against the uncaptured V-cycle on random vectors and refuses the speed-up if they
disagree beyond round-off.  The arithmetic is identical -- the same kernels in
the same order -- so agreement should be to the last bit, not merely to solver
tolerance.
"""
import cupy as cp


class GraphedVCycle:
    """Wraps a `ConsistentPMG` (or anything with `_v(r, 0)`) in a captured graph.

    Interface is the wrapped object's: `M(r) -> z`.  Falls back to the original
    call if capture is unavailable or fails, so a run never breaks because of it.
    """

    def __init__(self, pmg, like, verbose=False):
        self.pmg = pmg
        self.ok = False
        self._fallback = pmg.__call__
        try:
            self._capture(like)
            self.ok = True
            if verbose:
                print('  V-cycle: CUDA graph captured', flush=True)
        except Exception as e:                      # loudly, but not fatally
            print(f'  V-cycle: graph capture FAILED ({type(e).__name__}: {e}); '
                  f'falling back to the uncaptured path', flush=True)

    def _capture(self, like):
        self._pool = cp.cuda.MemoryPool()
        with cp.cuda.using_allocator(self._pool.malloc):
            self._in = cp.zeros(like.shape, dtype=like.dtype)
            self.pmg._v(self._in, 0)                # rule 2: warm the pool
            st = cp.cuda.Stream(non_blocking=True)
            with st:
                st.begin_capture()
                self._out = self.pmg._v(self._in, 0)
                self._graph = st.end_capture()
        self._st = st

    def __call__(self, r):
        if not self.ok:
            return self._fallback(r)
        self._in[...] = r                           # rule 1: fixed input buffer
        self._graph.launch()
        return self._out.copy()                     # caller may hold it; we reuse


def maybe_graph(pmg, like, enable=True, verbose=False):
    """Wrap `pmg` if graph capture is wanted and possible; otherwise return it.

    Returns the original object unchanged on the numpy backend, on a CuPy build
    without graph support, or for a preconditioner with no `_v` (the K path's
    HelmholtzPMG), so callers need no backend tests of their own.
    """
    if not enable or not hasattr(pmg, '_v'):
        return pmg
    try:
        import cupy as _cp
        if not isinstance(like, _cp.ndarray):
            return pmg
        if not hasattr(_cp.cuda.Stream, 'begin_capture'):
            print('  V-cycle: this CuPy has no stream capture; skipping', flush=True)
            return pmg
    except ImportError:
        return pmg
    g = GraphedVCycle(pmg, like, verbose=verbose)
    return g if g.ok else pmg
