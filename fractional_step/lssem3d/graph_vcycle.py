"""CUDA-graph replay of the consistent-projection V-cycle (torch backend).

WHY TORCH AND NOT CUPY.  CuPy refuses to record cuBLAS calls in a stream capture
("calling cuBLAS API during stream capture is currently unsupported"), and every
derivative in this solver is an einsum, which CuPy routes through cuBLAS -- so
there is no capturable subset at all.  PyTorch owns its cuBLAS handle and
workspace and captures them, which is why the driver gained a torch backend.
Moving to torch was itself worth 1.78x on an A100 (19.96 s/step to 11.22, and
uniformly across three phases that do completely different work, which is what a
per-launch overhead looks like).  This is the rest of that factor.

WHY THE V-CYCLE AND NOT THE CG LOOP.  The V-cycle is 11.12 of the 11.22 s step
and 97 % of a pressure iteration, and `ConsistentPMG._v` has no host reads --
fixed recursion depth, fixed Chebyshev degree, a dense coarse solve.  CG's
convergence test reads the residual to the host, so capturing the CG loop would
mean batching iterations and over-solving at the tail, which is what
`cupy_graph.py` had to do.  Capturing the preconditioner alone leaves the
iteration count and the convergence criterion exactly as they were.

THE RECIPE, and each step is load-bearing:

 1. **Warm up on a side stream.**  The first calls allocate, autotune cuBLAS and
    populate torch's caching allocator.  Capturing that would record one-off work
    into the graph, and cuBLAS workspace allocation during capture fails outright.
 2. **Static input buffer.**  Replay re-executes against recorded addresses, so
    the residual is copied into one tensor rather than passed by reference.
 3. **Hold the output.**  The tensor the capture pass produced lives in the
    graph's private pool and is overwritten by every replay; it is returned as a
    clone so a caller may keep it across iterations, which CG does.

CORRECTNESS IS CHECKED, NOT ASSUMED -- `colab/fs_graph_check.py` compares against
the uncaptured V-cycle and refuses the speed-up unless they agree to round-off.
Replay runs the same kernels in the same order, so the bar is bit-exactness.
"""


class GraphedVCycle:
    """Wraps a `ConsistentPMG` in a captured CUDA graph.  Interface: `M(r) -> z`."""

    def __init__(self, pmg, like, warmup=3, verbose=False):
        import torch
        self.torch = torch
        self.pmg = pmg
        self.ok = False
        self._fallback = pmg.__call__
        try:
            self._capture(like, warmup)
            self.ok = True
            if verbose:
                print('  V-cycle: CUDA graph captured (torch)', flush=True)
        except Exception as e:
            print(f'  V-cycle: graph capture FAILED ({type(e).__name__}: {e}); '
                  f'falling back to the uncaptured path', flush=True)

    def _capture(self, like, warmup):
        torch = self.torch
        if not torch.cuda.is_available():
            raise RuntimeError('no CUDA device; capture is meaningless on the host')
        self._in = torch.zeros_like(like)

        side = torch.cuda.Stream()                       # rule 1
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(warmup):
                self.pmg._v(self._in, 0)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):              # rules 2 and 3
            self._out = self.pmg._v(self._in, 0)

    def __call__(self, r):
        if not self.ok:
            return self._fallback(r)
        self._in.copy_(r)
        self._graph.replay()
        return self._out.clone()


def maybe_graph(pmg, like, enable=True, verbose=False):
    """Wrap `pmg` if capture is wanted and possible; otherwise return it unchanged.

    Returns the original object on the numpy backend, on CuPy (which cannot
    capture cuBLAS), and for a preconditioner with no `_v` -- the K path's
    HelmholtzPMG -- so callers need no backend tests of their own.
    """
    if not enable or not hasattr(pmg, '_v'):
        return pmg
    try:
        import torch
    except ImportError:
        return pmg
    if not isinstance(like, torch.Tensor):
        if type(like).__module__.startswith('cupy'):
            print('  V-cycle: graph capture needs the torch backend '
                  '(cupy cannot capture cuBLAS); skipping', flush=True)
        return pmg
    g = GraphedVCycle(pmg, like, verbose=verbose)
    return g if g.ok else pmg
