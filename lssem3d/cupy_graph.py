"""CUDA-graph-captured PCG for the CuPy backend.

WHY.  Measured on a Colab A100 (`CUPY_BACKEND.md`), one `normal_op` costs
**11.45 ms regardless of problem size** -- 0.53 M dof and 6.17 M dof time
identically, a 12x range in work with no change in wall clock.  That flatness
is the signature of a DISPATCH-bound loop: the ~200 Python-level array
operations behind one matvec cost more to *issue* than the GPU takes to run
them.  Inferred from the traffic (2.5 GB per matvec) and the measured
bandwidth (1356 GB/s), the A100's real work is ~1.85 ms, so **84% of the wall
clock is the host talking, not the GPU computing**.  No faster GPU fixes that.

CG replays an identical kernel sequence thousands of times, which is the
textbook case for CUDA graph capture: record the sequence once, replay it with
a single launch.

THE CONSTRAINT THAT SHAPES THE DESIGN.  Capture forbids host synchronisation
inside the captured region, and the reference `pcg` reads the residual to disk
-- to *host* -- every iteration to test convergence.  So this captures a BATCH
of `batch` iterations and tests convergence only at batch boundaries.  The cost
is over-solving by up to `batch`-1 iterations at the tail; the gain is removing
the per-iteration host cost from all the others.

TWO RULES, both of which produce wrong answers rather than errors if broken:

 1. **Fixed buffers.**  Every captured operation must write into
    pre-allocated arrays, because replay re-executes the recorded kernels
    against the recorded POINTERS.  `x = x + alpha*p` rebinds a name to fresh
    memory and the graph would keep updating the old block; `x += alpha*p`
    mutates in place, which is what capture needs.
 2. **Warm up before capturing.**  `cudaMalloc` is illegal during capture, but
    a CuPy memory-pool hit is not a CUDA call at all.  Running the loop once
    beforehand populates the pool so every allocation inside the capture is a
    cached pointer bump.  Skipping this is the usual way graph capture fails.

CORRECTNESS IS NOT ASSUMED.  `scratch/cupy_graph_check.py` solves the same
system with `solver3d.pcg` and with this, and compares -- the graph path has
to agree with the reference to solver tolerance, or it is not used.
"""
import cupy as cp

from . import device as DEV
from . import solver3d as S3

SPATIAL = (1, 2, 3)


def _dot_into(out, a, b, w):
    """out[...] = sum_over(a*b*w) -- the reference `_dot`, in place."""
    t = a*b if w is None else a*b*w
    out[...] = cp.sum(t, axis=SPATIAL).sum(axis=0)[None, None, None, None, :]


def pcg_graph(b, D, facx, facy, kz, nu, c, mesh=None, mask=None, M_inv=None,
              tol=1e-10, max_iter=2000, x0=None, wq=None, kap=0.0, rw=None,
              batch=20, capture=True):
    """Preconditioned CG with the inner loop replayed from a CUDA graph.

    Signature mirrors `solver3d.pcg`.  `batch` iterations run per convergence
    test; `capture=False` runs the identical arithmetic without a graph, which
    isolates the graph's contribution when benchmarking.
    """
    A = lambda v: S3.normal_op(v, D, facx, facy, kz, nu, c, mesh, mask, wq,
                               kap, rw)
    # to_device, exactly as the reference pcg does: multiplicity_weight
    # builds from np.ones and therefore comes back on the HOST.
    mw = (None if mesh is None else
          DEV.to_device(S3.multiplicity_weight(mesh, b.shape), b))
    P = (lambda r: r) if M_inv is None else (lambda r: r*M_inv)

    if mask is not None:
        b = b*mask
    x = cp.zeros_like(b) if x0 is None else x0.copy()
    if mask is not None:
        x *= mask

    r = b - A(x)
    z = P(r)
    p = z.copy()
    Ap = cp.zeros_like(b)
    sshape = (1, 1, 1, 1, b.shape[-1])
    rz = cp.zeros(sshape); rz_new = cp.zeros(sshape)
    denom = cp.zeros(sshape); alpha = cp.zeros(sshape); beta = cp.zeros(sshape)
    one = cp.ones(sshape)
    _dot_into(rz, r, z, mw)
    bn = cp.zeros(sshape); _dot_into(bn, b, b, mw)
    target = cp.maximum(tol*cp.sqrt(bn), 1e-300)

    def one_iteration():
        """In-place, buffer-stable -- see rule 1."""
        Ap[...] = A(p)
        _dot_into(denom, p, Ap, mw)
        alpha[...] = cp.where(cp.abs(denom) > 1e-300,
                              rz/cp.where(denom == 0, one, denom), 0.0*one)
        x[...] = x + alpha*p
        r[...] = r - alpha*Ap
        z[...] = P(r)
        _dot_into(rz_new, r, z, mw)
        beta[...] = cp.where(cp.abs(rz) > 1e-300,
                             rz_new/cp.where(rz == 0, one, rz), 0.0*one)
        p[...] = z + beta*p
        rz[...] = rz_new

    graph, it = None, 0
    while it < max_iter:
        if capture and graph is None:
            for _ in range(batch):              # rule 2: warm the pool
                one_iteration()
            it += batch
            rn = cp.sqrt(cp.zeros(sshape)); _dot_into(rn, r, r, mw)
            if bool(cp.all(cp.sqrt(rn) < target)):
                break
            st = cp.cuda.Stream(non_blocking=True)
            with st:
                st.begin_capture()
                for _ in range(batch):
                    one_iteration()
                graph = st.end_capture()
            continue
        if graph is not None:
            graph.launch()
        else:
            for _ in range(batch):
                one_iteration()
        it += batch
        rn = cp.zeros(sshape); _dot_into(rn, r, r, mw)
        if bool(cp.all(cp.sqrt(rn) < target)):
            break

    rn = cp.zeros(sshape); _dot_into(rn, r, r, mw)
    return x, it, cp.sqrt(rn).ravel()
