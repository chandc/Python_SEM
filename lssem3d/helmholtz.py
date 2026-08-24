"""Scalar 2-D Helmholtz on the SEM mesh — the primitive of the fractional-step path.

Every solve in FRACTIONAL_STEP_PLAN.md reduces to one operator, batched over
fields and Fourier modes:

    A psi  =  lambda * M psi  +  mu * K psi

with M the (diagonal) mass matrix and K the stiffness, so that A corresponds to
`lambda - mu * grad^2_xy` in strong form.  The two uses are

    velocity Helmholtz   lambda = c_k + nu*kz^2,   mu = nu
    pressure Poisson     lambda = kz^2,            mu = 1

so lambda varies per mode and mu is a scalar.  Both are SPD for lambda >= 0,
and strictly positive-definite except for the pressure at kz = 0, whose
constant mode is the usual null space.

WHY THIS IS THE WHOLE POINT.  The VVP least-squares operator couples all seven
fields at every node, which is what defeated fast diagonalisation
(CUPY_BACKEND.md: fdm.py inverts each field block exactly, drops 37% of the
operator as inter-field coupling, and loses to plain Jacobi at every order).
Here there is NO coupling to drop: each field is an independent scalar
Helmholtz, and on a tensor-product element

    lambda * M1(x)M1/(fx*fy) + mu * [ (fx/fy) K1(x)M1 + (fy/fx) M1(x)K1 ]

is exactly the separable form FDM was designed for.  The element inverse is
transform -> divide -> transform back at O(N^4) instead of O(N^6), and it is
EXACT rather than an approximation.
"""
import numpy as np

from . import device as DEV
from . import deriv as DV
from . import fdm
from . import solver3d as S3


def apply(U, D, facx, facy, wq, lam, mu, mesh=None, mask=None):
    """A = lam*M + mu*K applied to (nelem, n, n, F, nk).

    `lam` broadcasts against the trailing mode axis; `mu` is a scalar.
    mesh/mask follow solver3d.normal_op: assemble then mask, so the result is
    continuous and the operator is symmetric on the constrained space.
    """
    if mask is not None:
        U = U*mask
    w = wq[..., None, None]
    out = lam*(w*U) + mu*(DV.ddxT(w*DV.ddx(U, D, facx), D, facx)
                          + DV.ddyT(w*DV.ddy(U, D, facy), D, facy))
    if mesh is not None:
        out = S3.gs(mesh, out)
    return out*mask if mask is not None else out


def jacobi_diagonal(shape, D, facx, facy, wq, lam, mu, mesh=None, mask=None):
    """WRONG for an ASSEMBLED operator -- use jacobi_diagonal_analytic.

    Kept only so the mistake is not repeated.  Setting `e[:, i, j] = 1` sets
    that LOCAL index in every element at once, and a node shared between two
    elements then carries 1 in one element's storage and 0 in the other's --
    a DISCONTINUOUS vector, which the assembled operator is not defined on.
    Measured against the true diagonal (probed one global dof at a time, via gs
    of a one-hot): this is 5.0e-01 wrong, the analytic form 2.1e-16.

    The failure is invisible without a reference: the numbers look plausible,
    are positive, and give a preconditioner that converges -- just a worse one.
    """
    import warnings
    warnings.warn('jacobi_diagonal probes with discontinuous vectors and is '
                  'wrong for an assembled operator; use '
                  'jacobi_diagonal_analytic', RuntimeWarning, stacklevel=2)
    n = shape[1]
    d = np.zeros(shape)
    e = np.zeros(shape)
    for i in range(n):
        for j in range(n):
            e[:] = 0.0
            e[:, i, j] = 1.0
            col = apply(e, D, facx, facy, wq, lam, mu, mesh, mask)
            d[:, i, j] = col[:, i, j]
    return d


def jacobi_diagonal_analytic(mesh, N, wq, lam, mu, nfield, nk, mask=None):
    """diag(lam*M + mu*K) in CLOSED FORM -- no probing.

    With M = diag(wq), wq_ij = w_i w_j /(fx fy), and K = Kx + Ky assembled from
    the same 1-D pieces the FDM factors use,

        Kx[(i,j),(i,j)] = (fx/fy) w_j  sum_p w_p D[p,i]^2
        Ky[(i,j),(i,j)] = (fy/fx) w_i  sum_q w_q D[q,j]^2

    so the whole diagonal is

        lam * w_i w_j/(fx fy)
      + mu * [ (fx/fy) w_j S[i] + (fy/fx) w_i S[j] ],   S[i] = sum_p w_p D[p,i]^2

    `jacobi_diagonal` above gets the same numbers by applying the operator to
    (N+1)^2 unit vectors, which is fine once but is (N+1)^2 matvecs of setup --
    and a multigrid smoother needs this at EVERY level, per dt, so the closed
    form is what makes that affordable.  It is also assembly-exact: the sum
    over elements sharing a node is just gather-scatter of the local diagonals.
    """
    from lssem2d.lgl import diff_matrix, lgl_weights
    w = lgl_weights(N)
    D = diff_matrix(N)
    S = np.einsum('p,pi->i', w, D*D)          # sum_p w_p D[p,i]^2
    fx = np.asarray(mesh.facx, dtype=float)
    fy = np.asarray(mesh.facy, dtype=float)
    lam = np.asarray(lam, dtype=float).reshape(-1)
    if lam.size == 1:
        lam = np.repeat(lam, nk)
    n1 = N + 1
    d = np.empty((mesh.nelem, n1, n1, nfield, nk))
    for e in range(mesh.nelem):
        m0 = 1.0/(fx[e]*fy[e])
        stiff = mu*((fx[e]/fy[e])*np.outer(S, w) + (fy[e]/fx[e])*np.outer(w, S))
        mass = m0*np.outer(w, w)
        for k in range(nk):
            d[e, :, :, :, k] = (stiff + lam[k]*mass)[:, :, None]
    d = S3.gs(mesh, d)                        # assemble: shared nodes sum
    return d*mask if mask is not None else d


def jacobi_inverse(d, mask=None):
    inv = np.where(np.abs(d) > 1e-300, 1.0/np.where(d == 0, 1.0, d), 0.0)
    return inv*mask if mask is not None else inv


def fdm_preconditioner(mesh, N, lam, mu, mask, nfield, nk, like=None,
                       assemble=True):
    """Element-block FDM inverse of lam*M + mu*K, EXACT per element.

    In the M1-orthonormal eigenbasis (S^T M1 S = I, S^T K1 S = Lam) the element
    block is diagonal with entries

        lam/(fx*fy)  +  mu * [ (fx/fy)*Lam_i + (fy/fx)*Lam_j ]

    `assemble=False` skips the gather-scatter and returns the RAW element-local
    inverse.  That is EXACT for the unassembled operator and is how the shifts
    are verified; it is not a usable preconditioner for an assembled system.

    GATHER-SCATTER IS OTHERWISE NOT OPTIONAL.  CG's vectors live in discontinuous element
    storage but are constrained to the continuous subspace, since `apply` ends
    with gs.  A purely element-local inverse takes continuous input to
    DISCONTINUOUS output and the iterates then leave the space the operator is
    defined on -- worth 40000 iterations against 170 when it was found on the
    VVP path.  With Q the scatter and Q^T = gs, this is Q^T A_loc^-1 Q, i.e.
    additive Schwarz.
    """
    S, ev = fdm.reference_factors(N)
    n1 = N + 1
    fx = np.asarray(mesh.facx, dtype=float)
    fy = np.asarray(mesh.facy, dtype=float)
    lam = np.asarray(lam, dtype=float).reshape(-1)
    if lam.size == 1:
        lam = np.repeat(lam, nk)
    li, lj = ev[:, None], ev[None, :]
    d = np.empty((mesh.nelem, n1, n1, nfield, nk))
    for e in range(mesh.nelem):
        blk = mu*((fx[e]/fy[e])*li + (fy[e]/fx[e])*lj)
        for k in range(nk):
            d[e, :, :, :, k] = (blk + lam[k]/(fx[e]*fy[e]))[:, :, None]
    # clamp to the block's own smallest POSITIVE entry: the kz = 0 pressure
    # block is singular (constant mode), and clamping to a fraction of the
    # block MAXIMUM instead would amplify that mode by ~1e12.
    pos = np.where(d > 0, d, np.inf)
    floor = pos.min(axis=(1, 2), keepdims=True)
    d = np.maximum(d, np.where(np.isfinite(floor), floor, 1e-300))
    dinv = 1.0/d
    Sd = DEV.to_device(S, like) if like is not None else S
    dinv = DEV.to_device(dinv, like) if like is not None else dinv

    def M(r):
        g = r*mask if mask is not None else r
        g = DEV.einsum('ip,eijvk->epjvk', Sd, g)
        g = DEV.einsum('jq,epjvk->epqvk', Sd, g)
        g = g*dinv
        g = DEV.einsum('ip,epqvk->eiqvk', Sd, g)
        g = DEV.einsum('jq,eiqvk->eijvk', Sd, g)
        if assemble:
            g = S3.gs(mesh, g)
        return g*mask if mask is not None else g

    return M


def solve(b, D, facx, facy, wq, lam, mu, mesh, mask, M, tol=1e-10,
          max_iter=4000, check_every=None):
    """PCG for A x = b, per mode, in the multiplicity-weighted inner product.

    Not solver3d.pcg: that one is wired to normal_op.  The inner product must
    carry 1/multiplicity so a node stored once per owning element counts once
    -- omitting it silently mis-weights every interface node and breaks the
    symmetry CG needs.
    """
    xp = DEV.xp(b)
    mw = DEV.to_device(S3.multiplicity_weight(mesh, tuple(b.shape)), b)
    A = lambda v: apply(v, D, facx, facy, wq, lam, mu, mesh, mask)
    if check_every is None:
        check_every = 1 if xp is np else 10

    # THE INNER PRODUCT AS A GEMM.  This reduction keeps only the mode axis, so
    # it produces nk outputs from the whole field -- the same starved shape that
    # measured 17x off bandwidth on the least-squares path and turned out to be
    # two thirds of that solve (CUPY_BACKEND.md Phase 5).  Written as
    # (1 x M) @ (M x nk) cuBLAS fills the card instead of leaving most of it
    # idle.  Kept on the reduction path for numpy, where it is not a problem.
    nk_ = b.shape[-1]
    M_ = b.size//nk_ if hasattr(b, 'size') else int(np.prod(b.shape[:-1]))
    if DEV.is_cupy(b):
        ones = S3._ones_row(M_, b)

        def dot(a, c):
            return (ones @ (a*c*mw).reshape(M_, nk_)).reshape(-1)
    else:
        def dot(a, c):
            return xp.sum(a*c*mw, axis=(0, 1, 2, 3))
    x = DEV.zeros_like(b)
    r = b - A(x)
    z = M(r)
    p = DEV.clone(z)
    rz = dot(r, z)
    bn = xp.sqrt(dot(b, b))
    target = xp.maximum(tol*bn, 1e-300)
    one = xp.ones_like(rz)
    it = 0
    for it in range(1, max_iter + 1):
        Ap = A(p)
        den = dot(p, Ap)
        al = xp.where(abs(den) > 1e-300, rz/xp.where(den == 0, one, den),
                      0.0*one)
        x = x + al*p
        r = r - al*Ap
        # TESTING CONVERGENCE COSTS A HOST SYNC: the CPU drains the queue,
        # waits, reads one bit and re-issues.  Every iteration, that stall can
        # dominate the iteration itself.  Testing every K costs at most K-1
        # extra iterations out of hundreds, and skips K-1 reductions too.
        if it % check_every == 0 or it == max_iter:
            if bool(xp.all(xp.sqrt(dot(r, r)) < target)):
                break
        z = M(r)
        rzn = dot(r, z)
        be = xp.where(abs(rz) > 1e-300, rzn/xp.where(rz == 0, one, rz), 0.0*one)
        p = z + be*p
        rz = rzn
    # TRUE residual, not the recursive one -- over thousands of iterations the
    # recursion drifts from b - A x, and CG then declares victory on a number
    # that no longer describes the iterate.
    rt = xp.sqrt(dot(b - A(x), b - A(x)))
    rel = float(xp.max(rt/xp.maximum(bn, 1e-300)))
    return x, it, rel
