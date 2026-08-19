"""Batched per-mode normal-equation solve, and the RKW3/Crank-Nicolson driver.

NEW CODE.  lssem2d is not modified.

BATCHED, NOT LOOPED.  Every routine here takes the whole mode set at once --
arrays are (nelem, n, n, var, mode) with z last -- and the CG iterates on all
modes simultaneously.  A Python loop over k_z would throw away exactly the
decoupling that motivates the Fourier approach, three times per RKW3 step
(3D_DEVELOPMENT_PLAN.md sec 1.1).

Because the modes are independent, each carries its OWN convergence state.  The
inner products are therefore reduced over the spatial axes only, leaving a
per-mode scalar; a converged mode simply stops changing while its neighbours
continue.  Reducing over the mode axis as well -- the obvious mistake -- would
couple the modes through the stopping criterion and make the iteration count of
the worst mode apply to all of them.
"""
import numpy as np
from lssem2d.assembly import gather_scatter
from . import operator as OP
from .timestep import ALPHA, BETA, GAMMA, ZETA, NSTAGE

SPATIAL = (1, 2, 3)          # n, n, var -- reduce over these, keep (elem, mode)


def _dot(a, b, w=None):
    """Per-mode inner product: sum over space and fields, keep the mode axis.

    w is the multiplicity weight (see multiplicity_weight): without it every
    element-interface node is counted once per owning element and the inner
    product is not the one the assembled operator is symmetric in.
    """
    ab = a*b if w is None else a*b*w
    return np.sum(ab, axis=SPATIAL).sum(axis=0)[None, None, None, None, :]


def gs(mesh, U):
    """Gather-scatter Q^T Q over the (var, mode) batch.

    lssem2d.assembly.gather_scatter accepts 3-D or 4-D only; the 3D layout is
    5-D.  Q acts on the SPATIAL index alone, so folding (var, mode) into a
    single trailing axis reuses it exactly -- no modification to lssem2d, and no
    reimplementation of the connectivity.
    """
    nel, n, _, nv, nk = U.shape
    return gather_scatter(mesh, U.reshape(nel, n, n, nv*nk)).reshape(U.shape)


def multiplicity_weight(mesh, shape):
    """1/multiplicity, for the CG inner product.

    A node on an element interface is stored once per owning element, so a plain
    sum over the local array counts it twice (four times at a corner).  lssem2d
    divides by the multiplicity in every CG inner product "to guarantee
    symmetry"; the same is required here, and omitting it silently mis-weights
    every interface node.
    """
    mult = gs(mesh, np.ones(shape))
    return 1.0/np.where(mult < 1e-10, 1.0, mult)


def make_continuous(mesh, U):
    """Project a local array onto the C0-continuous subspace.

    Averages duplicated interface nodes.  Needed because the assembled operator
    annihilates the discontinuous part, so a manufactured solution built from
    random LOCAL values is not recoverable -- only its continuous projection is.
    Fields built by evaluating a smooth function at node coordinates are already
    continuous and need no projection.
    """
    mult = gs(mesh, np.ones_like(U))
    return gs(mesh, U)/np.where(mult < 1e-10, 1.0, mult)


def normal_op(Ur, D, facx, facy, kz, nu, c, mesh=None, mask=None, wq=None, kap=0.0):
    """A = M Q^T Q L0^T W L0 M applied to a split-real state.

    THE ASSEMBLY STEP IS NOT OPTIONAL.  Without gather_scatter the operator is
    element-LOCAL: interface nodes of neighbouring elements are independent
    degrees of freedom, C0 continuity is never imposed, and the system is
    massively under-determined.  Omitting it here made a fully boundary-
    conditioned cavity solve fail to converge in 20000 CG iterations, which is
    how it was found.  mesh=None reproduces the unassembled operator and is for
    single-element tests only.

    wq are the quadrature weights (mesh.wq).  They belong in the FORWARD
    operator only -- apply_LT is the unweighted transpose -- so that the product
    is the normal operator of J = int R^2 dOmega.  See operator.apply_L_complex.
    Passing wq=None solves a different (unweighted, nodal) least-squares problem
    and is for tests only.

    mask is 1 where a degree of freedom is free and 0 where it is prescribed;
    applying it on both sides keeps A symmetric, which is what CG needs.
    """
    if mask is not None:
        Ur = Ur*mask
    out = OP.apply_LT(OP.apply_L(Ur, D, facx, facy, kz, nu, c, wq, kap),
                      D, facx, facy, kz, nu, c, kap)
    if mesh is not None:
        out = gs(mesh, out)
    return out*mask if mask is not None else out


def jacobi_diagonal(shape, D, facx, facy, kz, nu, c, mesh=None, mask=None, wq=None, kap=0.0):
    """diag(L^T L) by probing with unit vectors.

    REFERENCE QUALITY, NOT PRODUCTION.  This costs one operator application per
    (node, field) -- 2*7*(N+1)^2 of them -- which is fine for validation and far
    too slow for a real run.  lssem2d earned a ~100x speed-up by replacing the
    same probing loop with an analytic diagonal (compute_jacobi_old ->
    compute_jacobi); the 3D analytic form is a later optimisation, and this
    routine is the thing it must be checked against when written.
    """
    diag = np.zeros(shape)
    n = shape[1]
    for f in range(OP.NVAR_R):
        for i in range(n):
            for j in range(n):
                e = np.zeros(shape)
                e[:, i, j, f, :] = 1.0
                diag[:, i, j, f, :] = normal_op(
                    e, D, facx, facy, kz, nu, c, mesh, mask, wq, kap)[:, i, j, f, :]
    return diag


def pcg(b, D, facx, facy, kz, nu, c, mesh=None, mask=None, M_inv=None,
        tol=1e-10, max_iter=2000, x0=None, wq=None, kap=0.0):
    """Preconditioned CG on A x = b, batched over modes.

    Returns (x, iters, resid) with resid the per-mode final residual norm.
    Convergence is per mode: a mode whose residual is already below tol
    contributes nothing further, and the loop exits when ALL modes are below.
    """
    x = np.zeros_like(b) if x0 is None else x0.copy()
    if mask is not None:
        b = b*mask
        x = x*mask
    A = lambda v: normal_op(v, D, facx, facy, kz, nu, c, mesh, mask, wq, kap)
    mw = None if mesh is None else multiplicity_weight(mesh, b.shape)
    P = (lambda r: r) if M_inv is None else (lambda r: r*M_inv)

    r = b - A(x)
    z = P(r)
    p = z.copy()
    rz = _dot(r, z, mw)
    b_norm = np.sqrt(_dot(b, b, mw))
    target = np.maximum(tol*b_norm, 1e-300)
    it = 0
    for it in range(1, max_iter + 1):
        Ap = A(p)
        denom = _dot(p, Ap, mw)
        alpha = np.where(np.abs(denom) > 1e-300, rz/np.where(denom == 0, 1, denom), 0.0)
        x = x + alpha*p
        r = r - alpha*Ap
        rn = np.sqrt(_dot(r, r, mw))
        if np.all(rn < target):
            break
        z = P(r)
        rz_new = _dot(r, z, mw)
        beta = np.where(np.abs(rz) > 1e-300, rz_new/np.where(rz == 0, 1, rz), 0.0)
        p = z + beta*p
        rz = rz_new
    return x, it, np.sqrt(_dot(r, r, mw)).ravel()


def rkw3_step(Uh, dt, rhs_explicit, solve_stage, N_prev=None):
    """One RKW3/Crank-Nicolson step.  Two registers: Uh and N_prev.

    rhs_explicit(Uh)  -> the explicit (convective) term for the current state
    solve_stage(rhs, c) -> the implicit solve with mass coefficient c

    The scheme is
        U^k = U^{k-1} + dt[ gamma_k N^{k-1} + zeta_k N^{k-2}
                          + alpha_k L^{k-1} + beta_k L^k ]
    and the implicit coefficient handed to solve_stage is c = 1/(beta_k*dt),
    NOT fac1/dt -- see timestep.a_mass_worst and plan sec 0.4.
    """
    if N_prev is None:
        N_prev = np.zeros_like(Uh)
    for k in range(NSTAGE):
        Nk = rhs_explicit(Uh)
        rhs = Uh + dt*(GAMMA[k]*Nk + ZETA[k]*N_prev)
        Uh = solve_stage(rhs, 1.0/(BETA[k]*dt), k)
        N_prev = Nk
    return Uh, N_prev
