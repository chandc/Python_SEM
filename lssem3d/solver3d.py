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
from . import operator as OP
from .timestep import ALPHA, BETA, GAMMA, ZETA, NSTAGE

SPATIAL = (1, 2, 3)          # n, n, var -- reduce over these, keep (elem, mode)


def _dot(a, b):
    """Per-mode inner product: sum over space and fields, keep elem+mode."""
    return np.sum(a*b, axis=SPATIAL).sum(axis=0)[None, None, None, None, :]


def normal_op(Ur, D, facx, facy, kz, nu, c, mask=None):
    """A = L^T L applied to a split-real state, optionally masked.

    mask is 1 where a degree of freedom is free and 0 where it is prescribed;
    applying it on both sides keeps A symmetric, which is what CG needs.
    """
    if mask is not None:
        Ur = Ur*mask
    out = OP.apply_LT(OP.apply_L(Ur, D, facx, facy, kz, nu, c),
                      D, facx, facy, kz, nu, c)
    return out*mask if mask is not None else out


def jacobi_diagonal(shape, D, facx, facy, kz, nu, c, mask=None):
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
                    e, D, facx, facy, kz, nu, c, mask)[:, i, j, f, :]
    return diag


def pcg(b, D, facx, facy, kz, nu, c, mask=None, M_inv=None, tol=1e-10,
        max_iter=2000, x0=None):
    """Preconditioned CG on A x = b, batched over modes.

    Returns (x, iters, resid) with resid the per-mode final residual norm.
    Convergence is per mode: a mode whose residual is already below tol
    contributes nothing further, and the loop exits when ALL modes are below.
    """
    x = np.zeros_like(b) if x0 is None else x0.copy()
    if mask is not None:
        b = b*mask
        x = x*mask
    A = lambda v: normal_op(v, D, facx, facy, kz, nu, c, mask)
    P = (lambda r: r) if M_inv is None else (lambda r: r*M_inv)

    r = b - A(x)
    z = P(r)
    p = z.copy()
    rz = _dot(r, z)
    b_norm = np.sqrt(_dot(b, b))
    target = np.maximum(tol*b_norm, 1e-300)
    it = 0
    for it in range(1, max_iter + 1):
        Ap = A(p)
        denom = _dot(p, Ap)
        alpha = np.where(np.abs(denom) > 1e-300, rz/np.where(denom == 0, 1, denom), 0.0)
        x = x + alpha*p
        r = r - alpha*Ap
        rn = np.sqrt(_dot(r, r))
        if np.all(rn < target):
            break
        z = P(r)
        rz_new = _dot(r, z)
        beta = np.where(np.abs(rz) > 1e-300, rz_new/np.where(rz == 0, 1, rz), 0.0)
        p = z + beta*p
        rz = rz_new
    return x, it, np.sqrt(_dot(r, r)).ravel()


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
