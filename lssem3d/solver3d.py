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


def normal_op(Ur, D, facx, facy, kz, nu, c, mesh=None, mask=None, wq=None, kap=0.0, rw=None):
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
    out = OP.apply_LT(OP.apply_L(Ur, D, facx, facy, kz, nu, c, wq, kap, rw),
                      D, facx, facy, kz, nu, c, kap)
    if mesh is not None:
        out = gs(mesh, out)
    return out*mask if mask is not None else out


def jacobi_diagonal(shape, D, facx, facy, kz, nu, c, mesh=None, mask=None,
                    wq=None, kap=0.0, assemble=True, rw=None):
    """diag(A) by probing with unit vectors, ASSEMBLED across elements.

    THE ASSEMBLY IS NOT OPTIONAL FOR CORRECTNESS.  A probe sets one LOCAL node
    index, so at a node shared by several elements it sees only ONE element's
    contribution, while the assembled operator A = M Q^T Q L^T W L M has all of
    them.  Measured on a uniform mesh: the raw probe returns exactly
    diag/multiplicity -- HALF the true value on an element edge, a QUARTER at a
    corner -- so 1/diag over-weights every element-boundary node by 2-4x.

    `gs` is the right correction rather than multiplying by the multiplicity:
    it SUMS each element's own contribution, which stays correct on a
    non-uniform mesh where the copies of a shared node differ.  Multiplying by
    the multiplicity would only be right when they happen to be equal.

    Worth 1.41-1.44x fewer CG iterations, measured across three grids
    (N=4/6/8, c=50/600/1200).  Pass assemble=False only to reproduce the old,
    incorrect behaviour.

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
                # PROBE THE UNASSEMBLED OPERATOR (mesh=None).  With the mesh
                # passed in, the probe vector -- local index (i,j) set in EVERY
                # element -- is DISCONTINUOUS at an interface: one copy of a
                # shared node is 1 while its twin is 0, and the neighbour's
                # opposite edge also carries a 1.  gs() then sums intra-element
                # off-diagonal couplings (west<->east through the D-matrix row)
                # into what is read back as "the diagonal".  Measured: 1.4%
                # error at interface dofs, worst on the c-independent pressure
                # and vorticity rows (velocity contamination falls like 1/c^2,
                # which is why a velocity-only spot check missed it entirely).
                # Unassembled, each local unit vector is a clean unit inside its
                # own element block, so the local diagonals are exact; the gs()
                # below then sums them, which IS the assembled diagonal because
                # L^T W L is element-block-diagonal.  Verified exact (0.0e+00).
                diag[:, i, j, f, :] = normal_op(
                    e, D, facx, facy, kz, nu, c, None, mask, wq, kap,
                    rw)[:, i, j, f, :]
    if assemble and mesh is not None:
        diag = gs(mesh, diag)
    return diag


def jacobi_inverse(diag, mask=None):
    """1/diag on free dofs, ZERO on prescribed ones -- never 1/eps.

    The idiom `1.0/np.maximum(diag, 1e-30)` puts 1e30 at every masked dof, whose
    diagonal is exactly 0.  It survives only because the masked residual is
    exactly zero every iteration, so 0 * 1e30 = 0; one round-off leak or one NaN
    and it detonates.  It would also clamp a NEGATIVE diagonal to 1e-30, turning
    a bug notification into a 1e30 multiplier on a LIVE dof.

    A non-positive diagonal of an SPD operator is a defect, so this asserts
    rather than sanitises.
    """
    live = np.abs(diag) > 0.0 if mask is None else (mask != 0.0) & (diag != 0.0)
    bad = live & (diag <= 0.0)
    if bad.any():
        raise ValueError(
            f'{int(bad.sum())} free dofs have a non-positive Jacobi diagonal '
            f'(min {diag[bad].min():.3e}) -- the operator is not SPD there')
    return np.where(live, 1.0/np.where(live, diag, 1.0), 0.0)


# (row, field, a, b, cval) for L0, read off apply_L0_complex row by row.
#   a    coefficient of d/dx of that field
#   b    coefficient of d/dy
#   cval coefficient of the field itself; 'ik' and 'nuik' are filled in per mode
#
#   0  kap*p + ux + vy + ik*w        4  c*u + px + nu*(ozy - ik*oy)
#   1  wy - ik*v - ox                5  c*v + py + nu*(ik*ox - ozx)
#   2  ik*u - wx - oy                6  c*w + ik*p + nu*(oyx - oxy)
#   3  vx - uy - oz                  7  oxx + oyy + ik*oz
_L0_TERMS = (
    (0, OP.U_,  1.0,  0.0, None),   (0, OP.V_,  0.0,  1.0, None),
    (0, OP.W_,  0.0,  0.0, 'ik'),   (0, OP.P_,  0.0,  0.0, 'kap'),
    (1, OP.W_,  0.0,  1.0, None),   (1, OP.V_,  0.0,  0.0, '-ik'),
    (1, OP.OX_, 0.0,  0.0, '-1'),
    (2, OP.U_,  0.0,  0.0, 'ik'),   (2, OP.W_, -1.0,  0.0, None),
    (2, OP.OY_, 0.0,  0.0, '-1'),
    (3, OP.V_,  1.0,  0.0, None),   (3, OP.U_,  0.0, -1.0, None),
    (3, OP.OZ_, 0.0,  0.0, '-1'),
    (4, OP.U_,  0.0,  0.0, 'c'),    (4, OP.P_,  1.0,  0.0, None),
    (4, OP.OZ_, 0.0,  'nu', None),  (4, OP.OY_, 0.0,  0.0, '-nuik'),
    (5, OP.V_,  0.0,  0.0, 'c'),    (5, OP.P_,  0.0,  1.0, None),
    (5, OP.OX_, 0.0,  0.0, 'nuik'), (5, OP.OZ_, '-nu', 0.0, None),
    (6, OP.W_,  0.0,  0.0, 'c'),    (6, OP.P_,  0.0,  0.0, 'ik'),
    (6, OP.OY_, 'nu', 0.0, None),   (6, OP.OX_, 0.0, '-nu', None),
    (7, OP.OX_, 1.0,  0.0, None),   (7, OP.OY_, 0.0,  1.0, None),
    (7, OP.OZ_, 0.0,  0.0, 'ik'),
)


def jacobi_diagonal_analytic(shape, D, facx, facy, kz, nu, c, mesh=None,
                             mask=None, wq=None, kap=0.0, rw=None):
    """diag(A) in closed form -- the production replacement for the probing loop.

    `jacobi_diagonal` costs 2*7*(N+1)^2 operator applications per stage, which
    measured 34-41% of total runtime and GROWS with N (it scales as N^2 while
    the solve's iteration count does not).  At N=16: 43 s of probing to set up a
    62 s solve.

    THE ALGEBRA.  Row r of L0 is  sum_v [ a*dx(U_v) + b*dy(U_v) + cval*U_v ],
    so at quadrature point (p,q),

        dR_r(p,q)/dU_v(i,j) = a*D[p,i]*facx*delta_qj
                            + b*D[q,j]*facy*delta_pi
                            + cval*delta_pi*delta_qj

    which is non-zero only on the row p=i or the column q=j.  Squaring and
    summing against the weights gives, per (r, v),

        wq[i,j] * |a*D[i,i]*facx + b*D[j,j]*facy + cval|^2      (p,q)=(i,j)
      + a^2*facx^2 * sum_{p!=i} wq[p,j]*D[p,i]^2                 the column
      + b^2*facy^2 * sum_{q!=j} wq[i,q]*D[q,j]^2                 the row

    The two sums are (N+1)-point contractions computed once for the whole mesh,
    so the cost is O(n^2) per element instead of O(n^2) OPERATOR APPLICATIONS.

    SPLIT-REAL.  For a complex coefficient alpha the split-real block is
    [[Re, -Im], [Im, Re]], whose column norms are both |alpha|^2 -- so the real
    and imaginary halves of a field share one diagonal value, and the modulus
    above is the whole story.  That is why this can be derived in complex form
    and written to both halves.

    Verified against `jacobi_diagonal` to machine precision -- that routine is
    exact (0.0 against a continuous-unit-vector ground truth), which is what
    makes it a usable oracle here.
    """
    nelem, n = shape[0], shape[1]
    nk = shape[-1]
    if wq is None:
        wq = np.ones((nelem, n, n))
    rw = np.ones(OP.NROW) if rw is None else np.asarray(rw)

    D2 = D*D
    dg = np.diag(D2)                                  # D[i,i]^2
    # column/row contractions, with the (p,q)=(i,j) term removed
    Sx = np.einsum('epj,pi->eij', wq, D2) - wq*dg[None, :, None]
    Sy = np.einsum('eiq,qj->eij', wq, D2) - wq*dg[None, None, :]
    Sx = Sx*(facx**2)[:, None, None]
    Sy = Sy*(facy**2)[:, None, None]

    Dii = np.diag(D)
    dxx = Dii[None, :, None]*facx[:, None, None]      # (e, i, 1)
    dyy = Dii[None, None, :]*facy[:, None, None]      # (e, 1, j)

    ik = 1j*np.asarray(kz).reshape(1, 1, 1, -1)
    lit = {'ik': ik, '-ik': -ik, 'nuik': nu*ik, '-nuik': -nu*ik,
           'kap': kap, 'c': c, '-1': -1.0, 'nu': nu, '-nu': -nu}
    num = lambda x: lit[x] if isinstance(x, str) else x

    diag = np.zeros(shape)
    for r, f, a, b, cv in _L0_TERMS:
        a, b = num(a), num(b)
        cval = 0.0 if cv is None else num(cv)
        loc = a*dxx[..., None] + b*dyy[..., None] + cval   # (e,i,j,k) complex
        contrib = rw[r]*(wq[..., None]*np.abs(loc)**2
                         + (a*a)*Sx[..., None] + (b*b)*Sy[..., None])
        diag[..., f, :] += contrib                    # real half
        diag[..., OP.NVAR + f, :] += contrib          # imaginary half

    if mask is not None:
        diag = diag*mask
    if mesh is not None:
        diag = gs(mesh, diag)
    return diag*mask if mask is not None else diag


def pcg(b, D, facx, facy, kz, nu, c, mesh=None, mask=None, M_inv=None,
        tol=1e-10, max_iter=2000, x0=None, wq=None, kap=0.0, rw=None):
    """Preconditioned CG on A x = b, batched over modes.

    Returns (x, iters, resid) with resid the per-mode final residual norm.
    Convergence is per mode: a mode whose residual is already below tol
    contributes nothing further, and the loop exits when ALL modes are below.
    """
    x = np.zeros_like(b) if x0 is None else x0.copy()
    if mask is not None:
        b = b*mask
        x = x*mask
    A = lambda v: normal_op(v, D, facx, facy, kz, nu, c, mesh, mask, wq, kap, rw)
    mw = None if mesh is None else multiplicity_weight(mesh, b.shape)
    P = (lambda r: r) if M_inv is None else (lambda r: r*M_inv)

    r = b - A(x)
    z = P(r)
    p = z.copy()
    rz = _dot(r, z, mw)
    b_norm = np.sqrt(_dot(b, b, mw))
    target = np.maximum(tol*b_norm, 1e-300)
    it = 0
    restarts = 0
    for it in range(1, max_iter + 1):
        Ap = A(p)
        denom = _dot(p, Ap, mw)
        alpha = np.where(np.abs(denom) > 1e-300, rz/np.where(denom == 0, 1, denom), 0.0)
        x = x + alpha*p
        r = r - alpha*Ap
        rn = np.sqrt(_dot(r, r, mw))
        if np.all(rn < target):
            # TRUE-RESIDUAL SAFEGUARD, ported from lssem2d.pcg_solve.
            #
            # `r` here is the RECURSIVE residual, updated as r - alpha*Ap.  Over
            # thousands of iterations it drifts away from the true b - A x, and
            # CG then declares victory on a number that no longer describes the
            # iterate.  The k_z study already runs this solver past 1e4
            # iterations, which is exactly the regime where that bites.
            #
            # Costs one extra matvec, and only when the recursive residual
            # claims convergence -- usually once per solve.
            r_true = b - A(x)
            rn_true = np.sqrt(_dot(r_true, r_true, mw))
            if np.all(rn_true < target):
                break
            # Drift: restart the recursion from the true residual.  Doing it for
            # every mode at once is correct even though only some drifted -- the
            # per-mode recurrences are independent, and for an undrifted mode
            # r_true is its recursive r to rounding.
            r = r_true
            z = P(r)
            p = z.copy()
            rz = _dot(r, z, mw)
            restarts += 1
            continue
        z = P(r)
        rz_new = _dot(r, z, mw)
        beta = np.where(np.abs(rz) > 1e-300, rz_new/np.where(rz == 0, 1, rz), 0.0)
        p = z + beta*p
        rz = rz_new
    # report the TRUE residual, not the recursive one it may have drifted from
    return x, it, np.sqrt(_dot(b - A(x), b - A(x), mw)).ravel()


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
