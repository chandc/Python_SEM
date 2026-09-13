"""p-multigrid preconditioner for the 3D VVP operator.

NEW CODE.  lssem2d is not modified; this is a port of `lssem2d/precond.py` to the
5-D `(elem, i, j, var, mode)` layout.

WHY.  Jacobi rescales pointwise.  Measured on this operator it needs ~5400 CG
iterations per stage solve at a 48^3 grid, and **block-Jacobi does not help** --
inverting the full 7x7 node block gave 1.10x / 1.00x / 0.89x on three grids, i.e.
nothing, and a net loss at the largest once its 16-20% apply cost is counted.
That is the signature the 2D module already documents: the VVP pressure lives in
a very soft direction of A (~8e3x softer than a generic direction there), and

    a diagonal preconditioner rescales pointwise and CANNOT touch a near-null
    global mode -- no matter how good the diagonal is.

Block-Jacobi is still pointwise; it fails for the same reason.  A polynomial
smoother damps a broad spectral band and a coarse solve removes what is left,
which is what a p-multigrid V-cycle does.

THREE LEVELS, e.g. p = 8 -> 4 -> 2, mirroring solver_pmg2.f90's 10 -> 4 -> 2.

FOUR RULES INHERITED FROM THE 2D PORT, each of which breaks the method quietly:

 1. **R = P^T.**  Restriction must be the adjoint of prolongation or the V-cycle
    is not symmetric, and CG loses its convergence guarantee.  An independent
    fine->coarse interpolation is NOT the adjoint.
 2. **Weight by 1/multiplicity before restricting.**  The state is in redundant
    local storage where every copy of a shared node already holds the assembled
    value, so a plain P^T counts each shared node once per owning element.
 3. **The coarse solve must be a FIXED LINEAR operator.**  Chebyshev, not CG:
    CG's polynomial depends on its right-hand side, which destroys the symmetry
    of the whole V-cycle when it is used inside a preconditioner.
 4. **The coarse operator must carry the same least-squares weighting as the
    fine one** -- here `rw` and `kap`.  In 2D, dropping them sent the coarse
    problem down a different weighting branch and CG needed ~2000 iterations per
    solve instead of tens.
"""
from copy import copy

import os

import numpy as np

from lssem2d.lgl import lgl_nodes, diff_matrix
from . import operator as OP
from . import bc as BC
from . import solver3d as S3
from . import device as DEV

# Optimised 4th-kind Chebyshev weights (Phillips & Fischer / Lottes, Table 5),
# the same table as solver_pmg2.f90's beta4.
_BETA4 = {
    1: [1.125],
    2: [1.02387287570313, 1.26408905371085],
    3: [1.00842544782028, 1.08867839208730, 1.33753125909618],
    4: [1.00391310427285, 1.04035811188593, 1.14863498546254, 1.38268869241000],
    5: [1.00212930146164, 1.02173711549260, 1.07872433192603, 1.19810065292663,
        1.41322542791682],
    6: [1.00128517255940, 1.01304293035233, 1.04678215124113, 1.11616489419675,
        1.23829020218444, 1.43524297106744],
    8: [1.00063462992460, 1.00641320535722, 1.02290561803222, 1.05409390982680,
        1.10600719133713, 1.18579117549170, 1.29938185158384, 1.45507550085384],
    10: [1.00037339349285, 1.00377761154979, 1.01348264656944, 1.03170645330584,
         1.06195838009669, 1.10748199530873, 1.17262062761420, 1.26146485689102,
         1.37844818254735, 1.47063496090587],
}


def p_interp(p_from, p_to):
    """1D Lagrange interpolation, LGL(p_from) nodes -> LGL(p_to) nodes."""
    xs, xt = lgl_nodes(p_from), lgl_nodes(p_to)
    n = len(xs)
    C = np.empty((len(xt), n))
    for a, x in enumerate(xt):
        w = np.ones(n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    w[i] *= (x - xs[j])/(xs[i] - xs[j])
        C[a] = w
    return C


def estimate_lambda_max(A, M_inv, shape, npow=20, seed=0):
    """Largest eigenvalue of M^-1 A by power iteration.

    Chebyshev needs only an UPPER bound on the spectrum -- no lower edge -- which
    is what makes the 4th-kind form robust when the spectrum is poorly known.
    """
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(shape)*(M_inv > 0 if not callable(M_inv) else 1.0)
    lam = 0.0
    for _ in range(npow):
        w = M_inv(A(v)) if callable(M_inv) else M_inv*A(v)
        nw = float(np.sqrt(np.sum(w*w)))
        if nw <= 1e-300:
            return 0.0
        lam = nw/max(float(np.sqrt(np.sum(v*v))), 1e-300)
        v = w/nw
    return float(lam)


class Chebyshev4:
    """Degree-`deg` 4th-kind Chebyshev polynomial in M^-1 A, applied from z = 0.

    Costs `deg` operator applications per preconditioner call -- that is the
    price of damping a band rather than a point.
    """

    name = 'chebyshev4'

    def __init__(self, A, M_inv, shape, deg=6, optimised=True, lam_max=None,
                 safety=1.3, npow=20):
        self.A, self.M_inv, self.deg = A, M_inv, int(deg)
        self.beta = (_BETA4.get(self.deg) if optimised else None) or [1.0]*self.deg
        if lam_max is None:
            lam_max = estimate_lambda_max(A, M_inv, shape, npow)
        self.rho = float(safety)*float(lam_max)
        self.n_applies = 0

    def _P(self, r):
        return self.M_inv(r) if callable(self.M_inv) else self.M_inv*r

    def __call__(self, r):
        if self.rho <= 0.0:
            return self._P(r)
        z = np.zeros_like(r)
        d = np.zeros_like(r)
        rf = r.copy()
        for k in range(1, self.deg + 1):
            c1 = (2.0*k - 3.0)/(2.0*k + 1.0)
            c2 = self.beta[k-1]*(8.0*k - 4.0)/((2.0*k + 1.0)*self.rho)
            d = c1*d + c2*self._P(rf)
            z += d
            rf = rf - self.A(d)
            self.n_applies += 1
        return z


def coarsen_mesh(m, pc):
    """Same elements and geometry, lower polynomial order.

    The coarse operator is REDISCRETISED at order pc rather than formed as a
    Galerkin product -- solver_pmg2.f90 offers both and the rediscretised path
    is far less code.  `periodic_x`/`periodic_y` ride along on the copy, so the
    coarse mesh keeps the seam.
    """
    mc = copy(m)
    mc.N = int(pc)
    mc.nterm = int(pc) + 1
    mc.gidx = -np.ones((m.nelem, mc.nterm, mc.nterm), dtype=int)
    mc.xnod = np.zeros((m.nelem, mc.nterm))
    mc.ynod = np.zeros((m.nelem, mc.nterm))
    mc.wq = np.zeros((m.nelem, mc.nterm, mc.nterm))
    mc.setup_derived()
    mc.compute_global_indices()
    return mc


class _Level:
    """One polynomial order: mesh, operator, diagonal, transfer to the next."""

    def __init__(self, mesh, nk, nz, nu, c, kz, kap, rw, pin_p, mask=None):
        self.m, self.nk = mesh, nk
        n = mesh.N + 1
        self.shape = (mesh.nelem, n, n, OP.NVAR_R, nk)
        self.D = diff_matrix(mesh.N)
        # AN EXPLICIT MASK OVERRIDES pin_p, and the finest level MUST use the
        # caller's own mask.  A preconditioner has to be defined on exactly the
        # space the operator is: if it pins a dof the solve leaves free it
        # returns zero there, M is singular on the space CG searches, and the
        # V-cycle silently under-performs.  Measured on a 3x3 N=8 Nz=16 mesh,
        # build_mask(pin_p=True) pinned 60 dofs the driver leaves free -- 32
        # pressure real parts plus their imaginary partners, across every mode
        # except k = 0 -- because pin_p pins pressure at EVERY mode while the
        # physics needs it pinned only at k = 0 (for k != 0 the ik*p term in the
        # z-momentum row already determines pressure uniquely).
        self.mask = (np.array(mask, copy=True) if mask is not None
                     else BC.build_mask(mesh, nk, pin_p=pin_p, nz=nz))
        self.A = lambda v: S3.normal_op(v, self.D, mesh.facx, mesh.facy, kz, nu,
                                        c, mesh, self.mask, mesh.wq, kap, rw)
        # Same operator WITHOUT the gather-scatter: elements stay decoupled, so
        # one probe returns every element's local block at once.  DirectCoarseE
        # uses this; nothing else should.
        self.A_un = lambda v: S3.normal_op(v, self.D, mesh.facx, mesh.facy, kz,
                                           nu, c, None, self.mask, mesh.wq,
                                           kap, rw)
        diag = S3.jacobi_diagonal_analytic(self.shape, self.D, mesh.facx,
                                           mesh.facy, kz, nu, c, mesh,
                                           self.mask, mesh.wq, kap, rw)
        self.M_inv = S3.jacobi_inverse(diag, self.mask)
        mult = S3.gs(mesh, np.ones(self.shape))
        self.mw = 1.0/np.where(mult < 1e-10, 1.0, mult)


def _factor_spd(A):
    """Factorise an SPD matrix and return a solve callable.

    Cholesky, not `np.linalg.pinv`.  The coarse operator is a Galerkin
    projection of an SPD operator and is explicitly symmetrised, so a
    pseudo-inverse is the wrong tool three ways: it costs a full SVD (roughly
    10-30x a Cholesky), it stores a DENSE inverse where a triangular factor
    needs half the memory, and its rcond silently truncates -- which would
    discard exactly the near-null directions a coarse solve exists to resolve.
    (Measured on a 3x3 p=2 mesh it truncated nothing, kappa being 5e6-3.4e8,
    but that is luck rather than design.)

    Falls back to a pseudo-inverse if the matrix is not positive definite,
    which means the level is under-pinned -- worth knowing rather than hiding.
    """
    try:
        from scipy.linalg import cho_factor, cho_solve
        cf = cho_factor(A, lower=True, check_finite=False)
        return lambda b: cho_solve(cf, b, check_finite=False)
    except Exception:
        try:
            Lc = np.linalg.cholesky(A)
            return lambda b: np.linalg.solve(
                Lc.T, np.linalg.solve(Lc, b))
        except np.linalg.LinAlgError:
            import warnings
            warnings.warn('coarse level is not positive definite -- falling '
                          'back to a pseudo-inverse; the level is probably '
                          'under-pinned', RuntimeWarning)
            Ai = np.linalg.pinv(A)
            return lambda b: Ai @ b


class DirectCoarse:
    """EXACT solve on the coarsest level, factorised once.

    A V-cycle needs the coarsest grid SOLVED, not smoothed.  Measured with a
    Chebyshev coarse "solve" the cycle converged at 0.99 per iteration, and
    raising its degree from 4 to 40 changed nothing (0.9898 -> 0.9897) -- a
    polynomial saturates well short of a solve on an ill-conditioned operator,
    so degree-independence does NOT mean the coarse level is fine.

    Cheap here because the MODES ARE INDEPENDENT: the coarse operator is block
    diagonal in k_z, so this factorises one small dense block per mode rather
    than one huge one.  At p = 2 on a 3x3 mesh that is ~700 dofs per mode.

    Built in the GLOBAL (continuous) basis, since the assembled operator acts on
    continuous fields: a basis vector is `gs` of a one-hot, deduplicated by
    support, and the inner product carries the multiplicity weight so each
    global node counts once.
    """

    name = 'direct'

    def __init__(self, level):
        self.lev = level
        shape, m, mask = level.shape, level.m, level.mask
        nk = shape[-1]
        self.B, self.lu = [], []      # `lu` holds SOLVE CALLABLES, see _factor_spd
        for k in range(nk):
            cols, seen = [], set()
            for e in range(shape[0]):
                for i in range(shape[1]):
                    for j in range(shape[2]):
                        for f in range(OP.NVAR_R):
                            if mask[e, i, j, f, k] == 0.0:
                                continue
                            ed = np.zeros(shape)
                            ed[e, i, j, f, k] = 1.0
                            g = S3.gs(m, ed)
                            key = tuple(np.flatnonzero(np.abs(g.ravel()) > 0.5))
                            if not key or key in seen:
                                continue
                            seen.add(key)
                            cols.append(g)
            if not cols:
                self.B.append(None); self.lu.append(None); continue
            B = np.stack([c.ravel() for c in cols], axis=1)
            mw = level.mw.ravel()
            Amat = np.empty((B.shape[1], B.shape[1]))
            for a in range(B.shape[1]):
                Av = level.A(B[:, a].reshape(shape)).ravel()
                Amat[:, a] = B.T @ (Av*mw)
            Amat = 0.5*(Amat + Amat.T)          # symmetrise against round-off
            self.B.append(B)
            self.lu.append(_factor_spd(Amat))
        self.mw = level.mw.ravel()

    def __call__(self, r):
        shape = self.lev.shape
        z = np.zeros(shape)
        rf = r.ravel()
        for k, (B, fac) in enumerate(zip(self.B, self.lu)):
            if B is None:
                continue
            g = B.T @ (rf*self.mw)
            z += (B @ fac(g)).reshape(shape)
        return z*self.lev.mask


class DirectCoarseE(DirectCoarse):
    """DirectCoarse with ELEMENT-LOCAL assembly -- same operator, O(p^2*nvar)
    probes instead of O(global dofs).

    DirectCoarse builds each mode's matrix by probing the ASSEMBLED operator
    once per global dof: for the minimal channel's coarse level (13x37 nodes x
    14 fields x 17 modes) that is ~114,000 full operator applies, which is why
    the channel has been running with `direct_coarse=False` and a Chebyshev
    coarse "solve" instead.

    But the unassembled operator M L0^T W L0 M is ELEMENT-BLOCK-DIAGONAL, and
    the modes never couple.  So a one-hot placed at local node (i,j,f) in EVERY
    element and EVERY mode comes back carrying that column of every element's
    and every mode's local block simultaneously.  That is `nloc = (p+1)^2 * 14`
    probes TOTAL -- 126 at p=2 -- independent of the mesh and of nk.  The local
    blocks are then assembled sparsely through `gidx` in the ordinary
    finite-element way and factorised per mode.

    Verified against DirectCoarse: identical action to roundoff (see
    `scratch/direct_coarse_check.py`).
    """

    name = 'directE'

    def __init__(self, level):
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
        self.lev = level
        m, shape, mask = level.m, level.shape, level.mask
        nelem, n, _, nvar, nk = shape
        nloc = n*n*nvar

        # ---- element-local blocks, nloc probes for the whole mesh ----
        Aloc = np.empty((nelem, nk, nloc, nloc))
        for col in range(nloc):
            i, j, f = np.unravel_index(col, (n, n, nvar))
            v = np.zeros(shape)
            v[:, i, j, f, :] = 1.0
            out = level.A_un(v)                       # (nelem,n,n,nvar,nk)
            Aloc[:, :, :, col] = np.moveaxis(out, -1, 1).reshape(nelem, nk, nloc)

        # ---- global dof numbering: node*nvar + field ----
        gidx = m.gidx                                  # (nelem, n, n)
        nnode = int(gidx.max()) + 1
        ndof = nnode*nvar
        gd = (gidx[..., None]*nvar
              + np.arange(nvar)).reshape(nelem, nloc)   # (nelem, nloc)
        rows = np.repeat(gd, nloc, axis=1).ravel()
        cols = np.tile(gd, (1, nloc)).ravel()

        self.gd, self.ndof, self.nvar = gd, ndof, nvar
        self.lu = []
        for k in range(nk):
            data = Aloc[:, k].reshape(nelem, nloc*nloc).ravel()
            A = sp.coo_matrix((data, (rows, cols)), shape=(ndof, ndof)).tocsr()
            # Prescribed dofs assemble to an empty row/col; give them a unit
            # diagonal so the factorisation is non-singular.  Their residual is
            # zero (r is masked), so they solve to zero and the final *mask
            # keeps them there.
            d = np.asarray(A.diagonal())
            dead = np.abs(d) <= 1e-300
            if dead.any():
                A = A + sp.diags(dead.astype(float))
            A = (A + A.T)*0.5
            self.lu.append(spla.splu(sp.csc_matrix(A)))

        # gather/scatter weights: mw = 1/multiplicity, so summing a dof's copies
        # and weighting recovers the single assembled value (DirectCoarse does
        # the same thing via B^T diag(mw)).
        self.mwl = level.mw.reshape(nelem, n, n, nvar, nk)

    def __call__(self, r):
        shape = self.lev.shape
        nelem, n, _, nvar, nk = shape
        nloc = n*n*nvar
        rw_ = (r*self.mwl).reshape(nelem, nloc, nk)
        z = np.empty((nelem, nloc, nk))
        flat_gd = self.gd.ravel()
        for k in range(nk):
            g = np.bincount(flat_gd, weights=rw_[:, :, k].ravel(),
                            minlength=self.ndof)
            zk = self.lu[k].solve(g)
            z[:, :, k] = zk[self.gd]
        return z.reshape(shape)*self.lev.mask


class PMG:
    """Recursive p-multigrid V-cycle, usable as a `pcg` preconditioner.

    `orders` is the p-hierarchy, finest first, e.g. (8, 4, 2) for three levels.
    Each level pre-smooths, restricts the residual, recurses, prolongs the
    correction and post-smooths; the coarsest level applies a higher-degree
    Chebyshev instead of recursing.
    """

    name = 'pmg'

    def __init__(self, mesh, nk, nz, nu, c, kz, kap=0.0, rw=None,
                 orders=(8, 4, 2), deg=6, coarse_deg=10, pin_p=True,
                 optimised=True, direct_coarse=True, mask=None):
        assert len(orders) >= 2 and list(orders) == sorted(orders, reverse=True), \
            f'orders must be descending, got {orders}'
        assert orders[0] == mesh.N, f'finest order {orders[0]} != mesh.N {mesh.N}'
        self.levels, self.P, self.R = [], [], []
        meshes = [mesh] + [coarsen_mesh(mesh, p) for p in orders[1:]]
        for li, mm in enumerate(meshes):
            if li == 0 and mask is not None:
                lm = mask                     # the caller's own, exactly
            elif mask is not None:
                # Coarse levels: reproduce the caller's convention rather than
                # pin_p's.  Pressure is pinned at k = 0 only; at k != 0 it is
                # already determined, and pinning it there removes a dof the
                # fine operator keeps.
                lm = BC.build_mask(mm, nk, pin_p=False, nz=nz)
                BC.pin_dof(mm, lm, OP.P_, 0)
            else:
                lm = None
            self.levels.append(_Level(mm, nk, nz, nu, c, kz, kap, rw, pin_p,
                                      mask=lm))
        for lf, lc in zip(orders[:-1], orders[1:]):
            P = p_interp(lc, lf)          # coarse -> fine, nodal interpolation
            self.P.append(P)
            self.R.append(P.T)            # restriction is the ADJOINT (rule 1)
        self.smooth = [Chebyshev4(l.A, l.M_inv, l.shape, deg=deg,
                                  optimised=optimised)
                       for l in self.levels[:-1]]
        # coarsest: SOLVED, not smoothed.  A direct factorisation is also a
        # fixed linear operator, so rule 3 is satisfied a fortiori.
        lc = self.levels[-1]
        self.coarse = ((DirectCoarseE(lc) if str(direct_coarse) == 'element'
                        else DirectCoarse(lc)) if direct_coarse else
                       Chebyshev4(lc.A, lc.M_inv, lc.shape, deg=coarse_deg,
                                  optimised=optimised))

    def _restrict(self, x, lev):
        """Fine residual -> coarse.  Multiplicity-weighted (rule 2), then P^T,
        then re-assembled on the coarse mesh."""
        R = self.R[lev]
        xw = x*self.levels[lev].mw
        t = np.einsum('bj,eijvk->eibvk', R, xw)
        c = np.einsum('ai,eibvk->eabvk', R, t)
        return S3.gs(self.levels[lev+1].m, c)*self.levels[lev+1].mask

    def _prolong(self, xc, lev):
        P = self.P[lev]
        t = np.einsum('bj,eijvk->eibvk', P, xc)
        return np.einsum('ai,eibvk->eabvk', P, t)*self.levels[lev].mask

    def _vcycle(self, r, lev):
        if lev == len(self.levels) - 1:
            return self.coarse(r)
        z = self.smooth[lev](r)                                # pre-smooth
        res = r - self.levels[lev].A(z)
        ec = self._vcycle(self._restrict(res, lev), lev + 1)    # recurse
        z = z + self._prolong(ec, lev)                          # correct
        res = r - self.levels[lev].A(z)
        return z + self.smooth[lev](res)                        # post-smooth

    def __call__(self, r):
        return self._vcycle(r, 0)


# =============================================================================
# Overlapping vertex-patch additive Schwarz (statically condensed), per mode
# =============================================================================
#
# Port of scratch/vertex_schwarz2d.py (VERTEX_SCHWARZ_IMPLEMENTATION.md,
# SCHWARZ_SEM_TUTORIAL.md, BALANCED_CONDENSED_PLAN.md sec 1.4) to the 5-D
# (elem, i, j, var, mode) layout.  The global operator stays matrix-free; the
# only dense objects are per-element interior factors and per-patch edge Schur
# complements, one set per Fourier mode (the modes never couple).
#
#     M^-1 r = P_c A_c^-1 P_c^T r  +  sum_v R_v^T (R_v A R_v^T)^-1 R_v r
#
# patch(v) = every dof of the four elements sharing mesh vertex v;
# R_v A R_v^T is the ASSEMBLED operator on those dofs, so the ring of
# neighbouring elements contributes (a patch built from its own elements alone
# is singular).  Condensation orders each patch [I_1..I_4 | E]: the element
# interior blocks are factored once and shared by the four patches that use
# the element; only the edge Schur complement S_v is per patch.  Exact: the
# condensed and dense solves agree to round-off (tested).
#
# Why it works where Jacobi/PMG stall: above c* ~ nu p^4/h^2 the (u, omega)
# pair is coupled and the divergence-free kernel is invisible to pointwise
# smoothers; a patch solve is exact on the coupled pair inside the patch and
# the p = 2 coarse term carries the global part.  2D: 19-21 CG iterations flat
# for N = 5..20 against Jacobi's 435-4010 (LOW_MEMORY_PATCH_SOLVERS.md 3.1).
#
# Build cost: one probe of the UNASSEMBLED per-mode operator per local column
# (n*n*14 probes, each returning every element's column at once, exactly as
# DirectCoarseE) -- done one mode at a time so the transient (nelem, nloc,
# nloc) block array is per mode, not per run.  With explicit convection the
# operator is fixed for the run: build once per stage value of c.


def _element_blocks_mode(mesh, D, kz_k, nu, c, kap, rw, mask_k, wq):
    """A_e = M L0_e^T W L0_e M for one mode, every element: (nelem, nloc, nloc),
    local column index (i*n + j)*NVAR_R + var.  `mask_k` is (nelem,n,n,14,1)
    sliced from the FULL-nk mask (never rebuilt for a subset: bc.build_mask
    zeroes the imaginary half of the column it believes is k = 0)."""
    nelem, n = mesh.nelem, mesh.N + 1
    nvar = OP.NVAR_R
    nloc = n*n*nvar
    blocks = np.empty((nelem, nloc, nloc))
    v = np.zeros((nelem, n, n, nvar, 1))
    for col in range(nloc):
        i, j, f = np.unravel_index(col, (n, n, nvar))
        v[:] = 0.0
        v[:, i, j, f, 0] = 1.0
        out = S3.normal_op(v, D, mesh.facx, mesh.facy, kz_k, nu, c, None,
                           mask_k, wq, kap, rw)
        blocks[:, :, col] = out.reshape(nelem, nloc)
    return blocks


def _chol(K):
    """Equilibrated Cholesky solve callable for an SPD block.  Symmetric Jacobi
    scaling first: the pressure rows carry the momentum weight (1/c^2 ~ 1e-8
    legacy) against O(1) constraint rows and unscaled potrf loses those pivots."""
    import scipy.linalg as sla
    sc = 1.0/np.sqrt(np.diag(K))
    Ks = K*sc[:, None]*sc[None, :]
    try:
        cf = sla.cho_factor(Ks, lower=True, check_finite=False)
        return lambda b: sc*sla.cho_solve(cf, sc*b, check_finite=False), Ks.shape[0]*(Ks.shape[0]+1)//2
    except sla.LinAlgError:
        lu = sla.lu_factor(Ks, check_finite=False)
        return lambda b: sc*sla.lu_solve(lu, sc*b, check_finite=False), Ks.shape[0]**2


class VertexSchwarz3D:
    """Vertex-patch additive Schwarz preconditioner for the per-mode VVP
    operator, callable r -> z for `pcg(M_inv=...)`.

    Parameters mirror `PMG`: (mesh, nk, nz, nu, c, kz, kap, rw), plus
      mask      the caller's own FULL-nk mask (recommended; else built from pin_p)
      coarse    'element' -> exact p = 2 coarse term via DirectCoarseE (default);
                None -> patches only (one-level; iterations grow with h)
      condense  True (default) -> static condensation; False -> dense patch
                blocks (reference for the exactness test; 8-13x the memory)
      pc        coarse polynomial order (2)

    Apply is NumPy; a device-resident residual is moved to the host and back
    (functional, not fast -- the batched apply is the port's step 4.5).
    """

    name = 'vschwarz'

    def __init__(self, mesh, nk, nz, nu, c, kz, kap=0.0, rw=None, mask=None,
                 pin_p=False, coarse='element', condense=True, pc=2,
                 verbose=False):
        import time
        t0 = time.perf_counter()
        n, N = mesh.N + 1, mesh.N
        nvar = OP.NVAR_R
        nloc = n*n*nvar
        self.m, self.nk, self.c, self.condense = mesh, nk, c, condense
        self.D = diff_matrix(N)
        self.shape = (mesh.nelem, n, n, nvar, nk)
        self.mask = (np.array(mask, copy=True) if mask is not None
                     else BC.build_mask(mesh, nk, pin_p=pin_p, nz=nz))
        kz = np.asarray(kz, dtype=float)
        mult = S3.gs(mesh, np.ones(self.shape))
        self.mw = 1.0/np.where(mult < 1e-10, 1.0, mult)
        gid = mesh.gidx
        nnode = int(gid.max()) + 1
        self.g = (gid[..., None]*nvar + np.arange(nvar)).astype(np.int64)   # (nelem,n,n,nvar)
        self.ndof = nnode*nvar
        gflat = self.g.reshape(mesh.nelem, -1)
        loc_int = np.zeros((n, n, nvar), bool)
        loc_int[1:N, 1:N, :] = True
        loc_int = loc_int.ravel()

        # ---- mode-independent topology: patches (corner -> elements) and rings
        corners, node_elems = {}, {}
        for e in range(mesh.nelem):
            for (a, b) in ((0, 0), (0, N), (N, 0), (N, N)):
                corners.setdefault(int(gid[e, a, b]), []).append(e)
            for gn in np.unique(gid[e]):
                node_elems.setdefault(int(gn), []).append(e)
        topo = []
        for es in corners.values():
            ring = sorted({e2 for e in es for gn in np.unique(gid[e])
                           for e2 in node_elems[int(gn)]})
            topo.append((es, ring))
        self.npatch = len(topo)

        # ---- per mode: element interior factors + patch Schur complements
        self.modes = []            # per mode: dict(elem=..., patches=...) or None
        self.bytes = 0
        for k in range(nk):
            mask_k = np.ascontiguousarray(self.mask[..., k:k+1])
            free = np.zeros(self.ndof, bool)
            np.logical_or.at(free, self.g.ravel(), mask_k[..., 0].ravel() > 0.5)
            if not free.any():
                self.modes.append(None)
                continue
            blocks = _element_blocks_mode(mesh, self.D, kz[k:k+1], nu, c, kap,
                                          rw, mask_k, mesh.wq)
            if not condense:
                patches = []
                for es, ring in topo:
                    dofs = np.unique(np.concatenate([gflat[e] for e in es]))
                    dofs = dofs[free[dofs]]
                    loc = -np.ones(self.ndof, dtype=np.int64)
                    loc[dofs] = np.arange(dofs.size)
                    K = np.zeros((dofs.size, dofs.size))
                    for e in ring:
                        li = loc[gflat[e]]
                        keep = li >= 0
                        K[np.ix_(li[keep], li[keep])] += blocks[e][np.ix_(keep, keep)]
                    K = 0.5*(K + K.T)
                    solve, nb = _chol(K)
                    self.bytes += 8*nb
                    patches.append((dofs, solve))
                self.modes.append(dict(patches=patches))
                continue
            # element interior factor and interior->edge coupling, shared
            eint, eedge, esolve, eKIB = {}, {}, {}, {}
            for e in range(mesh.nelem):
                ge = gflat[e]
                fr = free[ge]
                I = np.flatnonzero(loc_int & fr)
                B = np.flatnonzero(~loc_int & fr)
                K = blocks[e]
                if I.size:
                    esolve[e], nb = _chol(K[np.ix_(I, I)])
                    self.bytes += 8*(nb + I.size*B.size)
                else:
                    esolve[e] = lambda b: b
                eKIB[e] = K[np.ix_(I, B)].copy()
                eint[e], eedge[e] = ge[I], ge[B]
            patches = []
            for es, ring in topo:
                Bp = np.unique(np.concatenate([eedge[e] for e in es]))
                locB = -np.ones(self.ndof, dtype=np.int64)
                locB[Bp] = np.arange(Bp.size)
                S = np.zeros((Bp.size, Bp.size))
                for e in ring:                     # K_EE from every element touching the patch edge dofs
                    li = locB[gflat[e]]
                    kp = li >= 0
                    S[np.ix_(li[kp], li[kp])] += blocks[e][np.ix_(kp, kp)]
                emaps = []
                for e in es:                       # - K_EI K_II^-1 K_IE of the patch's own elements
                    cols = locB[eedge[e]]
                    KIB = eKIB[e]
                    if KIB.shape[0]:
                        X = np.column_stack([esolve[e](KIB[:, j]) for j in range(KIB.shape[1])])
                        S[np.ix_(cols, cols)] -= KIB.T @ X
                    emaps.append((e, cols))
                S = 0.5*(S + S.T)
                solve, nb = _chol(S)
                self.bytes += 8*nb
                patches.append((Bp, emaps, solve))
            self.modes.append(dict(eint=eint, eedge=eedge, esolve=esolve,
                                   eKIB=eKIB, patches=patches))
            del blocks

        # ---- coarse term: exact p = pc solve with PMG's transfers
        self.coarse = None
        if coarse:
            mm = coarsen_mesh(mesh, pc)
            if mask is not None:
                # the caller's convention: pressure pinned at k = 0 only
                lm = BC.build_mask(mm, nk, pin_p=False, nz=nz)
                BC.pin_dof(mm, lm, OP.P_, 0)
                BC.pin_dof(mm, lm, OP.NVAR + OP.P_, 0)
            else:
                lm = None
            lev = _Level(mm, nk, nz, nu, c, kz, kap, rw, pin_p, mask=lm)
            self.coarse = DirectCoarseE(lev)
            self.lev_c = lev
            self.P = p_interp(pc, N)
            self.R = self.P.T
        self.setup_time = time.perf_counter() - t0
        if verbose:
            print(f'  VertexSchwarz3D: {self.npatch} patches x {nk} modes, '
                  f'{"condensed" if condense else "dense"}, stored {self.bytes/1e6:.1f} MB, '
                  f'setup {self.setup_time:.1f}s ({nloc} probes/mode)', flush=True)

    # -- two-level pieces (PMG rules 1-2: R = P^T, multiplicity-weighted) --
    def _restrict(self, x):
        xw = x*self.mw
        t = np.einsum('bj,eijvk->eibvk', self.R, xw)
        cc = np.einsum('ai,eibvk->eabvk', self.R, t)
        return S3.gs(self.lev_c.m, cc)*self.lev_c.mask

    def _prolong(self, xc):
        t = np.einsum('bj,eijvk->eibvk', self.P, xc)
        return np.einsum('ai,eibvk->eabvk', self.P, t)*self.mask

    def _apply_host(self, r):
        r = r*self.mask
        z = np.zeros(self.shape)
        gr = self.g.ravel()
        for k, md in enumerate(self.modes):
            if md is None:
                continue
            rg = np.bincount(gr, weights=(r[..., k]*self.mw[..., k]).ravel(),
                             minlength=self.ndof)
            zg = np.zeros(self.ndof)
            if not self.condense:
                for dofs, solve in md['patches']:
                    zg[dofs] += solve(rg[dofs])
            else:
                eint, esolve, eKIB = md['eint'], md['esolve'], md['eKIB']
                for Bp, emaps, solve in md['patches']:
                    rB = rg[Bp].copy()
                    ys = []
                    for e, cols in emaps:
                        y = esolve[e](rg[eint[e]])
                        ys.append(y)
                        rB[cols] -= eKIB[e].T @ y
                    zB = solve(rB)
                    zg[Bp] += zB
                    for (e, cols), y in zip(emaps, ys):
                        zg[eint[e]] += y - esolve[e](eKIB[e] @ zB[cols])
            z[..., k] = zg[self.g]
        z *= self.mask
        if self.coarse is not None:
            z = z + self._prolong(self.coarse(self._restrict(r)))
        return z

    def __call__(self, r):
        if DEV.is_tensor(r) or DEV.is_cupy(r):
            return DEV.to_device(self._apply_host(DEV.to_host(r)), r)
        return self._apply_host(np.asarray(r))


# =============================================================================
# Batched, factor-sharing vertex-patch Schwarz (torch, fp64)
# =============================================================================
#
# Same preconditioner as VertexSchwarz3D, restructured for the device
# (BALANCED_CONDENSED_PLAN.md step 4.5):
#
#  * PADDING.  Every element block keeps all n*n*14 local dofs and every patch
#    keeps all dofs of its (2N+1)^2 (or (2N+1)(N+1) at a wall) node grid; a
#    prescribed dof gets a unit diagonal and zero coupling.  All blocks of a
#    kind then have one size, so one batched solve serves all of them.
#  * SHARING.  With identical elements and constant coefficients (no
#    convection in the implicit operator) the element matrix depends only on
#    the mode and on which local dofs are masked.  Elements are grouped by
#    mask pattern and the group's blocks are CHECKED equal (else the group is
#    split); patch Schur complements are built in a canonical patch-grid
#    ordering and deduplicated by comparison.  On the 6x18 channel that is
#    3-4 element types and 3-4 patch types per mode instead of 108 / 114.
#  * ONE BACK-SUBSTITUTION PER ELEMENT.  The interior correction is linear in
#    the edge solution, so the four patches containing an element are summed
#    first (C_e = sum of their edge solutions on e's edge dofs) and the
#    interior solve is done once: z_I = 4 y - K_II^-1 K_IE C_e.
#  * APPLY = index gathers, batched torch.cholesky_solve with many right-hand
#    sides (level-3 work), index_add_ scatters.  fp64 throughout.  The coarse
#    term is the same DirectCoarseE on the host.
#
# Device: kernels_torch.device() (LSSEM3D_DEVICE, else CUDA if present, else
# CPU) -- the Mac tests the batched path on the CPU in fp64.


def _build_coarse(mesh, nk, nz, nu, c, kz, kap, rw, pin_p, mask, pc):
    mm = coarsen_mesh(mesh, pc)
    if mask is not None:
        lm = BC.build_mask(mm, nk, pin_p=False, nz=nz)
        BC.pin_dof(mm, lm, OP.P_, 0)
        BC.pin_dof(mm, lm, OP.NVAR + OP.P_, 0)
    else:
        lm = None
    lev = _Level(mm, nk, nz, nu, c, kz, kap, rw, pin_p, mask=lm)
    P = p_interp(pc, mesh.N)
    return lev, DirectCoarseE(lev), P


class _DenseCoarseDevice:
    """Exact p = pc coarse solve with a DENSE fp64 Cholesky factor per mode,
    resident on the device, so the coarse term costs no host round trip.
    Channel at p = 2: 6216 dofs per mode, 0.3 GB per mode in double -- fine on
    a 40 GB A100, so it is the default when the device is CUDA; on the CPU the
    sparse host solver (DirectCoarseE) stays the default."""

    def __init__(self, level, dev, dtype=None):
        import torch
        m, shape, mask = level.m, level.shape, level.mask
        nelem, n, _, nvar, nk = shape
        nloc = n*n*nvar
        Aloc = np.empty((nelem, nk, nloc, nloc))
        for col in range(nloc):
            i, j, f = np.unravel_index(col, (n, n, nvar))
            v = np.zeros(shape); v[:, i, j, f, :] = 1.0
            Aloc[:, :, :, col] = np.moveaxis(level.A_un(v), -1, 1).reshape(nelem, nk, nloc)
        gidx = m.gidx
        nnode = int(gidx.max()) + 1
        ndof = nnode*nvar
        gd = (gidx[..., None]*nvar + np.arange(nvar)).reshape(nelem, nloc)
        rows = np.repeat(gd, nloc, axis=1).ravel(); cols = np.tile(gd, (1, nloc)).ravel()
        # The EXPLICIT INVERSE is stored, not the Cholesky factor: a
        # single-right-hand-side triangular solve is a sequential level-2
        # operation and 34 of them (two per mode) at n = 6216 cost ~100 ms on an
        # A100 regardless of its bandwidth; one batched GEMV with the inverse
        # reads the same 5.3 GB once and is bandwidth-bound (~4 ms on an A100,
        # ~50 ms on the GB10).  The inverse of an SPD matrix from its Cholesky
        # factor is accurate to kappa*eps, ample for a preconditioner.
        # STORAGE PRECISION.  The coarse inverse is read in full on every CG
        # iteration -- 5.3 GB for the production channel -- and on a bandwidth-
        # bound device that read IS the apply (4.3 ms of 8.3 on an A100).
        # Halving it exactly would need a batched symmetric matvec, which torch
        # does not have (cholesky_solve's triangular solves are sequential and
        # were 12x slower; that is why the explicit inverse is stored at all).
        # What is available is storage precision: the preconditioner is an
        # approximation by construction and does not enter the answer -- CG
        # requires only that it be a fixed SPD operator -- so holding the coarse
        # inverse in fp32 halves the traffic while the state, the operator and
        # the solution stay fp64.  Off by default: it changes the preconditioner,
        # so it must be measured (iterations, and the channel's step-matches-
        # Jacobi gate) before it is trusted.
        st_dtype = torch.float64 if dtype is None else dtype
        Ainv = torch.empty((nk, ndof, ndof), dtype=st_dtype, device=dev)
        for k in range(nk):
            A = np.zeros((ndof, ndof))
            np.add.at(A, (rows, cols), Aloc[:, k].reshape(-1))
            d = np.diag(A); dead = np.abs(d) <= 1e-300
            A[dead, dead] = 1.0
            A = 0.5*(A + A.T)
            Ainv[k] = torch.cholesky_inverse(torch.linalg.cholesky(torch.as_tensor(A, device=dev))).to(st_dtype)
        self.Ainv, self.ndof, self.gd = Ainv, ndof, torch.as_tensor(gd.ravel(), device=dev)
        self.st_dtype = st_dtype
        self.mw = torch.as_tensor(level.mw, device=dev)
        self.mask = torch.as_tensor(level.mask, device=dev)
        self.shape = shape
        self.bytes = Ainv.element_size()*Ainv.numel()

    def __call__(self, r):
        import torch
        nelem, n, _, nvar, nk = self.shape
        rw = (r*self.mw).reshape(nelem*n*n*nvar, nk).T.contiguous()          # (nk, nlocal)
        g = torch.zeros((nk, self.ndof), dtype=r.dtype, device=r.device)
        g.index_add_(1, self.gd, rw)
        if self.st_dtype != g.dtype:                 # fp32 factor, fp64 everything else
            z = torch.bmm(self.Ainv, g[:, :, None].to(self.st_dtype))[:, :, 0].to(g.dtype)
        else:
            z = torch.bmm(self.Ainv, g[:, :, None])[:, :, 0]                   # (nk, ndof): one batched GEMV
        return z[:, self.gd].T.reshape(self.shape)*self.mask


class VertexSchwarzBatched3D:
    name = 'vsbatch'

    def __init__(self, mesh, nk, nz, nu, c, kz, kap=0.0, rw=None, mask=None,
                 pin_p=False, coarse='element', pc=2, device=None, verbose=False,
                 share=True, coarse_dense=None, coarse_fp32=False):
        import time
        import torch
        from . import backend as BK
        t0 = time.perf_counter()
        n, N = mesh.N + 1, mesh.N
        nvar = OP.NVAR_R
        nloc = n*n*nvar
        nelem = mesh.nelem
        self.m, self.nk, self.c = mesh, nk, c
        self.D = diff_matrix(N)
        self.shape = (nelem, n, n, nvar, nk)
        self.mask = (np.array(mask, copy=True) if mask is not None
                     else BC.build_mask(mesh, nk, pin_p=pin_p, nz=nz))
        kz = np.asarray(kz, dtype=float)
        mult = S3.gs(mesh, np.ones(self.shape))
        self.mw = 1.0/np.where(mult < 1e-10, 1.0, mult)
        gid = mesh.gidx
        nnode = int(gid.max()) + 1
        self.g = (gid[..., None]*nvar + np.arange(nvar)).astype(np.int64)
        self.ndof = nnode*nvar
        gflat = self.g.reshape(nelem, -1)
        loc_int = np.zeros((n, n, nvar), bool); loc_int[1:N, 1:N, :] = True
        loc_int = loc_int.ravel()
        I_loc, E_loc = np.flatnonzero(loc_int), np.flatnonzero(~loc_int)
        self.nI, self.nE = I_loc.size, E_loc.size
        eint_g, eedge_g = gflat[:, I_loc], gflat[:, E_loc]
        self.dev = torch.device(device) if device is not None else __import__('lssem3d.kernels_torch', fromlist=['device']).device()
        self.timing = None; self._t0 = None
        T = lambda a: torch.as_tensor(np.ascontiguousarray(a), device=self.dev)
        Ti = lambda a: torch.as_tensor(np.ascontiguousarray(a, dtype=np.int64), device=self.dev)

        # ---------------- topology (mode independent) ----------------
        corners, node_elems = {}, {}
        slot_of = {(N, N): 0, (0, N): 1, (N, 0): 2, (0, 0): 3}           # element is SW/SE/NW/NE of the vertex
        for e in range(nelem):
            for (a, b) in slot_of:
                corners.setdefault(int(gid[e, a, b]), []).append((slot_of[(a, b)], e, a, b))
            for gn in np.unique(gid[e]):
                node_elems.setdefault(int(gn), []).append(e)
        slot_off = {0: (0, 0), 1: (N, 0), 2: (0, N), 3: (N, N)}
        px = getattr(mesh, 'periodic_x', None); py = getattr(mesh, 'periodic_y', None)
        xc = 0.5*(mesh.xnod[:, 0] + mesh.xnod[:, -1]); yc = 0.5*(mesh.ynod[:, 0] + mesh.ynod[:, -1])
        def rel(e2, xv, yv):
            dx, dy = xc[e2] - xv, yc[e2] - yv
            if px: dx = (dx + px/2) % px - px/2
            if py: dy = (dy + py/2) % py - py/2
            return (round(float(dx), 6), round(float(dy), 6))
        patches = []      # dict(slots={s: e}, edofs, ring, present, ringrel=[(dx,dy,e2)])
        for v, lst in corners.items():
            node = -np.ones((2*N+1, 2*N+1), dtype=np.int64)
            slots = {}
            s0, e0, a0, b0 = lst[0]
            xv, yv = float(mesh.xnod[e0, a0]), float(mesh.ynod[e0, b0])
            for s, e, a, b in lst:
                ox, oy = slot_off[s]
                node[ox:ox+n, oy:oy+n] = gid[e]
                slots[s] = e
            II, JJ = np.meshgrid(np.arange(2*N+1), np.arange(2*N+1), indexing='ij')
            edge = (node >= 0) & ((II % N == 0) | (JJ % N == 0))
            enodes = node[edge]
            _, first = np.unique(enodes, return_index=True)
            enodes = enodes[np.sort(first)]
            edofs = (enodes[:, None]*nvar + np.arange(nvar)).ravel()
            ring = sorted({e2 for e in slots.values() for gn in np.unique(gid[e]) for e2 in node_elems[int(gn)]})
            patches.append(dict(slots=slots, edofs=edofs, ring=ring, present=tuple(sorted(slots)),
                                ringrel=[rel(e2, xv, yv) + (e2,) for e2 in ring]))
        self.npatch = len(patches)
        npe = np.zeros(nelem, dtype=np.int64)
        for P_ in patches:
            for e in P_['slots'].values(): npe[e] += 1

        # ---------------- element types per mode from the mask pattern ----------------
        mloc_all = self.mask.reshape(nelem, nloc, nk) > 0.5                   # (nelem, nloc, nk)
        etype = np.zeros((nk, nelem), dtype=np.int64); ereps = []
        for k in range(nk):
            keys = {}
            for e in range(nelem):
                key = mloc_all[e, :, k].tobytes()
                if key not in keys: keys[key] = len(keys)
                etype[k, e] = keys[key]
            reps = np.zeros(len(keys), dtype=np.int64)
            for e in range(nelem - 1, -1, -1): reps[etype[k, e]] = e
            ereps.append(reps)
        ntypes = [len(r) for r in ereps]

        # ---------------- probe: all modes at once, representatives kept, sharing verified ----------------
        # With identical elements and constant coefficients (no convection in the
        # implicit operator) an element's block depends only on its mask pattern;
        # every column of every non-representative element is compared against
        # its representative as it comes out of the probe, so the assumption is
        # checked, not trusted (the check is what makes a graded or curved mesh
        # fail loudly instead of quietly).
        use_dev = self.dev.type == 'cuda' and BK.get_backend() in ('torch', 'cuda')
        if use_dev:
            Dd, fx, fy, kzd, md, wqd = (T(self.D), T(mesh.facx), T(mesh.facy), T(kz), T(self.mask), T(mesh.wq))
            v = torch.zeros(self.shape, dtype=torch.float64, device=self.dev)
            rep_blocks = [torch.zeros((ntypes[k], nloc, nloc), dtype=torch.float64, device=self.dev) for k in range(nk)]
            et_t = [Ti(etype[k]) for k in range(nk)]; er_t = [Ti(ereps[k]) for k in range(nk)]
            maxdiff = torch.zeros(nk, dtype=torch.float64, device=self.dev)
        else:
            Dd, fx, fy, kzd, md, wqd = self.D, mesh.facx, mesh.facy, kz, self.mask, mesh.wq
            v = np.zeros(self.shape)
            rep_blocks = [np.zeros((ntypes[k], nloc, nloc)) for k in range(nk)]
            maxdiff = np.zeros(nk)
        for col in range(nloc):
            i, j, f = np.unravel_index(col, (n, n, nvar))
            v[:] = 0.0
            v[:, i, j, f, :] = 1.0
            out = S3.normal_op(v, Dd, fx, fy, kzd, nu, c, None, md, wqd, kap, rw)
            o = out.reshape(nelem, nloc, nk)
            for k in range(nk):
                ok = o[:, :, k]
                if use_dev:
                    rr = ok[er_t[k]]                                            # (ntypes, nloc)
                    rep_blocks[k][:, :, col] = rr
                    maxdiff[k] = torch.maximum(maxdiff[k], (ok - rr[et_t[k]]).abs().max())
                else:
                    rr = ok[ereps[k]]
                    rep_blocks[k][:, :, col] = rr
                    maxdiff[k] = max(maxdiff[k], np.abs(ok - rr[etype[k]]).max())
        if use_dev:
            maxdiff = maxdiff.cpu().numpy(); rep_blocks = [b.cpu().numpy() for b in rep_blocks]
        scale = max(np.abs(b).max() for b in rep_blocks)
        if share and maxdiff.max() > 1e-11*scale:
            raise RuntimeError(f'element blocks differ within a mask type (max {maxdiff.max():.2e} vs scale {scale:.2e}): '
                               'non-uniform mesh or non-constant coefficients -- use VertexSchwarz3D')
        self.probe_time = time.perf_counter() - t0

        # ---------------- per mode: element factors, patch Schur complements (types only) ----------------
        self.modes = []
        self.bytes = 0
        self.n_etypes, self.n_ptypes = [], []
        locB = -np.ones(self.ndof, dtype=np.int64)
        for k in range(nk):
            free = np.zeros(self.ndof, bool)
            np.logical_or.at(free, gflat.ravel(), mloc_all[:, :, k].ravel())
            if not free.any():
                self.modes.append(None); self.n_etypes.append(0); self.n_ptypes.append(0)
                continue
            # element types: padded block, interior factor, couplings, Schur contribution Q_t
            efac, Kt, Qt = [], [], []
            for t in range(ntypes[k]):
                K = rep_blocks[k][t].copy()
                dead = ~mloc_all[ereps[k][t], :, k]
                K[dead, dead] = 1.0
                Kt.append(K)
                KII = K[np.ix_(I_loc, I_loc)]
                sc = 1.0/np.sqrt(np.diag(KII))
                L = np.linalg.cholesky(KII*sc[:, None]*sc[None, :])
                KEI = K[np.ix_(E_loc, I_loc)]
                Y = sc[:, None]*np.linalg.solve(L.T, np.linalg.solve(L, sc[:, None]*KEI.T))   # K_II^-1 K_IE
                Qt.append(KEI @ Y)
                # EXPLICIT INVERSE of the scaled block, applied as a GEMM: a batched
                # triangular solve runs as thousands of small MAGMA kernels per apply
                # (profiled: 62 % of device time, ~3200 launches on the GB10) while a
                # GEMM is a few cutlass launches on the fp64 tensor cores.  Storage
                # is the same (the factor was stored as a full matrix); accuracy
                # kappa*eps of the equilibrated block, ample for a preconditioner.
                Linv = torch.cholesky_inverse(T(L))
                efac.append((T(sc), Linv, T(KEI), T(KEI.T.copy())))
                self.bytes += 8*(L.size + 2*KEI.size)
            # patch types by signature; Schur complement assembled for the representative only
            sigs, ptype = {}, np.zeros(self.npatch, dtype=np.int64)
            for pi, P_ in enumerate(patches):
                sig = (P_['present'], tuple(int(etype[k, P_['slots'][s]]) for s in P_['present']),
                       tuple(sorted((dx, dy, int(etype[k, e2])) for dx, dy, e2 in P_['ringrel'])), P_['edofs'].size)
                if sig not in sigs: sigs[sig] = (len(sigs), pi)
                ptype[pi] = sigs[sig][0]
            def assemble(pi):
                P_ = patches[pi]; edofs = P_['edofs']; nb = edofs.size
                locB[:] = -1; locB[edofs] = np.arange(nb)
                S = np.zeros((nb, nb))
                for e2 in P_['ring']:
                    li = locB[gflat[e2]]; kp = li >= 0
                    S[np.ix_(li[kp], li[kp])] += Kt[etype[k, e2]][np.ix_(kp, kp)]
                cols_by_slot = {}
                for s_, e in P_['slots'].items():
                    cols = locB[eedge_g[e]]
                    S[np.ix_(cols, cols)] -= Qt[etype[k, e]]
                    cols_by_slot[s_] = cols
                dead = ~free[edofs]
                S[dead, :] = 0.0; S[:, dead] = 0.0; S[dead, dead] = 1.0
                return 0.5*(S + S.T), cols_by_slot
            pfac, pplan = [], []
            for sig, (t, pi) in sigs.items():
                S, cols_by_slot = assemble(pi)
                ps = np.flatnonzero(ptype == t)
                if share and ps.size > 1:                                       # spot check one more patch of the type
                    S2, _ = assemble(int(ps[-1]))
                    if np.abs(S - S2).max() > 1e-11*np.abs(S).max():
                        raise RuntimeError('patch Schur complements differ within a signature type -- use VertexSchwarz3D')
                sc = 1.0/np.sqrt(np.diag(S))
                Sinv = torch.cholesky_inverse(torch.linalg.cholesky(T(S*sc[:, None]*sc[None, :])))
                pfac.append((T(sc), Sinv))
                self.bytes += 8*Sinv.numel()
                gB = np.stack([patches[p]['edofs'] for p in ps])
                slots = [(Ti(np.array([patches[p]['slots'][s_] for p in ps])), Ti(cols_by_slot[s_])) for s_ in sorted(cols_by_slot)]
                pplan.append(dict(ps=ps, gB=Ti(gB), n=ps.size, nb=S.shape[0], slots=slots))
            eplan = []
            for t in range(ntypes[k]):
                es = np.flatnonzero(etype[k] == t)
                eplan.append(dict(es=Ti(es), gI=Ti(eint_g[es]), n=es.size))
            self.modes.append(dict(efac=efac, eplan=eplan, pfac=pfac, pplan=pplan,
                                   gI=Ti(eint_g), gE=Ti(eedge_g), etype=Ti(etype[k]),
                                   npe=T(npe.astype(float))))
            self.n_etypes.append(ntypes[k]); self.n_ptypes.append(len(sigs))
        self.gflat_t = Ti(gflat)
        self.g_t = Ti(self.g.ravel())
        self.mask_t = T(self.mask)
        self.mw_t = T(self.mw)
        self._group_modes()

        # ---------------- coarse term ----------------
        self.coarse = None; self.coarse_dev = None
        if coarse:
            self.lev_c, host_coarse, self.P = _build_coarse(mesh, nk, nz, nu, c, kz, kap, rw, pin_p, mask, pc)
            self.R = self.P.T
            if coarse_dense is None:
                coarse_dense = (self.dev.type == 'cuda')
            if coarse_dense:
                import torch as _t
                self.coarse_dev = _DenseCoarseDevice(
                    self.lev_c, self.dev, dtype=(_t.float32 if coarse_fp32 else None))
                self.P_t, self.R_t = T(self.P), T(self.R)
                self.mask_c_t = T(self.lev_c.mask)
                self.bytes += self.coarse_dev.bytes
            else:
                self.coarse = host_coarse
        self.setup_time = time.perf_counter() - t0
        if verbose:
            print(f'  VertexSchwarzBatched3D: {self.npatch} patches x {nk} modes, element types/mode {self.n_etypes}, '
                  f'patch types/mode {self.n_ptypes}, factors {self.bytes/1e6:.1f} MB on {self.dev}, '
                  f'coarse {"dense on device" if self.coarse_dev is not None else "sparse on host"}, '
                  f'setup {self.setup_time:.1f}s (probes {self.probe_time:.1f}s, {"device" if use_dev else "host"} operator)', flush=True)

    def _restrict(self, x):
        xw = x*self.mw
        t = np.einsum('bj,eijvk->eibvk', self.R, xw)
        cc = np.einsum('ai,eibvk->eabvk', self.R, t)
        return S3.gs(self.lev_c.m, cc)*self.lev_c.mask

    def _prolong(self, xc):
        t = np.einsum('bj,eijvk->eibvk', self.P, xc)
        return np.einsum('ai,eibvk->eabvk', self.P, t)*self.mask

    def _coarse_device(self, r):
        """Coarse correction entirely on the device: restrict, dense solve, prolong."""
        import torch
        xw = r*self.mw_t
        t = torch.einsum('bj,eijvk->eibvk', self.R_t, xw)
        cc = torch.einsum('ai,eibvk->eabvk', self.R_t, t)
        rc = DEV.gs_torch(self.lev_c.m, cc)*self.mask_c_t
        zc = self.coarse_dev(rc)
        t = torch.einsum('bj,eijvk->eibvk', self.P_t, zc)
        return torch.einsum('ai,eibvk->eabvk', self.P_t, t)*self.mask_t

    def _group_modes(self):
        """Stack the factors of modes that share the same index plans, so one
        batched call serves all of them (17 modes -> 2-3 groups on the channel:
        k = 0 with its pin, the Nyquist mode, and everything else), and build
        FLAT index plans for the group: elements and patches reordered so each
        type is a contiguous slice, one padded gather map for all patch edge
        dofs, and one (src, dst) scatter map for every (patch, slot) coupling.
        The apply is then a fixed sequence of ~10 gathers/scatters plus one
        GEMM per type, instead of ~25 index operations per patch type."""
        import torch
        Ti = lambda a: torch.as_tensor(np.ascontiguousarray(a, dtype=np.int64), device=self.dev)
        nelem = self.shape[0]
        groups = {}
        for k, md in enumerate(self.modes):
            if md is None:
                continue
            sig = (tuple(int(p['n']) for p in md['eplan']),
                   tuple((int(p['n']), int(p['nb'])) for p in md['pplan']),
                   bytes(md['etype'].cpu().numpy()),
                   tuple(bytes(p['ps']) for p in md['pplan']))
            groups.setdefault(sig, []).append(k)
        self.groups = []
        for sig, ks in groups.items():
            m0 = self.modes[ks[0]]
            efac = [tuple(torch.stack([self.modes[k]['efac'][t][j] for k in ks]) for j in range(4))
                    for t in range(len(m0['eplan']))]
            pfac = [tuple(torch.stack([self.modes[k]['pfac'][t][j] for k in ks]) for j in range(2))
                    for t in range(len(m0['pplan']))]
            # ---- elements: sorted by type, contiguous slices ----
            eorder = np.concatenate([pl['es'].cpu().numpy() for pl in m0['eplan']])
            pos = np.empty(nelem, dtype=np.int64); pos[eorder] = np.arange(nelem)
            eslices, e0 = [], 0
            for pl in m0['eplan']:
                eslices.append((e0, e0 + int(pl['n']))); e0 += int(pl['n'])
            gI_sorted = m0['gI'].cpu().numpy()[eorder]                     # (nelem, nI)
            npe_sorted = m0['npe'].cpu().numpy()[eorder]
            # ---- patches: sorted by type, padded edge-dof gather map, slot scatter map ----
            nb_max = max(int(pl['nb']) for pl in m0['pplan'])
            npatch = sum(int(pl['n']) for pl in m0['pplan'])
            gB_all = np.full((npatch, nb_max), self.ndof, dtype=np.int64)  # pad -> dummy zero dof
            pslices, src, dst, p0 = [], [], [], 0
            for pl in m0['pplan']:
                n_t, nb = int(pl['n']), int(pl['nb'])
                gB_all[p0:p0+n_t, :nb] = pl['gB'].cpu().numpy()
                ar = np.arange(self.nE)
                for es, cols in pl['slots']:
                    es_np, cols_np = es.cpu().numpy(), cols.cpu().numpy()
                    src.append((pos[es_np][:, None]*self.nE + ar[None, :]).ravel())
                    dst.append(((p0 + np.arange(n_t))[:, None]*nb_max + cols_np[None, :]).ravel())
                pslices.append((p0, p0 + n_t, nb)); p0 += n_t
            self.groups.append(dict(ks=torch.as_tensor(ks, device=self.dev), nm=len(ks), efac=efac, pfac=pfac,
                                    eslices=eslices, pslices=pslices, gI=Ti(gI_sorted), gI_flat=Ti(gI_sorted.ravel()),
                                    npe=torch.as_tensor(npe_sorted, device=self.dev),
                                    gB=Ti(gB_all), gB_flat=Ti(gB_all.ravel()), nb_max=nb_max, npatch=npatch,
                                    src=Ti(np.concatenate(src)), dst=Ti(np.concatenate(dst))))
        self.n_groups = len(self.groups)
        self.use_graph = (self.dev.type == 'cuda' and os.environ.get('LSSEM3D_VSB_GRAPH', '1') not in ('0', 'false', 'False'))
        self._graph = None

    def _tick(self, key):
        """Optional phase profiler: set self.timing = {} to accumulate synchronized
        milliseconds per phase (device work included) across applies."""
        if self.timing is None:
            return
        import time, torch
        if self.dev.type == 'cuda':
            torch.cuda.synchronize(self.dev)
        now = time.perf_counter()
        if self._t0 is not None:
            self.timing[key] = self.timing.get(key, 0.0) + (now - self._t0)*1e3
        self._t0 = now

    def _apply_patches(self, r):
        """r: torch (nelem, n, n, 14, nk) on self.dev -> z same shape (patch part only).
        Flat plans: per mode group, 2 gathers + 1 scatter for the edge assembly,
        one GEMM (+ scale/transposes) per block type, 3 scatters back."""
        import torch
        nelem = self.shape[0]
        self._t0 = None; self._tick('start')
        z = torch.zeros_like(r)
        rm = (r*self.mask_t*self.mw_t).reshape(-1, self.nk)               # (nlocal, nk)
        for G in self.groups:
            ks, nm = G['ks'], G['nm']
            rg = torch.zeros((nm, self.ndof + 1), dtype=r.dtype, device=self.dev)   # +1: dummy zero dof for padding
            rg.index_add_(1, self.g_t, rm[:, ks].T.contiguous())
            self._tick('gather')
            Y = torch.empty((nm, nelem, self.nI), dtype=r.dtype, device=self.dev)   # elements in sorted order
            F = torch.empty((nm, nelem, self.nE), dtype=r.dtype, device=self.dev)
            for t, (e0, e1) in enumerate(G['eslices']):
                sc, Linv, KEI, KIE = G['efac'][t]                             # (nm,nI) (nm,nI,nI) (nm,nE,nI) (nm,nI,nE)
                B = rg[:, G['gI'][e0:e1]].transpose(1, 2)*sc[:, :, None]      # (nm, nI, n_t)
                Yt = (Linv @ B)*sc[:, :, None]                                # K_II^-1 B, one batched GEMM
                Y[:, e0:e1] = Yt.transpose(1, 2)
                F[:, e0:e1] = (KEI @ Yt).transpose(1, 2)                     # K_EI y
            self._tick('interior')
            # edge right-hand sides of ALL patches at once: gather, then one scatter of the couplings
            rB = rg[:, G['gB']]                                               # (nm, npatch, nb_max)
            rBf = rB.view(nm, -1)
            rBf.index_add_(1, G['dst'], -F.view(nm, -1)[:, G['src']])
            zB = torch.zeros_like(rB)
            for t, (p0, p1, nb) in enumerate(G['pslices']):
                sc, Sinv = G['pfac'][t]                                       # (nm,nb) (nm,nb,nb)
                X = (rB[:, p0:p1, :nb]*sc[:, None, :]).transpose(1, 2)        # (nm, nb, n_t)
                zB[:, p0:p1, :nb] = ((Sinv @ X)*sc[:, :, None]).transpose(1, 2)
            zg = torch.zeros((nm, self.ndof + 1), dtype=r.dtype, device=self.dev)
            zg.index_add_(1, G['gB_flat'], zB.view(nm, -1))
            C = torch.zeros((nm, nelem*self.nE), dtype=r.dtype, device=self.dev)
            C.index_add_(1, G['src'], zB.view(nm, -1)[:, G['dst']])
            C = C.view(nm, nelem, self.nE)
            self._tick('patch')
            zI = torch.empty_like(Y)
            for t, (e0, e1) in enumerate(G['eslices']):
                sc, Linv, KEI, KIE = G['efac'][t]
                Gm = (KIE @ C[:, e0:e1].transpose(1, 2))*sc[:, :, None]      # (nm, nI, n_t)
                W = (Linv @ Gm)*sc[:, :, None]
                zI[:, e0:e1] = G['npe'][None, e0:e1, None]*Y[:, e0:e1] - W.transpose(1, 2)
            zg.index_add_(1, G['gI_flat'], zI.view(nm, -1))
            z[..., ks] = zg[:, :self.ndof][:, self.g_t].T.reshape(self.shape[:-1] + (nm,))
            self._tick('backsub+scatter')
        return z*self.mask_t

    def _apply_device(self, rt):
        z = self._apply_patches(rt)
        if self.coarse_dev is not None:
            z = z + self._coarse_device(rt)
            self._tick('coarse')
        return z

    def __call__(self, r):
        import torch
        was_tensor = DEV.is_tensor(r)
        rt = r.to(self.dev) if was_tensor else torch.as_tensor(np.asarray(r), device=self.dev)
        if self.use_graph and self.timing is None:
            # CUDA graph: the apply is a fixed sequence of ~100 kernels on static
            # buffers; capturing it once removes the per-launch host cost (which
            # was ~half of the A100 apply).  Warm-up on a side stream, then replay.
            if self._graph is None:
                self._static_in = rt.clone()
                s_ = torch.cuda.Stream(self.dev)
                s_.wait_stream(torch.cuda.current_stream(self.dev))
                with torch.cuda.stream(s_):
                    for _ in range(2):
                        self._apply_device(self._static_in)
                torch.cuda.current_stream(self.dev).wait_stream(s_)
                self._graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self._graph):
                    self._static_out = self._apply_device(self._static_in)
            self._static_in.copy_(rt)
            self._graph.replay()
            z = self._static_out.clone()
        else:
            z = self._apply_device(rt)
        if self.coarse_dev is None and self.coarse is not None:
            rh = DEV.to_host(r) if was_tensor else np.asarray(r)
            zc = self._prolong(self.coarse(self._restrict(rh)))
            z = z + torch.as_tensor(zc, device=self.dev)
        if was_tensor:
            return z.to(r.device)
        return z.cpu().numpy()
