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

import numpy as np

from lssem2d.lgl import lgl_nodes, diff_matrix
from . import operator as OP
from . import bc as BC
from . import solver3d as S3

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

    def __init__(self, mesh, nk, nz, nu, c, kz, kap, rw, pin_p):
        self.m, self.nk = mesh, nk
        n = mesh.N + 1
        self.shape = (mesh.nelem, n, n, OP.NVAR_R, nk)
        self.D = diff_matrix(mesh.N)
        self.mask = BC.build_mask(mesh, nk, pin_p=pin_p, nz=nz)
        self.A = lambda v: S3.normal_op(v, self.D, mesh.facx, mesh.facy, kz, nu,
                                        c, mesh, self.mask, mesh.wq, kap, rw)
        diag = S3.jacobi_diagonal_analytic(self.shape, self.D, mesh.facx,
                                           mesh.facy, kz, nu, c, mesh,
                                           self.mask, mesh.wq, kap, rw)
        self.M_inv = S3.jacobi_inverse(diag, self.mask)
        mult = S3.gs(mesh, np.ones(self.shape))
        self.mw = 1.0/np.where(mult < 1e-10, 1.0, mult)


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
                 optimised=True):
        assert len(orders) >= 2 and list(orders) == sorted(orders, reverse=True), \
            f'orders must be descending, got {orders}'
        assert orders[0] == mesh.N, f'finest order {orders[0]} != mesh.N {mesh.N}'
        self.levels, self.P, self.R = [], [], []
        meshes = [mesh] + [coarsen_mesh(mesh, p) for p in orders[1:]]
        for mm in meshes:
            self.levels.append(_Level(mm, nk, nz, nu, c, kz, kap, rw, pin_p))
        for lf, lc in zip(orders[:-1], orders[1:]):
            P = p_interp(lc, lf)          # coarse -> fine, nodal interpolation
            self.P.append(P)
            self.R.append(P.T)            # restriction is the ADJOINT (rule 1)
        self.smooth = [Chebyshev4(l.A, l.M_inv, l.shape, deg=deg,
                                  optimised=optimised)
                       for l in self.levels[:-1]]
        # coarsest: a FIXED linear operator (rule 3)
        lc = self.levels[-1]
        self.coarse = Chebyshev4(lc.A, lc.M_inv, lc.shape, deg=coarse_deg,
                                 optimised=optimised)

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
