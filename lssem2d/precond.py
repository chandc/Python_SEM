"""Preconditioners for the LSSEM VVP operator.

Three options, selectable at the call site:

  Jacobi       z = M_inv * r                      (the existing behaviour)
  Chebyshev4   degree-k 4th-kind Chebyshev polynomial in D^-1 A
  PMG2         two-level p-multigrid: Chebyshev on the fine order,
               coarse solve at order pmin, p-interpolation between

Why bother: the VVP outflow pressure lives in a very soft direction of A
(measured ~8e3x softer than a generic direction on the Chan mesh).  A diagonal
preconditioner rescales pointwise and cannot touch such near-null modes, so the
solver stops with them unresolved and the converged state becomes path
dependent.  A polynomial smoother damps a broad spectral band; a coarse solve
removes what is left.  This mirrors SEM_08_bfs / solver_pmg2.f90, which uses
4th-kind Chebyshev (degree 10) over a 3-level p-hierarchy p=10 -> 4 -> 2.

Reference for the optimised weights: Phillips & Fischer / Lottes, "Optimal
Chebyshev smoothers", Table 5 -- reproduced in solver_pmg2.f90 as beta4.
"""
import numpy as np

from .solver import apply_A
from .lssem import SolverState
from .lgl import lgl_nodes, diff_matrix
from .assembly import gather_scatter


# --- optimised 4th-kind Chebyshev weights, beta4[deg][k], k = 1..deg ----------
# Same table as solver_pmg2.f90.  beta = 1 everywhere reduces to plain 4th-kind.
_BETA4 = {
    1:  [1.12500000000000],
    2:  [1.02387287570313, 1.26408905371085],
    3:  [1.00842544782028, 1.08867839208730, 1.33753125909618],
    4:  [1.00391310427285, 1.04035811188593, 1.14863498546254, 1.38268869241000],
    5:  [1.00212930146164, 1.02173711549260, 1.07872433192603, 1.19810065292663,
         1.41322542791682],
    6:  [1.00128517255940, 1.01304293035233, 1.04678215124113, 1.11616489419675,
         1.23829020218444, 1.43524297106744],
    7:  [1.00083464397912, 1.00843949430122, 1.03008707768713, 1.07408384092003,
         1.15036186707366, 1.27116474046139, 1.45186658649364],
}


def estimate_lambda_max(state, fu, fv, M_inv, pin_p=False, npow=20, seed=0):
    """Power iteration on D^-1 A.  Returns lambda_max (float)."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(M_inv.shape) * (M_inv > 0)
    nv = np.sqrt(np.sum(v * v))
    if nv == 0.0:
        return 1.0
    v /= nv
    lam = 1.0
    for _ in range(npow):
        w = M_inv * apply_A(state, v, fu, fv, pin_p=pin_p)
        nw = np.sqrt(np.sum(w * w))
        if nw == 0.0:
            break
        lam = nw
        v = w / nw
    return float(lam)


class Jacobi:
    """z = M_inv * r  (the existing diagonal preconditioner)."""

    name = "jacobi"

    def __init__(self, M_inv):
        self.M_inv = M_inv

    def __call__(self, r):
        return self.M_inv * r


class Chebyshev4:
    """Degree-`deg` 4th-kind Chebyshev polynomial in D^-1 A, applied from z=0.

    Needs only an upper bound on the spectrum (no lower edge), which is what
    makes 4th-kind robust when the spectrum is poorly known.  Costs `deg`
    operator applications per preconditioner call.
    """

    name = "chebyshev4"

    def __init__(self, state, fu, fv, M_inv, pin_p=False, deg=6,
                 optimised=True, lam_max=None, safety=1.3, npow=20):
        self.state, self.fu, self.fv = state, fu, fv
        self.M_inv, self.pin_p = M_inv, pin_p
        self.deg = int(deg)
        self.beta = (_BETA4.get(self.deg) if optimised else None) or [1.0]*self.deg
        if lam_max is None:
            lam_max = estimate_lambda_max(state, fu, fv, M_inv, pin_p, npow)
        self.rho = float(safety) * float(lam_max)
        self.n_applies = 0

    def __call__(self, r):
        rho, Mi = self.rho, self.M_inv
        if rho <= 0.0:
            return Mi * r
        z = np.zeros_like(r)
        d = np.zeros_like(r)
        rf = r.copy()
        for k in range(1, self.deg + 1):
            c1 = (2.0*k - 3.0) / (2.0*k + 1.0)
            c2 = self.beta[k-1] * (8.0*k - 4.0) / ((2.0*k + 1.0) * rho)
            d = c1 * d + c2 * (Mi * rf)
            z += d
            rf -= apply_A(self.state, d, self.fu, self.fv, pin_p=self.pin_p)
            self.n_applies += 1
        return z


# ---------------------------------------------------------------------------
#  two-level p-multigrid
# ---------------------------------------------------------------------------
def _p_interp(p_from, p_to):
    """1D Lagrange interpolation matrix, LGL(p_from) nodes -> LGL(p_to) nodes."""
    xs = lgl_nodes(p_from)
    xt = lgl_nodes(p_to)
    n = len(xs)
    C = np.empty((len(xt), n))
    for a, x in enumerate(xt):
        w = np.ones(n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    w[i] *= (x - xs[j]) / (xs[i] - xs[j])
        C[a] = w
    return C


class DirectCoarse:
    """EXACT coarse solve: assemble the coarse operator once, factorise, reuse.

    Why this may matter.  PMG2 uses a polynomial smoother on the fine level AND
    a polynomial on the coarse level, so NOTHING in the cycle actually inverts
    the soft direction.  Polynomial smoothers damp a spectral BAND; the soft end
    is exactly what a coarse solve is meant to remove -- and the soft outflow
    pressure mode (~8e3x softer than generic) is the reason PMG2 exists at all.

    Like Chebyshev4 this is a FIXED LINEAR operator, so it is admissible as a CG
    preconditioner.  An inner CG would NOT be: its polynomial depends on the
    right-hand side, which destroys the symmetry the outer CG needs.

    ASSEMBLY IS BY GLOBAL PROBING of apply_A -- one application per free coarse
    DOF.  That is correct BY CONSTRUCTION: whatever apply_A does (masking, the
    Dong OBC boundary term, the least-squares weights) is in the matrix, with no
    second implementation to drift.  The cost is O(ndof_coarse) applications of
    a small operator, paid once per linearisation.  At pc = 2 that is 260 DOF on
    the Poiseuille mesh and 828 on Gartling's 11x4 -- cheap.  It scales with the
    ELEMENT count, not with p, so it is the mesh that limits this, not the order.
    """

    name = "direct"

    def __init__(self, state, fu, fv, pin_p=False):
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
        m = state.mesh
        n = m.N + 1
        self.flat = m.gidx.ravel()
        ngl = int(self.flat.max()) + 1
        order = np.argsort(self.flat, kind='stable')
        fs = self.flat[order]
        lo = np.searchsorted(fs, np.arange(ngl), 'left')
        hi = np.searchsorted(fs, np.arange(ngl), 'right')
        self.copies = [order[lo[j]:hi[j]] for j in range(ngl)]
        # After gather_scatter every local copy of a node holds the SAME
        # assembled value, so one representative per node reads the global vector.
        self.rep = np.array([c[0] for c in self.copies])
        self.ngl = ngl

        mask = state.get_global_mask(pin_p=pin_p)
        self.free = (mask.reshape(-1, 4)[self.rep].reshape(-1) > 0)
        gd_free = np.nonzero(self.free)[0]

        U = np.zeros((m.nelem, n, n, 4))
        Uf = U.reshape(-1, 4)
        cols = np.empty((gd_free.size, gd_free.size))
        for k, gd in enumerate(gd_free):
            j, f = divmod(int(gd), 4)
            U[:] = 0.0
            # The global basis function is the SUM of its local copies, so the
            # probe must be set on every copy, not just the representative.
            Uf[self.copies[j], f] = 1.0
            out = apply_A(state, U, fu, fv, pin_p=pin_p)
            cols[:, k] = out.reshape(-1, 4)[self.rep].reshape(-1)[self.free]

        self.asym = float(np.abs(cols - cols.T).max() / max(np.abs(cols).max(), 1e-300))
        Ac = 0.5*(cols + cols.T)
        # The coarse operator inherits any null space the FINE one has -- with
        # pin_p=False on a closed domain the pressure is defined only up to a
        # constant, and splu then fails with "Factor is exactly singular".  That
        # is real, not a bug.  A preconditioner does not need the null direction
        # resolved, so shift the diagonal by a relative epsilon and carry on; the
        # shift is ~1e-12 of the diagonal scale and cannot affect a Krylov method
        # whose operator has that direction anyway.
        self._dense = Ac
        self.shift = 0.0
        try:
            self.lu = spla.splu(sp.csc_matrix(Ac))
        except RuntimeError:
            self.shift = 1e-12*float(np.abs(np.diag(Ac)).max())
            self.lu = spla.splu(sp.csc_matrix(Ac + self.shift*np.eye(Ac.shape[0])))
        self.ndof = gd_free.size

    def __call__(self, rc):
        rg = rc.reshape(-1, 4)[self.rep].reshape(-1)
        xg = np.zeros_like(rg)
        xg[self.free] = self.lu.solve(rg[self.free])
        return xg.reshape(self.ngl, 4)[self.flat].reshape(rc.shape)


class AMGCoarse(DirectCoarse):
    """Coarse solve by ONE AMG V-cycle instead of a direct factorisation.

    Same assembly as DirectCoarse -- probing apply_A, correct by construction --
    but the factorisation is replaced by a single smoothed-aggregation V-cycle.
    One V-cycle is a FIXED LINEAR OPERATOR, so it remains admissible as a CG
    preconditioner (see PMG2's contract); an AMG *solve* to a tolerance would not
    be, for the same reason an inner CG is not.

    WHY BOTHER, when a direct solve is exact.  DirectCoarse costs O(ndof_coarse)
    to assemble and factorise, and that scales with the ELEMENT count.  Under
    h-refinement it stops being free -- which is exactly the regime p-multigrid
    does nothing about, since it coarsens in p alone.  This is the standard
    remedy: p-ladder down to p_c, then AMG on the low-order operator.  It is what
    Pazner uses for his coarse solver R_0 (FOSLS_2D_PLAN sec F2h(ii)).

    THE NEAR-NULL SPACE IS NOT OPTIONAL.  FOSLS_2D_PLAN sec F2 measured it: with
    pyamg's default single all-ones vector the iteration count grows almost
    exactly like Jacobi (2.28x vs 2.40x under h-refinement).  One constant PER
    FIELD is what makes it h-independent.  The DOF layout here is node-major with
    4 fields interleaved, so field v occupies indices where idx % 4 == v.
    """

    name = "amgcoarse"

    def __init__(self, state, fu, fv, pin_p=False, **amg_kw):
        import pyamg
        import scipy.sparse as sp
        super().__init__(state, fu, fv, pin_p)
        Ac = self._dense                      # kept by DirectCoarse for reuse
        nf = Ac.shape[0]
        gd = np.nonzero(self.free)[0]
        B = np.zeros((nf, 4))
        for v in range(4):
            B[gd % 4 == v, v] = 1.0           # one constant per FIELD
        kw = dict(B=B, max_coarse=20, smooth='energy')
        kw.update(amg_kw)
        self.ml = pyamg.smoothed_aggregation_solver(sp.csr_matrix(Ac), **kw)
        self.levels = len(self.ml.levels)
        self.opcx = float(self.ml.operator_complexity())
        self._M = self.ml.aspreconditioner()

    def __call__(self, rc):
        rg = rc.reshape(-1, 4)[self.rep].reshape(-1)
        xg = np.zeros_like(rg)
        xg[self.free] = self._M.matvec(rg[self.free])
        return xg.reshape(self.ngl, 4)[self.flat].reshape(rc.shape)


class PMG2:
    """Two-level p-multigrid V-cycle used as a CG preconditioner.

    fine order p, coarse order pc (or a SEQUENCE of orders for >2 levels):
      pre-smooth (Chebyshev)  ->  restrict residual  ->  coarse solve (CG)
      ->  prolong correction  ->  post-smooth (Chebyshev)

    The coarse operator is REDISCRETISED at order pc on the same elements
    (solver_pmg2.f90 offers this as pmg_galerkin=.false.; its production path
    is the Galerkin variant, which is more accurate and much more code).
    """

    name = "pmg2"

    def __init__(self, state, fu, fv, M_inv, pin_p=False, pc=2, deg=6,
                 optimised=True, coarse_deg=10, coarse_solver='chebyshev'):
        # pc may be a SEQUENCE to build a deeper hierarchy: pc=(4, 2) gives
        # p -> 4 -> 2, i.e. three levels, which is what solver_pmg2.f90 runs
        # (p = 10 -> 4 -> 2).  The recursion works because __call__(r) already
        # has exactly the coarse-solver signature -- residual in, correction out
        # -- so a PMG2 built on the coarse state IS a valid coarse solver.
        _rest = ()
        if isinstance(pc, (tuple, list)):
            pc, _rest = int(pc[0]), tuple(pc[1:])
        self.smooth = Chebyshev4(state, fu, fv, M_inv, pin_p, deg=deg,
                                 optimised=optimised)
        self.state, self.fu, self.fv, self.pin_p = state, fu, fv, pin_p
        m = state.mesh
        self.pf = m.N
        self.pc = int(pc)

        # --- coarse mesh: same elements/geometry, lower order -------------
        from copy import copy
        mc = copy(m)
        mc.N = self.pc
        mc.nterm = self.pc + 1
        mc.gidx = -np.ones((m.nelem, mc.nterm, mc.nterm), dtype=int)
        mc.xnod = np.zeros((m.nelem, mc.nterm))
        mc.ynod = np.zeros((m.nelem, mc.nterm))
        mc.wq = np.zeros((m.nelem, mc.nterm, mc.nterm))
        mc.setup_derived()
        mc.compute_global_indices()
        self.mc = mc
        # The coarse operator MUST carry the same least-squares weighting as the
        # fine one.  Omitting w_mom/w_mass sends ls_coeffs down its LEGACY branch
        # on the coarse grid -- (fac1, dt) instead of (fac1*w_mass/dt, w_mom) --
        # so the coarse problem is weighted differently from the fine one and the
        # V-cycle returns a correction to the wrong operator.  With
        # w_mass = w_mom = 1 at dt = 0.1 that is a factor of 10 on BOTH
        # coefficients, and CG needed ~2000 iterations per solve instead of tens.
        # dtau matters for the same reason: it sits on the momentum diagonal.
        self.sc = SolverState(mc, diff_matrix(self.pc), state.nu, state.dt,
                              state.fac1,
                              w_mom=getattr(state, 'w_mom', None),
                              w_mass=getattr(state, 'w_mass', None),
                              dtau=getattr(state, 'dtau', None))

        # --- Dong OBC parameters MUST be carried down too -----------------
        # obc_active() is decided by the MESH (bc == 6 edges), which copy(m)
        # preserves, so apply_A adds the boundary term on BOTH levels.  But the
        # coefficients live on the STATE and default to (w, D0, delta, U0) =
        # (1.0, 0.0, None, 1.0).  Without this loop the coarse rows lose the
        # nu*D0*du/dt term entirely -- D0 sits on the boundary diagonal, so
        # this is precisely the w_mom/w_mass/dtau failure above in a new place:
        # measured (1.0, 1.0, 0.05, 1.0) fine against (1.0, 0.0, None, 1.0)
        # coarse.  See scratch/pmg_coarse_probe.py.
        for _a in ('obc_w', 'obc_D0', 'obc_delta', 'obc_U0', 'obc_picard'):
            if hasattr(state, _a):
                setattr(self.sc, _a, getattr(state, _a))

        # coarse linearisation velocities: restrict fu, fv
        # Prolongation P: coarse -> fine (nodal interpolation).
        # Restriction MUST be its adjoint (R = P^T), otherwise the V-cycle is
        # not symmetric and CG loses its convergence guarantee.  Using an
        # independent fine->coarse interpolation breaks this.
        self.P = _p_interp(self.pc, self.pf)          # (nt_f, nt_c)
        self.R = self.P.T                             # (nt_c, nt_f)
        self.Rnod = _p_interp(self.pf, self.pc)       # nodal, for fu/fv only
        self.fuc = np.ascontiguousarray(self._sample_scalar(fu))
        self.fvc = np.ascontiguousarray(self._sample_scalar(fv))
        self.sc.update_linearisation(self.fuc, self.fvc)
        self.gmc = self.sc.get_global_mask(pin_p=self._map_pin(pin_p))
        multf = gather_scatter(m, np.ones((m.nelem, m.nterm, m.nterm, 4)))
        self.mwf = 1.0 / np.where(multf < 1e-10, 1.0, multf)
        multc = gather_scatter(mc, np.ones((mc.nelem, mc.nterm, mc.nterm, 4)))
        self.mwc = 1.0 / np.where(multc < 1e-10, 1.0, multc)
        from .solver import compute_jacobi
        self.Mic = compute_jacobi(self.sc, self.fuc, self.fvc,
                                  pin_p=self._map_pin(pin_p))
        # Coarse solver.  Either choice must be a FIXED LINEAR operator -- CG
        # is not (its polynomial depends on the right-hand side), which would
        # destroy the symmetry of the whole V-cycle.
        #   'chebyshev' (default)  fixed-degree polynomial -- SMOOTHS the coarse
        #                          problem, does not solve it
        #   'direct'               assemble + factorise -- actually inverts it,
        #                          including the soft modes a polynomial cannot
        #                          reach.  See DirectCoarse.
        if _rest:
            # Deeper hierarchy: the coarse level is itself a V-cycle.  It is a
            # fixed linear operator as long as its own coarse solver is, so the
            # CG symmetry requirement is preserved all the way down.
            self.coarse = PMG2(self.sc, self.fuc, self.fvc, self.Mic,
                               self._map_pin(pin_p), pc=_rest, deg=deg,
                               optimised=optimised, coarse_deg=coarse_deg,
                               coarse_solver=coarse_solver)
        elif str(coarse_solver).lower() in ('amg', 'amgcoarse'):
            self.coarse = AMGCoarse(self.sc, self.fuc, self.fvc,
                                    self._map_pin(pin_p))
        elif str(coarse_solver).lower() in ('direct', 'lu', 'exact'):
            self.coarse = DirectCoarse(self.sc, self.fuc, self.fvc,
                                       self._map_pin(pin_p))
        else:
            self.coarse = Chebyshev4(self.sc, self.fuc, self.fvc, self.Mic,
                                     self._map_pin(pin_p), deg=coarse_deg,
                                     optimised=optimised)

    def _map_pin(self, pin_p):
        if not pin_p or not isinstance(pin_p, tuple):
            return pin_p
        e, i, j = pin_p
        mi = 0 if i == 0 else (self.pc if i == self.pf else self.pc // 2)
        mj = 0 if j == 0 else (self.pc if j == self.pf else self.pc // 2)
        return (e, mi, mj)

    def _restrict(self, x):
        # x is in redundant LOCAL storage where every copy of a shared node
        # already holds the assembled value.  A plain P^T would therefore count
        # each shared node once per owning element.  Divide by the multiplicity
        # first so the transfer is the true adjoint of _prolong.
        xw = x * self.mwf
        t = np.einsum('bj,eijk->eibk', self.R, xw)
        c = np.einsum('ai,eibk->eabk', self.R, t)
        return gather_scatter(self.mc, c)

    def _prolong(self, xc):
        t = np.einsum('bj,eijk->eibk', self.P, xc)
        return np.einsum('ai,eibk->eabk', self.P, t)

    def _coarse_solve(self, rc):
        return self.coarse(rc * self.gmc) * self.gmc

    def _sample_scalar(self, s):
        """Nodal sampling of a scalar field onto the coarse nodes.
        Used only for the linearisation velocities, where we want the
        function value, not an adjoint transfer."""
        t = np.einsum('bj,eij->eib', self.Rnod, s)
        return np.einsum('ai,eib->eab', self.Rnod, t)

    def __call__(self, r):
        z = self.smooth(r)                                   # pre-smooth
        res = r - apply_A(self.state, z, self.fu, self.fv, pin_p=self.pin_p)
        ec = self._coarse_solve(self._restrict(res))         # coarse solve
        z = z + self._prolong(ec)                            # prolong
        res = r - apply_A(self.state, z, self.fu, self.fv, pin_p=self.pin_p)
        z = z + self.smooth(res)                             # post-smooth
        return z


def make(kind, state, fu, fv, M_inv, pin_p=False, **kw):
    """Factory: kind in {'jacobi', 'chebyshev4', 'pmg2'}."""
    k = (kind or 'jacobi').lower()
    if k in ('jacobi', 'diag', 'none'):
        return Jacobi(M_inv)
    if k in ('chebyshev4', 'cheby4', 'cheb'):
        return Chebyshev4(state, fu, fv, M_inv, pin_p, **kw)
    if k in ('pmg2', 'pmg', 'multigrid'):
        return PMG2(state, fu, fv, M_inv, pin_p, **kw)
    raise ValueError(f"unknown preconditioner {kind!r}")
