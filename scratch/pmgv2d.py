"""PMGV: the lssem2d p-multigrid V-cycle (PMG2) with the overlapping
vertex-patch Schwarz as the SMOOTHER on every level whose order is <= p_patch,
Chebyshev on the finer levels, direct solve at the bottom.

    orders  N -> pc[0] -> pc[1] -> ... -> pc[-1] (direct)
    smoother(level p) = damped additive vertex-patch Schwarz  if p <= p_patch
                      = Chebyshev4(deg)                        otherwise

Damping: omega = 1/lambda_max(M_AS A) from a short power iteration, so that
the symmetric pre/post sweeps keep the V-cycle SPD for CG.  The same object
is a valid `precond` for pcg_solve and for the state.precond_factory hook.
"""
import numpy as np
from lssem2d.precond import PMG2, Chebyshev4
from lssem2d.solver import apply_A
from vertex_schwarz2d import VertexSchwarz2D


class PatchSmoother:
    name = 'vschwarz-smoother'

    def __init__(self, state, fu, fv, pin_p=False, npow=12, safety=1.05):
        self.vs = VertexSchwarz2D(state, fu, fv, pin_p=pin_p, coarse=None)
        rng = np.random.default_rng(0)
        v = rng.standard_normal(self.vs.mask.shape)*self.vs.mask; v /= np.linalg.norm(v)
        lam = 1.0
        for _ in range(npow):
            w = self.vs(apply_A(state, v, fu, fv, pin_p=pin_p))
            lam = np.linalg.norm(w); v = w/lam
        self.omega = 1.0/(safety*lam); self.lam_max = lam

    def __call__(self, r):
        return self.omega*self.vs(r)


class PMGV(PMG2):
    name = 'pmgv'

    def __init__(self, state, fu, fv, M_inv, pin_p=False, pc=(8, 4, 2), deg=4,
                 p_patch=8, coarse_solver='direct', optimised=True, coarse_deg=10):
        rest = ()
        if isinstance(pc, (tuple, list)):
            pc, rest = int(pc[0]), tuple(pc[1:])
        # two-level PMG2 first; a placeholder Chebyshev coarse when deeper levels follow
        super().__init__(state, fu, fv, M_inv, pin_p, pc=pc, deg=deg, optimised=optimised,
                         coarse_deg=coarse_deg,
                         coarse_solver=(coarse_solver if not rest else 'chebyshev'))
        if rest:
            self.coarse = PMGV(self.sc, self.fuc, self.fvc, self.Mic, self._map_pin(pin_p),
                               pc=rest, deg=deg, p_patch=p_patch, coarse_solver=coarse_solver,
                               optimised=optimised, coarse_deg=coarse_deg)
        self.patch_level = state.mesh.N <= p_patch
        if self.patch_level:
            self.smooth = PatchSmoother(state, fu, fv, pin_p)

    def describe(self, indent='  '):
        s = f'{indent}p={self.pf}: {"vertex-patch Schwarz (omega=%.3f, %d patches, max %d dofs)" % (self.smooth.omega, self.smooth.vs.npatch, self.smooth.vs.maxdofs) if self.patch_level else "Chebyshev4 deg %d" % self.smooth.deg}'
        c = self.coarse
        return s + '\n' + (c.describe(indent) if isinstance(c, PMGV) else f'{indent}p={self.pc}: {type(c).__name__}')
