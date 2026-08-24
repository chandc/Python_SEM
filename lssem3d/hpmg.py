"""p-multigrid for the scalar Helmholtz/Poisson — the coarse grid FDM lacks.

WHY THIS EXISTS.  `helmholtz.fdm_preconditioner` is one-level additive Schwarz:
an EXACT element-local inverse plus gather-scatter, with no coarse grid.  On
the Re_tau = 180 channel Poisson that buys nothing over a plain diagonal, and
both grow with element count:

    elements   Jacobi   FDM
    2x4           254   231
    6x12          734   826      ~sqrt(elements)

which is the signature.  For Poisson the slow modes are GLOBAL and smooth, so
exactness inside an element cannot touch them.

AND THIS IS THE CASE WHERE MULTIGRID WORKS.  3D_STATUS.md sec 7K closed p-MG
for the VVP LEAST-SQUARES operator after measuring 7.4x fewer iterations for
~27x the cost per iteration, and sec 7K.2 explained why: that operator's slow
modes are ROUGH, which a coarse grid cannot reach.  Poisson's are smooth.  The
closure does not transfer, and the machinery it left behind -- p_interp,
coarsen_mesh, Chebyshev4 -- is operator-agnostic and reused here unchanged.

The smoother needs a diagonal at EVERY level, per dt, so it uses
helmholtz.jacobi_diagonal_analytic: closed form, and exact to 2.1e-16 against a
true diagonal probed one global dof at a time.
"""
import numpy as np

from . import device as DEV
from . import helmholtz as HH
from . import solver3d as S3
from .precond import p_interp, coarsen_mesh, Chebyshev4
from lssem2d.lgl import diff_matrix


class _Lvl:
    def __init__(self, mesh, p, lam, mu, nfield_c, nk, nz, wall, pin_kz0):
        from . import project as PJ
        self.m, self.p = mesh, p
        self.D = diff_matrix(p)
        self.mask = PJ.build_masks(mesh, nk, nz, nfield_c, wall=wall)
        if pin_kz0:
            ind = np.zeros(self.mask.shape)
            ind[0, 0, 0, 0, 0] = 1.0
            self.mask[..., 0, 0] *= (S3.gs(mesh, ind)[..., 0, 0] < 0.5)
        self.shape = self.mask.shape
        self.A = lambda v: HH.apply(v, self.D, mesh.facx, mesh.facy, mesh.wq,
                                    lam, mu, mesh, self.mask)
        d = HH.jacobi_diagonal_analytic(mesh, p, mesh.wq, lam, mu,
                                        2*nfield_c, nk, self.mask)
        self.M_inv = HH.jacobi_inverse(d, self.mask)
        self.mw = S3.multiplicity_weight(mesh, self.shape)


class _Direct:
    """Exact coarsest solve, in the GLOBAL (continuous) basis, per mode.

    Per mode because the operator is block diagonal in k_z -- one small dense
    factorisation each rather than one huge one.  Cholesky, not pinv: the
    matrix is a Galerkin projection of an SPD operator, and pinv's rcond would
    silently truncate exactly the near-null directions a coarse solve exists to
    resolve.
    """

    def __init__(self, lvl):
        from .precond import _factor_spd
        shape, m, mask = lvl.shape, lvl.m, lvl.mask
        nk = shape[-1]
        self.B, self.fac, self.lvl = [], [], lvl
        self.mw = lvl.mw.ravel()
        for k in range(nk):
            cols, seen = [], set()
            for e in range(shape[0]):
                for i in range(shape[1]):
                    for j in range(shape[2]):
                        for f in range(shape[3]):
                            if mask[e, i, j, f, k] == 0.0:
                                continue
                            oh = np.zeros(shape)
                            oh[e, i, j, f, k] = 1.0
                            g = S3.gs(m, oh)
                            key = tuple(np.flatnonzero(np.abs(g.ravel()) > .5))
                            if not key or key in seen:
                                continue
                            seen.add(key)
                            cols.append(g)
            if not cols:
                self.B.append(None); self.fac.append(None); continue
            B = np.stack([c.ravel() for c in cols], axis=1)
            A = np.empty((B.shape[1], B.shape[1]))
            for a in range(B.shape[1]):
                A[:, a] = B.T @ (lvl.A(B[:, a].reshape(shape)).ravel()*self.mw)
            self.B.append(B)
            self.fac.append(_factor_spd(0.5*(A + A.T)))

    def __call__(self, r):
        z = np.zeros(self.lvl.shape)
        rf = r.ravel()
        for B, f in zip(self.B, self.fac):
            if B is None:
                continue
            z += (B @ f(B.T @ (rf*self.mw))).reshape(self.lvl.shape)
        return z*self.lvl.mask


class HelmholtzPMG:
    """V-cycle preconditioner for lam*M + mu*K.  Callable: z = M^-1 r."""

    def __init__(self, mesh, N, lam, mu, nfield_c, nk, nz, orders=None,
                 deg=6, wall=False, pin_kz0=True, like=None):
        orders = orders or tuple(o for o in (N, max(2, N//2), 2)
                                 if o <= N and o >= 2)
        orders = tuple(sorted(set(orders), reverse=True))
        meshes = [mesh] + [coarsen_mesh(mesh, p) for p in orders[1:]]
        self.lv = [_Lvl(mm, p, lam, mu, nfield_c, nk, nz, wall, pin_kz0)
                   for mm, p in zip(meshes, orders)]
        self.P = [p_interp(c, f) for f, c in zip(orders[:-1], orders[1:])]
        self.R = [P.T for P in self.P]
        self.sm = [Chebyshev4(l.A, l.M_inv, l.shape, deg=deg)
                   for l in self.lv[:-1]]
        self.coarse = _Direct(self.lv[-1])
        self.like = like

    def _restrict(self, x, i):
        t = np.einsum('bj,eijvk->eibvk', self.R[i], x*self.lv[i].mw)
        c = np.einsum('ai,eibvk->eabvk', self.R[i], t)
        return S3.gs(self.lv[i+1].m, c)*self.lv[i+1].mask

    def _prolong(self, xc, i):
        t = np.einsum('bj,eijvk->eibvk', self.P[i], xc)
        return np.einsum('ai,eibvk->eabvk', self.P[i], t)*self.lv[i].mask

    def _v(self, r, i):
        if i == len(self.lv) - 1:
            return self.coarse(r)
        z = self.sm[i](r)
        z = z + self._prolong(self._v(self._restrict(r - self.lv[i].A(z), i),
                                      i + 1), i)
        return z + self.sm[i](r - self.lv[i].A(z))

    def __call__(self, r):
        host = DEV.to_host(r) if hasattr(DEV, 'to_host') else r
        if not isinstance(host, np.ndarray):
            host = host.get() if hasattr(host, 'get') else np.asarray(host)
        z = self._v(host, 0)
        return DEV.to_device(z, r) if not isinstance(r, np.ndarray) else z
