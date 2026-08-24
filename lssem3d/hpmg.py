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
    """One polynomial order.  Everything the V-cycle touches lives on `like`.

    The masks, the analytic diagonal and the multiplicity weight are all built
    in NumPy -- they are closed-form setup, evaluated once -- and moved once.
    What must NOT stay on the host is anything the operator reads per
    application: D, facx, facy, wq and mask.  Leaving those behind is what made
    the first version cost 275x a matvec.
    """

    def __init__(self, mesh, p, lam, mu, nfield_c, nk, nz, wall, pin_kz0,
                 like=None):
        from . import project as PJ
        self.m, self.p = mesh, p
        Dh = diff_matrix(p)
        mask = PJ.build_masks(mesh, nk, nz, nfield_c, wall=wall)
        if pin_kz0:
            ind = np.zeros(mask.shape)
            ind[0, 0, 0, 0, 0] = 1.0
            mask[..., 0, 0] *= (S3.gs(mesh, ind)[..., 0, 0] < 0.5)
        self.shape = mask.shape
        dh = HH.jacobi_diagonal_analytic(mesh, p, mesh.wq, lam, mu,
                                         2*nfield_c, nk, mask)
        mwh = S3.multiplicity_weight(mesh, self.shape)
        to = (lambda a: a) if like is None else (lambda a: DEV.to_device(a, like))
        self.D, self.mask = to(Dh), to(mask)
        self.fx, self.fy, self.wq = to(mesh.facx), to(mesh.facy), to(mesh.wq)
        self.M_inv = to(HH.jacobi_inverse(dh, mask))
        self.mw = to(mwh)
        self.mask_h = mask
        # lam is per-mode (kz^2) and is MULTIPLIED into the operator, so it is
        # read every application and has to move as well.  mu is a scalar.
        lam_d = to(np.asarray(lam, dtype=float)) if np.ndim(lam) else lam
        self.A = lambda v: HH.apply(v, self.D, self.fx, self.fy, self.wq,
                                    lam_d, mu, mesh, self.mask)


class _Direct:
    """Exact coarsest solve, in the GLOBAL (continuous) basis, per mode.

    Per mode because the operator is block diagonal in k_z -- one small dense
    factorisation each rather than one huge one.  Cholesky, not pinv: the
    matrix is a Galerkin projection of an SPD operator, and pinv's rcond would
    silently truncate exactly the near-null directions a coarse solve exists to
    resolve.
    """

    def __init__(self, lvl, like=None):
        from .precond import _factor_spd
        shape, m, mask = lvl.shape, lvl.m, lvl.mask_h
        nk = shape[-1]
        self.B, self.fac, self.lvl = [], [], lvl
        self.mw = np.asarray(S3.multiplicity_weight(m, shape)).ravel()
        self.Dh, self.fxh, self.fyh = diff_matrix(lvl.p), m.facx, m.facy
        self.wqh, self.lam, self.mu = m.wq, lvl.lam, lvl.mu
        self.mwd = lvl.mw.reshape(-1)
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
            Ah = lambda v: HH.apply(v, self.Dh, self.fxh, self.fyh, self.wqh,
                                    self.lam, self.mu, m, mask)
            for a in range(B.shape[1]):
                A[:, a] = B.T @ (Ah(B[:, a].reshape(shape)).ravel()*self.mw)
            # store the explicit INVERSE, not a factorisation callable: a
            # triangular solve is host-only, a dense matmul is not, and the
            # coarse level is small enough that n^2 costs little.  Built from a
            # Cholesky, so the cost is the factorisation's, not an SVD's.
            fac = _factor_spd(0.5*(A + A.T))
            Ai = np.stack([fac(e) for e in np.eye(B.shape[1])], axis=1)
            to = (lambda a: a) if like is None else (
                lambda a: DEV.to_device(a, like))
            self.B.append(to(B))
            self.fac.append(to(0.5*(Ai + Ai.T)))

    def __call__(self, r):
        z = DEV.zeros_like(r)
        rf = (r*self.lvl.mw).reshape(-1)
        for B, Ai in zip(self.B, self.fac):
            if B is None:
                continue
            z = z + (B @ (Ai @ (B.T @ rf))).reshape(self.lvl.shape)
        return z*self.lvl.mask


class HelmholtzPMG:
    """V-cycle preconditioner for lam*M + mu*K.  Callable: z = M^-1 r."""

    def __init__(self, mesh, N, lam, mu, nfield_c, nk, nz, orders=None,
                 deg=6, wall=False, pin_kz0=True, like=None):
        orders = orders or tuple(o for o in (N, max(2, N//2), 2)
                                 if o <= N and o >= 2)
        orders = tuple(sorted(set(orders), reverse=True))
        meshes = [mesh] + [coarsen_mesh(mesh, p) for p in orders[1:]]
        self.lv = []
        for mm, p in zip(meshes, orders):
            l = _Lvl(mm, p, lam, mu, nfield_c, nk, nz, wall, pin_kz0, like)
            l.lam, l.mu = lam, mu
            self.lv.append(l)
        to = (lambda a: a) if like is None else (lambda a: DEV.to_device(a, like))
        self.P = [to(p_interp(c, f)) for f, c in zip(orders[:-1], orders[1:])]
        self.R = [to(p_interp(c, f).T) for f, c in zip(orders[:-1], orders[1:])]
        self.sm = []
        for l in self.lv[:-1]:
            # estimate_lambda_max starts from a HOST random vector; give it a
            # device one so the power iteration never leaves the device
            v = DEV.to_device(np.random.default_rng(0).standard_normal(l.shape),
                              l.mask)*l.mask
            lm = 0.0
            for _ in range(20):
                w = l.M_inv*l.A(v)
                nw = float(DEV.sqrt((w*w).sum()))
                if nw <= 1e-300:
                    break
                lm, v = nw, w/nw
            self.sm.append(Chebyshev4(l.A, l.M_inv, l.shape, deg=deg,
                                      lam_max=lm))
        self.coarse = _Direct(self.lv[-1], like)
        self.like = like

    def _restrict(self, x, i):
        t = DEV.einsum('bj,eijvk->eibvk', self.R[i], x*self.lv[i].mw)
        c = DEV.einsum('ai,eibvk->eabvk', self.R[i], t)
        return S3.gs(self.lv[i+1].m, c)*self.lv[i+1].mask

    def _prolong(self, xc, i):
        t = DEV.einsum('bj,eijvk->eibvk', self.P[i], xc)
        return DEV.einsum('ai,eibvk->eabvk', self.P[i], t)*self.lv[i].mask

    def _v(self, r, i):
        if i == len(self.lv) - 1:
            return self.coarse(r)
        z = self.sm[i](r)
        z = z + self._prolong(self._v(self._restrict(r - self.lv[i].A(z), i),
                                      i + 1), i)
        return z + self.sm[i](r - self.lv[i].A(z))

    def __call__(self, r):
        """Device-resident when built with `like`: no transfer, no NumPy."""
        return self._v(r, 0)
