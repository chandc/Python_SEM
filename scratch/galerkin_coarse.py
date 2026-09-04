"""Is REDISCRETISATION the reason the coarse correction does not remove p-growth?

coarsen_mesh rediscretises: A_c = L_c^T W_c L_c at order pc.  For a normal-
equation system that is NOT the Galerkin projection:

    L_c^T W_c L_c   !=   P^T (L^T W L) P

Variational multigrid theory needs the second one.  With the first, the coarse
solve answers a different question than the fine residual asked, so the
correction helps by a constant (measured 1.47x -> 1.31x) but cannot make the
count p-independent.

Two-level test, coarse p=2, coarse operator assembled BOTH ways and solved
EXACTLY, so the only difference is rediscretised vs Galerkin.
"""
import os, sys, time
for _v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '8')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); os.chdir(_R)
os.environ.setdefault('LSSEM3D_BACKEND', 'numba')
import numpy as np
TOL, CAP = 1e-8, 40000


class TwoLevel:
    """Chebyshev pre/post + EXACT coarse solve. `galerkin` picks the operator."""

    def __init__(self, pmg, Afine, galerkin):
        self.pmg, self.A, self.lv = pmg, Afine, pmg.levels[0]
        self.lc = pmg.levels[-1]
        self.smooth = pmg.smooth[0]
        import scipy.linalg as sla
        # coarse global basis
        from lssem3d import operator as OP, solver3d as S3
        sh = self.lc.shape
        cols, seen = [], set()
        for e in range(sh[0]):
            for i in range(sh[1]):
                for j in range(sh[2]):
                    for f in range(OP.NVAR_R):
                        if self.lc.mask[e, i, j, f, 0] == 0.0: continue
                        ed = np.zeros(sh); ed[e, i, j, f, 0] = 1.0
                        g = S3.gs(self.lc.m, ed)
                        k = tuple(np.flatnonzero(np.abs(g.ravel()) > 0.5))
                        if not k or k in seen: continue
                        seen.add(k); cols.append(g)
        self.B = np.stack([c.ravel() for c in cols], axis=1)
        self.mwc = self.lc.mw.ravel()
        n = self.B.shape[1]
        Ac = np.empty((n, n))
        for a in range(n):
            v = self.B[:, a].reshape(sh)
            if galerkin:                      # P^T A P -- the true projection
                y = self._R(self.A(self._P(v)))
            else:                             # rediscretised, what PMG uses
                y = self.lc.A(v)
            Ac[:, a] = self.B.T @ (y.ravel()*self.mwc)
        Ac = 0.5*(Ac + Ac.T)
        self.fac = sla.cho_factor(Ac + 1e-14*np.eye(n)*abs(Ac).max(), lower=True)
        self.sla = sla

    def _P(self, xc):   return self.pmg._prolong(xc, 0)
    def _R(self, x):    return self.pmg._restrict(x, 0)

    def _csolve(self, r):
        g = self.B.T @ (r.ravel()*self.mwc)
        return (self.B @ self.sla.cho_solve(self.fac, g)).reshape(self.lc.shape)

    def __call__(self, r):
        z = self.smooth(r)
        res = r - self.A(z)
        z = z + self._P(self._csolve(self._R(res)))
        res = r - self.A(z)
        return z + self.smooth(res)


def main(ex=3, ey=3, nz=32, cc=5405.4, kcol=1):
    import lssem3d; lssem3d.set_backend(os.environ['LSSEM3D_BACKEND'])
    from lssem2d.mesh import build_channel
    from lssem2d.lgl import diff_matrix
    from lssem3d import bc as BC, operator as OP, precond as P3, solver3d as S3, fourier as FR
    nk = nz//2 + 1
    kz = np.array([float(FR.wavenumbers(nz, 0.34*np.pi)[kcol])])
    print(f'{ex}x{ey}, k_z={kz[0]:.2f}, c={cc:g}, two-level p->2, coarse solved EXACTLY\n')
    print(f'{"p":>3} {"jacobi":>8} {"rediscretised":>14} {"GALERKIN":>10} {"gain":>7}')
    out = []
    for N in (8, 12, 16, 20):
        m = build_channel(np.pi, 2.0, ex, ey, N, bcs=(0, 0, 1, 1))
        m.periodic_x = np.pi; m.compute_global_indices()
        mf = BC.build_mask(m, nk, pin_p=False, nz=nz)
        BC.pin_dof(m, mf, OP.P_, 0); BC.pin_dof(m, mf, OP.NVAR+OP.P_, 0)
        mask = np.ascontiguousarray(mf[..., kcol:kcol+1])
        nu, D = 1/180., diff_matrix(N)
        rw = OP.momentum_row_weights(cc)
        sh = (m.nelem, N+1, N+1, OP.NVAR_R, 1)
        kwn = dict(mesh=m, mask=mask, wq=m.wq, kap=0.0, rw=rw)
        A = lambda x: S3.normal_op(x, D, m.facx, m.facy, kz, nu, cc, **kwn)
        b = A(np.random.default_rng(0).standard_normal(sh)*mask); b /= np.linalg.norm(b)
        run = lambda M: int(np.max(S3.pcg(b, D, m.facx, m.facy, kz, nu, cc, m,
                                          mask, M, TOL, CAP, None, m.wq, 0.0, rw)[1]))
        pmg = P3.PMG(m, 1, nz, nu, cc, kz, kap=0.0, rw=rw, orders=(N, 2), deg=6,
                     pin_p=True, direct_coarse='element', mask=mask)
        itj = run(pmg.levels[0].M_inv)
        itr = run(TwoLevel(pmg, A, galerkin=False))
        itg = run(TwoLevel(pmg, A, galerkin=True))
        print(f'{N:3d} {itj:8d} {itr:14d} {itg:10d} {itr/max(itg,1):6.2f}x', flush=True)
        out.append((itr, itg))
    r = np.array(out, float)
    print(f'\n  growth p=8->20:  rediscretised {r[-1,0]/r[0,0]:.2f}x   '
          f'GALERKIN {r[-1,1]/r[0,1]:.2f}x')
    print('  target: p-INDEPENDENT, i.e. ~1.0x (2D ladder achieves 1.05x)')

if __name__ == '__main__':
    main()
