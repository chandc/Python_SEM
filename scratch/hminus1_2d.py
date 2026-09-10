"""Stage-B prototype of the H^-1 momentum norm (Bramble-Lazarov-Pasciak /
Bochev-Gunzburger (4.25)-(4.27)) for the 2D FOSLS cavity, ACCURACY ONLY:
assembled operators, direct solves, no preconditioner questions.

Per time step (BDF, one Picard/Newton sub-iteration like newton_step):
    residual rows (unweighted, per element node)  R = (R_u, R_v, R_con, R_vor)
    legacy functional   J_L2 = sum wq [ R_u^2 + R_v^2 + R_con^2 + R_vor^2 ],
                        with R_u = fac1 u - hist + dt N_u(u)          (weight dt on momentum)
    H^-1 functional     J_-1 = |R'_u|^2_{-1,h} + |R'_v|^2_{-1,h} + sum wq [R_con^2 + R_vor^2],
                        R' = R_mom/dt = (fac1 u - hist)/dt + N(u)     (weight 1 on momentum)
    with the discrete negative norm of a residual field r:
        |r|^2_{-1,h} = f^T (K_int^-1 + beta M^-1) f,   f = load vector = Q^T(wq r) on interior nodes,
        K = assembled SEM Laplacian (Dirichlet), M = lumped mass, beta ~ h^2 (BLP's h^2 I term).
The step solves  A dU = -grad J  with A = L^T Wtilde L  assembled from element-local
probes of apply_L (324 probes at N = 8), dense LU.  Gate: with the legacy weighting
this A equals fosls_assemble's to round-off.

    python scratch/hminus1_2d.py <legacy|hm1> <dt> <nstep> [beta (absolute pointwise weight)] [out_tag]
"""
import os, sys, time
for _v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '4')
os.environ.setdefault('CAV_RE', '1000')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, 'scratch')); os.chdir(_R)
import numpy as np, scipy.sparse as sp, scipy.linalg as sla
import lssem2d; lssem2d.set_backend('numpy')
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState, apply_L, ls_coeffs
from lssem2d.mesh import build_channel
from lssem2d.bc import apply_bc
from lssem2d.operators import dUdx, dUdy
from lssem2d.assembly import gather_scatter
from pmg_ghia_cavity import centreline_u, centreline_v, GHIA_X, GHIA_V

RE, EX, N = 1000.0, 4, int(os.environ.get('HM1_N', 8))
NV = 4


class Assembler:
    def __init__(self, m, st):
        self.m, self.st = m, st; n = m.N + 1; self.n = n
        gid = m.gidx; self.ng = int(gid.max()) + 1; self.ndof = self.ng*NV
        self.g = (gid[..., None]*NV + np.arange(NV)).astype(np.int64)          # local -> global dof
        self.gloc = gid                                                          # local node -> global node
        nde = n*n*NV; self.nde = nde
        # Q: global dof -> local storage (nelem*nde x ndof)
        rows = np.arange(m.nelem*nde); cols = self.g.reshape(-1)
        self.Q = sp.csr_matrix((np.ones(rows.size), (rows, cols)), shape=(rows.size, self.ndof))
        # node assembly for scalar fields: Qn (nelem*n*n x ng)
        self.Qn = sp.csr_matrix((np.ones(m.nelem*n*n), (np.arange(m.nelem*n*n), gid.reshape(-1))), shape=(m.nelem*n*n, self.ng))
        self.wq = m.wq.reshape(-1)                                               # per local node
        # scalar SEM Laplacian K and lumped mass M on global nodes (for the H^-1 norm)
        D = diff_matrix(m.N); Ke = np.zeros((m.nelem, n*n, n*n))
        for c in range(n*n):
            i, j = np.unravel_index(c, (n, n)); P = np.zeros((m.nelem, n, n)); P[:, i, j] = 1.0
            px = dUdx(P, D, m.facx); py = dUdy(P, D, m.facy)
            # K[:, :, c] = sum_q wq (dphi_c . dphi_r): need dphi_r too -> build via gradient basis below
            Ke[:, :, c] = 0.0
        # gradient basis: G_x[e, q, c] = d phi_c / dx at node q  (nodal derivative matrices)
        Gx = np.zeros((m.nelem, n*n, n*n)); Gy = np.zeros_like(Gx)
        for c in range(n*n):
            i, j = np.unravel_index(c, (n, n)); P = np.zeros((m.nelem, n, n)); P[:, i, j] = 1.0
            Gx[:, :, c] = dUdx(P, D, m.facx).reshape(m.nelem, -1); Gy[:, :, c] = dUdy(P, D, m.facy).reshape(m.nelem, -1)
        W = m.wq.reshape(m.nelem, -1)
        Ke = np.einsum('eqr,eq,eqc->erc', Gx, W, Gx) + np.einsum('eqr,eq,eqc->erc', Gy, W, Gy)
        gn = gid.reshape(m.nelem, -1)
        self.K = sp.coo_matrix((Ke.ravel(), (np.repeat(gn, n*n, axis=1).ravel(), np.tile(gn, (1, n*n)).ravel())), shape=(self.ng, self.ng)).tocsr()
        self.Mlump = np.asarray(self.Qn.T @ self.wq).ravel()

    def probe_L(self, fu, fv):
        """Element-local residual blocks Le[e] (nde x nde): unweighted rows R = apply_L/wq."""
        m, n, nde = self.m, self.n, self.nde
        Le = np.empty((m.nelem, nde, nde)); U = np.zeros((m.nelem, n, n, NV))
        for c in range(nde):
            i, j, f = np.unravel_index(c, (n, n, NV)); U[:] = 0; U[:, i, j, f] = 1.0
            su = apply_L(self.st, U, fu, fv)/self.m.wq[..., None]
            Le[:, :, c] = su.reshape(m.nelem, nde)
        return Le

    def Lglobal(self, Le):
        """L: global dofs -> local rows (nelem*nde), = blockdiag(Le) Q."""
        m, nde = self.m, self.nde
        rows = np.repeat(np.arange(m.nelem*nde).reshape(m.nelem, nde), nde, axis=1).ravel()
        cols = np.tile(self.g.reshape(m.nelem, nde), (1, nde)).ravel()
        return sp.coo_matrix((Le.ravel(), (rows, cols)), shape=(m.nelem*nde, self.ndof)).tocsr()


def run(kind, dt, nstep, beta_h2=1.0, tag=None, steady=0.0, init=None):
    m = build_channel(1.0, 1.0, EX, EX, N, bcs=(1, 1, 1, 2)); m.compute_global_indices()
    st = SolverState(m, diff_matrix(N), nu=1.0/RE, dt=dt, fac1=1.0)
    n = N + 1; A_ = Assembler(m, st)
    mask = st.get_global_mask(pin_p=True)                                          # local, 0 = fixed
    free = np.zeros(A_.ndof, bool); np.logical_or.at(free, A_.g.ravel(), mask.ravel() > 0.5)
    fr = np.flatnonzero(free)
    # interior nodes for the H^-1 norm: nodes where u AND v are free
    node_free = free.reshape(A_.ng, NV); ni = np.flatnonzero(node_free[:, 0] & node_free[:, 1])
    Kint = A_.K[ni][:, ni].toarray()
    beta = beta_h2                                    # ABSOLUTE weight of the pointwise L2 term (BLP's h^2 I)
    Sint = np.linalg.inv(Kint)                        # B = K_int^-1 on the load vectors (dense, small at N=8)
    Sfull = sp.coo_matrix((Sint.ravel(), (np.repeat(ni, ni.size), np.tile(ni, ni.size))), shape=(A_.ng, A_.ng)).tocsr()
    # row selectors in local storage
    nloc = m.nelem*n*n; rowf = np.tile(np.arange(NV), nloc); nodeof = np.repeat(np.arange(nloc), NV)
    def rows_of(f): return np.flatnonzero(rowf == f)
    R_u, R_v, R_c, R_w = (rows_of(f) for f in range(NV))
    Wd = sp.diags(np.repeat(A_.wq, NV))                                              # quadrature weights per local row
    # node-assembly of a per-node row field (local rows of one component -> global nodes)
    Qn = A_.Qn
    U = np.zeros((m.nelem, n, n, NV)) if init is None else np.load(init)['U'].copy(); hist = [U.copy()]
    if init is not None: print(f'  init from {init}', flush=True)
    t0 = time.perf_counter(); rec = []
    for s in range(nstep):
        t = (s+1)*dt
        if len(hist) == 1: st.fac1 = 1.0; alpha = [1.0]
        else: st.fac1 = 1.5; alpha = [2.0, -0.5]
        a_mass, a_flux, hs = ls_coeffs(st)
        U = apply_bc(m, U, time=t, pin_p=True)
        # nonlinear residual rows (unweighted): R = L(U; U/2) - hist
        st.update_linearisation(U[..., 0]/2, U[..., 1]/2)
        Rn = (apply_L(st, U, U[..., 0]/2, U[..., 1]/2)/m.wq[..., None])
        for k, al in enumerate(alpha):
            Rn[..., 0] -= hs*al*hist[k][..., 0]; Rn[..., 1] -= hs*al*hist[k][..., 1]
        Rn = Rn.reshape(-1)
        fu, fv = np.ascontiguousarray(U[..., 0]), np.ascontiguousarray(U[..., 1]); st.update_linearisation(fu, fv)
        L = A_.Lglobal(A_.probe_L(fu, fv))                                           # rows: local residuals, cols: global dofs
        if kind == 'legacy':
            A = (L.T @ Wd @ L); b = -(L.T @ (Wd @ Rn))
        else:
            # momentum rows scaled to weight 1 (divide by a_flux = dt), measured in the discrete H^-1 norm
            Lm = [L[R_u]/a_flux, L[R_v]/a_flux]; Rm = [Rn[R_u]/a_flux, Rn[R_v]/a_flux]
            Wn = sp.diags(A_.wq)
            F = [Qn.T @ Wn @ Lc for Lc in Lm]                                        # load-vector maps (ng x ndof)
            fres = [Qn.T @ (Wn @ Rc) for Rc in Rm]
            Lc_ = L[np.concatenate([R_c, R_w])]; Wc_ = sp.diags(np.concatenate([A_.wq, A_.wq]))
            A = Lc_.T @ Wc_ @ Lc_; b = -(Lc_.T @ (Wc_ @ np.concatenate([Rn[R_c], Rn[R_w]])))
            # |r|^2_{-1,h} = beta * sum wq r^2 (pointwise, ALL nodes -- kills the checkerboard
            # pressure null modes of a purely weak test)  +  f^T K_int^-1 f
            for Fc, fc, Lc, Rc in zip(F, fres, Lm, Rm):
                A = A + Fc.T @ Sfull @ Fc + beta*(Lc.T @ Wn @ Lc); b = b - Fc.T @ (Sfull @ fc) - beta*(Lc.T @ (Wn @ Rc))
        Af = A.toarray()[np.ix_(fr, fr)] if sp.issparse(A) else A[np.ix_(fr, fr)]
        dU = np.zeros(A_.ndof); dU[fr] = sla.solve(Af, b[fr], assume_a='sym')
        U = U + dU[A_.g]*mask
        hist.insert(0, U); hist = hist[:2]
        rate = np.abs(hist[0] - hist[1]).max()/dt if len(hist) > 1 else np.nan
        if (s+1) % max(1, nstep//10) == 0 or s == 0:
            print(f'  {kind:6s} dt={dt:g} step {s+1:5d} t={t:7.3f} max|u|={np.abs(U[...,0]).max():.4f} max|v|={np.abs(U[...,1]).max():.4f} max|dU|/dt={rate:.2e} [{time.perf_counter()-t0:.0f}s]', flush=True)
        if steady > 0 and rate < steady: print(f'  STEADY at step {s+1}'); break
    y, u = centreline_u(U, m, N); x, v = centreline_v(U, m, N)
    gh = np.load('cavity_re1000_data.npz'); gy, gu = gh['ghia_y'], gh['ghia_u']; o = np.argsort(GHIA_X)
    rms_u = float(np.sqrt(np.mean((np.interp(gy, y, u) - gu)**2))); rms_v = float(np.sqrt(np.mean((np.interp(GHIA_X[o], x, v) - GHIA_V[o])**2)))
    sel = (y > 0.75) & (y < 0.95); d = np.diff(u[sel])
    zz = (np.abs(d).mean(), np.abs(d).max(), int((np.sign(d[1:]) != np.sign(d[:-1])).sum()), d.size-1)
    tag = tag or (f'{kind}_dt{dt:g}' + (f'_b{beta_h2:g}' if kind == 'hm1' else ''))
    np.savez_compressed(f'scratch/hm1/{tag}.npz', U=U, y=y, u=u, x=x, v=v, rms_u=rms_u, rms_v=rms_v, t=t, kind=kind, dt=dt, beta_h2=beta_h2, N=N)
    print(f'DONE {tag}: t={t:.2f} rms vs Ghia u {rms_u:.4f} v {rms_v:.4f} | zigzag |du| mean {zz[0]:.4f} max {zz[1]:.4f} sign changes {zz[2]}/{zz[3]}', flush=True)
    return U


def gate():
    """Legacy assembled A from this module == fosls_assemble's A."""
    import fosls_assemble as FA
    m = build_channel(1.0, 1.0, EX, EX, N, bcs=(1, 1, 1, 2)); m.compute_global_indices()
    st = SolverState(m, diff_matrix(N), nu=1.0/RE, dt=0.1, fac1=1.0)
    n = N+1; fu = np.zeros((m.nelem, n, n)); fv = np.zeros_like(fu); st.update_linearisation(fu, fv)
    A_ = Assembler(m, st); L = A_.Lglobal(A_.probe_L(fu, fv)); Wd = sp.diags(np.repeat(A_.wq, NV))
    A1 = (L.T @ Wd @ L).toarray(); A2, free, g = FA.assemble(st, fu, fv, pin_p=False)
    A2 = A2.toarray(); fr = np.flatnonzero(free)
    print('gate: max|A_mine - A_FA| / max|A| on free dofs =', np.abs(A1[np.ix_(fr, fr)] - A2[np.ix_(fr, fr)]).max()/np.abs(A2).max())
    # Laplacian sanity: K applied to a linear function is ~0 at INTERIOR nodes
    xg = np.zeros(A_.ng); xg[m.gidx.reshape(-1)] = m.xnod[:, :, None].repeat(n, 2).reshape(-1)
    bnd = np.zeros(A_.ng, bool)
    for e in range(m.nelem):
        for side in (m.gidx[e, 0, :], m.gidx[e, -1, :], m.gidx[e, :, 0], m.gidx[e, :, -1]):
            pass
    xs = m.xnod.reshape(-1); ys = m.ynod.reshape(-1)
    xn = np.zeros(A_.ng); yn = np.zeros(A_.ng); xn[m.gidx.reshape(-1)] = m.xnod[:, :, None].repeat(n, 2).reshape(-1); yn[m.gidx.reshape(-1)] = m.ynod[:, None, :].repeat(n, 1).reshape(-1)
    interior = (xn > 1e-9) & (xn < 1-1e-9) & (yn > 1e-9) & (yn < 1-1e-9)
    r = A_.K @ xg; print('gate: |K x|_interior / |K|_max (should be ~0):', np.abs(r[interior]).max()/np.abs(A_.K).max(), ' boundary:', np.abs(r[~interior]).max()/np.abs(A_.K).max())


if __name__ == '__main__':
    os.makedirs('scratch/hm1', exist_ok=True)
    if sys.argv[1:] == ['gate']: gate()
    else:
        kind, dt, nstep = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
        beta = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
        run(kind, dt, nstep, beta_h2=beta, tag=(sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] != '-' else None), steady=1e-7,
            init=(sys.argv[6] if len(sys.argv) > 6 else None))
