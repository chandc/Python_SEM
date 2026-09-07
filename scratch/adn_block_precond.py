"""ADN / operator-preconditioning test on the ASSEMBLED per-mode FOSLS operator.

Mardal-Winther: if the FOSLS form is norm-equivalent to a product norm
|||U|||^2 = |u|_X^2 + |w|_Y^2 + |p|_Z^2 with constants (C1, C2) independent of
h, p and c, then the block-diagonal Riesz map B = diag(X, Y, Z) gives
cond(B^-1 A) <= C2/C1.  The STRONGEST block-diagonal preconditioner is the
exact block diagonal of A itself, so:

    if  cond(blockdiag(A)^-1 A)  is flat in p and c  -> the ADN route is open:
        the job reduces to approximating three decoupled blocks;
    if  it grows                                     -> the coupling is the
        problem and no block-diagonal (norm-based) preconditioner can fix it.

Assembled sparsely per mode through gidx (same code path as DirectCoarseE),
masked dofs removed, cond from dense generalised eigenvalues (small) or from
the CG Lanczos Ritz values (large).
"""
import os, sys, time
for _v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '8')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); os.chdir(_R)
os.environ.setdefault('LSSEM3D_BACKEND', 'numpy')
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla, scipy.linalg as sla

NVAR = 14
VEL = [0,1,2,7,8,9]; VOR = [3,4,5,10,11,12]; PRE = [6,13]


def assemble(level, k):
    """Assembled sparse A for mode column k of `level`, plus the free-dof mask."""
    m, shape, mask = level.m, level.shape, level.mask
    nelem, n, _, nvar, nk = shape
    nloc = n*n*nvar
    Aloc = np.empty((nelem, nloc, nloc))
    for col in range(nloc):
        i, j, f = np.unravel_index(col, (n, n, nvar))
        v = np.zeros(shape); v[:, i, j, f, :] = 1.0
        out = level.A_un(v)[..., k]
        Aloc[:, :, col] = out.reshape(nelem, nloc)
    gidx = m.gidx; nnode = int(gidx.max())+1; ndof = nnode*nvar
    gd = (gidx[..., None]*nvar + np.arange(nvar)).reshape(nelem, nloc)
    rows = np.repeat(gd, nloc, axis=1).ravel(); cols = np.tile(gd, (1, nloc)).ravel()
    A = sp.coo_matrix((Aloc.reshape(-1), (rows, cols)), shape=(ndof, ndof)).tocsr()
    A = (A + A.T)*0.5
    free = np.abs(A.diagonal()) > 1e-300
    field = np.tile(np.arange(nvar), nnode)
    return A, free, field


def blockdiag_solver(A, groups):
    """Exact solver for the block-diagonal part of A over dof groups."""
    lus = []
    for g in groups:
        Ag = A[g][:, g].tocsc()
        lus.append((g, spla.splu(Ag)))
    def solve(r):
        z = np.zeros_like(r)
        for g, lu in lus: z[g] = lu.solve(r[g])
        return z
    return solve


def pcg_ritz(A, Minv, b, tol=1e-10, maxit=20000):
    """PCG with Lanczos Ritz-value extraction -> (iters, cond estimate)."""
    x = np.zeros_like(b); r = b.copy(); z = Minv(r); p = z.copy()
    rz = r@z; nb = np.linalg.norm(b); alphas, betas = [], []
    for it in range(1, maxit+1):
        Ap = A@p; a = rz/(p@Ap); x += a*p; r -= a*Ap
        alphas.append(a)
        if np.linalg.norm(r) < tol*nb: break
        z = Minv(r); rz_new = r@z
        if not np.isfinite(rz_new) or rz_new <= 0: break      # indefinite preconditioner
        bt = rz_new/rz; betas.append(bt); rz = rz_new
        p = z + bt*p
    # Lanczos tridiagonal from CG coefficients
    n = len(alphas); d = np.empty(n); e = np.empty(max(n-1, 0))
    for i in range(n):
        d[i] = 1/alphas[i] + (betas[i-1]/alphas[i-1] if i > 0 else 0.0)
        if i < n-1: e[i] = np.sqrt(betas[i])/alphas[i]
    ev = sla.eigvalsh_tridiagonal(d, e) if n > 1 else np.array([d[0]])
    return it, ev.max()/ev.min()


def main(ex=2, ey=2, nz=16):
    import lssem3d; lssem3d.set_backend('numpy')
    from lssem2d.mesh import build_channel
    from lssem3d import bc as BC, operator as OP, fourier as FR
    from lssem3d.precond import _Level
    nk = nz//2+1; kz_all = FR.wavenumbers(nz, 0.34*np.pi)
    print(f'{ex}x{ey} elements, nu=1/180, tol 1e-10; cond from CG Ritz values')
    hdr = f'{"c":>7} {"kz":>6} {"p":>3} {"ndof":>6} | {"jacobi":>14} | {"blk7":>14} | {"blk3 u|w|p":>14} | {"blk2 uw|p":>14} | {"cond(A_uu)":>10} {"cond(A_ww)":>10} {"cond(A_pp)":>10}'
    print(hdr)
    for cc in (1.0, 5405.4):
        for kcol in (0, 1):
            for N in (4, 6, 8, 12):
                m = build_channel(np.pi, 2.0, ex, ey, N, bcs=(0,0,1,1))
                m.periodic_x = np.pi; m.compute_global_indices()
                mfull = BC.build_mask(m, nk, pin_p=False, nz=nz)
                BC.pin_dof(m, mfull, OP.P_, 0); BC.pin_dof(m, mfull, OP.NVAR+OP.P_, 0)
                mask = np.ascontiguousarray(mfull[..., kcol:kcol+1])
                kz = np.array([float(kz_all[kcol])])
                rw = OP.momentum_row_weights(cc)
                lev = _Level(m, 1, nz, 1/180., cc, kz, 0.0, rw, False, mask=mask)
                A, free, field = assemble(lev, 0)
                A = A[free][:, free].tocsr(); field = field[free]; nd = A.shape[0]
                rng = np.random.default_rng(0); b = A@rng.standard_normal(nd); b /= np.linalg.norm(b)
                D = A.diagonal()
                res = []
                res.append(pcg_ritz(A, lambda r: r/D, b))
                for groups in ([np.flatnonzero(field == f) for f in range(NVAR)],
                               [np.flatnonzero(np.isin(field, g)) for g in (VEL, VOR, PRE)],
                               [np.flatnonzero(np.isin(field, g)) for g in (VEL+VOR, PRE)]):
                    groups = [g for g in groups if g.size]
                    res.append(pcg_ritz(A, blockdiag_solver(A, groups), b))
                def bcond(g):
                    idx = np.flatnonzero(np.isin(field, g)); Ag = A[idx][:, idx].toarray()
                    w = sla.eigvalsh(Ag); w = w[w > 1e-14*w.max()]; return w.max()/w.min()
                cu, cw, cp = (bcond(g) for g in (VEL, VOR, PRE))
                print(f'{cc:7.0f} {kz[0]:6.2f} {N:3d} {nd:6d} | ' +
                      ' | '.join(f'{it:5d} {cd:8.1e}' for it, cd in res) +
                      f' | {cu:10.1e} {cw:10.1e} {cp:10.1e}', flush=True)
            print()

if __name__ == '__main__':
    main()
