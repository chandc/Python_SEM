"""The whole preconditioned spectrum of M A -- no sampling.

M is SPD (scratch/pmg_symmetry.py), so CG is governed by cond(M A) and nothing
else.  The earlier eigenvector test sampled 118 of 1806 directions and found all
of them well reduced; that is consistent with 402 CG iterations only if the bad
directions were among the ~1700 NOT sampled.  So build M A densely and look at
every eigenvalue.

In the global basis B with multiplicity weight, pick(u) = B^T diag(mw) u, so
    T = B^T diag(mw) M A B
is the preconditioned operator in coefficient space.  Its eigenvalues are real
and are exactly what CG sees.  Predicted CG ~ 0.5*sqrt(cond)*ln(2/tol).
"""
import os, sys, time
for _v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '8')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); os.chdir(_R)
os.environ.setdefault('LSSEM3D_BACKEND', 'numba')
import numpy as np

def main(N=8, ex=2, ey=2, cc=5405.4):
    import lssem3d; lssem3d.set_backend(os.environ['LSSEM3D_BACKEND'])
    from lssem2d.mesh import build_channel
    from lssem2d.lgl import diff_matrix
    from lssem3d import bc as BC, operator as OP, precond as P3, solver3d as S3

    nz, nk = 4, 1
    m = build_channel(2.0*np.pi, 2.0, ex, ey, N, bcs=(0, 0, 1, 1))
    m.periodic_x = 2.0*np.pi; m.compute_global_indices()
    kz = np.array([5.882])
    mask = BC.build_mask(m, nk, pin_p=False, nz=nz); BC.pin_dof(m, mask, OP.P_, 0)
    nu, D = 1/180., diff_matrix(N)
    rw = OP.momentum_row_weights(cc)
    shape = (m.nelem, N+1, N+1, OP.NVAR_R, nk)
    mwa = S3.multiplicity_weight(m, shape)
    A = lambda x: S3.normal_op(x, D, m.facx, m.facy, kz, nu, cc, m, mask,
                               m.wq, 0.0, rw)

    cols, seen, ids = [], set(), []
    for e in range(shape[0]):
        for i in range(shape[1]):
            for j in range(shape[2]):
                for f in range(OP.NVAR_R):
                    if mask[e, i, j, f, 0] == 0.0: continue
                    ed = np.zeros(shape); ed[e, i, j, f, 0] = 1.0
                    g = S3.gs(m, ed)
                    key = tuple(np.flatnonzero(np.abs(g.ravel()) > 0.5))
                    if not key or key in seen: continue
                    seen.add(key); cols.append(g); ids.append(f)
    B = np.stack([c.ravel() for c in cols], axis=1); nd = B.shape[1]
    ids = np.array(ids)
    print(f'N={N} {ex}x{ey} k_z=5.88 c={cc:g}: {nd} dof', flush=True)
    Bw = (B*mwa.ravel()[:, None]).T                     # = B^T diag(mw)

    for tag, M in (('Jacobi', None),
                   ('PMG (8,4,2) deg=6', P3.PMG(m, nk, nz, nu, cc, kz, kap=0.0,
                        rw=rw, orders=(8,4,2), deg=6, pin_p=True,
                        direct_coarse='element', mask=mask))):
        if M is None:
            lv = P3._Level(m, nk, nz, nu, cc, kz, 0.0, rw, False, mask=mask)
            ap = lambda r: lv.M_inv*r
        else:
            ap = M
        t0 = time.perf_counter()
        T = np.empty((nd, nd))
        for a in range(nd):
            T[:, a] = Bw @ ap(A(B[:, a].reshape(shape))).ravel()
        ev = np.linalg.eigvals(T).real
        ev = ev[ev > 1e-14*ev.max()]
        cond = ev.max()/ev.min()
        pred = 0.5*np.sqrt(cond)*np.log(2/1e-8)
        print(f'  {tag:20s} built {time.perf_counter()-t0:5.1f}s  '
              f'lam {ev.min():.3e} .. {ev.max():.3e}  cond {cond:.3e}  '
              f'CG predicted ~{pred:.0f}', flush=True)
        lo = np.sort(ev)[:12]
        print(f'    12 smallest: ' + ' '.join(f'{v:.2e}' for v in lo), flush=True)

if __name__ == '__main__':
    main()
