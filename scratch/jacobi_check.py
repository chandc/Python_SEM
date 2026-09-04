"""Is the PRODUCTION Jacobi diagonal correct?

2000 CG iterations for one Fourier mode is excessive, and jacobi_diagonal's own
docstring says of the analytic form: "this routine is the thing it must be
checked against when written."  So check it, three ways:

  1. jacobi_diagonal_analytic  -- production (channel3d.make_precond uses it)
  2. jacobi_diagonal           -- reference, probes the UNASSEMBLED operator
                                  then gs()es; docstring says "verified exact"
  3. diag of the densely assembled A in the global basis -- ground truth

A wrong diagonal costs iterations silently: it stays SPD, so CG still converges,
just slowly.  That is exactly the symptom.
"""
import os, sys
for _v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '4')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); os.chdir(_R)
os.environ.setdefault('LSSEM3D_BACKEND', 'numpy')
import numpy as np

def main(N=6, ex=2, ey=2, cc=5405.4):
    import lssem3d; lssem3d.set_backend('numpy')
    from lssem2d.mesh import build_channel
    from lssem2d.lgl import diff_matrix
    from lssem3d import bc as BC, operator as OP, solver3d as S3, fourier as FR
    nz, nk = 32, 17
    m = build_channel(np.pi, 2.0, ex, ey, N, bcs=(0, 0, 1, 1))
    m.periodic_x = np.pi; m.compute_global_indices()
    mf = BC.build_mask(m, nk, pin_p=False, nz=nz)
    BC.pin_dof(m, mf, OP.P_, 0); BC.pin_dof(m, mf, OP.NVAR+OP.P_, 0)
    kcol = 1
    mask = np.ascontiguousarray(mf[..., kcol:kcol+1])
    kz = np.array([float(FR.wavenumbers(nz, 0.34*np.pi)[kcol])])
    nu, D = 1/180., diff_matrix(N)
    rw = OP.momentum_row_weights(cc)
    sh = (m.nelem, N+1, N+1, OP.NVAR_R, 1)
    kw = dict(mesh=m, mask=mask, wq=m.wq, kap=0.0, rw=rw)

    d_prod = S3.jacobi_diagonal_analytic(sh, D, m.facx, m.facy, kz, nu, cc, **kw)
    d_ref  = S3.jacobi_diagonal(sh, D, m.facx, m.facy, kz, nu, cc, **kw)

    # ground truth: diagonal of the assembled operator in the global basis
    mwa = S3.multiplicity_weight(m, sh)
    A = lambda x: S3.normal_op(x, D, m.facx, m.facy, kz, nu, cc, **kw)
    cols, seen, loc = [], set(), []
    for e in range(sh[0]):
        for i in range(sh[1]):
            for j in range(sh[2]):
                for f in range(OP.NVAR_R):
                    if mask[e, i, j, f, 0] == 0.0: continue
                    ed = np.zeros(sh); ed[e, i, j, f, 0] = 1.0
                    g = S3.gs(m, ed)
                    k = tuple(np.flatnonzero(np.abs(g.ravel()) > 0.5))
                    if not k or k in seen: continue
                    seen.add(k); cols.append(g); loc.append((e, i, j, f))
    B = np.stack([c.ravel() for c in cols], axis=1)
    true = np.array([B[:, a] @ (A(B[:, a].reshape(sh)).ravel()*mwa.ravel())
                     for a in range(B.shape[1])])
    pv = np.array([d_prod[e, i, j, f, 0] for (e, i, j, f) in loc])
    rv = np.array([d_ref[e, i, j, f, 0] for (e, i, j, f) in loc])

    def cmp(a, b, na, nb):
        ok = np.abs(b) > 0
        r = np.abs(a[ok]-b[ok])/np.abs(b[ok])
        print(f'  {na:28s} vs {nb:18s} max rel {r.max():.3e}  '
              f'median {np.median(r):.3e}  >1% at {int((r>0.01).sum())}/{ok.sum()} dofs')

    print(f'N={N} {ex}x{ey} k_z={kz[0]:.2f} c={cc:g}, {B.shape[1]} global dof\n')
    cmp(pv, true, 'jacobi_diagonal_analytic', 'TRUE diag(A)')
    cmp(rv, true, 'jacobi_diagonal (ref)', 'TRUE diag(A)')
    cmp(pv, rv,  'jacobi_diagonal_analytic', 'reference')
    print(f'\n  ratio analytic/true: min {np.min(pv/true):.4f}  max {np.max(pv/true):.4f}')
    # which fields are wrong?
    fl = np.array([f for (_, _, _, f) in loc])
    bad = np.abs(pv-true)/np.abs(true) > 0.01
    if bad.any():
        import collections
        cnt = collections.Counter(fl[bad])
        names = {OP.U_:'u',OP.V_:'v',OP.W_:'w',OP.OX_:'ox',OP.OY_:'oy',OP.OZ_:'oz',OP.P_:'p'}
        print('  wrong dofs by field: ' + ', '.join(
            f'{names.get(f%OP.NVAR,f)}{"(im)" if f>=OP.NVAR else ""}:{n}'
            for f, n in sorted(cnt.items())))

if __name__ == '__main__':
    main()
