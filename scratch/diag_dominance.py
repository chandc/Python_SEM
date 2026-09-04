"""Does small dt make A diagonally dominant, so Jacobi should do well?

The intuition: c = 1/(beta*dt) is large for small dt, the mass term dominates,
diagonal dominance follows, Jacobi wins.  That is right for the VELOCITY block
and the question is whether it holds for the system.

Watch what the 1/c^2 momentum weighting actually leaves behind as c -> infinity:

    (1/c^2)(c u + grad p + nu curl om)^2  ->  (u + grad p/c + ...)^2  ->  u^2

so the momentum rows contribute a clean MASS matrix on u -- diagonal, dominant,
exactly as expected.  But the CONSTRAINT rows carry weight 1 and are unaffected
by c at all:

    R0 = div u          -> derivative-squared, no diagonal mass term
    R1..R3 = om - curl u -> om^2 (mass) AND (curl u)^2 (derivative)

and pressure has NO mass term anywhere: it enters only through grad p/c in the
momentum rows, so its block scales as 1/c^2 and VANISHES as dt -> 0.

Measures, per field and vs c: row diagonal dominance |a_ii|/sum_j|a_ij|,
cond(D^-1 A), and what the softest mode is made of.
"""
import os, sys
for _v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '4')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); os.chdir(_R)
os.environ.setdefault('LSSEM3D_BACKEND', 'numpy')
import numpy as np

def main(N=6, ex=2, ey=2, kcol=1):
    import lssem3d; lssem3d.set_backend('numpy')
    from lssem2d.mesh import build_channel
    from lssem2d.lgl import diff_matrix
    from lssem3d import bc as BC, operator as OP, solver3d as S3, fourier as FR
    nz, nk = 32, 17
    m = build_channel(np.pi, 2.0, ex, ey, N, bcs=(0, 0, 1, 1))
    m.periodic_x = np.pi; m.compute_global_indices()
    mf = BC.build_mask(m, nk, pin_p=False, nz=nz)
    BC.pin_dof(m, mf, OP.P_, 0); BC.pin_dof(m, mf, OP.NVAR+OP.P_, 0)
    mask = np.ascontiguousarray(mf[..., kcol:kcol+1])
    kz = np.array([float(FR.wavenumbers(nz, 0.34*np.pi)[kcol])])
    nu, D = 1/180., diff_matrix(N)
    sh = (m.nelem, N+1, N+1, OP.NVAR_R, 1)
    mwa = S3.multiplicity_weight(m, sh)

    cols, seen, fld = [], set(), []
    for e in range(sh[0]):
        for i in range(sh[1]):
            for j in range(sh[2]):
                for f in range(OP.NVAR_R):
                    if mask[e, i, j, f, 0] == 0.0: continue
                    ed = np.zeros(sh); ed[e, i, j, f, 0] = 1.0
                    g = S3.gs(m, ed)
                    k = tuple(np.flatnonzero(np.abs(g.ravel()) > 0.5))
                    if not k or k in seen: continue
                    seen.add(k); cols.append(g); fld.append(f % OP.NVAR)
    B = np.stack([c.ravel() for c in cols], axis=1); fld = np.array(fld)
    names = {OP.U_:'u', OP.V_:'v', OP.W_:'w', OP.OX_:'ox', OP.OY_:'oy',
             OP.OZ_:'oz', OP.P_:'p'}
    print(f'N={N} {ex}x{ey} k_z={kz[0]:.2f}, {B.shape[1]} dof\n')
    print(f'{"c":>8} {"weighting":>15} | {"cond(D^-1A)":>12} | ' +
          ' '.join(f'{"dd_"+names[q]:>7}' for q in (OP.U_, OP.OX_, OP.P_)) +
          ' | softest mode')
    print('-'*94)
    import itertools
    for cc, wm in itertools.product((1.0, 100.0, 5405.4, 50000.0), (None, 'raw')):
        # wm='raw' sets rw[4:7]=1, i.e. NO 1/c^2 weighting -- the momentum rows
        # keep their c^2|u|^2 term, which is the term that would make A
        # diagonally dominant as dt -> 0.
        rw = (OP.momentum_row_weights(cc) if wm is None
              else OP.momentum_row_weights(cc, w_mom=cc*cc))
        kw = dict(mesh=m, mask=mask, wq=m.wq, kap=0.0, rw=rw)
        A = lambda x: S3.normal_op(x, D, m.facx, m.facy, kz, nu, cc, **kw)
        Ad = np.empty((B.shape[1],)*2)
        for a in range(B.shape[1]):
            Ad[:, a] = B.T @ (A(B[:, a].reshape(sh)).ravel()*mwa.ravel())
        Ad = 0.5*(Ad + Ad.T)
        dg = np.abs(np.diag(Ad))
        off = np.abs(Ad).sum(axis=1) - dg
        dd = dg/np.maximum(off, 1e-300)          # >1 = diagonally dominant row
        d = np.diag(Ad).copy(); d[d <= 0] = 1.0
        S = np.diag(1/np.sqrt(d))
        w, V = np.linalg.eigh(S @ Ad @ S); w = w[w > 1e-13*w.max()]
        q = S @ V[:, 0]; tot = float(q @ q)
        comp = {names[f]: 100*float((q[fld == f]**2).sum())/tot for f in set(fld)}
        top = sorted(comp.items(), key=lambda kv: -kv[1])[:2]
        tag = 'weighted 1/c^2' if wm is None else 'RAW (no 1/c^2)'
        print(f'{cc:8.0f} {tag:>15} | {w[-1]/w[0]:12.3e} | ' +
              ' '.join(f'{np.median(dd[fld == q_]):7.3f}' for q_ in (OP.U_, OP.OX_, OP.P_)) +
              ' | ' + ', '.join(f'{k} {v:.0f}%' for k, v in top))
    print('\ndd = median over that field\'s rows of |a_ii| / sum_j!=i |a_ij|.')
    print('dd > 1 means diagonally dominant rows.  Watch u vs p as c grows.')

if __name__ == '__main__':
    main()
