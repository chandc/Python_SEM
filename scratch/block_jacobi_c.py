"""Would BLOCK Jacobi help the time-dependent operator?

Different question from sec 7S.3a, which asked whether block smoothing rescues
w7=1 (it recovers 2% -- that near-null space is global, unreachable pointwise).

Here the argument is stronger: sec 2.1 shows pressure loses its own diagonal
entirely as dt -> 0 (dd_p -> 0.000) while remaining COUPLED to velocity through
grad p in the momentum rows.  A block containing p and u together could supply
what the point diagonal cannot.

The catch: grad p is a DERIVATIVE, so in a spectral element p at node i couples
to u at EVERY node in i's row -- not just at i.  A 14x14 point block therefore
captures only the D[i,i] part of that coupling.  An ELEMENT block captures all
of it.  So measure three preconditioners at production weighting, vs c:

    point   diag(A)                    -- what the channel runs
    node    14x14 per node             -- all fields at one node
    elem    additive Schwarz, one block per element (all its dofs)

cond(M^-1 A) after each, plus the softest mode, so we can see whether the
pressure direction is what improves.
"""
import os, sys
for _v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '4')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); os.chdir(_R)
os.environ.setdefault('LSSEM3D_BACKEND', 'numpy')
import numpy as np
import scipy.linalg as sla


def sym_cond(A, Ph):
    e = np.linalg.eigvalsh(Ph @ A @ Ph); e = e[e > 1e-13*e.max()]
    return e[-1]/e[0], e


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

    cols, seen, node, fld, elems = [], set(), [], [], []
    for e in range(sh[0]):
        for i in range(sh[1]):
            for j in range(sh[2]):
                for f in range(OP.NVAR_R):
                    if mask[e, i, j, f, 0] == 0.0: continue
                    ed = np.zeros(sh); ed[e, i, j, f, 0] = 1.0
                    g = S3.gs(m, ed)
                    k = tuple(np.flatnonzero(np.abs(g.ravel()) > 0.5))
                    if not k or k in seen: continue
                    seen.add(k); cols.append(g); node.append((e, i, j))
                    fld.append(f % OP.NVAR)
                    elems.append(set(np.flatnonzero(np.abs(g).reshape(sh[0], -1).sum(1) > 0.5)))
    B = np.stack([c.ravel() for c in cols], axis=1); nd = B.shape[1]
    fld = np.array(fld)
    names = {OP.U_:'u', OP.V_:'v', OP.W_:'w', OP.OX_:'ox', OP.OY_:'oy',
             OP.OZ_:'oz', OP.P_:'p'}
    ngrp = {}
    for a, nd_ in enumerate(node): ngrp.setdefault(nd_, []).append(a)
    egrp = {}
    for a, es in enumerate(elems):
        for e in es: egrp.setdefault(e, []).append(a)
    print(f'N={N} {ex}x{ey} k_z={kz[0]:.2f}, {nd} dof, {len(ngrp)} node blocks '
          f'(<=14), {len(egrp)} element blocks (~{np.mean([len(v) for v in egrp.values()]):.0f})\n')
    print(f'{"c":>8} | {"point":>10} {"node 14x14":>11} {"elem Schwarz":>13} | '
          f'{"node gain":>9} {"elem gain":>9} | softest after elem')
    print('-'*100)

    for cc in (1.0, 100.0, 5405.4, 50000.0):
        rw = OP.momentum_row_weights(cc)
        kw = dict(mesh=m, mask=mask, wq=m.wq, kap=0.0, rw=rw)
        A = lambda x: S3.normal_op(x, D, m.facx, m.facy, kz, nu, cc, **kw)
        Ad = np.empty((nd, nd))
        for a in range(nd):
            Ad[:, a] = B.T @ (A(B[:, a].reshape(sh)).ravel()*mwa.ravel())
        Ad = 0.5*(Ad + Ad.T)

        d = np.diag(Ad).copy(); d[d <= 0] = 1.0
        c_pt, _ = sym_cond(Ad, np.diag(1/np.sqrt(d)))

        def blk(groups):
            P = np.zeros_like(Ad)
            for idx in groups.values():
                ix = np.ix_(idx, idx)
                w, V = np.linalg.eigh(0.5*(Ad[ix]+Ad[ix].T))
                w = np.maximum(w, 1e-14*max(w.max(), 1.0))
                P[ix] += V @ np.diag(1.0/w) @ V.T          # additive
            w, V = np.linalg.eigh(0.5*(P+P.T))
            w = np.maximum(w, 1e-300)
            return V @ np.diag(np.sqrt(w)) @ V.T
        c_nd, _ = sym_cond(Ad, blk(ngrp))
        Phe = blk(egrp); c_el, ee = sym_cond(Ad, Phe)
        w2, V2 = np.linalg.eigh(Phe @ Ad @ Phe)
        q = Phe @ V2[:, np.argmax(w2 > 1e-13*w2.max())]
        tot = float(q @ q) or 1.0
        comp = {names[f]: 100*float((q[fld == f]**2).sum())/tot for f in set(fld)}
        top = sorted(comp.items(), key=lambda kv: -kv[1])[:2]
        print(f'{cc:8.0f} | {c_pt:10.3e} {c_nd:11.3e} {c_el:13.3e} | '
              f'{c_pt/c_nd:8.2f}x {c_pt/c_el:8.2f}x | ' +
              ', '.join(f'{k} {v:.0f}%' for k, v in top), flush=True)
    print('\ngain > 1 means the block preconditioner beats point Jacobi.')

if __name__ == '__main__':
    main()
