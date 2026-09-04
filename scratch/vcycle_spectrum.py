"""WHICH error does the V-cycle fail to remove?

Per mode (the operator never couples k_z, so this is the natural unit):

  1. build A densely in the global basis for one k_z
  2. eigendecompose the JACOBI-PRECONDITIONED operator D^-1/2 A D^-1/2 -- that is
     the spectrum CG actually sees
  3. for selected eigenvectors v, run ONE V-cycle and measure how much of v
     survives:  e = v - M(A v),  reduction = ||e||/||v||
  4. report the field composition of whatever survives

A textbook multigrid gives small reduction everywhere: the smoother kills the
top of the spectrum, the coarse grid kills the bottom.  If instead the survivors
sit in the MIDDLE and are made of omega, that is sec 7J's vorticity near-null
space -- error that is neither high-frequency nor low-order-polynomial, so
neither mechanism sees it and CG grinds it out alone.  That would explain the
p-degradation (110 -> 228 -> 379) and why the coarse solve barely matters.
"""
import os, sys, time
for _v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '8')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); os.chdir(_R)
os.environ.setdefault('LSSEM3D_BACKEND', 'numba')
import numpy as np

def main(N=8, ex=2, ey=2, kz_val=5.882, cc=5405.4):
    import lssem3d; lssem3d.set_backend(os.environ['LSSEM3D_BACKEND'])
    from lssem2d.mesh import build_channel
    from lssem2d.lgl import diff_matrix
    from lssem3d import bc as BC, operator as OP, precond as P3, solver3d as S3

    nz, nk = 4, 1
    m = build_channel(2.0*np.pi, 2.0, ex, ey, N, bcs=(0, 0, 1, 1))
    m.periodic_x = 2.0*np.pi; m.compute_global_indices()
    kz = np.array([kz_val])
    mask = BC.build_mask(m, nk, pin_p=False, nz=nz); BC.pin_dof(m, mask, OP.P_, 0)
    nu, D = 1/180., diff_matrix(N)
    rw = OP.momentum_row_weights(cc)
    shape = (m.nelem, N+1, N+1, OP.NVAR_R, nk)
    lv = P3._Level(m, nk, nz, nu, cc, kz, 0.0, rw, False, mask=mask)
    mw = lv.mw

    # ---- global basis: gs of one-hots, deduplicated by support ----
    cols, seen, first = [], set(), []
    for e in range(shape[0]):
        for i in range(shape[1]):
            for j in range(shape[2]):
                for f in range(OP.NVAR_R):
                    if mask[e, i, j, f, 0] == 0.0: continue
                    ed = np.zeros(shape); ed[e, i, j, f, 0] = 1.0
                    g = S3.gs(m, ed)
                    key = tuple(np.flatnonzero(np.abs(g.ravel()) > 0.5))
                    if not key or key in seen: continue
                    seen.add(key); cols.append(g); first.append((e, i, j, f))
    B = np.stack([c.ravel() for c in cols], axis=1)
    nd = B.shape[1]
    print(f'N={N} {ex}x{ey} k_z={kz_val:g} c={cc:g}: {nd} global dof', flush=True)

    t0 = time.perf_counter()
    Ad = np.empty((nd, nd))
    for a in range(nd):
        Ad[:, a] = B.T @ (lv.A(B[:, a].reshape(shape)).ravel()*mw.ravel())
    Ad = 0.5*(Ad + Ad.T)
    print(f'  A assembled in {time.perf_counter()-t0:.1f}s', flush=True)

    d = np.diag(Ad).copy(); d[d <= 0] = 1.0
    S = np.diag(1.0/np.sqrt(d))
    w, V = np.linalg.eigh(S @ Ad @ S)          # spectrum CG sees
    keep = w > 0; w, V = w[keep], V[:, keep]
    print(f'  cond(D^-1 A) = {w[-1]/w[0]:.3e}', flush=True)

    M = P3.PMG(m, nk, nz, nu, cc, kz, kap=0.0, rw=rw, orders=(8, 4, 2), deg=6,
               pin_p=True, direct_coarse='element', mask=mask)

    idx = np.unique(np.concatenate([np.arange(0, min(60, len(w))),
                                    np.linspace(0, len(w)-1, 60).astype(int)]))
    rows = []
    for i in idx:
        c = S @ V[:, i]                        # back to the unscaled basis
        u = (B @ c).reshape(shape)
        nu0 = np.linalg.norm(u)
        e = u - M(lv.A(u))
        red = np.linalg.norm(e)/max(nu0, 1e-300)
        comp = [float((u[..., q, :]**2).sum()) for q in
                (OP.U_, OP.V_, OP.W_, OP.OX_, OP.OY_, OP.OZ_, OP.P_)]
        tot = sum(comp) or 1.0
        rows.append((i, w[i], red, 100*(comp[3]+comp[4]+comp[5])/tot,
                     100*(comp[0]+comp[1]+comp[2])/tot, 100*comp[6]/tot))
    print(f'\n{"rank":>5} {"lambda":>11} {"survives":>9} | {"om%":>6} {"vel%":>6} {"p%":>6}')
    for i, l, r, o, v, p in rows:
        flag = '   <-- NOT REDUCED' if r > 0.9 else ''
        print(f'{i:5d} {l:11.3e} {r:9.4f} | {o:6.1f} {v:6.1f} {p:6.1f}{flag}')
    bad = [r for r in rows if r[2] > 0.9]
    print(f'\n{len(bad)}/{len(rows)} sampled modes survive a full V-cycle (>0.9)')
    if bad:
        print(f'  their mean composition: om {np.mean([b[3] for b in bad]):.1f}%  '
              f'vel {np.mean([b[4] for b in bad]):.1f}%  p {np.mean([b[5] for b in bad]):.1f}%')
        print(f'  their lambda range: {min(b[1] for b in bad):.2e} .. {max(b[1] for b in bad):.2e}'
              f'   (full spectrum {w[0]:.2e} .. {w[-1]:.2e})')

if __name__ == '__main__':
    main()
