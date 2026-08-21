"""Dense spectrum of the preconditioned VVP operator -- the invariant measurement.

    uv run --quiet python scratch/spectrum.py

WHY THIS EXISTS.  Four preconditioner conclusions in a row were drawn from
convenient-but-invalid measurements and each was wrong:

  1. "Jacobi is resolution-independent"  -- the RHS was b = A x with x RANDOM, so
     the problem got rougher as it got finer.  A rough RHS is dominated by the
     well-conditioned high end of the spectrum and hides the growing low end.
  2. "PMG is actively harmful on smooth modes" -- compared ||e - P(A e)|| across
     two preconditioners with DIFFERENT SCALINGS.  A preconditioner is not A^-1;
     its output scale is arbitrary, so that comparison means nothing.
  3. "The coarse solve is too weak" -- refuted by replacing Chebyshev with an
     exact direct solve, which changed the convergence factor from 0.9904 to
     0.9899, i.e. not at all.
  4. "The coarse operator is 100x worse conditioned" -- compared p=6 at
     nu=1/180, c=525 against p=3 at nu=1/100, c=600.  Different parameters.

Every one was caught by the NEXT test contradicting it.  The common fault was
reaching for whatever was convenient rather than something invariant to the
thing being varied.  This module computes the quantity that settles all four --
the dense spectrum of M^-1 A -- and it should be the first stop, not the last.

It is affordable because the modes decouple: one dense block per k_z, built in
the GLOBAL (continuous) basis, which is the space the assembled operator acts on.
"""
import os, sys, json
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem3d import operator as OP, solver3d as S3, bc as BC


def global_basis(mesh, mask, shape, mode=0):
    """Continuous basis: gs of a one-hot, deduplicated by support."""
    cols, seen = [], set()
    for idx in np.argwhere(mask[..., mode] != 0):
        e, i, j, f = idx
        ed = np.zeros(shape)
        ed[e, i, j, f, mode] = 1.0
        g = S3.gs(mesh, ed)
        key = tuple(np.flatnonzero(np.abs(g.ravel()) > 0.5))
        if key and key not in seen:
            seen.add(key)
            cols.append(g)
    return np.stack([c.ravel() for c in cols], axis=1)


def spectrum(p, ex=2, nu=1/100., c=600., precond=True, rowweight=True):
    """Eigenvalues of M^-1 A (or A) at order p, single mode."""
    m = build_channel(1., 1., ex, ex, p, bcs=(1, 1, 1, 2))
    D = diff_matrix(p)
    kz = np.zeros(1)
    mask = BC.build_mask(m, 1, pin_p=True, nz=1)
    shape = (m.nelem, p+1, p+1, OP.NVAR_R, 1)
    rw = OP.momentum_row_weights(c) if rowweight else None
    kw = dict(mesh=m, mask=mask, wq=m.wq, kap=0.0, rw=rw)
    Mi = (S3.jacobi_inverse(S3.jacobi_diagonal_analytic(
        shape, D, m.facx, m.facy, kz, nu, c, **kw), mask) if precond else 1.0)
    B = global_basis(m, mask, shape)
    mw = S3.multiplicity_weight(m, shape).ravel()
    M = np.empty((B.shape[1],)*2)
    for a in range(B.shape[1]):
        Av = S3.normal_op(B[:, a].reshape(shape), D, m.facx, m.facy, kz, nu,
                          c, **kw).ravel()
        M[:, a] = B.T @ ((Mi.ravel() if precond else 1.0)*Av*mw)
    M = 0.5*(M + M.T)
    ev = np.sort(np.real(np.linalg.eigvals(M)))
    return ev[np.abs(ev) > 1e-14], B.shape[1]


if __name__ == '__main__':
    print('Dense spectrum of M^-1 A, 2x2 elements, k_z = 0, nu=1/100, c=600')
    print('IDENTICAL parameters at every order -- the mistake in claim 4 above\n')
    print(f"{'p':>4}{'dofs':>8}{'lam_min':>12}{'lam_max':>11}{'cond':>12}"
          f"{'sqrt(cond)':>12}")
    out = []
    for p in (2, 3, 4, 6, 8, 10):
        ev, n = spectrum(p)
        cond = ev.max()/ev.min()
        out.append(dict(p=p, dofs=int(n), lam_min=float(ev.min()),
                        lam_max=float(ev.max()), cond=float(cond)))
        print(f'{p:>4}{n:>8}{ev.min():>12.3e}{ev.max():>11.3e}{cond:>12.3e}'
              f'{np.sqrt(cond):>12.0f}', flush=True)
    json.dump(out, open('scratch/spectrum.json', 'w'), indent=1)
    print('\n  cond grows ~p^4-p^6 -- Jacobi DOES degrade with order, as SEM must.')
    print('  But cond ~ 1.8e8 even at p = 2: the ill-conditioning is intrinsic to')
    print('  the least-squares VVP operator at EVERY order, so p-coarsening')
    print('  cannot hand multigrid an easy coarse problem.')
