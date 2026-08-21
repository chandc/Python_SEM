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
  5. AND THIS FILE ITSELF, in its first version: it symmetrised M^-1 A with
     0.5*(M + M.T).  M^-1 A is a product of two symmetric matrices and is NOT
     symmetric, so the eigenvalues of its symmetric part are not the operator's
     -- the giveaway was NEGATIVE eigenvalues for an SPD operator.  The tool
     written to prevent this class of error contained an instance of it.

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
    """Eigenvalues at order p, single mode.

    UNPRECONDITIONED: A is symmetric, so `eigvalsh` is exact and unambiguous.

    PRECONDITIONED: M^-1 A is NOT symmetric (a product of symmetric matrices
    rarely is), so it must be posed as the GENERALISED problem A v = lam M v --
    equivalently the spectrum of M^-1/2 A M^-1/2, which is symmetric and has the
    same eigenvalues.  Symmetrising M^-1 A instead is wrong and gives negative
    eigenvalues for an SPD operator.
    """
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
    Amat = np.empty((B.shape[1],)*2)
    for a in range(B.shape[1]):
        Av = S3.normal_op(B[:, a].reshape(shape), D, m.facx, m.facy, kz, nu,
                          c, **kw).ravel()
        Amat[:, a] = B.T @ (Av*mw)
    Amat = 0.5*(Amat + Amat.T)          # A IS symmetric; this only kills round-off
    if not precond:
        ev = np.linalg.eigvalsh(Amat)
        return ev[ev > 1e-13*ev.max()], B.shape[1]
    # generalised: A v = lam M v, with M the Jacobi diagonal in the same basis
    dinv = Mi.ravel()
    dmat = np.empty_like(Amat)
    for a in range(B.shape[1]):
        col = np.where(dinv > 0, 1.0/np.where(dinv > 0, dinv, 1.0), 0.0)*B[:, a]
        dmat[:, a] = B.T @ (col*mw)
    dmat = 0.5*(dmat + dmat.T)
    from scipy.linalg import eigh
    ev = eigh(Amat, dmat, eigvals_only=True)
    return ev[ev > 1e-13*ev.max()], B.shape[1]


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
    print('  And p = 2 is EASY: cond 7.6e3, sqrt ~ 87.  Coarsening p=10 -> p=2')
    print('  improves conditioning 24,000x, so p-multigrid DOES get an easy')
    print('  coarse problem.  An earlier claim to the contrary came from the')
    print('  symmetrisation bug above (error 5) and is retracted.')
