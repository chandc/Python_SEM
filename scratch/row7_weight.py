"""Does down-weighting the vorticity-divergence row hold up?

    uv run --quiet python scratch/row7_weight.py

THE FINDING TO TEST.  At k_z = 0 the softest eigenmodes of the preconditioned
operator are ENTIRELY omega_x, omega_y (energy fraction 0.50/0.50, zero in
u,v,w,omega_z,p).  Those two fields appear in exactly one row that 2D does not
have: R7 = div(omega) = 0.  It contributes a derivative-squared term to their
Jacobi diagonal while contributing NOTHING to A for modes already satisfying
div(omega) = 0 -- big denominator, zero numerator, near-null cluster.

Measured at N=6, 2x2, k_z=0:

    row-7 weight    1        1e-2     1e-4     0
    cond            5.55e6   5.97e4   1.01e4   1.01e4

That is the answer to "why didn't 2D have this problem": 2D has no div(omega)
row at all.

WHAT THIS SCRIPT CHECKS, because one spectrum at one N is not a result:
  1. does the conditioning gain hold across N?
  2. does it hold at k_z != 0, where R7 also contains i*k*omega_z and is NOT
     confined to the transverse fields?
  3. does it convert into actual CG iterations and wall time?
  4. is div(omega) still satisfied?  The row is being down-weighted, not
     deleted, and if the constraint degrades the gain is not free.
"""
import os, sys, json
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
import time
from scipy.linalg import eigh
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem3d import operator as OP, solver3d as S3, bc as BC, fourier as FR, deriv as DV

NU, C = 1/100., 600.


def basis(m, mask, shape, mode=0):
    cols, seen = [], set()
    for idx in np.argwhere(mask[..., mode] != 0):
        e, i, j, f = idx
        ed = np.zeros(shape); ed[e, i, j, f, mode] = 1.0
        g = S3.gs(m, ed)
        key = tuple(np.flatnonzero(np.abs(g.ravel()) > 0.5))
        if key and key not in seen:
            seen.add(key); cols.append(g)
    return np.stack([c.ravel() for c in cols], axis=1)


def cond_at(p, w7, ex=2, nz=1, mode=0):
    m = build_channel(1., 1., ex, ex, p, bcs=(1, 1, 1, 2))
    D = diff_matrix(p)
    nk = nz//2 + 1 if nz > 1 else 1
    kz = FR.wavenumbers(nz, 2*np.pi)[:nk] if nz > 1 else np.zeros(1)
    mask = BC.build_mask(m, nk, pin_p=True, nz=max(nz, 1))
    shape = (m.nelem, p+1, p+1, OP.NVAR_R, nk)
    rw = OP.momentum_row_weights(C); rw[7] = w7
    kw = dict(mesh=m, mask=mask, wq=m.wq, kap=0.0, rw=rw)
    Mi = S3.jacobi_inverse(S3.jacobi_diagonal_analytic(
        shape, D, m.facx, m.facy, kz, NU, C, **kw), mask)
    dv = Mi.ravel()
    B = basis(m, mask, shape, mode)
    mw = S3.multiplicity_weight(m, shape).ravel()
    A = np.empty((B.shape[1],)*2); Mg = np.empty_like(A)
    for a in range(B.shape[1]):
        A[:, a] = B.T @ (S3.normal_op(B[:, a].reshape(shape), D, m.facx, m.facy,
                                      kz, NU, C, **kw).ravel()*mw)
        col = np.where(dv > 0, 1.0/np.where(dv > 0, dv, 1.0), 0.0)*B[:, a]
        Mg[:, a] = B.T @ (col*mw)
    A = 0.5*(A+A.T); Mg = 0.5*(Mg+Mg.T)
    w = eigh(A, Mg, eigvals_only=True); w = w[w > 1e-13*w.max()]
    return w.min(), w.max()/w.min()


if __name__ == '__main__':
    print('1. CONDITIONING vs N  (2x2 elements, k_z = 0)')
    print(f"{'p':>4}{'w7=1':>13}{'w7=1e-4':>13}{'gain':>9}")
    for p in (4, 6, 8, 10):
        _, c1 = cond_at(p, 1.0)
        _, c2 = cond_at(p, 1e-4)
        print(f'{p:>4}{c1:>13.3e}{c2:>13.3e}{c1/c2:>8.0f}x', flush=True)

    print('\n2. CONDITIONING at k_z != 0  (R7 also contains i*k*omega_z there)')
    print(f"{'mode':>6}{'w7=1':>13}{'w7=1e-4':>13}{'gain':>9}")
    for mode in (0, 1, 2):
        _, c1 = cond_at(6, 1.0, nz=8, mode=mode)
        _, c2 = cond_at(6, 1e-4, nz=8, mode=mode)
        print(f'{mode:>6}{c1:>13.3e}{c2:>13.3e}{c1/c2:>8.0f}x', flush=True)

    print('\n3. CG ITERATIONS and 4. div(omega) on a real solve  (3x3, nz=8)')
    print(f"{'w7':>8}{'its':>8}{'wall s':>9}{'rms div(om)':>14}{'rel to |om|':>13}")
    p, ex, nz = 8, 3, 8
    m = build_channel(1., 1., ex, ex, p, bcs=(1, 1, 1, 2))
    D = diff_matrix(p); nk = nz//2+1; kz = FR.wavenumbers(nz, 2*np.pi)
    mask = BC.build_mask(m, nk, pin_p=True, nz=nz)
    shape = (m.nelem, p+1, p+1, OP.NVAR_R, nk)
    xs = S3.make_continuous(m, np.random.default_rng(0).standard_normal(shape))*mask
    for w7 in (1.0, 1e-2, 1e-4, 0.0):
        rw = OP.momentum_row_weights(C); rw[7] = w7
        kw = dict(mesh=m, mask=mask, wq=m.wq, kap=0.0, rw=rw)
        b = S3.normal_op(xs, D, m.facx, m.facy, kz, NU, C, **kw)
        Mi = S3.jacobi_inverse(S3.jacobi_diagonal_analytic(
            shape, D, m.facx, m.facy, kz, NU, C, **kw), mask)
        t0 = time.perf_counter()
        x, it, _ = S3.pcg(b, D, m.facx, m.facy, kz, NU, C, m, mask, Mi, 1e-6,
                          40000, None, m.wq, 0.0, rw)
        t = time.perf_counter()-t0
        Uc = OP.to_complex(x)
        dom = (DV.ddx(Uc[..., OP.OX_, :], D, m.facx)
               + DV.ddy(Uc[..., OP.OY_, :], D, m.facy)
               + 1j*kz*Uc[..., OP.OZ_, :])
        omag = max(float(np.sqrt(np.mean(sum(np.abs(Uc[..., f, :])**2
                                             for f in (OP.OX_, OP.OY_, OP.OZ_))))), 1e-30)
        d = float(np.sqrt(np.mean(np.abs(dom)**2)))
        print(f'{w7:>8g}{it:>8}{t:>9.1f}{d:>14.3e}{d/omag:>13.3e}', flush=True)
