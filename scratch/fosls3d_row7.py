"""Does down-weighting R_7 break the FOSLS norm-equivalence?  And how big is the
w7 = 1 near-null cluster?

    uv run --quiet python scratch/fosls3d_row7.py

THE OBJECTION THIS TESTS.  FOSLS carries div(omega) = 0 as a row because it is
what gives H^1 control of omega: ||grad omega|| is bounded by ||curl omega|| +
||div omega||, the momentum rows supply curl omega through the viscous term and
the definition rows supply omega itself, but NOTHING ELSE supplies div omega.
sec 7J calls R_7 "redundant" -- true at the CONTINUOUS level, where it is implied by
omega = curl u -- and down-weights it to 1e-4, so its contribution to J is scaled
by w7^2 = 1e-8 and is effectively gone.  That is contrary to the FOSLS recipe,
and this measures the price.

TWO MEASUREMENTS:

  E1  the ELLIPTICITY CONSTANT c2/c1 from A q = lambda H q, H the H^1 inner
      product, for w7 = 1 and 1e-4.  FOSLS_2D_PLAN sec F1 measured this in 2D --
      which has NO R_7 at all -- so the 3D constant has never been measured
      either way.  If c2/c1 degrades at w7 = 1e-4 the norm-equivalence really is
      being traded for speed.

  E2  the SPECTRUM OF THE JACOBI-PRECONDITIONED operator at w7 = 1, to size the
      near-null cluster sec 7J identified (ranks 0-1 at lambda ~ 8.3e-07, 100% of
      their energy in omega_x, omega_y).  Deflation is worth building only if the
      cluster is DISCRETE -- a handful of modes then a gap.  A continuum means a
      small deflation space buys little, which is what sec 7K.2's "constant slice
      of the slow modes" hints at.

The point of E2: sec 7J's mechanism is that R_7 "loads the JACOBI DIAGONAL with
derivative-squared terms while contributing nothing to A for a divergence-free
vorticity field".  So the cluster lives in D^-1 A, not in A -- R_7 makes the
OPERATOR more elliptic and the PRECONDITIONED operator worse.  If so, the
principled fix is to keep w7 = 1 and deflate, not to down-weight.

Small mesh on purpose: dense generalised eigenproblems, run on the Mac so the
Spark channel run is undisturbed.
"""
import os
import sys

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R)
os.chdir(_R)

import numpy as np
import scipy.linalg as sla

from lssem2d.lgl import diff_matrix
from lssem2d.mesh import build_channel
from lssem3d import backend, bc as BC, operator as OP, solver3d as S3
from lssem3d.deriv import ddx as dUdx, ddy as dUdy

NU, C = 1.0/180.0, 525.0
OUT = os.path.join(_R, 'scratch', 'fosls3d_row7.npz')


def setup(N=4, ex=2, ey=2, kz_val=0.0):
    m = build_channel(2.0*np.pi, 2.0, ex, ey, N, bcs=(0, 0, 1, 1))
    m.compute_global_indices()
    D = diff_matrix(N)
    kz = np.array([kz_val])
    mask = BC.build_mask(m, 1, pin_p=True)
    return m, D, kz, mask


def dense_A(m, D, kz, mask, w7):
    """A on the FREE global DOF, by probing normal_op with a global basis."""
    rw = OP.momentum_row_weights(C, w7=w7)
    shape = (m.nelem, m.N+1, m.N+1, OP.NVAR_R, 1)
    A = lambda x: S3.normal_op(x, D, m.facx, m.facy, kz, NU, C, m, mask,
                               m.wq, 0.0, rw)
    mw = S3.multiplicity_weight(m, shape)
    cols, seen, nodes = [], set(), []
    for e in range(shape[0]):
        for i in range(shape[1]):
            for j in range(shape[2]):
                for f in range(OP.NVAR_R):
                    if mask[e, i, j, f, 0] == 0.0:
                        continue
                    ed = np.zeros(shape); ed[e, i, j, f, 0] = 1.0
                    g = S3.gs(m, ed)
                    key = tuple(np.flatnonzero(np.abs(g.ravel()) > 0.5))
                    if not key or key in seen:
                        continue
                    seen.add(key); cols.append(g); nodes.append((e, i, j))
    B = np.stack([c.ravel() for c in cols], axis=1)
    dense_A.nodes = nodes
    mwf = mw.ravel()
    Ad = np.empty((B.shape[1], B.shape[1]))
    for a in range(B.shape[1]):
        Ad[:, a] = B.T @ (A(B[:, a].reshape(shape)).ravel()*mwf)
    return 0.5*(Ad + Ad.T), B, mwf, shape


def dense_H(m, D, kz, B, mwf, shape):
    """H^1 inner product, block-diagonal over the 14 split-real fields.

    Per field  (f,g)_1 = int (f g + grad_xy f . grad_xy g + kz^2 f g), the
    Fourier-transformed H^1 norm.  Built through the SAME basis B as A so any
    assembly error is shared rather than differing between the two.
    """
    wq = m.wq
    n = m.N + 1
    k2 = float(kz[0])**2

    def Hop(x):
        out = np.empty_like(x)
        for f in range(OP.NVAR_R):
            q = np.ascontiguousarray(x[..., f, 0])
            qx = dUdx(q, D, m.facx); qy = dUdy(q, D, m.facy)
            a = np.einsum('pi,epj->eij', D, qx*wq)*m.facx.reshape(-1, 1, 1)
            b = np.einsum('qj,eiq->eij', D, qy*wq)*m.facy.reshape(-1, 1, 1)
            out[..., f, 0] = (1.0 + k2)*q*wq + a + b
        return S3.gs(m, out)

    Hd = np.empty((B.shape[1], B.shape[1]))
    for a in range(B.shape[1]):
        Hd[:, a] = B.T @ (Hop(B[:, a].reshape(shape)).ravel()*mwf)
    return 0.5*(Hd + Hd.T)


def main(N=4, ex=2, ey=2):
    backend.set_backend('numpy')
    m, D, kz, mask = setup(N, ex, ey)
    print(f'3D FOSLS, N={N}, {ex}x{ey} elements, k_z=0, nu=1/180, c={C:g}\n')

    rows = []
    for w7 in (1.0, 1.0e-2, 1.0e-4):
        A, B, mwf, shape = dense_A(m, D, kz, mask, w7)
        H = dense_H(m, D, kz, B, mwf, shape)
        ev = sla.eigvalsh(A, H)
        c1, c2 = ev[0], ev[-1]
        # Jacobi-preconditioned spectrum, for the cluster
        d = np.diag(A).copy(); d[d <= 0] = 1.0
        Dm = np.diag(1.0/np.sqrt(d))
        evj = np.linalg.eigvalsh(Dm @ A @ Dm)
        evj = evj[evj > 0]
        print(f'w7 = {w7:<8g} dof {A.shape[0]:4d}   c1 = {c1:.4e}  c2 = {c2:.4e}  '
              f'c2/c1 = {c2/c1:.4e}   cond(D^-1 A) = {evj[-1]/evj[0]:.3e}')
        rows.append((w7, c1, c2, c2/c1, evj[-1]/evj[0]))
        if w7 == 1.0:
            spec = evj.copy()

    print(f'\n  E2 -- the w7=1 Jacobi-preconditioned spectrum, smallest 14:')
    print('     ' + '  '.join(f'{v:.3e}' for v in spec[:7]))
    print('     ' + '  '.join(f'{v:.3e}' for v in spec[7:14]))
    g = spec[1:15]/spec[:14]
    k = int(np.argmax(g)) + 1
    print(f'\n  largest gap ratio lambda_(k+1)/lambda_k = {g.max():.2f} at k = {k}')
    print(f'  deflating {k} modes would leave cond {spec[-1]/spec[k]:.3e} '
          f'against {spec[-1]/spec[0]:.3e}  ->  {spec[0]*0+spec[-1]/spec[k]:.0f} vs '
          f'{spec[-1]/spec[0]:.0f}')
    print(f'  i.e. a factor {(spec[-1]/spec[0])/(spec[-1]/spec[k]):.1f} on cond, '
          f'~{np.sqrt((spec[-1]/spec[0])/(spec[-1]/spec[k])):.1f}x on CG iterations')
    np.savez_compressed(OUT, rows=np.array(rows), spec_w7_1=spec, N=N, ex=ex, ey=ey)
    print(f'\nsaved -> {OUT}')


if __name__ == '__main__':
    main()
