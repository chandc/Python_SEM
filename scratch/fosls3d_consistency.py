"""Is the 3D operator implemented consistently with what FOSLS states?

    uv run --quiet python scratch/fosls3d_consistency.py

3D_FORMULATION.md states the operator as

    A = M Q^T Q L0^T (rho W) L0 M

i.e. A is the HESSIAN of  J = sum_r rho_r |R_r|^2 W  over the eight rows R_0..R_7.
That claim has three testable consequences, and FOSLS rests on all three:

  G1  HESSIAN     <v, A u> == <L v, rho W L u>.  If this fails, A is not the
                  normal operator of the stated functional and every FOSLS
                  result -- norm equivalence, the ellipticity constant, J as an
                  error estimator -- is about a different operator.
  G2  SYMMETRY    <v, A u> == <u, A v>.  Follows from G1, but tested separately
                  because a masking or gather-scatter error can break it while
                  leaving G1 intact on unmasked data.
  G3  POSITIVE    <u, A u> > 0 for u != 0 in the free space.

The inner product is the MULTIPLICITY-WEIGHTED one (solver3d._dot with mw).
That is not a detail: the assembled operator is symmetric in that inner product
and not in the naive one, because gather_scatter appears once in A but the
element-local storage holds one copy of a shared node per owning element.

This is the 3D counterpart of FOSLS_2D_PLAN sec F0, which gated the 2D operator
at 2.2e-16 asymmetry before any FOSLS measurement was trusted.
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

from lssem2d.lgl import diff_matrix
from lssem2d.mesh import build_channel
from lssem3d import backend, bc as BC, operator as OP, solver3d as S3

NU, C = 1.0/180.0, 525.0


def setup(N=6, ex=2, ey=2, nk=3, seed=0, rowweight=True):
    m = build_channel(2.0*np.pi, 2.0, ex, ey, N, bcs=(0, 0, 1, 1))
    D = diff_matrix(N)
    kz = np.arange(nk, dtype=float)
    mask = BC.build_mask(m, nk, pin_p=True)
    rw = OP.momentum_row_weights(C) if rowweight else None
    mw = S3.multiplicity_weight(m, mask.shape)
    return m, D, kz, mask, rw, mw


def rand(m, mask, seed):
    rng = np.random.default_rng(seed)
    x = S3.make_continuous(m, rng.standard_normal(mask.shape))
    return x*mask


def main(N=6, ex=2, ey=2, nk=3):
    backend.set_backend('numpy')
    m, D, kz, mask, rw, mw = setup(N, ex, ey, nk)
    u = rand(m, mask, 0)
    v = rand(m, mask, 1)
    A = lambda x: S3.normal_op(x, D, m.facx, m.facy, kz, NU, C, m, mask,
                               m.wq, 0.0, rw)
    L = lambda x: OP.apply_L(x*mask, D, m.facx, m.facy, kz, NU, C, m.wq, 0.0, rw)
    # UNWEIGHTED and UNROW-WEIGHTED: apply_LT carries neither wq nor rw (it has
    # no rw argument at all), so A = L0^T (rho W L0 u) has rho exactly ONCE.
    # Passing rw here too would square it -- which is what an earlier version of
    # this gate did, and it read 3.2e-03 instead of round-off.
    L0 = lambda x: OP.apply_L(x*mask, D, m.facx, m.facy, kz, NU, C, None, 0.0, None)

    print(f'3D FOSLS operator consistency   N={N}, {ex}x{ey} elements, '
          f'nk={nk}, row weights ON\n')
    print(f'  state shape {mask.shape}   free DOF {int(mask.sum())}\n')

    # G1 -- A is the Hessian of J.  <v, A u> must equal <L v, rho W L u>.
    # apply_L already carries wq and rw, so <Lp v, L u> is <L0 v, rho W L0 u>.
    lhs = float(np.sum(S3._dot(v, A(u), mw)))
    rhs = float(np.sum(S3._dot(L0(v), L(u))))
    g1 = abs(lhs - rhs)/max(abs(lhs), 1e-300)
    print(f'  G1 HESSIAN   <v,Au> = {lhs: .10e}')
    print(f'               <Lv,rhoW Lu> = {rhs: .10e}     rel {g1:.3e}   '
          f'{"PASS" if g1 < 1e-12 else "FAIL"}')

    # G2 -- symmetry in the multiplicity-weighted inner product
    a = float(np.sum(S3._dot(v, A(u), mw)))
    b = float(np.sum(S3._dot(u, A(v), mw)))
    g2 = abs(a - b)/max(abs(a), 1e-300)
    print(f'  G2 SYMMETRY  <v,Au> vs <u,Av>                 rel {g2:.3e}   '
          f'{"PASS" if g2 < 1e-12 else "FAIL"}')

    # G3 -- positive definiteness on the free space
    quads = [float(np.sum(S3._dot(x, A(x), mw)))
             for x in (rand(m, mask, s) for s in range(6))]
    g3 = min(quads)
    print(f'  G3 POSITIVE  min <u,Au> over 6 random u = {g3:.6e}   '
          f'{"PASS" if g3 > 0 else "FAIL"}')

    # G1b -- the Hessian gate again with row weights OFF, so the result cannot
    # be an artefact of how rho is threaded through.
    m3, D3, kz3, mask3, rw3, mw3 = setup(N, ex, ey, nk, rowweight=False)
    u3, v3 = rand(m3, mask3, 0), rand(m3, mask3, 1)
    A3 = lambda x: S3.normal_op(x, D3, m3.facx, m3.facy, kz3, NU, C, m3, mask3,
                                m3.wq, 0.0, None)
    l3 = float(np.sum(S3._dot(v3, A3(u3), mw3)))
    r3 = float(np.sum(S3._dot(OP.apply_L(v3*mask3, D3, m3.facx, m3.facy, kz3,
                                         NU, C, None, 0.0, None),
                              OP.apply_L(u3*mask3, D3, m3.facx, m3.facy, kz3,
                                         NU, C, m3.wq, 0.0, None))))
    g1b = abs(l3 - r3)/max(abs(l3), 1e-300)
    print(f'  G1b HESSIAN (row weights OFF)                 rel {g1b:.3e}   '
          f'{"PASS" if g1b < 1e-12 else "FAIL"}')

    # G4 -- the same, with row weights OFF, to show the gates are not an
    # artefact of one weighting choice.
    m2, D2, kz2, mask2, rw2, mw2 = setup(N, ex, ey, nk, rowweight=False)
    u2, v2 = rand(m2, mask2, 0), rand(m2, mask2, 1)
    A2 = lambda x: S3.normal_op(x, D2, m2.facx, m2.facy, kz2, NU, C, m2, mask2,
                                m2.wq, 0.0, rw2)
    a2 = float(np.sum(S3._dot(v2, A2(u2), mw2)))
    b2 = float(np.sum(S3._dot(u2, A2(v2), mw2)))
    g4 = abs(a2 - b2)/max(abs(a2), 1e-300)
    print(f'  G4 SYMMETRY (row weights OFF)                 rel {g4:.3e}   '
          f'{"PASS" if g4 < 1e-12 else "FAIL"}')

    ok = (g1 < 1e-12 and g1b < 1e-12 and g2 < 1e-12 and g3 > 0
          and g4 < 1e-12)
    print(f'\n  {"ALL GATES PASS -- A IS the Hessian of the stated functional"
                if ok else "*** A GATE FAILED ***"}')
    return ok


if __name__ == '__main__':
    main()
