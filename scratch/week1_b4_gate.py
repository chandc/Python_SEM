"""Go/no-go for b4 (a GenEO spectral coarse space): how many bad modes survive
the patch preconditioner, and does the count grow with the mesh?

    uv run --quiet python scratch/week1_b4_gate.py

WHY THIS COMES FIRST.  3D_STATUS.md sec 7S.3 already killed deflation, and for a
reason that would kill GenEO in the same way: the soft set of the operator is a
CONSTANT FRACTION of the dofs (13% at N = 4, 6, 8), it grows with the mesh, and
deflating a fixed number of modes bought 1.1x.  A GenEO coarse space is exactly
"the modes below a threshold", so if the bad set after the patch preconditioner
is still a constant fraction, GenEO's coarse space is as large as the problem
and b4 dies the same death -- before anyone spends three weeks on it.

But 7S.3 measured the soft set of the JACOBI-preconditioned operator.  The patch
solves already remove the LOCAL kernel modes; what should remain is a small
number of GLOBAL smooth divergence-free modes, and that number is what decides
b4.  So: form M^-1 A densely, take its spectrum, and count.

THE GATE.  Bad-mode count roughly FIXED as h and p refine -> GenEO works, the
coarse space is small, and kappa = O(1) is reachable.  Count growing in
proportion to the dofs -> b4 is deflation again and is dropped.
"""
import os
import sys
import time

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '8')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R)
sys.path.insert(0, os.path.join(_R, 'scratch'))
os.chdir(_R)

import numpy as np
import scipy.linalg as sla

import lssem2d
from lssem2d import solver as S
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState
from lssem2d.mesh import build_channel
from vertex_schwarz2d import VertexSchwarzCondensed2D, make_coarse

RE = 1000.0


def spectrum(N, ex, dt=1.85e-4, coarse_pc=2):
    m = build_channel(1.0, 1.0, ex, ex, N, bcs=(1, 1, 1, 2))
    m.compute_global_indices()
    st = SolverState(m, diff_matrix(N), nu=1.0/RE, dt=dt, fac1=1.0)
    n = N + 1
    fu = np.zeros((m.nelem, n, n)); fv = np.zeros_like(fu)
    st.update_linearisation(fu, fv)
    mult = S.gather_scatter(m, np.ones((m.nelem, n, n, 4)))
    mw = 1.0/np.where(mult < 1e-10, 1.0, mult)
    Mi = S.compute_jacobi(st, fu, fv, pin_p=True)
    mask = st.get_global_mask(pin_p=True)
    co = make_coarse(st, fu, fv, Mi, True, pc=coarse_pc)
    M = VertexSchwarzCondensed2D(st, fu, fv, pin_p=True, coarse=co)

    # free dofs in the redundant element layout, reduced to unique global ones
    g = M.g.reshape(-1)
    free_g = np.flatnonzero(M.free)
    ndof = free_g.size
    loc = -np.ones(M.ndof, np.int64); loc[free_g] = np.arange(ndof)

    def to_local(vg):
        """global vector -> element-redundant field"""
        full = np.zeros(M.ndof); full[free_g] = vg
        return full[M.g]*mask

    def to_global(field):
        return np.bincount(g, weights=(field*mw).ravel(), minlength=M.ndof)[free_g]

    t0 = time.perf_counter()
    P = np.empty((ndof, ndof))
    e = np.zeros(ndof)
    for j in range(ndof):
        e[j] = 1.0
        x = to_local(e)
        P[:, j] = to_global(M(S.apply_A(st, x, fu, fv, pin_p=True)))
        e[j] = 0.0
    tb = time.perf_counter() - t0
    ev = np.sort(np.real(sla.eigvals(P)))
    return ev, ndof, tb


if __name__ == '__main__':
    lssem2d.set_backend('numba')
    print('spectrum of M^-1 A, M = condensed vertex patch + p=2 coarse, '
          'c = 5405 (2D cavity Re=1000)\n')
    print(f'{"mesh":>6} {"N":>3} {"free":>6} | {"lambda_min":>10} {"lambda_max":>10} '
          f'{"kappa":>9} | {"<0.1 max":>9} {"<0.01 max":>9} {"% of dof":>9} | build')
    for ex, N in ((2, 6), (2, 8), (4, 6), (4, 8), (6, 6)):
        ev, nd, tb = spectrum(N, ex)
        lo, hi = ev[0], ev[-1]
        n1 = int((ev < 0.1*hi).sum()); n2 = int((ev < 0.01*hi).sum())
        print(f'{f"{ex}x{ex}":>6} {N:3d} {nd:6d} | {lo:10.3e} {hi:10.3e} {hi/lo:9.1f} '
              f'| {n1:9d} {n2:9d} {100*n2/nd:8.2f}% | {tb:.0f}s', flush=True)
