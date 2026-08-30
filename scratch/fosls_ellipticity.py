"""F1 -- is the LSSEM functional H^1-norm-equivalent?  The FOSLS ellipticity constant.

    uv run --quiet python scratch/fosls_ellipticity.py

THE ONE NUMBER THE WHOLE THEORY RESTS ON.  FOSLS requires constants
c1 <= c2 with

    c1 ||Q||_1^2  <=  F(Q; 0)  <=  c2 ||Q||_1^2 ,

and ellipticity, AMG's h-independence and the a-posteriori error bound all follow
from c2/c1 being O(1) AND BOUNDED INDEPENDENTLY OF h AND N.  McCormick's summary
states the conclusions; it does not state that the constant depends on the ROW
SCALING, and ours was chosen for time-stepping reasons (sec 5.1), not for norm
equivalence.

So: solve the generalised eigenproblem

    A q = lambda H q ,      H = block-diag over the 4 fields of (K + M)

with K the SEM stiffness and M the mass matrix -- i.e. H is the discrete H^1
inner product.  Then c1 = lambda_min, c2 = lambda_max, and c2/c1 IS the FOSLS
ellipticity constant.

WHAT DISTINGUISHES THE TWO OUTCOMES, and it is the point of the sweep:

  * c2/c1 FLAT in h and N  -> the functional IS norm-equivalent.  H1 refuted;
    C3 should then hold and any multigrid failure is the solver's fault.
  * c2/c1 GROWING          -> the weights break norm equivalence, and NO
    preconditioner can deliver h-independence until the functional is fixed.
    LOR-AMG (F2) also loses its basis, since LOR-SEM equivalence is an H^1 result.

H IS BUILT THE SAME WAY A IS -- by probing element-locally, then assembling
through gidx -- so that any error in the assembly machinery cancels between the
two rather than masquerading as physics.
"""
import os
import sys

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp

from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState
from lssem2d.mesh import build_channel
from lssem2d.operators import dUdx, dUdy

import fosls_assemble as FA

NV = FA.NV


def h1_blocks(state):
    """Element blocks of the H^1 inner product, one scalar operator per field.

    (u, v)_1 = int (u v + grad u . grad v).  Probed element-locally and assembled
    exactly as A is, so the two share any assembly error rather than differing by
    one.
    """
    m = state.mesh
    n = m.N + 1
    nde = n*n*NV
    D = state.D
    wq = m.wq
    blocks = np.zeros((m.nelem, nde, nde))
    U = np.zeros((m.nelem, n, n))
    for i in range(n):
        for j in range(n):
            U[:] = 0.0
            U[:, i, j] = 1.0
            ux = dUdx(U, D, m.facx)
            uy = dUdy(U, D, m.facy)
            # column of (M + K) for a scalar field, in weak form:
            #   (M q)_ab = w_ab q_ab ;  (K q)_ab = Dx^T W Dx q + Dy^T W Dy q
            colM = U*wq
            colK = (dUdx(ux*wq, D, m.facx, transpose=True)
                    if False else _kt(ux, uy, D, m, wq))
            col = (colM + colK).reshape(m.nelem, n*n)
            for v in range(NV):
                blocks[:, v::NV, ((i*n) + j)*NV + v] = col
    return blocks


def _kt(ux, uy, D, m, wq):
    """Dx^T W Dx + Dy^T W Dy applied to one scalar field, via explicit transposes."""
    n = m.N + 1
    a = np.einsum('pi,epj->eij', D, ux*wq)*m.facx.reshape(-1, 1, 1)
    b = np.einsum('qj,eiq->eij', D, uy*wq)*m.facy.reshape(-1, 1, 1)
    return a + b


def assemble_h1(state):
    m = state.mesh
    n = m.N + 1
    blocks = h1_blocks(state)
    gid = m.gidx
    ng = int(gid.max()) + 1
    g = np.empty((m.nelem, n, n, NV), dtype=np.int64)
    for v in range(NV):
        g[..., v] = gid*NV + v
    gf = g.reshape(m.nelem, -1)
    rows = np.repeat(gf[:, :, None], gf.shape[1], axis=2).ravel()
    cols = np.repeat(gf[:, None, :], gf.shape[1], axis=1).ravel()
    return sp.coo_matrix((blocks.ravel(), (rows, cols)),
                         shape=(ng*NV, ng*NV)).tocsr(), g


def ellipticity(N, ex, ey, nu=1/100., dt=1.0, w_mom=None, w_mass=None, pin_p=True):
    m = build_channel(2.0, 1.0, ex, ey, N, bcs=(1, 1, 1, 2))
    m.compute_global_indices()
    st = SolverState(m, diff_matrix(N), nu=nu, dt=dt, fac1=1.0,
                     w_mom=w_mom, w_mass=w_mass)
    n = N + 1
    fu = np.zeros((m.nelem, n, n)); fv = np.zeros((m.nelem, n, n))
    st.update_linearisation(fu, fv)
    A, free, g = FA.assemble(st, fu, fv, pin_p=pin_p)
    H, _ = assemble_h1(st)
    Af = np.asarray(A[free][:, free].todense())
    Hf = np.asarray(H[free][:, free].todense())
    ev = sla.eigvalsh(Af, Hf)
    return ev[0], ev[-1], int(free.sum())


if __name__ == "__main__":
    print("F1 -- FOSLS ellipticity constant  c2/c1  (A q = lambda H q)\n")

    print("--- h-refinement at FIXED N=4 (the C3 test: must be FLAT) ---")
    print(f"{'mesh':>8} {'free':>6} {'c1':>11} {'c2':>11} {'c2/c1':>11} {'step':>7}")
    prev = None
    for ex, ey in ((1, 1), (2, 2), (3, 3), (4, 4), (6, 6), (8, 8)):
        c1, c2, nf = ellipticity(4, ex, ey)
        r = c2/c1
        step = "" if prev is None else f"{r/prev:6.2f}x"
        print(f"{f'{ex}x{ey}':>8} {nf:6d} {c1:11.3e} {c2:11.3e} {r:11.3e} {step:>7}")
        prev = r

    print("\n--- p-refinement at FIXED 2x2 mesh ---")
    print(f"{'N':>8} {'free':>6} {'c1':>11} {'c2':>11} {'c2/c1':>11} {'step':>7}")
    prev = None
    for N in (2, 4, 6, 8, 10):
        c1, c2, nf = ellipticity(N, 2, 2)
        r = c2/c1
        step = "" if prev is None else f"{r/prev:6.2f}x"
        print(f"{N:8d} {nf:6d} {c1:11.3e} {c2:11.3e} {r:11.3e} {step:>7}")
        prev = r

    # At dt = 1 with fac1 = 1 the three weightings COINCIDE (a_mass = a_flux = 1),
    # so a sweep there varies nothing -- the first version of this script did
    # exactly that and reported three identical numbers as if they were a result.
    # dt is what separates them:
    #   legacy   a_mass = 1,     a_flux = dt
    #   w_mom=1  a_mass = 1/dt,  a_flux = 1
    print("\n--- row weighting vs dt (N=4, 2x2).  dt -> inf is the ELLIPTIC limit ---")
    print(f"{'dt':>8} {'legacy c2/c1':>14} {'w_mom=1 c2/c1':>15} {'a_mass':>10}")
    for dt in (0.01, 0.1, 1.0, 10.0, 100.0, 1e4):
        a, b, _ = ellipticity(4, 2, 2, dt=dt)
        c, d, _ = ellipticity(4, 2, 2, dt=dt, w_mom=1.0)
        print(f"{dt:8g} {b/a:14.3e} {d/c:15.3e} {1.0:10g}" if False else
              f"{dt:8g} {b/a:14.3e} {d/c:15.3e} {1.0/dt:10.3g}")


def c3_test():
    """THE C3 TEST: h-refinement in the ELLIPTIC LIMIT, correctly scaled.

    The h-sweep above ran at dt = 1, where legacy and w_mom=1 COINCIDE
    (a_mass = a_flux = 1) -- so it says nothing about the weighting.  FOSLS
    theory is about the steady elliptic system, which is dt -> inf, and there the
    dt sweep showed the two diverge completely: legacy c2/c1 -> 1.9e10 while
    w_mom=1 saturates at 1.4e4.

    So this is the measurement that matters: with the CORRECTLY SCALED functional,
    in the limit the theory addresses, is c2/c1 independent of h?
    """
    print('\n--- C3 TEST: h-refinement, w_mom=1, dt=1e4 (elliptic limit) ---')
    print(f'{"mesh":>8} {"free":>6} {"c1":>11} {"c2":>11} {"c2/c1":>11} {"step":>7}')
    prev = None
    for ex, ey in ((1, 1), (2, 2), (3, 3), (4, 4), (6, 6)):
        c1, c2, nf = ellipticity(4, ex, ey, dt=1e4, w_mom=1.0)
        r = c2/c1
        step = '' if prev is None else f'{r/prev:6.2f}x'
        print(f'{f"{ex}x{ey}":>8} {nf:6d} {c1:11.3e} {c2:11.3e} {r:11.3e} {step:>7}')
        prev = r
    print('\n--- same, legacy weighting (for contrast) ---')
    print(f'{"mesh":>8} {"c2/c1":>11} {"step":>7}')
    prev = None
    for ex, ey in ((1, 1), (2, 2), (3, 3), (4, 4)):
        c1, c2, _ = ellipticity(4, ex, ey, dt=1e4)
        r = c2/c1
        step = '' if prev is None else f'{r/prev:6.2f}x'
        print(f'{f"{ex}x{ey}":>8} {r:11.3e} {step:>7}')
        prev = r




def nu_scaling():
    """Is the residual constant ~1.5e4 a VARIABLE-SCALING artefact?

    c2/c1 saturates in h at ~1.55e4 with w_mom=1 -- bounded, as FOSLS predicts,
    but far above the O(1)-O(100) a well-scaled FOSLS system achieves.  The rows
    suggest why: momentum carries  p_x + nu*om_y  while the vorticity definition
    carries  om + u_y - v_x.  With nu = 0.01 the SAME variable omega enters one
    row 100x weaker than the other, so no single H^1 norm can bound both tightly.

    Classical FOSLS for Stokes avoids this by rescaling the variables (Bochev &
    Gunzburger) -- e.g. working with nu*omega.  If that is the cause here,
    c2/c1 should scale like nu^-2.  If it does not, the constant is structural
    and rescaling will not help.
    """
    print('\n--- nu-dependence of c2/c1 (N=4, 2x2, w_mom=1, dt=1e4) ---')
    print(f'{"nu":>10} {"c1":>11} {"c2":>11} {"c2/c1":>11} {"vs nu^-2":>10}')
    ref = None
    for nu in (1/10., 1/30., 1/100., 1/300., 1/1000.):
        c1, c2, _ = ellipticity(4, 2, 2, nu=nu, dt=1e4, w_mom=1.0)
        r = c2/c1
        pred = '' if ref is None else f'{r/(ref[1]*(ref[0]/nu)**2):9.2f}x'
        if ref is None:
            ref = (nu, r)
        print(f'{nu:10.4g} {c1:11.3e} {c2:11.3e} {r:11.3e} {pred:>10}')
    print('  "vs nu^-2" = measured / (nu^-2 extrapolation from the first row).')
    print('  ~1.0 means the constant IS a variable-scaling artefact.')


if __name__ == "__main__":
    c3_test()
    nu_scaling()
