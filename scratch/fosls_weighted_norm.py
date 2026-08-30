"""F1b-i -- is the nu^-2 blow-up in the OPERATOR, or in my yardstick?

    uv run --quiet python scratch/fosls_weighted_norm.py

F1 measured c2/c1 against an UNWEIGHTED H^1 norm on all four fields and found
c2/c1 ~ nu^-2.  But FOSLS theory does not use an unweighted norm: it proves
equivalence to a WEIGHTED product norm, with the weights chosen to make the
constant uniform in the parameters (Bochev & Gunzburger for Stokes).  So the
nu-dependence may be an artefact of the yardstick rather than a defect in the
operator -- and if so, nothing in the solver needs changing and F1b evaporates.

This costs one generalised eigensolve on the matrix F0 already built.  No PDE
solve, no validation case, no code change.  It should be tried before anything
that alters the functional.

THE CANDIDATE WEIGHTS.  omega enters momentum as nu*om_y and the vorticity
definition as om, so the two rows see the SAME variable a factor nu apart.  Three
yardsticks, chosen to bracket the possibilities:

    unweighted   diag(1, 1, 1, 1)        what F1 used
    nu on omega  diag(1, 1, 1, nu)       measure omega in units of nu
    nu^2         diag(1, 1, 1, nu^2)     if the mismatch is quadratic

If any of these makes c2/c1 nu-INDEPENDENT, the operator was never the problem.
"""
import os
import sys

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R)
sys.path.insert(0, os.path.join(_R, 'scratch'))

import numpy as np
import scipy.linalg as sla

from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState
from lssem2d.mesh import build_channel

import fosls_assemble as FA
import fosls_ellipticity as FE

NV = FA.NV


def constants(nu, N=4, ex=2, ey=2, dt=1e4, wts=(1.0, 1.0, 1.0, 1.0)):
    """c1, c2 for A against a field-weighted H^1 norm diag(wts) x H."""
    m = build_channel(2.0, 1.0, ex, ey, N, bcs=(1, 1, 1, 2))
    m.compute_global_indices()
    st = SolverState(m, diff_matrix(N), nu=nu, dt=dt, fac1=1.0, w_mom=1.0)
    n = N + 1
    fu = np.zeros((m.nelem, n, n)); fv = np.zeros((m.nelem, n, n))
    st.update_linearisation(fu, fv)
    A, free, g = FA.assemble(st, fu, fv, pin_p=True)
    H, _ = FE.assemble_h1(st)

    # field-dependent scaling of the NORM only; the operator is untouched
    ng = H.shape[0]//NV
    s = np.tile(np.asarray(wts, dtype=float), ng)
    Af = np.asarray(A[free][:, free].todense())
    Hf = np.asarray(H[free][:, free].todense())*np.outer(s[free], s[free])
    ev = sla.eigvalsh(Af, Hf)
    return ev[0], ev[-1]


def _sweep():
    NUS = (1/10., 1/100., 1/1000.)
    # FIRST ATTEMPT WAS BACKWARDS.  Weighting omega DOWN (by nu, nu^2, sqrt(nu))
    # made the constant worse -- 7e6x spread against the unweighted 2748x.  That
    # is diagnostic: shrinking H_omega raises lambda_MAX, so omega sits at the
    # STIFF end of the spectrum, not the soft end.  The correction is to weight
    # it UP, which is also what the row structure suggests: omega enters momentum
    # as nu*om_y, so in momentum units omega is naturally an O(1/nu) quantity.
    yardsticks = (
        ('unweighted        ', lambda nu: (1.0, 1.0, 1.0, 1.0)),
        ('1/nu on omega     ', lambda nu: (1.0, 1.0, 1.0, 1.0/nu)),
        ('1/sqrt(nu) on om  ', lambda nu: (1.0, 1.0, 1.0, 1.0/np.sqrt(nu))),
        ('1/nu on omega, p  ', lambda nu: (1.0, 1.0, 1.0/nu, 1.0/nu)),
        ('1/nu om, 1/sq(nu)p', lambda nu: (1.0, 1.0, 1.0/np.sqrt(nu), 1.0/nu)),
    )
    print('F1b-i -- c2/c1 under different H^1 yardsticks (N=4, 2x2, elliptic limit)\n')
    hdr = ' '.join(f'{f"nu={nu:g}":>12}' for nu in NUS)
    print(f'{"yardstick":>19} {hdr}   {"spread":>9}')
    for lbl, w in yardsticks:
        rs = []
        for nu in NUS:
            c1, c2 = constants(nu, wts=w(nu))
            rs.append(c2/c1)
        spread = max(rs)/min(rs)
        cells = ' '.join(f'{r:12.3e}' for r in rs)
        flag = '  <-- nu-INDEPENDENT' if spread < 3.0 else ''
        print(f'{lbl:>19} {cells}   {spread:8.1f}x{flag}')
    print('\n  spread = max/min across two decades of nu.  ~1 means the constant is')
    print('  uniform in nu and the operator was never the problem -- only the norm.')



def invariance_check(nu=1/100.):
    """Is c2/c1 INVARIANT under variable rescaling?  If so, F1b cannot work.

    F1b proposed rescaling omega to remove the nu^-2.  But a variable rescaling
    q = D q~ maps the generalised eigenproblem

        A q = lambda H q     ->     (D A D) q~ = lambda (D H D) q~

    so lambda is UNCHANGED when the norm is transformed consistently.  Rescaling
    the variable is only visible if the norm is left alone -- which is what the
    yardstick sweep above did, and it made things WORSE in both directions.

    If this check confirms the invariance, F1b is refuted as conceived: no change
    of variables can move the FOSLS constant, and only a change to the ROW weights
    (which alters the functional, hence the answer) or to the first-order SYSTEM
    itself can.  Worth one measurement before spending a day on it.
    """
    import scipy.sparse as sp
    m = build_channel(2.0, 1.0, 2, 2, 4, bcs=(1, 1, 1, 2))
    m.compute_global_indices()
    st = SolverState(m, diff_matrix(4), nu=nu, dt=1e4, fac1=1.0, w_mom=1.0)
    fu = np.zeros((m.nelem, 5, 5)); fv = np.zeros((m.nelem, 5, 5))
    st.update_linearisation(fu, fv)
    A, free, g = FA.assemble(st, fu, fv, pin_p=True)
    H, _ = FE.assemble_h1(st)
    Af = np.asarray(A[free][:, free].todense())
    Hf = np.asarray(H[free][:, free].todense())
    ng = H.shape[0]//NV

    print(f'\ninvariance check (nu={nu:g}):')
    base = sla.eigvalsh(Af, Hf)
    print(f'  unscaled                     c2/c1 = {base[-1]/base[0]:.6e}')
    for w in (1/nu, nu, 10.0):
        s = np.tile(np.array([1.0, 1.0, 1.0, w]), ng)[free]
        S = np.outer(s, s)
        ev = sla.eigvalsh(Af*S, Hf*S)          # BOTH transformed
        print(f'  omega scaled by {w:8.3g}     c2/c1 = {ev[-1]/ev[0]:.6e}'
              f'   ratio {(ev[-1]/ev[0])/(base[-1]/base[0]):.3e}')
    print('  ratio == 1 confirms invariance: a change of variables cannot move it.')


if __name__ == '__main__':
    _sweep()
    invariance_check()
