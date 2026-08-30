"""F4' -- the least-squares functional as an a-posteriori error estimator.

    uv run --quiet python scratch/fosls_estimator.py

F1 established c1||e||_1^2 <= J(Q) <= c2||e||_1^2 with c2/c1 bounded and
h-independent.  That makes J -- the quantity already minimised, already computed
by kov.residual() and never used -- a COMPUTABLE two-sided error bound needing
no exact solution.  The project has no a-posteriori estimate at all; accuracy is
measured only where an analytic answer exists.

THE SUCCESS MEASURE, stated before the run.  "J correlates with error" is too
weak -- any monotone function would.  The theory gives a sharper test: the
EFFECTIVITY INDEX

    theta = sqrt(J(Q)) / ||e||_1

must satisfy sqrt(c1) <= theta <= sqrt(c2).  Three gates:

  G1 BOUNDED.        theta lies inside [sqrt(c1), sqrt(c2)] measured in F1.
                     A theory check: if it escapes, the ellipticity constants are
                     wrong or the functional is not what F1 measured.

  G2 ASYMPTOTICALLY CONSTANT.  theta -> const as N grows.  THIS IS THE ONE THAT
                     MATTERS.  An estimator whose effectivity drifts cannot be
                     used to decide anything -- you would not know whether a
                     falling J meant a better solution or a coarser yardstick.
                     Gate: theta varies < 2x over the resolutions where the
                     solution is converging.

  G3 SENSITIVE.      J must RISE when a known defect is injected, roughly in
                     proportion.  This is the minchan_001 test: that run looked
                     healthy on every logged diagnostic while carrying 11%
                     divergence.  An estimator that does not react to the failure
                     mode we actually suffered is of no use, however elegant.

G3 is the practical gate; G1 and G2 are what license it.
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

from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState
from lssem2d.operators import dUdx, dUdy

import kov as K


def gauge_fix(U, E, mk):
    """Remove the pressure gauge before differencing.

    p is determined only up to a CONSTANT (it is pinned at a single node), so the
    raw difference carries an O(1) offset that swamps everything else.  Measured:
    ||e||_1 pinned at 0.993 for every N while the velocity error fell to 1e-10 --
    the constant, not the solution.  kov.py subtracts the mean from both fields
    before differencing; this does the same.
    """
    D = U - E
    D[..., 2] -= (U[..., 2][mk].mean() - E[..., 2][mk].mean())
    return D


def h1_norm(st, E):
    """||E||_1 = sqrt( int (E^2 + |grad E|^2) ), summed over the four fields.

    Element-wise quadrature, NOT multiplicity-weighted: the integral over Omega
    is the SUM of the element integrals, so a shared node legitimately
    contributes to both with each element's own weight.
    """
    wq = st.mesh.wq
    tot = 0.0
    for f in range(4):
        e = np.ascontiguousarray(E[..., f])
        ex = dUdx(e, st.D, st.mesh.facx)
        ey = dUdy(e, st.D, st.mesh.facy)
        tot += float(np.sum((e*e + ex*ex + ey*ey)*wq))
    return np.sqrt(tot)


def solve(nex, ney, N, tol=1e-11, cap=200, cg_tol=1e-12):
    """Steady Kovasznay Newton loop.

    Reimplemented here rather than calling kov.run, which returns only scalars --
    F4' needs the STATE and the FIELD to form ||e||_1 and J.  kov.py is another
    session's validated driver and is not modified; its helpers (build, fields,
    exact, residual) are reused so the setup cannot drift from the validation.
    """
    import lssem2d.solver as S
    m = K.build(nex, ney, N)
    n = N + 1
    st = SolverState(m, diff_matrix(N), nu=K.NU, dt=1.0, fac1=1.0,
                     w_mom=1.0, w_mass=0.0)
    E, X, Y = K.fields(m, N)
    mk = K.uniq_mask(X, Y)
    U = np.zeros((m.nelem, n, n, 4)); U[..., 0] = 1.0
    hist = [U]
    _p = S.pcg_solve

    def pcg(state, b, fu, fv, M, mw, **kw):
        kw['tol'] = cg_tol; kw['max_iter'] = 200000
        return _p(state, b, fu, fv, M, mw, **kw)
    S.pcg_solve = pcg
    try:
        for _ in range(cap):
            Up = hist[0].copy()
            U = S.step_bdf(st, hist, time=0.0, max_newton=1, newton_tol=1e-16,
                           newton_factor=0.0, exact_solution=K.exact, pin_p=True,
                           cgsfac=0.0, cg_max_iter=200000, cg_tol=cg_tol)
            if not np.all(np.isfinite(U)):
                return None
            if float(np.max(np.abs(U - Up))) < tol:
                break
    finally:
        S.pcg_solve = _p
    rms, J = K.residual(st, U)
    return st, U, E, mk, J


def _eps(U, E, mk):
    """rms velocity error over UNIQUE nodes -- kov.py's convention."""
    du = U[..., 0] - E[..., 0]; dv = U[..., 1] - E[..., 1]
    return float(np.sqrt(np.mean(du[mk]**2 + dv[mk]**2)))


if __name__ == '__main__':
    print("F4' -- effectivity of J as an error estimator (Kovasznay, Re=40)\n")

    # --- G1 / G2: is theta bounded, and does it settle?
    print(f'{"N":>4} {"J":>12} {"||e||_1":>11} {"theta":>11} {"eps_rms":>11}')
    rows = []
    for N in (4, 6, 8, 10, 12):
        out = solve(4, 4, N)
        if out is None:
            print(f'{N:4d}   diverged'); continue
        st, U, E, mk, J = out
        e1 = h1_norm(st, gauge_fix(U, E, mk))
        th = np.sqrt(J)/e1
        rows.append((N, J, e1, th, _eps(U, E, mk)))
        print(f'{N:4d} {J:12.4e} {e1:11.4e} {th:11.4f} {_eps(U, E, mk):11.3e}')

    # THE GATE MUST BE READ ONLY WHERE THE ERROR IS STILL FALLING.  Beyond N=8
    # this case is at the round-off floor: J ~ 5e-19 and eps_rms stops improving
    # at 1.1e-10, so BOTH sides of theta = sqrt(J)/||e||_1 are noise and their
    # ratio means nothing.  Taking "the last three N" -- the obvious choice --
    # selects exactly the meaningless points.  Converging = error fell >10x from
    # the previous resolution.
    # Detect on eps_rms -- kov.py's own convergence metric -- not on ||e||_1.
    # At N=10 ||e||_1 still fell 19x while eps_rms moved only 2.3x, so ||e||_1
    # would have admitted a saturated point.  J there is 5.7e-19, i.e. zero.
    conv = [rows[0]] + [b for a, b in zip(rows, rows[1:]) if b[4] < a[4]/10.0]
    if len(conv) > 1:
        th = [r[3] for r in conv]
        ns = ' '.join(f'N={r[0]}' for r in conv)
        print(f'\n  converging regime ({ns}) -- beyond it J and ||e|| are both at')
        print(f'  the round-off floor and theta is a ratio of two noise levels.')
        print(f'\n  G2  theta = {" ".join(f"{t:.4f}" for t in th)}'
              f'   spread {max(th)/min(th):.2f}x   '
              f'{"PASS" if max(th)/min(th) < 2.0 else "FAIL"}   (gate < 2x)')
        lo, hi = np.sqrt(6.4e-5), np.sqrt(1.0)
        ok = all(lo <= t <= hi for t in th)
        print(f'  G1  theta in [sqrt(c1), sqrt(c2)] = [{lo:.4f}, {hi:.4f}]'
              f'   {"PASS" if ok else "FAIL"}')
        print(f'      (c1, c2 from F1 at nu=1e-2, 2x2, N=4; this case is nu=1/40'
              f' on 4x4, so they bracket rather than bound exactly)')

    # --- G3: does J react to the defect that went undetected for 14 hours?
    print('\n  G3  sensitivity to an injected divergence defect')
    st, U, E, mk, J0 = solve(4, 4, 8)
    print(f'{"amp":>8} {"J":>12} {"J/J0":>9} {"rel div":>10} {"eps_rms":>11}')
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(U[..., 0].shape)
    for amp in (0.0, 1e-4, 1e-3, 1e-2, 1e-1):
        Ud = U.copy()
        Ud[..., 0] += amp*noise          # NOT solenoidal -- minchan_001's defect
        _, Jd = K.residual(st, Ud)
        ux = dUdx(np.ascontiguousarray(Ud[..., 0]), st.D, st.mesh.facx)
        vy = dUdy(np.ascontiguousarray(Ud[..., 1]), st.D, st.mesh.facy)
        dv = float(np.sqrt(np.mean((ux + vy)**2)))
        print(f'{amp:8.0e} {Jd:12.4e} {Jd/J0:9.2f} {dv:10.3e} '
              f'{_eps(Ud, E, mk):11.3e}')
    print('\n  G3 asks only that J RISE with the defect.  minchan_001 carried 11%')
    print('  divergence for 900 steps while every logged diagnostic looked healthy.')
