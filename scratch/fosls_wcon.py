"""Does weighting the CONTINUITY row buy anything?  A w_con sweep.

    uv run --quiet python scratch/fosls_wcon.py

WHY THIS IS THE RIGHT LEVER TO TRY.  FOSLS_2D_PLAN sec F1b measured c2/c1 to be
INVARIANT under rescaling omega, because a change of VARIABLES maps A q = lam H q
to (DAD)q~ = lam (DHD)q~ and cannot move the ratio.  It recorded that only a
change to the ROWS -- which alters the answer -- or to the first-order SYSTEM
could.  w_con is a row change, so unlike F1b it should actually move the number.

WHAT IT SHOULD BUY.  Lesson L5: div u is only PENALISED in a least-squares
formulation, never enforced.  minchan_001 ran 14 h carrying relative divergence
of 1.1e-01.  Raising w_con trades the momentum and vorticity residuals for the
divergence one.

TWO MEASURES, because one alone would mislead:
  c2/c1     the FOSLS ellipticity constant (the F1 machinery), which sets the
            preconditioned iteration count via sqrt(c2/c1).
  div/accy  relative divergence and error against the EXACT Poiseuille solution,
            which is exactly representable in this basis.

Already established before running: A stays symmetric to 1e-16 and SPD at every
w_con tested, and cond(A) grows as w_con^2 while lambda_min moves 0.3%.
"""
import os
import sys

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, 'scratch'))
os.chdir(_R)

import numpy as np
import lssem2d
lssem2d.set_backend('numpy')

import fosls_ellipticity as FE
from lssem2d.lgl import diff_matrix, lgl_weights
from lssem2d.lssem import SolverState
from lssem2d.mesh import build_channel
from lssem2d.solver import apply_A, compute_jacobi, pcg_solve
from lssem2d.assembly import gather_scatter

OUT = os.path.join(_R, 'scratch', 'fosls_wcon.npz')
WCONS = (0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0)


def sweep_ellipticity(N=6, ex=3, ey=3):
    print('c2/c1 against w_con  (steady, w_mom=1, nu=1/100)\n')
    print(f'{"w_con":>7} {"c1":>12} {"c2":>10} {"c2/c1":>12} {"sqrt":>9}')
    rows = []
    for w in WCONS:
        c1, c2, _nf = FE.ellipticity(N, ex, ey, dt=1e4, w_mom=1.0, w_con=w)
        print(f'{w:7.2f} {c1:12.4e} {c2:10.4f} {c2/c1:12.4e} {np.sqrt(c2/c1):9.1f}')
        rows.append((w, c1, c2, c2/c1))
    return np.array(rows)


def divergence(N=8, ex=6, ey=2, L=12.0, wcons=WCONS, cap=400):
    """Steady Poiseuille, DRIVEN -- not manufactured.

    An earlier version built b = A(w_con) @ U_exact and solved.  That is
    circular: U = A^-1 A U_exact = U_exact for EVERY w_con, and the divergence
    and error columns came out identical to four decimals across a 250x range of
    w_con.  Lesson L17 -- check that the thing you varied actually varied.

    Here the problem is FIXED (parabolic inlet, free outlet, no-slip walls) and
    only the least-squares weighting changes, so the minimiser genuinely moves.
    """
    import lssem2d.solver as S
    print('\n\nSteady Poiseuille, DRIVEN -- divergence against w_con\n')
    print(f'{"w_con":>7} {"steps":>7} {"status":>9} {"rel div":>11} '
          f'{"err vs exact":>13} {"max|u|":>8}')
    rows = []
    inl = lambda x, y, t: 6.0*np.asarray(y, dtype=float)*(1.0 - np.asarray(y, dtype=float))
    for w in wcons:
        m = build_channel(L_x=L, L_y=1.0, E_x=ex, E_y=ey, N=N, bcs=(3, 4, 1, 1))
        st = SolverState(m, diff_matrix(N), nu=1/100., dt=0.5, fac1=1.0,
                         w_mom=1.0, w_mass=1.0, w_con=w)
        n = N + 1
        Y = np.stack([np.repeat(m.ynod[e][None, :], n, axis=0)
                      for e in range(m.nelem)])
        Ue = np.zeros((m.nelem, n, n, 4))
        Ue[..., 0] = 6.0*Y*(1.0 - Y)
        Ue[..., 3] = -(6.0 - 12.0*Y)

        U = np.zeros((m.nelem, n, n, 4)); h = [U.copy()]
        status, d = 'cap', np.nan
        for k in range(cap):
            prev = h[0].copy()
            U = S.step_bdf(st, h, time=(k+1)*0.5, max_newton=1,
                           newton_tol=1e-13, newton_factor=1e-6,
                           custom_inlet=inl, pin_p=False, cgsfac=1e-8,
                           cg_tol=1e-10, cg_max_iter=200000)
            if not np.all(np.isfinite(U)):
                status = f'NaN@{k}'; break
            d = float(np.abs(U - prev).max())
            if d < 1e-11:
                status = f'conv@{k}'; break
        from lssem2d.operators import dUdx as _dx, dUdy as _dy
        ux = _dx(np.ascontiguousarray(U[..., 0]), st.D, m.facx)
        vy = _dy(np.ascontiguousarray(U[..., 1]), st.D, m.facy)
        wq = m.wq
        div = float(np.sqrt((wq*(ux + vy)**2).sum()))
        nrm = float(np.sqrt((wq*(U[..., 0]**2 + U[..., 1]**2)).sum()))
        # compare only u and omega, which the exact solution pins; p is gauge-free
        err = float(np.abs(U[..., [0, 1, 3]] - Ue[..., [0, 1, 3]]).max()
                    / np.abs(Ue[..., [0, 1, 3]]).max())
        print(f'{w:7.2f} {k:7d} {status:>9} {div/max(nrm,1e-300):11.4e} '
              f'{err:13.4e} {np.abs(U[...,0]).max():8.4f}')
        rows.append((w, k, div/max(nrm, 1e-300), err))
    return np.array(rows)


def main():
    e = sweep_ellipticity()
    d = divergence()
    np.savez_compressed(OUT, ellipticity=e, divergence=d, wcons=np.array(WCONS))
    print(f'\nsaved -> {OUT}')


if __name__ == '__main__':
    main()
