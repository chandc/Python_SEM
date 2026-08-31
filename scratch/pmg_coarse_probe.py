"""Does PMG2's COARSE operator match the fine one once Dong OBC is active?

    uv run --quiet python scratch/pmg_coarse_probe.py

WHY ASK.  precond.py already documents this exact failure mode for the
least-squares weights:

    "Omitting w_mom/w_mass sends ls_coeffs down its LEGACY branch on the coarse
     grid ... so the coarse problem is weighted differently from the fine one
     and the V-cycle returns a correction to the wrong operator ... CG needed
     ~2000 iterations per solve instead of tens."

PMG2.__init__ forwards w_mom, w_mass and dtau for that reason.  It forwards
NOTHING about the Dong OBC.  But apply_A includes the boundary term whenever
obc_active(state) fires, and obc_active is decided by the MESH (bc == 6 edges),
which copy(m) preserves.  So the coarse operator plausibly carries an OBC term
built from DEFAULT parameters -- obc_w = 1.0, obc_D0 = 0.0, obc_delta = None --
while the fine one uses whatever the caller set.

That would be the same bug in a new place, and it must be settled BEFORE any
coarse-solver experiment: a direct solve of the wrong coarse operator is still
a correction to the wrong operator.
"""
import os
import sys

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R)

import numpy as np

from lssem2d import obc, precond as P
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState
from lssem2d.mesh import build_channel

NU = 1.0/100.0


def build(N=8, Ex=6, Ey=2, L=12.0, dt=0.5, D0=1.0, delta=0.05):
    """Poiseuille with a Dong outlet -- dong_obc_test.py stage0 shape."""
    m = build_channel(L_x=L, L_y=1.0, E_x=Ex, E_y=Ey, N=N, bcs=(3, 6, 1, 1))
    st = SolverState(m, diff_matrix(N), nu=NU, dt=dt, fac1=1.0,
                     w_mom=1.0, w_mass=1.0)
    st.obc_D0 = D0
    st.obc_delta = delta
    return m, st


def main():
    m, st = build()
    n = m.N + 1
    fu = np.zeros((m.nelem, n, n)); fv = np.zeros((m.nelem, n, n))
    st.update_linearisation(fu, fv)
    from lssem2d.solver import compute_jacobi
    M_inv = compute_jacobi(st, fu, fv, pin_p=False)

    print(f'fine   N={m.N}  obc_active={obc.obc_active(st)}')
    print(f'       params (w, D0, delta, U0) = {obc._params(st)}')

    pmg = P.make('pmg2', st, fu, fv, M_inv, False, pc=2, deg=4, coarse_deg=10)
    sc = pmg.sc
    print(f'coarse N={sc.mesh.N}  obc_active={obc.obc_active(sc)}')
    print(f'       params (w, D0, delta, U0) = {obc._params(sc)}')

    same = obc._params(st) == obc._params(sc)
    print(f'\nOBC parameters match across levels: {same}')
    if obc.obc_active(st) != obc.obc_active(sc):
        print('*** the two levels disagree on whether OBC is even ACTIVE ***')
    elif not same:
        print('*** COARSE OPERATOR IS A DIFFERENT PROBLEM -- the V-cycle')
        print('    returns a correction to the wrong operator, exactly the')
        print('    failure precond.py documents for w_mom/w_mass. ***')
    return same


if __name__ == '__main__':
    main()
