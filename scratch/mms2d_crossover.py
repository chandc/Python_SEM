"""Test the crossover prediction, rather than merely fitting it.

From the transient MMS (scratch/mms2d_temporal.py, ZIGZAG_CURE_RESEARCH.md 4.11):

    || U_h(T) - U(T) ||  ~  C_2 dt^2 + Phi ||R_h||,
    C_2 = 5.46e-3 (both weightings),  Phi = 0.19 legacy / ~0.05 balanced,

so the two terms cross, and the weighting starts to matter, below

    dt* = sqrt( Phi ||R_h|| / C_2 ).

That formula already accounts for every row of the N = 6, 8, 10, 12 table
a posteriori: dt* = 0.42, 0.046, 3.8e-3, 2.6e-4 against a sweep that ran
7.8e-4 <= dt <= 2.5e-2 -- floor-bound at N = 6 and 8, crossing in view at
N = 10 (the legacy curve departs between 6.3e-3 and 3.1e-3), and nothing
visible at N = 12 because its dt* is a factor of three below the smallest
step run.

THE PREDICTION UNDER TEST.  Continue the N = 12 row past dt* and the
separation must appear, at the sizes the formula names:

    dt         C_2 dt^2    legacy = +3.65e-10   balanced = +9.6e-11   ratio
    3.906e-4   8.33e-10        1.20e-9              9.29e-10          1.29
    1.953e-4   2.08e-10        5.73e-10             3.04e-10          1.88
    9.766e-5   5.21e-11        4.17e-10             1.48e-10          2.82

A NULL RESULT WOULD ALSO BE INFORMATIVE, and there is a specific way this can
fail that is not the theory's fault: at these levels (1e-10) the solver and
round-off floors are near.  The signature to watch for is BOTH weightings
flattening together at the same level -- that is arithmetic, not the
weighting.  Hence cgsfac = 1e-14 here rather than the sweep's 1e-12, and the
run reports both so the two floors can be told apart.

    uv run python scratch/mms2d_crossover.py
"""
import os
import sys

import numpy as np

_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R)
sys.path.insert(0, os.path.join(_R, 'scratch'))
os.chdir(_R)

from mms2d_temporal import march, TEND

C2 = 5.46e-3                      # fitted from the N=12 row, both weightings
RH12 = 1.922e-09                  # ||R_h|| at N = 12, E = 2
PHI = dict(legacy=0.19, balanced=0.05)


def predict(dt, wname, rh=RH12):
    return C2*dt*dt + PHI[wname]*rh


if __name__ == '__main__':
    N, E = 12, 2
    dts = [float(x) for x in os.environ.get('DTS', '3.90625e-4,1.953125e-4,9.765625e-5').split(',')]
    cg = float(os.environ.get('CGSFAC', '1e-14'))
    print(f'N = {N}, {E}x{E} elements, T = {TEND}, CG to {cg:g} relative')
    print(f'||R_h|| = {RH12:.3e};  dt* = sqrt(Phi*||R_h||/C_2) = '
          f'{np.sqrt(PHI["legacy"]*RH12/C2):.2e} (legacy)\n')
    print(f'{"dt":>11} {"steps":>6} | {"legacy":>11} {"predicted":>11} | '
          f'{"balanced":>11} {"predicted":>11} | {"ratio":>6} {"pred":>6}')
    for dt in dts:
        row = [f'{dt:11.4e} {int(round(TEND/dt)):6d} |']
        got = {}
        for wname in ('legacy', 'balanced'):
            e = march(N, E, dt, wname, cgsfac=cg)
            got[wname] = e['uv']
            row.append(f'{e["uv"]:11.4e} {predict(dt, wname):11.4e} |')
        row.append(f'{got["legacy"]/got["balanced"]:6.2f} '
                   f'{predict(dt, "legacy")/predict(dt, "balanced"):6.2f}')
        print(' '.join(row), flush=True)
