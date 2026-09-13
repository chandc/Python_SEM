"""What the balanced weighting costs in ellipticity -- the row F1 never had.

    uv run --quiet python scratch/ellipticity_weightings.py

F1 (FOSLS_2D_PLAN.md) measured the FOSLS ellipticity constant c2/c1 of the 2D
VVP functional against the discrete H^1 inner product, and established the thing
the whole theory rests on: in the elliptic limit the constant SATURATES under
h-refinement (1.18e4 -> 1.55e4 from 1x1 to 6x6, step 1.01x), so the steady
functional is H^1-norm-equivalent with an h-independent constant, exactly as
McCormick's FOSLS framework requires.  It swept two weightings, legacy and
w_mom = 1, and found each optimal in its own dt regime.

It could not sweep the third, because the third did not exist yet.  The balanced
weighting w_mom = w_mass = sqrt(dt) is the unique choice that keeps the momentum
equation in the fixed point (PAPER_DRAFT.md sec 3), and the obvious question
about it is whether accuracy has been bought with conditioning: it sits between
two weightings whose constants differ by six orders at production dt.  This
script answers that with the same machinery, so the numbers are comparable to
F1's row for row.

    c1 ||Q||_1^2  <=  F(Q; 0)  <=  c2 ||Q||_1^2 ,   c1 = lambda_min, c2 = lambda_max
    of  A q = lambda H q  with H the assembled discrete H^1 inner product.

sqrt(c2/c1) is the CG iteration-count proxy and is printed beside it: F1 found
it predicts the 3D channel's ~4000 iterations to within 1.5x.
"""
import os
import sys

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '4')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R)
sys.path.insert(0, os.path.join(_R, 'scratch'))
os.chdir(_R)

import numpy as np

from fosls_ellipticity import ellipticity

# The three members of the family, as (label, w_mom, w_mass) given dt.
# legacy: a_flux = dt, a_mass = fac1      (w = None is the code's own default)
# balanced: w_mom = w_mass = sqrt(dt)     -> ma = fac1
# unit:   w_mom = w_mass = 1              -> ma = fac1/dt
WEIGHTINGS = (
    ('legacy', lambda dt: (None, None)),
    ('balanced', lambda dt: (np.sqrt(dt), np.sqrt(dt))),
    ('unit', lambda dt: (1.0, 1.0)),
)


def sweep_dt(N=4, ex=4, ey=4, nu=1/100., dts=(1e-3, 1e-2, 1e-1, 1.0, 1e2, 1e4)):
    print(f'--- c2/c1 vs dt   ({ex}x{ey} elements, N={N}, nu={nu:g}) ---')
    print(f'{"dt":>8} | ' + ' | '.join(f'{n:>21s}' for n, _ in WEIGHTINGS))
    print(f'{"":>8} | ' + ' | '.join(f'{"c2/c1":>11s} {"sqrt":>9s}' for _ in WEIGHTINGS))
    out = {}
    for dt in dts:
        row = [f'{dt:8.0e} |']
        for name, wf in WEIGHTINGS:
            wm, ws = wf(dt)
            c1, c2, nf = ellipticity(N, ex, ey, nu=nu, dt=dt, w_mom=wm, w_mass=ws)
            r = c2/c1
            out[(dt, name)] = r
            row.append(f'{r:11.3e} {np.sqrt(r):9.0f} |')
        print(' '.join(row), flush=True)
    return out


def h_refine(N=4, dt=1e-3, nu=1/100., meshes=((1, 1), (2, 2), (4, 4), (6, 6))):
    """The C3 test at PRODUCTION dt rather than in the elliptic limit: does the
    balanced constant saturate in h the way the elliptic-limit one does?"""
    print(f'\n--- h-refinement at dt = {dt:g} (C3 test; must be FLAT) ---')
    print(f'{"mesh":>8} {"free":>6} | ' + ' | '.join(f'{n:>17s}' for n, _ in WEIGHTINGS))
    prev = {}
    for ex, ey in meshes:
        row = []
        nf_ = 0
        for name, wf in WEIGHTINGS:
            wm, ws = wf(dt)
            c1, c2, nf = ellipticity(N, ex, ey, nu=nu, dt=dt, w_mom=wm, w_mass=ws)
            nf_ = nf
            r = c2/c1
            step = '' if name not in prev else f'{r/prev[name]:5.2f}x'
            prev[name] = r
            row.append(f'{r:11.3e} {step:>5s} |')
        print(f'{f"{ex}x{ey}":>8} {nf_:6d} | ' + ' '.join(row), flush=True)


if __name__ == '__main__':
    r = sweep_dt()
    print('\nreading: the balanced weighting is bounded by the two it sits between;')
    for dt in (1e-3, 1e-2):
        b, l, u = (r[(dt, n)] for n in ('balanced', 'legacy', 'unit'))
        print(f'  dt = {dt:g}:  balanced/legacy = {b/l:8.1f}x   unit/balanced = {u/b:8.1f}x'
              f'   predicted CG ratio balanced/legacy = {np.sqrt(b/l):.1f}x')
    h_refine()
