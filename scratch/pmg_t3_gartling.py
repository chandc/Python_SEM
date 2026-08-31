"""T3 -- does the DirectCoarse ordering hold on a case where the soft mode bites?

    python scratch/pmg_t3_gartling.py <preconditioner> <ic>     # one config
    python scratch/pmg_t3_gartling.py collect                   # reduce results

WHY THIS CASE.  T2 confirmed the ordering on Poiseuille -- jacobi 1.26e-06,
cheby 1.55e-07, direct 1.07e-08 -- but the magnitudes were small, and Poiseuille
was chosen as a GATE precisely because it is benign.  The ~8e3x soft direction
precond.py quotes was measured on the CHAN MESH, not on a channel.  PMG_ALGORITHM
sec 7 therefore blocks any default change until Gartling reconfirms it.

Gartling Re = 800 BFS, Chan's 11x4 grid at N = 7 (44 elements, 9048 global DOF),
steady form (w_mass = 0), Dong outlet.  Two measures, not one:

  SPREAD      max pairwise ||U_a - U_b||_inf over initial conditions that
              converge for EVERY preconditioner -- the T2 measure.

              THE LINEAR TOLERANCE HAD TO BE TIGHTENED TO MEASURE THIS.
              dong_gartling.py runs a deliberately loose solve (cg_tol = 1e-6),
              and from a P+Z warm start |dU| plateaus at 1.5e-10 -- the noise
              floor of that solve.  A spread cannot be resolved below the floor
              of the solve that produced it, and T2's direct-coarse spread was
              1.07e-08.  So cg_tol = 1e-10 here, as in T2.  The physics is
              unaffected: a tighter solve cannot move reattachment.

              AND THE STEADY TOLERANCE MUST BE REACHABLE.  The first T3 attempt
              used tol = 1e-11 and ALL TWELVE configurations hit the step cap:
              with cg_tol = 1e-10 and pseudo-time cgsfac = 1e-3, |dU| plateaus
              between 2e-9 and 4e-8 and never reaches 1e-11.  Comparing states at
              a common STEP COUNT rather than a common |dU| conflates "where does
              it end up" with "how far has it got" -- and jacobi was 17x less
              converged than direct at step 300, which would have inflated its
              apparent spread.  tol = 1e-9, cap = 2000.
  REATTACH    lower-wall reattachment against Gartling's benchmark 6.10
              (this repo measures 6.100 with P+Z at N = 7).  A preconditioner
              that changes the PHYSICS is a different and more serious finding
              than one that changes the last few digits.

NOTE ON THE OBC FIX.  The steady form sets obc_D0 = 0, so c_b = nu*D0*fac1/dt is
zero on BOTH levels and the coarse-propagation fix is inert here.  T4 (short BFS,
genuine backflow, D0 != 0) is where that fix should show.

INITIAL CONDITIONS are all physically sensible.  T2's `noisy` (0.3-amplitude
random) drove the outlet into a formulation blow-up -- jacobi and cheby both hit
NaN at exactly step 67 -- and measured nothing about the solver.  Not repeated.
"""
import os
import sys
import time

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
SC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SC)
sys.path.insert(0, ROOT); sys.path.insert(0, SC)
os.chdir(ROOT)

import numpy as np

import lssem2d
lssem2d.set_backend('numpy')
from fgrid import load
from lssem2d import precond as P
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState
import lssem2d.solver as S
from gartling_run import inlet_profile, features, NU

NX, N = 11, 7
GARTLING_REATT = 6.10                      # benchmark; repo gets 6.100 with P+Z
OUT = f'{SC}/pmg_t3_gartling'
PZ = f'{SC}/gartling_steady_nx{NX}_N{N}.npz'
_PCG = S.pcg_solve


def build():
    m, _, _ = load(f'grids/gartling_nx{NX}_N{N}_grid.dat')
    for e in range(m.nelem):
        if m.bc[e, 1] == 4:
            m.bc[e, 1] = 6                 # Dong outlet
    st = SolverState(m, diff_matrix(m.N), nu=NU, dt=1.0, fac1=1.0,
                     w_mom=1.0, w_mass=0.0)
    st.obc_D0 = 0.0
    st.obc_delta = None
    return m, st


def make_ic(kind, m):
    n = m.N + 1
    U = np.zeros((m.nelem, n, n, 4))
    if kind == 'zero':
        return U
    if kind == 'uniform':
        U[..., 0] = 1.0
        return U
    if kind == 'inlet':                    # inlet profile extended downstream
        for e in range(m.nelem):
            U[e, :, :, 0] = inlet_profile(m.ynod[e])[None, :]
        return U
    if kind == 'pzseed':                   # warm start from the P+Z solution
        return np.load(PZ)['U'].copy()
    raise ValueError(kind)


def run_one(pre, ic, cap=2000, tol=1e-9, cg_tol=1e-10):
    m, st = build()
    def wrapped(state, b, fu, fv, M_inv, mw, pin_p=False, max_iter=5000,
                tol=1e-6, cgsfac=0.0, precond=None):
        if pre != 'jacobi' and precond is None:
            kw = dict(pc=2, deg=4)
            kw['coarse_solver'] = 'direct' if pre == 'direct' else 'chebyshev'
            if pre != 'direct':
                kw['coarse_deg'] = 10
            precond = P.make('pmg2', state, fu, fv, M_inv, pin_p, **kw)
        return _PCG(state, b, fu, fv, M_inv, mw, pin_p=pin_p,
                    max_iter=max_iter, tol=tol, cgsfac=cgsfac, precond=precond)
    S.pcg_solve = wrapped
    try:
        U = make_ic(ic, m); h = [U.copy()]
        inl = lambda x, y, t: inlet_profile(y)
        status, d, t0 = 'CAP', np.nan, time.perf_counter()
        for s in range(cap):
            prev = h[0].copy()
            U = S.step_bdf(st, h, time=0.0, max_newton=1, newton_tol=1e-12,
                           newton_factor=0.0, custom_inlet=inl, pin_p=False,
                           cgsfac=1e-3, cg_tol=cg_tol, cg_max_iter=200000,
                           line_search=True)
            if not np.all(np.isfinite(U)):
                status = f'NaN@{s}'; break
            d = float(np.abs(U - prev).max())
            if np.abs(U[..., 0]).max() > 20.0:
                status = f'BLEWUP@{s}'; break
            if s % 25 == 0:
                print(f'      step {s:4d} |dU|={d:.3e} '
                      f'{time.perf_counter()-t0:6.1f}s', flush=True)
            if d < tol:
                status = f'conv@{s}'; break
        wall = time.perf_counter() - t0
    finally:
        S.pcg_solve = _PCG
    lo, us, ur = features(U, m, diff_matrix(m.N))
    return U, status, d, wall, lo, us, ur


def one(pre, ic):
    U, status, d, wall, lo, us, ur = run_one(pre, ic)
    f = f'{OUT}_{pre}_{ic}.npz'
    np.savez_compressed(f, U=U, status=status, dU=d, wall=wall,
                        lo_reatt=lo, up_sep=us, up_reatt=ur, pre=pre, ic=ic)
    print(f'DONE {pre:7s} {ic:8s} {status:10s} |dU|={d:.2e} {wall:7.1f}s '
          f'reatt={lo:.4f} (Gartling {GARTLING_REATT})', flush=True)


def collect():
    import glob
    st, meta = {}, {}
    for f in sorted(glob.glob(f'{OUT}_*.npz')):
        z = np.load(f, allow_pickle=True)
        k = (str(z['pre']), str(z['ic']))
        st[k] = z['U']
        meta[k] = (str(z['status']), float(z['dU']), float(z['wall']),
                   float(z['lo_reatt']), float(z['up_sep']), float(z['up_reatt']))
    pres = ['jacobi', 'cheby', 'direct']
    pres = [p for p in pres if any(k[0] == p for k in st)]
    ics = sorted({k[1] for k in st})
    print(f'\n{"pre":>8} {"ic":>9} {"status":>10} {"|dU|":>10} {"wall":>9} '
          f'{"lo_reatt":>9} {"err vs 6.10":>12}')
    for p_ in pres:
        for i_ in ics:
            if (p_, i_) in meta:
                s_, d_, w_, lo, _, _ = meta[(p_, i_)]
                print(f'{p_:>8} {i_:>9} {s_:>10} {d_:10.2e} {w_:8.1f}s '
                      f'{lo:9.4f} {abs(lo-GARTLING_REATT)/GARTLING_REATT*100:11.3f}%')
    common = [i for i in ics
              if all((p_, i) in meta and 'conv' in meta[(p_, i)][0] for p_ in pres)]
    print(f'\n  ICs converged for every preconditioner: {common}')
    print(f'\n{"preconditioner":>16} {"spread":>12} {"reatt spread":>14} {"mean reatt":>12}')
    for p_ in pres:
        sp = 0.0
        for a in range(len(common)):
            for b in range(a+1, len(common)):
                sp = max(sp, float(np.abs(st[(p_, common[a])]
                                          - st[(p_, common[b])]).max()))
        rs = [meta[(p_, i)][3] for i in common]
        print(f'{p_:>16} {sp:12.3e} {(max(rs)-min(rs)) if rs else 0:14.3e} '
              f'{np.mean(rs) if rs else 0:12.4f}')
    print(f'\n  Gartling benchmark reattachment = {GARTLING_REATT}; '
          f'this repo measures 6.100 with P+Z at N=7.')


if __name__ == '__main__':
    if len(sys.argv) == 3:
        one(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 2 and sys.argv[1] == 'collect':
        collect()
    else:
        print(__doc__)
