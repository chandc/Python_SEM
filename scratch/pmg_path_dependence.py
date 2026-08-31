"""Does the coarse solve remove PATH DEPENDENCE of the converged state?

    uv run --quiet python scratch/pmg_path_dependence.py

THIS, NOT ITERATION COUNT, IS WHY PMG2 EXISTS.  precond.py:

    "the VVP outflow pressure lives in a very soft direction of A (measured
     ~8e3x softer than a generic direction).  A diagonal preconditioner
     rescales pointwise and cannot touch such near-null modes, so the solver
     stops with them unresolved and THE CONVERGED STATE BECOMES PATH DEPENDENT."

That is a CORRECTNESS claim and iteration counts cannot test it.  The test is:
drive the same steady problem to convergence from several different initial
conditions and measure the SPREAD of the converged states.  A preconditioner
that resolves the soft mode should collapse the spread; one that does not
should leave each trajectory parked wherever its transient happened to end.

  jacobi          diagonal -- cannot touch the soft mode.  Expected: largest spread.
  pmg2 + cheby10  polynomial on BOTH levels -- damps a band, inverts nothing.
  pmg2 + DIRECT   exact coarse solve -- actually inverts the coarse soft modes.

TEST PROBLEM: plane Poiseuille with the Dong outlet, bcs = (3, 6, 1, 1), the
shape dong_obc_test.py stage0 uses.  Chosen because the exact solution is known
AND zeroes the Dong rows (dong_seeded.py), so a converged state that differs
between runs is unambiguously wrong rather than merely different.

The preconditioner is injected by wrapping pcg_solve, the pattern
scratch/ls_diag2.py already uses: step_bdf takes no precond argument, and the
preconditioner must be REBUILT every Newton step because fu, fv change.
"""
import os
import sys
import time

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R)

import numpy as np

from lssem2d import precond as P, solver as S
from lssem2d.lgl import diff_matrix, lgl_weights
from lssem2d.lssem import SolverState
from lssem2d.mesh import build_channel

NU = 1.0/100.0
OUT = os.path.join(_R, 'scratch', 'pmg_path_dependence.npz')
_PCG = S.pcg_solve


# Same profile dong_obc_test.py uses, and the same (x, y, t) signature
# lssem2d.bc.apply_bc calls custom_inlet with.
def inlet_parabolic(x, y, t):
    y = np.asarray(y, dtype=float)
    return 6.0*y*(1.0 - y)


def make_ic(kind, m, N, rng):
    """Different transients, same boundary data -- so any spread is the solver."""
    U = np.zeros((m.nelem, N+1, N+1, 4))
    if kind == 'zero':
        return U
    yy = m.ynod[:, None, :, None] if m.ynod.ndim == 2 else None
    Y = np.zeros((m.nelem, N+1, N+1))
    for e in range(m.nelem):
        Y[e] = m.ynod[e][None, :]
    if kind == 'uniform':
        U[..., 0] = 1.0
    elif kind == 'parabolic':
        U[..., 0] = 6.0*Y*(1.0 - Y)
    elif kind == 'noisy':
        U[..., 0] = 6.0*Y*(1.0 - Y) + 0.3*rng.standard_normal(Y.shape)
    return U


def run_one(kind_ic, kind_pre, N=8, Ex=6, Ey=2, L=12.0, dt=0.5,
            D0=1.0, delta=0.05, cap=400, tol=1e-11):
    m = build_channel(L_x=L, L_y=1.0, E_x=Ex, E_y=Ey, N=N, bcs=(3, 6, 1, 1))
    st = SolverState(m, diff_matrix(N), nu=NU, dt=dt, fac1=1.0,
                     w_mom=1.0, w_mass=1.0)
    st.obc_D0, st.obc_delta = D0, delta

    def wrapped(state, b, fu, fv, M_inv, mw, pin_p=False, max_iter=5000,
                tol=1e-6, cgsfac=0.0, precond=None):
        if kind_pre != 'jacobi' and precond is None:
            kw = dict(pc=2, deg=4)
            if kind_pre == 'direct':
                kw['coarse_solver'] = 'direct'
            else:
                kw['coarse_deg'] = 10
            precond = P.make('pmg2', state, fu, fv, M_inv, pin_p, **kw)
        return _PCG(state, b, fu, fv, M_inv, mw, pin_p=pin_p,
                    max_iter=max_iter, tol=tol, cgsfac=cgsfac, precond=precond)

    S.pcg_solve = wrapped
    try:
        rng = np.random.default_rng(0)
        U = make_ic(kind_ic, m, N, rng)
        h = [U.copy()]
        inl = inlet_parabolic
        status, d, hist = 'cap', np.nan, []
        t0 = time.perf_counter()
        for s in range(cap):
            prev = h[0].copy()
            U = S.step_bdf(st, h, time=(s+1)*dt, max_newton=1,
                           newton_tol=1e-13, newton_factor=1e-6,
                           custom_inlet=inl, pin_p=False, cgsfac=1e-8,
                           cg_tol=1e-10, cg_max_iter=200000)
            if not np.all(np.isfinite(U)):
                status = f'NaN@{s}'; break
            d = float(np.abs(U - prev).max())
            hist.append(d)
            if s % 25 == 0:
                print(f'        step {s:4d}  |dU|={d:.3e}  '
                      f'{time.perf_counter()-t0:6.1f}s', flush=True)
            if d < tol:
                status = f'conv@{s}'; break
        wall = time.perf_counter() - t0
    finally:
        S.pcg_solve = _PCG
    return U, status, d, wall, np.asarray(hist)


def main():
    ics = ('zero', 'uniform', 'parabolic', 'noisy')
    pres = ('jacobi', 'cheby', 'direct')
    res, states = {}, {}
    for pre in pres:
        print(f'\n--- {pre}', flush=True)
        for ic in ics:
            U, status, d, wall, hist = run_one(ic, pre)
            states[(pre, ic)] = U
            res[(pre, ic)] = (status, d, wall)
            print(f'    ic={ic:10s} {status:10s} |dU|={d:.2e}  {wall:6.1f}s',
                  flush=True)
            # Incremental: a 2h blind run that had to be killed produced NOTHING.
            np.savez_compressed(OUT.replace('.npz', f'_{pre}_{ic}.npz'),
                                U=U, status=status, dU=d, wall=wall, hist=hist)

    print(f'\n{"preconditioner":>16} {"max pairwise spread of converged U":>36}')
    summary = {}
    for pre in pres:
        good = [ic for ic in ics if 'conv' in res[(pre, ic)][0]]
        sp = 0.0
        for a in range(len(good)):
            for b in range(a+1, len(good)):
                sp = max(sp, float(np.abs(states[(pre, good[a])]
                                          - states[(pre, good[b])]).max()))
        summary[pre] = (sp, len(good))
        print(f'{pre:>16} {sp:36.3e}   ({len(good)}/{len(ics)} converged)')

    np.savez_compressed(OUT,
                        pres=list(pres), ics=list(ics),
                        spread=[summary[p][0] for p in pres],
                        nconv=[summary[p][1] for p in pres],
                        **{f'U_{p}_{i}': states[(p, i)] for p in pres for i in ics})
    print(f'\nsaved -> {OUT}')
    print('\n  A LARGE spread means the solver left the soft outflow mode')
    print('  unresolved and each trajectory parked somewhere different.')



def one(pre, ic):
    """Single configuration -- so the 12 can run as 12 independent processes.

    They share nothing, and only `noisy` is slow, so running them serially
    means waiting on the one pathological case.  With OMP_NUM_THREADS=1 each
    process is single-threaded and they pack onto cores cleanly.
    """
    U, status, d, wall, hist = run_one(ic, pre)
    f = OUT.replace('.npz', f'_{pre}_{ic}.npz')
    np.savez_compressed(f, U=U, status=status, dU=d, wall=wall, hist=hist,
                        pre=pre, ic=ic)
    print(f'DONE {pre:8s} {ic:10s} {status:10s} |dU|={d:.2e} {wall:7.1f}s -> {f}',
          flush=True)


def collect():
    """Reduce whatever single-config npz files exist into the spread table."""
    import glob
    st, meta = {}, {}
    for f in sorted(glob.glob(OUT.replace('.npz', '_*.npz'))):
        z = np.load(f, allow_pickle=True)
        k = (str(z['pre']), str(z['ic']))
        st[k], meta[k] = z['U'], (str(z['status']), float(z['dU']), float(z['wall']))
    pres = sorted({k[0] for k in st}); ics = sorted({k[1] for k in st})
    print(f'\n{"pre":>10} {"ic":>10} {"status":>10} {"|dU|":>10} {"wall":>9}')
    for p_ in pres:
        for i_ in ics:
            if (p_, i_) in meta:
                s_, d_, w_ = meta[(p_, i_)]
                print(f'{p_:>10} {i_:>10} {s_:>10} {d_:10.2e} {w_:8.1f}s')
    print(f'\n{"preconditioner":>16} {"converged":>10} {"max pairwise spread":>22}')
    for p_ in pres:
        good = [i for i in ics if (p_, i) in st and 'conv' in meta[(p_, i)][0]]
        sp = 0.0
        for a in range(len(good)):
            for b in range(a+1, len(good)):
                sp = max(sp, float(np.abs(st[(p_, good[a])]
                                          - st[(p_, good[b])]).max()))
        print(f'{p_:>16} {len(good):>4}/{len(ics):<5} {sp:22.3e}')
    print('\n  Large spread = the soft outflow mode was left unresolved and each')
    print('  trajectory parked somewhere different.  THIS is what PMG2 is for.')


if __name__ == '__main__':
    if len(sys.argv) == 3:
        one(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 2 and sys.argv[1] == 'collect':
        collect()
    else:
        main()
