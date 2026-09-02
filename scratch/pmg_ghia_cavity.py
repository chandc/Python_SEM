"""p-multigrid + direct coarse solve on the GHIA Re=1000 lid-driven cavity.

    python scratch/pmg_ghia_cavity.py <precond> <N> [refresh]   # one config
    python scratch/pmg_ghia_cavity.py collect

WHY.  PMG_ALGORITHM sec 6.6 found a halving p-multigrid ladder p-INDEPENDENT to
N = 30 (1.41x over N=5..30) where fixed 2- and 3-level hierarchies degrade 7.90x
and 5.56x.  But sec 6.6 is an OPERATOR study: Stokes, zero linearisation,
manufactured RHS.  Its own scope note says so.

This is the real thing.  Ghia Re = 1000 lid-driven cavity, 1x1, moving lid,
bcs = (wall, wall, wall, lid), dt = 1, pin_p = True -- and a genuine nonlinear
convection term, so fu/fv are nonzero and the preconditioner is REBUILT every
Newton step (the linearisation changes, so it must be).  That rebuild is the
cost sec 6.6 never paid: DirectCoarse assembles and factorises per step.

THREE THINGS ARE MEASURED, because iteration count alone would mislead:
  CG its/step   does p-independence survive convection?
  wall          does it survive paying for the rebuild every step?
  RMS vs Ghia   does the ANSWER stay right?  A preconditioner that changes the
                converged state is not a faster solver, it is a different one.

FREEZING THE FACTORISATION.  sec 6.7 found the ladder needs 43x fewer iterations
than Jacobi at N=16 but only 2.01x less WALL, because DirectCoarse re-assembles
and re-factorises every Newton step -- ~200 factorisations per run.  `refresh`
controls that: 1 rebuilds every step (sec 6.7), k rebuilds every k steps, and a
huge value builds ONCE and freezes.

A frozen preconditioner is built on a SNAPSHOT SolverState, not on the live one.
That matters: apply_A takes fu/fv explicitly but reads dfu_dx/dfv_dy from the
STATE, so holding fu/fv while the state keeps being re-linearised would give a
mismatched operator -- frozen in one half, live in the other.  The snapshot
freezes both.  CG only requires M^-1 fixed WITHIN a solve, so refreshing between
solves is legitimate either way.

Reference: Ghia, Ghia & Shin (1982), Tables I and II, Re = 1000 --
u(y) on x = 0.5 and v(x) on y = 0.5, 17 points each.  The u table is verified
against the repo's stored cavity_re1000_data.npz (max diff 5e-05, which is its
4-dp rounding).
"""
import os
import sys
import time

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
_SC = os.path.dirname(os.path.abspath(__file__))
_R = os.path.dirname(_SC)
sys.path.insert(0, _R); sys.path.insert(0, _SC)
os.chdir(_R)

import numpy as np
import lssem2d
# Backend is selectable.  The NumPy path is the reference; numba is the same
# algorithm compiled -- measured identical iteration counts, ~2x on solve and
# ~8.8x on the DirectCoarse build, which is hundreds of small apply_A probes and
# so exactly where NumPy per-call overhead dominates.
lssem2d.set_backend(os.environ.get('CAV_BACKEND', 'numpy'))

from lssem2d import precond as P, solver as S
from lssem2d.lgl import diff_matrix, lgl_nodes
from lssem2d.lssem import SolverState
from lssem2d.mesh import build_channel

RE = float(os.environ.get('CAV_RE', 1000.0))
DT = float(os.environ.get('CAV_DT', 1.0))
EX = int(os.environ.get('CAV_EX', 4))

# Ghia, Ghia & Shin (1982) Table II -- v on the horizontal centreline y = 0.5.
GHIA_X = np.array([1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594,
                   0.8047, 0.5000, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781,
                   0.0703, 0.0625, 0.0000])
GHIA_V = np.array([0.00000, -0.21388, -0.27669, -0.33714, -0.39188, -0.51550,
                   -0.42665, -0.31966, 0.02526, 0.32235, 0.33075, 0.37095,
                   0.32627, 0.30353, 0.29012, 0.27485, 0.00000])
MAXSTEP = int(os.environ.get('CAV_MAXSTEP', 300))
STEADY = 1.0e-8
OUT = f'{_SC}/pmg_ghia_cavity'
_PCG = S.pcg_solve


def ladder(N):
    seq, p = [], N
    while p > 2:
        p = max(2, p//2)
        seq.append(p)
    return tuple(seq)


def _lag(N, xi):
    xs = lgl_nodes(N)
    L = np.ones(N+1)
    for i in range(N+1):
        for j in range(N+1):
            if i != j:
                L[i] *= (xi - xs[j])/(xs[i] - xs[j])
    return L


def centreline_u(U, m, N):
    """u(y) on the VERTICAL line x = 0.5 -- interpolate in x, sweep y."""
    ys, us = [], []
    for e in range(m.nelem):
        x0, x1 = m.xnod[e, 0], m.xnod[e, -1]
        if not (x0 - 1e-12 <= 0.5 <= x1 + 1e-12):
            continue
        L = _lag(N, 2.0*(0.5 - x0)/(x1 - x0) - 1.0)
        for j in range(N+1):
            ys.append(m.ynod[e, j]); us.append(float(L @ U[e, :, j, 0]))
    o = np.argsort(ys); ys, us = np.array(ys)[o], np.array(us)[o]
    k = np.concatenate(([True], np.diff(ys) > 1e-9))
    return ys[k], us[k]


def centreline_v(U, m, N):
    """v(x) on the HORIZONTAL line y = 0.5 -- interpolate in y, sweep x."""
    xs_, vs = [], []
    for e in range(m.nelem):
        y0, y1 = m.ynod[e, 0], m.ynod[e, -1]
        if not (y0 - 1e-12 <= 0.5 <= y1 + 1e-12):
            continue
        L = _lag(N, 2.0*(0.5 - y0)/(y1 - y0) - 1.0)
        for i in range(N+1):
            xs_.append(m.xnod[e, i]); vs.append(float(L @ U[e, i, :, 1]))
    o = np.argsort(xs_); xs_, vs = np.array(xs_)[o], np.array(vs)[o]
    k = np.concatenate(([True], np.diff(xs_) > 1e-9))
    return xs_[k], vs[k]


def snapshot(st, fu, fv):
    """A SolverState with the linearisation FROZEN, for a frozen preconditioner."""
    s2 = SolverState(st.mesh, st.D, nu=st.nu, dt=st.dt, fac1=st.fac1,
                     w_mom=getattr(st, 'w_mom', None),
                     w_mass=getattr(st, 'w_mass', None),
                     dtau=getattr(st, 'dtau', None),
                     w_con=getattr(st, 'w_con', None))
    s2.update_linearisation(np.ascontiguousarray(fu), np.ascontiguousarray(fv))
    return s2


def run(kind, N, refresh=1):
    m = build_channel(1.0, 1.0, EX, EX, N, bcs=(1, 1, 1, 2))
    m.compute_global_indices()
    st = SolverState(m, diff_matrix(N), nu=1.0/RE, dt=DT, fac1=1.0)
    n = N + 1
    pc = {'p2': 2, 'p3': (4, 2), 'lad': ladder(N),
          'amg': ladder(N)}.get(kind)
    tot, nbuild, tbuild, cache, calls = [0], [0], [0.0], [None], [0]

    def wrapped(state, b, fu, fv, M_inv, mw, pin_p=False, max_iter=5000,
                tol=1e-6, cgsfac=0.0, precond=None):
        if pc is not None and precond is None:
            if cache[0] is None or (calls[0] % refresh) == 0:
                t0 = time.perf_counter()
                snap = snapshot(state, fu, fv)
                cache[0] = P.make('pmg2', snap, np.ascontiguousarray(fu),
                                  np.ascontiguousarray(fv), M_inv, pin_p,
                                  pc=pc, deg=4,
                                  coarse_solver=('amg' if kind == 'amg'
                                                 else 'direct'))
                tbuild[0] += time.perf_counter() - t0
                nbuild[0] += 1
            calls[0] += 1
            precond = cache[0]
        out = _PCG(state, b, fu, fv, M_inv, mw, pin_p=pin_p, max_iter=max_iter,
                   tol=tol, cgsfac=cgsfac, precond=precond)
        if isinstance(out, tuple) and len(out) > 1:
            it = out[1]
            tot[0] += int(it[0] if isinstance(it, (list, tuple)) else it)
        return out

    S.pcg_solve = wrapped
    try:
        U = np.zeros((m.nelem, n, n, 4)); h = [U.copy()]
        t0 = time.perf_counter(); status, d = 'CAP', np.nan
        for s in range(MAXSTEP):
            prev = h[0].copy()
            U = S.step_bdf(st, h, time=0.0, max_newton=1, newton_tol=1e-12,
                           newton_factor=0.0, pin_p=True, cgsfac=0.0,
                           cg_tol=1e-8, cg_max_iter=20000)
            if not np.all(np.isfinite(U)):
                status = f'NaN@{s}'; break
            d = float(np.abs(U - prev).max())
            if s % 25 == 0:
                print(f'    step {s:4d} |dU|={d:.3e} {time.perf_counter()-t0:7.1f}s',
                      flush=True)
            if d < STEADY:
                status = f'conv@{s}'; break
        wall = time.perf_counter() - t0
    finally:
        S.pcg_solve = _PCG

    gh = np.load(f'{_R}/cavity_re1000_data.npz')
    gy, gu = gh['ghia_y'], gh['ghia_u']
    y, u = centreline_u(U, m, N)
    x, v = centreline_v(U, m, N)
    rms_u = float(np.sqrt(np.mean((np.interp(gy, y, u) - gu)**2)))
    o = np.argsort(GHIA_X)
    rms_v = float(np.sqrt(np.mean((np.interp(GHIA_X[o], x, v) - GHIA_V[o])**2)))
    return (U, status, d, wall, tot[0], s+1, rms_u, rms_v,
            (int(m.gidx.max())+1)*4, nbuild[0], tbuild[0])


def one(kind, N, refresh=1):
    refresh = int(refresh)
    (U, status, d, wall, cg, steps, rms_u, rms_v, gdof,
     nbuild, tbuild) = run(kind, int(N), refresh)
    tag = 'frozen' if refresh > 10**6 else f'r{refresh}'
    bk = os.environ.get('CAV_BACKEND', 'numpy')
    f = f'{OUT}_Re{int(RE)}_E{EX}_{bk}_{kind}_N{N}_{tag}.npz'
    np.savez_compressed(f, U=U, status=status, dU=d, wall=wall, cg=cg,
                        steps=steps, rms_u=rms_u, rms_v=rms_v, gdof=gdof,
                        kind=kind, N=int(N), refresh=refresh, tag=tag,
                        nbuild=nbuild, tbuild=tbuild)
    print(f'DONE {kind:4s} N={N:>2} {tag:>7}  {status:10s} steps={steps:4d} '
          f'CG/step={cg/max(steps,1):7.1f}  {wall:7.1f}s  builds={nbuild:4d} '
          f'({tbuild:6.1f}s)  rms_u={rms_u:.4e} rms_v={rms_v:.4e}', flush=True)


def collect():
    import glob
    r = {}
    for f in sorted(glob.glob(f'{OUT}_*.npz')):
        z = np.load(f, allow_pickle=True)
        r[(str(z['kind']), int(z['N']))] = (int(z['steps']), int(z['cg']),
                                            float(z['wall']), float(z['rms']),
                                            int(z['gdof']), str(z['status']))
    kinds = [k for k in ('jac', 'p2', 'p3', 'lad') if any(x[0] == k for x in r)]
    Ns = sorted({x[1] for x in r})
    print(f'\nGhia Re=1000 cavity, {EX}x{EX} elements, dt=1, steady tol {STEADY:g}\n')
    print(f'{"N":>3} {"gDOF":>7} {"kind":>5} {"steps":>6} {"CG/step":>9} '
          f'{"wall_s":>8} {"RMS vs Ghia":>12} {"status":>10}')
    for N in Ns:
        for k in kinds:
            if (k, N) in r:
                st_, cg, w, rms, g, stt = r[(k, N)]
                print(f'{N:3d} {g:7d} {k:>5} {st_:6d} {cg/max(st_,1):9.1f} '
                      f'{w:8.1f} {rms:12.4e} {stt:>10}')
    print('\n  CG/step flat in N  => p-independence survives real convection.')
    print('  RMS must match across preconditioners: a preconditioner that')
    print('  changes the converged state is a different solver, not a faster one.')


if __name__ == '__main__':
    if len(sys.argv) in (3, 4):
        one(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) == 4 else 1)
    elif len(sys.argv) == 2 and sys.argv[1] == 'collect':
        collect()
    else:
        print(__doc__)
