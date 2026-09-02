"""p-multigrid + direct coarse solve on the GHIA Re=1000 lid-driven cavity.

    python scratch/pmg_ghia_cavity.py <precond> <N>     # one config
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

Reference: Ghia, Ghia & Shin (1982), u(y) on x = 0.5, 17 points.
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
lssem2d.set_backend('numpy')

from lssem2d import precond as P, solver as S
from lssem2d.lgl import diff_matrix, lgl_nodes
from lssem2d.lssem import SolverState
from lssem2d.mesh import build_channel

RE, DT, EX = 1000.0, 1.0, 4
MAXSTEP, STEADY = 300, 1.0e-8
OUT = f'{_SC}/pmg_ghia_cavity'
_PCG = S.pcg_solve


def ladder(N):
    seq, p = [], N
    while p > 2:
        p = max(2, p//2)
        seq.append(p)
    return tuple(seq)


def centreline_u(U, m, N):
    """u(y) on the vertical line x = 0.5, by Lagrange interpolation in x."""
    xs = lgl_nodes(N)
    ys, us = [], []
    for e in range(m.nelem):
        x0, x1 = m.xnod[e, 0], m.xnod[e, -1]
        if not (x0 - 1e-12 <= 0.5 <= x1 + 1e-12):
            continue
        xi = 2.0*(0.5 - x0)/(x1 - x0) - 1.0
        L = np.ones(N+1)
        for i in range(N+1):
            for j in range(N+1):
                if i != j:
                    L[i] *= (xi - xs[j])/(xs[i] - xs[j])
        for j in range(N+1):
            ys.append(m.ynod[e, j]); us.append(float(L @ U[e, :, j, 0]))
    o = np.argsort(ys); ys, us = np.array(ys)[o], np.array(us)[o]
    k = np.concatenate(([True], np.diff(ys) > 1e-9))
    return ys[k], us[k]


def run(kind, N):
    m = build_channel(1.0, 1.0, EX, EX, N, bcs=(1, 1, 1, 2))
    m.compute_global_indices()
    st = SolverState(m, diff_matrix(N), nu=1.0/RE, dt=DT, fac1=1.0)
    n = N + 1
    pc = {'p2': 2, 'p3': (4, 2), 'lad': ladder(N)}.get(kind)
    tot = [0]

    def wrapped(state, b, fu, fv, M_inv, mw, pin_p=False, max_iter=5000,
                tol=1e-6, cgsfac=0.0, precond=None):
        if pc is not None and precond is None:
            # rebuilt EVERY Newton step: fu, fv change, so the coarse operator
            # and its factorisation change with them.
            precond = P.make('pmg2', state, fu, fv, M_inv, pin_p, pc=pc,
                             deg=4, coarse_solver='direct')
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
    rms = float(np.sqrt(np.mean((np.interp(gy, y, u) - gu)**2)))
    return U, status, d, wall, tot[0], s+1, rms, (int(m.gidx.max())+1)*4


def one(kind, N):
    U, status, d, wall, cg, steps, rms, gdof = run(kind, int(N))
    f = f'{OUT}_{kind}_N{N}.npz'
    np.savez_compressed(f, U=U, status=status, dU=d, wall=wall, cg=cg,
                        steps=steps, rms=rms, gdof=gdof, kind=kind, N=int(N))
    print(f'DONE {kind:4s} N={N:>2}  {status:10s} steps={steps:4d} '
          f'CG={cg:7d} ({cg/max(steps,1):7.1f}/step)  {wall:7.1f}s  '
          f'rms={rms:.4e}', flush=True)


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
    if len(sys.argv) == 3:
        one(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 2 and sys.argv[1] == 'collect':
        collect()
    else:
        print(__doc__)
