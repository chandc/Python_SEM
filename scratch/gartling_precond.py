"""Preconditioner study on Gartling's backward-facing step (Re = 800).

WHY THIS CASE, beside the cavity and the channel.  It is the third geometry and
it differs from both in ways the preconditioner might care about:

  * an INFLOW/OUTFLOW problem rather than a closed or periodic one.  The 'pz'
    outlet prescribes p and masks omega on the outflow plane, so the patch
    blocks there see a different free-dof pattern from anything in the other two
    cases; the 'free' outlet instead leaves the plane unknown and pins pressure
    at a single corner;
  * a RE-ENTRANT CORNER at the step, the geometric singularity whose effect on
    the time step sec 6.3 measures and whose effect on the SOLVER is measured
    here;
  * NON-UNIFORM elements, and graded variants of the same grid ('g' suffix), so
    the factor-sharing shortcut of the 3D implementation does not apply -- a
    useful boundary on that claim.

WHAT IS MEASURED.  CG iterations per Newton step for Jacobi against the
condensed vertex-patch preconditioner, sweeping polynomial order at fixed mesh
and mesh at fixed order, marched unsteadily from the converged steady field so
that the large-c regime (small dt) is genuinely exercised.

    python scratch/gartling_precond.py --nx 11 --N 6 --dt 0.01 --steps 5
"""
import argparse
import os
import sys
import time

import numpy as np

_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R)
sys.path.insert(0, os.path.join(_R, 'scratch'))
os.chdir(_R)

import lssem2d.solver as S
from lssem2d.lssem import ls_coeffs
import gartling_run as GR
from vertex_schwarz2d import VertexSchwarzCondensed2D, make_coarse
from pmg_ghia_cavity import snapshot


def measure(NX, N, dt, steps, precond, outlet='pz', wmom=1.0, wmass=1.0,
            cgsfac=1e-8, refresh=1000):
    m, D, st, bc2, pin = GR.build(NX, N, steady=False, outlet=outlet,
                                  wmom=wmom, wmass=wmass)
    st.dt = dt
    S.apply_bc = bc2
    n = m.N + 1
    ic = f'{_R}/scratch/gartling_steady_nx{NX}_N{N}.npz'
    U = np.load(ic, allow_pickle=True)['U'].copy()
    h = [U.copy(), U.copy()]                      # BDF2 from a fixed point
    build_t = [0.0]
    if precond == 'patch':
        cache = {'n': 0, 'pre': None}

        def factory(s_, fu, fv, Mi, pp):
            if cache['pre'] is None or cache['n'] % refresh == 0:
                t0 = time.perf_counter()
                snap = snapshot(s_, fu, fv)
                fu_, fv_ = np.ascontiguousarray(fu), np.ascontiguousarray(fv)
                cache['pre'] = VertexSchwarzCondensed2D(
                    snap, fu_, fv_, pin_p=pp, coarse=make_coarse(snap, fu_, fv_, Mi, pp))
                build_t[0] += time.perf_counter() - t0
            cache['n'] += 1
            return cache['pre']
        st.precond_factory = factory
    its, orig = [], S.pcg_solve

    def counted(*a, **k):
        o = orig(*a, **k)
        its.append(int(o[1]))
        return o
    S.pcg_solve = counted
    inl = lambda x, y, t: GR.inlet_profile(y)
    t0 = time.perf_counter()
    try:
        for s in range(steps):
            S.step_bdf(st, h, time=(s + 1)*dt, max_newton=2, newton_tol=1e-12,
                       newton_factor=0.0, custom_inlet=inl, pin_p=pin,
                       cgsfac=cgsfac, cg_tol=1e-14, cg_max_iter=200000)
    finally:
        S.pcg_solve = orig
    wall = time.perf_counter() - t0
    ndof = m.nelem*n*n*4
    return dict(it=float(np.mean(its)), itmax=int(np.max(its)), wall=wall,
                build=build_t[0], ndof=ndof,
                mem=(cache['pre'].bytes/1e6 if precond == 'patch' else 0.0))


def sweep(cases, dt, steps, label, outlet='pz'):
    print(f'\n--- {label}  (dt = {dt:g}, {steps} steps from the steady field, outlet={outlet})')
    print(f'{"grid":>7} {"N":>3} {"ndof":>7} | {"jacobi it":>10} {"max":>7} {"wall":>7} | '
          f'{"patch it":>9} {"max":>5} {"wall":>7} {"build":>7} {"MB":>7} | {"ratio":>7} {"speedup":>8}')
    for NX, N in cases:
        r = {}
        for pc in ('jacobi', 'patch'):
            try:
                r[pc] = measure(NX, N, dt, steps, pc, outlet=outlet)
            except Exception as e:
                print(f'{NX:>7} {N:3d}  {pc} FAILED: {str(e)[:70]}', flush=True)
                r[pc] = None
        if r['jacobi'] and r['patch']:
            j, q = r['jacobi'], r['patch']
            print(f'{NX:>7} {N:3d} {j["ndof"]:7d} | {j["it"]:10.1f} {j["itmax"]:7d} {j["wall"]:6.1f}s | '
                  f'{q["it"]:9.1f} {q["itmax"]:5d} {q["wall"]:6.1f}s {q["build"]:6.1f}s {q["mem"]:7.1f} | '
                  f'{j["it"]/q["it"]:7.0f} {j["wall"]/q["wall"]:8.1f}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweep', action='store_true')
    ap.add_argument('--nx', default='11'); ap.add_argument('--N', type=int, default=6)
    ap.add_argument('--dt', type=float, default=0.01); ap.add_argument('--steps', type=int, default=5)
    ap.add_argument('--outlet', default='pz')
    a = ap.parse_args()
    if a.sweep:
        sweep([('11', 5), ('11', 6), ('11', 7)], 0.01, 3, 'order at fixed mesh (11 x 4 elements)')
        sweep([('11', 6), ('13g', 6), ('18', 6)], 0.01, 3, 'mesh at fixed order (N = 6)')
        for dt in (0.1, 0.01, 0.001):
            sweep([('11', 6)], dt, 2, f'time step: c = fac1/dt = {1.5/dt:g}')
        sweep([('11', 6)], 0.01, 3, 'free outlet (pressure pinned at one corner)', outlet='free')
        raise SystemExit
    print(f'Gartling BFS Re=800, grid nx{a.nx} N={a.N}, outlet={a.outlet}, '
          f'dt={a.dt:g} ({a.steps} steps from the steady field)')
    print(f'{"precond":>8} {"CG/solve":>9} {"max":>7} {"wall":>8} {"build":>8} {"stored MB":>10}')
    for pc in ('jacobi', 'patch'):
        r = measure(a.nx, a.N, a.dt, a.steps, pc, outlet=a.outlet)
        print(f'{pc:>8} {r["it"]:9.1f} {r["itmax"]:7d} {r["wall"]:7.1f}s {r["build"]:7.1f}s {r["mem"]:10.1f}', flush=True)
    print(f'  (ndof = {r["ndof"]})')
