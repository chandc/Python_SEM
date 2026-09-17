"""Vortex shedding past a cylinder at Re = 100 on the curvilinear box mesh.

    uv run python scratch/curvi_cyl_re100.py [--re 100] [--dt 0.1] [--tend 200]

The first case in this project that needs BOTH halves of the curvilinear port at
once: a body the mesh can only represent because elements bend, and a wake that
has to leave the domain through a real outflow condition.  Re = 100 is above the
shedding threshold (Re ~ 47), so this is time-accurate, not a march to steady.

FORCES COME FROM p AND omega DIRECTLY, WITH NO DERIVATIVE AND NO NUMERICAL
NORMAL.  On a no-slip wall the velocity vanishes, so the only surviving velocity
gradient is the wall-normal shear, and the traction collapses to

    sigma . n  =  -p n  +  nu * omega * t,     t = z x n

because omega IS du_t/dn there.  In a velocity-pressure code that shear is a
numerical derivative of the solution; in the VVP least-squares formulation omega
is a SOLVED VARIABLE, so the wall stress is read off the state.  And the body is
a circle, so n = (cos th, sin th) is analytic -- no boundary-normal machinery is
needed for the forces either, which is what keeps plan step 8 off this path.

    C_D = 2 F_x,  C_L = 2 F_y   (rho = 1, U = 1, D = 1)

SHEDDING MUST BE TRIGGERED.  The mesh and the boundary conditions are symmetric
about y = 0 to round-off, so the unstable global mode has to grow from ~1e-16 --
at a growth rate of ~0.12 per time unit that is nearly 300 time units of pure
waiting.  A 5 % perturbation in v, EVEN in y (the base flow's v is odd, so an
even perturbation is exactly the symmetry-breaking component), cuts that to a
few tens of time units and changes nothing about the saturated state.

Published Re = 100: St 0.164-0.167, mean C_D 1.32-1.35, C_L amplitude 0.32-0.34
(Williamson 1989; Liu, Zheng & Sung 1998).
"""
import argparse
import os
import sys
import time as _time

_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R)
sys.path.insert(0, os.path.join(_R, 'scratch'))
os.chdir(_R)

import numpy as np

import lssem2d
from lssem2d import curvi
from lssem2d.lgl import diff_matrix, lgl_weights
from lssem2d.lssem import SolverState
import lssem2d.solver as S
from vertex_schwarz2d import VertexSchwarzCondensed2D, make_coarse
from pmg_ghia_cavity import snapshot

ap = argparse.ArgumentParser()
ap.add_argument('--re', type=float, default=100.0)
ap.add_argument('--dt', type=float, default=0.1)
ap.add_argument('--tend', type=float, default=150.0)
# RELATIVE CG tolerance, not the absolute 1e-9 first used.  Measured on
# this case: absolute 1e-9 needs 96.1 CG per solve, relative 1e-6 needs
# 64.6 -- a 1.5x saving with |u|max identical to four decimals over four
# steps.  An absolute floor stops discriminating once |b| falls, which is
# exactly this regime (CG_TOLERANCE_FLOOR.md).
ap.add_argument('--cgsfac', type=float, default=1e-6)
ap.add_argument('--hours', type=float, default=24.0)
ap.add_argument('--refresh', type=int, default=40)
ap.add_argument('--out', default=os.path.join(_R, 'scratch', '_cyl_re100'))
ap.add_argument('--restart-from', default='',
                help='seed from another run\'s field (for a dt sweep)')
# ARTIFICIAL COMPRESSIBILITY, the documented remedy for refining dt.  The
# continuity row carries weight 1 while momentum carries a_mass = w_mass*fac1/dt,
# so halving dt drives momentum up against a constraint that never moves.
# GARTLING_VALIDATION.md sec 6 measured the consequence over 34 runs with no
# crossover: a_mass <= 6.05 bounded, a_mass >= 12.1 divergent.  This cylinder at
# dt = 0.2 and 0.1 has a_mass = 3.35 and 4.74 and ran; at dt = 0.05 it is 6.71
# and it diverged at t = 157 -- with |u|max still 1.322 while C_L reached -3.47,
# i.e. the pressure went first, exactly as an under-weighted pressure block does.
# dtau_p = 1/a_mass makes the two rows scale together at every dt.
ap.add_argument('--ac', action='store_true',
                help='artificial compressibility, dtau_p = 1/a_mass')
A = ap.parse_args()


def wall_nodes(m, D):
    """(element, node) list on the cylinder, with analytic normal and arc weight."""
    w = lgl_weights(m.N)
    xs = np.matmul(m.X, D.T)
    ys = np.matmul(m.Y, D.T)
    out = []
    for e in range(m.nelem):
        if m.bc[e, 0] != 1:
            continue
        x, y = m.X[e, 0, :], m.Y[e, 0, :]
        r = np.hypot(x, y)
        nx, ny = x/r, y/r                      # outward from the body
        ws = np.hypot(xs[e, 0, :], ys[e, 0, :])*w
        out.append((e, nx, ny, ws))
    return out


def forces(U, wn, nu):
    """C_D, C_L from -p n + nu omega t integrated over the body."""
    fx = fy = 0.0
    for e, nx, ny, ws in wn:
        p, om = U[e, 0, :, 2], U[e, 0, :, 3]
        tx, ty = -ny, nx                       # t = z x n
        fx += float((ws*(-p*nx + nu*om*tx)).sum())
        fy += float((ws*(-p*ny + nu*om*ty)).sum())
    return 2.0*fx, 2.0*fy


def main():
    lssem2d.set_backend('numpy')
    os.makedirs(A.out, exist_ok=True)
    nu = 1.0/A.re
    m = curvi.build_cylinder_box()
    m.compute_global_indices()
    D = diff_matrix(m.N)
    n = m.nterm
    w = np.sqrt(A.dt)
    st = SolverState(m, D, nu=nu, dt=A.dt, fac1=1.0, w_mom=w, w_mass=w)
    if A.ac:
        # AC's reference is the previous SUB-ITERATE, not the previous time
        # level, and solver._drop_pseudo removes kappa_p*p from the residual, so
        # at sub-iteration convergence the term vanishes identically and time
        # accuracy is preserved.  That REQUIRES the sub-iterations to converge --
        # with max_newton = 1 there is nothing to converge and it degenerates
        # into the physical-time compressible form.
        from lssem2d.lssem import ls_coeffs
        # fac1 MUST be the value step_bdf will install, not the one SolverState
        # was built with.  step_bdf sets fac1 = 1.5 for BDF2 (1.0 for BDF1) at
        # the top of every step, so reading it beforehand gives a_mass too small
        # by exactly that factor -- 4.47 instead of 6.71 here, a plausible number
        # computed at the wrong moment.
        st.fac1 = 1.5
        a_mass, a_flux, _ = ls_coeffs(st)
        st.dtau_p = 1.0/a_mass
        print(f'artificial compressibility ON: a_mass = {a_mass:.4f}, '
              f'a_flux = {a_flux:.4f}, dtau_p = 1/a_mass = {st.dtau_p:.4f}',
              flush=True)
    wn = wall_nodes(m, D)
    ndof = (int(m.gidx.max()) + 1)*4

    cache = {'n': 0, 'pre': None}

    def factory(s_, fu, fv, Mi, pp):
        if cache['pre'] is None or cache['n'] % A.refresh == 0:
            snap = snapshot(s_, fu, fv)
            fu_, fv_ = np.ascontiguousarray(fu), np.ascontiguousarray(fv)
            cache['pre'] = VertexSchwarzCondensed2D(
                snap, fu_, fv_, pin_p=pp,
                coarse=make_coarse(snap, fu_, fv_, Mi, pp))
        cache['n'] += 1
        return cache['pre']
    st.precond_factory = factory

    ck = os.path.join(A.out, 'chk_latest.npz')
    if A.restart_from and not os.path.exists(ck):
        # SEED FROM A SATURATED STATE.  A dt sweep does not need to re-grow the
        # instability from a seed -- 50 time units of exponential growth that
        # says nothing about dt.  Restarting on the limit cycle costs a couple of
        # periods of re-settling instead.
        #
        # BOTH BDF LEVELS ARE SET EQUAL, deliberately.  The stored history is
        # spaced at the OLD dt, and step_bdf assumes a uniform step; feeding it a
        # mismatched level makes the first step wrong in a way that is silent.
        # Equal levels are a first-order start whose transient the limit cycle
        # absorbs within a period, and the analysis discards that anyway.
        z = np.load(A.restart_from, allow_pickle=True)
        U = z['U0'].copy()
        h = [U, U.copy()]
        t = float(z['t'])
        hist = []                      # force history starts fresh at the new dt
        print(f'seeded from {A.restart_from} at t = {t:.3f}; '
              f'BDF history reset (dt changed), force history cleared',
              flush=True)
    elif os.path.exists(ck):
        z = np.load(ck)
        h = [z['U0'].copy(), z['U1'].copy()]
        t = float(z['t'])
        hist = [tuple(r) for r in z['hist']]
        print(f'resuming from {ck} at t = {t:.3f}', flush=True)
    else:
        U = np.zeros((m.nelem, n, n, 4))
        U[..., 0] = 1.0
        # symmetry-breaking seed: v perturbation EVEN in y, behind the body
        U[..., 1] += 0.05*np.exp(-((m.X - 1.2)**2 + m.Y**2)/0.6)
        h = [U.copy(), U.copy()]
        t = 0.0
        hist = []

    inlet = lambda x, y, tt: 1.0
    print(f'cylinder Re = {A.re:g}, dt = {A.dt:g}, to t = {A.tend:g}  '
          f'({int(round((A.tend-t)/A.dt))} steps)')
    print(f'{m.nelem} elements, N = {m.N}, {ndof:,} dof; Dong outflow, '
          f'symmetry top/bottom\n')
    t0 = _time.perf_counter()
    nstep = int(round((A.tend - t)/A.dt))
    for i in range(nstep):
        S.step_bdf(st, h, time=t + A.dt, max_newton=(3 if A.ac else 2), newton_tol=1e-11,
                   newton_factor=0.0, custom_inlet=inlet, pin_p=False,
                   cgsfac=A.cgsfac, cg_tol=1e-12, cg_max_iter=4000,
                   line_search=False)
        t += A.dt
        cd, cl = forces(h[0], wn, nu)
        hist.append((t, cd, cl))
        if (i + 1) % 20 == 0:
            el = _time.perf_counter() - t0
            print(f'  t = {t:7.2f}  C_D = {cd:8.4f}  C_L = {cl:+8.4f}  '
                  f'|u|max = {np.abs(h[0][..., 0:2]).max():6.3f}  '
                  f'[{el:.0f}s, {el/(i+1):.2f} s/step]', flush=True)
        if (i + 1) % 100 == 0 or i == nstep - 1:
            np.savez(os.path.join(A.out, 'chk_tmp.npz'), U0=h[0], U1=h[1],
                     t=t, hist=np.array(hist))
            os.replace(os.path.join(A.out, 'chk_tmp.npz'), ck)
        if _time.perf_counter() - t0 > A.hours*3600:
            print(f'\n  wall budget {A.hours} h reached at t = {t:.2f}', flush=True)
            break
    print(f'\nstopped at t = {t:.3f} after {(_time.perf_counter()-t0)/3600:.2f} h')
    np.savez(os.path.join(A.out, 'final.npz'), U0=h[0], U1=h[1], t=t,
             hist=np.array(hist), X=m.X, Y=m.Y, re=A.re, dt=A.dt)
    return 0


if __name__ == '__main__':
    sys.exit(main())
