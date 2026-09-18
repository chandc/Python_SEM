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
                help='artificial compressibility at the measured rule '
                     'kappa_p = a_mass/2')
ap.add_argument('--kfrac', type=float, default=0.5,
                help='kappa_p as a fraction of a_mass (0.5 is the rule)')
ap.add_argument('--N', type=int, default=8, help='polynomial order')
# LATERAL DOMAIN.  Qu et al. (2013) cases D1-D4 make all three global
# quantities linear in the blockage ratio D/H with R^2 >= 0.996, and
# extrapolating that fit to our D/H = 0.05 reproduces our St to 0.03 % -- so the
# 2 % Strouhal offset is blockage, not discretisation (CYLINDER_RE100.md sec 4).
# Testing that means widening the box AND NOTHING ELSE, which is why this goes
# through `nested_lateral_edges`: raising H and ny_side naively would rescale
# every interior lateral division as well, confounding the domain with the
# lateral resolution.  Behr et al. (1995) used nested meshes for this reason.
ap.add_argument('--hfull', type=float, default=20.0,
                help='full lateral extent (default 20, the original box)')
ap.add_argument('--ny-extra', type=int, default=2, dest='ny_extra',
                help='elements appended beyond the reference half-height')
# UPSTREAM EXTENT.  The lateral test (H_full 20 -> 40) moved St by 0.0004
# against a predicted 0.0028, so lateral blockage explains under 20 % of the
# offset from the literature and the box is adequate sideways.  Xu is the last
# domain parameter no sweep has varied: ours is 10, against Posdziech &
# Grundmann's recommended minimum of 20.  An inlet imposing uniform u = 1 that
# close pins the stagnation streamline where it should still be adjusting,
# which raises BOTH St and C_D -- the signature that survives the lateral fix.
ap.add_argument('--lu', type=float, default=10.0,
                help='upstream extent Xu (default 10, the original box)')
ap.add_argument('--nx-extra', type=int, default=2, dest='nx_extra',
                help='elements prepended beyond the reference inlet')
# NEWTON SUB-ITERATIONS.  The default (3 with AC, 2 without) is NOT a
# convergence criterion: newton_factor = 0.0 makes the ratio test unreachable
# and newton_tol = 1e-11 is never met, so the cap is simply what runs.  Measured
# on this case, the two regimes are completely different:
#   AC ON  -- the Jacobian carries +kappa_p that the residual does not (apply_L
#             is also the operator on the increment; _drop_pseudo removes the
#             term from the residual because its reference is the current
#             iterate).  That is a damped fixed point, not Newton: increments
#             fall 4e-1, 1.2e-2, 9.1e-3, then contract by a flat 0.87 per
#             iteration.  Reaching 1e-6 would take ~85 sub-iterations.
#   AC OFF -- the Jacobian is consistent and it is real Newton: 1.1e+1, 3.5e-1,
#             3.3e-4, 1.6e-4, 5.1e-7, converging to 3.5e-10 by the seventh.
# AC also buys a 5x better conditioned linear system (16 PCG iterations against
# 85), which is the trade it exists to make.  At dt = 0.1, a_mass = 4.74 sits
# below GARTLING_VALIDATION.md's bounded threshold of 6.05, so AC is not needed
# for stability here and running without it gives a genuinely CONVERGED
# reference -- which no AC run can provide at any affordable iteration count.
ap.add_argument('--newton', type=int, default=0,
                help='Newton sub-iterations per step (0 = the old default, '
                     '3 with --ac and 2 without)')
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



def _transfer_to_wider(Us, m):
    """Copy a narrow-box field onto the nested wide mesh; free stream elsewhere.

    Matching is on ROUNDED PHYSICAL COORDINATES, the same key
    `compute_global_indices` uses, so a node either matches exactly or is
    genuinely new.  Anything else -- nearest neighbour, interpolation -- would
    hide a mesh that is not actually nested, and a silently non-nested transfer
    is exactly the failure this experiment cannot afford.
    """
    ms = curvi.build_cylinder_box(N=Us.shape[1] - 1)          # the seed's mesh
    if ms.nelem != Us.shape[0]:
        raise ValueError(f'seed has {Us.shape[0]} elements, the default box has '
                         f'{ms.nelem}; the seed is not from the reference mesh')
    key = lambda X, Y: np.round(np.stack([X.ravel(), Y.ravel()], 1), 9)
    src = {tuple(r): i for i, r in enumerate(key(ms.X, ms.Y))}
    Uf = Us.reshape(-1, Us.shape[-1])

    U = np.zeros((m.nelem, m.nterm, m.nterm, 4))
    U[..., 0] = 1.0                                            # free stream
    Ud = U.reshape(-1, 4)
    hit = 0
    for i, r in enumerate(key(m.X, m.Y)):
        j = src.get(tuple(r))
        if j is not None:
            Ud[i] = Uf[j]
            hit += 1
    got = len({tuple(r) for r in key(ms.X, ms.Y)} &
              {tuple(r) for r in key(m.X, m.Y)})
    if got != len(src):
        raise ValueError(f'the meshes are NOT nested: {len(src) - got} of '
                         f'{len(src)} seed nodes have no match in the wide mesh')
    print(f'transferred the seed onto the wider box: {hit:,} of {Ud.shape[0]:,} '
          f'nodes copied exactly, {Ud.shape[0] - hit:,} filled with free stream',
          flush=True)
    return U


def main():
    lssem2d.set_backend('numpy')
    os.makedirs(A.out, exist_ok=True)
    nu = 1.0/A.re
    kw = {}
    if abs(A.hfull - 20.0) > 1e-12:
        kw['ys_side'] = curvi.nested_lateral_edges(A.hfull/2.0, A.ny_extra)
        print(f'lateral domain widened: H_full = {A.hfull:g} '
              f'(+{A.ny_extra} nested elements per side), lateral edges '
              f'{np.round(kw["ys_side"], 4).tolist()}', flush=True)
    if abs(A.lu - 10.0) > 1e-12:
        kw['xs_upstream'] = curvi.nested_upstream_edges(A.lu, A.nx_extra)
        print(f'upstream domain extended: Xu = {A.lu:g} '
              f'(+{A.nx_extra} nested elements), upstream edges '
              f'{np.round(kw["xs_upstream"], 4).tolist()}', flush=True)
    m = curvi.build_cylinder_box(N=A.N, **kw)
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
        # kappa_p = a_mass/2 IS THE MEASURED RULE, and it is a WINDOW, not a
        # floor.  ARTIFICIAL_COMPRESSIBILITY.md sec 4: at a_mass = 60, kappa_p =
        # 30 survived while 15 was too little and 45 and 60 both DIVERGED.  The
        # ls_pseudo_p docstring's 1/a_mass (= kappa_p = a_mass) is the
        # CONDITIONING optimum from the cavity study; stability and conditioning
        # want different values and stability wins.
        kappa_p = A.kfrac*a_mass
        st.dtau_p = 1.0/kappa_p
        print(f'artificial compressibility ON: a_mass = {a_mass:.4f}, '
              f'a_flux = {a_flux:.4f}, kappa_p = {A.kfrac:g}*a_mass = '
              f'{kappa_p:.4f}, dtau_p = {st.dtau_p:.4f}', flush=True)
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
        if U.shape[1] != m.nterm:
            # P-INTERPOLATE THE SEED.  The mesh geometry is identical, only the
            # order differs, so the transfer is the 1D Lagrange interpolation
            # between the two GLL node sets applied on each tensor axis --
            # exactly precond._p_interp.  Re-growing the wake from rest at every
            # N instead would cost 50+ time units per run and say nothing about
            # resolution.
            from lssem2d.precond import _p_interp
            Np = U.shape[1] - 1
            T = _p_interp(Np, m.N)                     # (nterm_new, nterm_old)
            U = np.einsum('ai,eijf,bj->eabf', T, U, T)
            print(f'p-interpolated the seed from N = {Np} to N = {m.N}',
                  flush=True)
        if U.shape[0] != m.nelem:
            # TRANSFER ONTO A WIDER BOX.  The meshes are NESTED by construction
            # (`nested_lateral_edges`), so every node of the narrow mesh exists
            # in the wide one at the same coordinates and the transfer is an
            # exact copy -- no interpolation, no accuracy lost in the wake.  The
            # strips that only the wide mesh has are filled with free stream,
            # which is what the flow is doing out there: at |y| = 10 the
            # disturbance is already down to the level the symmetry condition
            # was pretending it had reached.
            U = _transfer_to_wider(U, m)
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
        S.step_bdf(st, h, time=t + A.dt, max_newton=(A.newton or (3 if A.ac else 2)), newton_tol=1e-11,
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
