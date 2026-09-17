"""The curvilinear cavity against Ghia, Ghia & Shin (1982) at Re = 1000.

    uv run python scratch/curvi_ghia.py

Gates G0-G5 compare the curvilinear code against analytic answers and against
its own affine path.  This compares it against the standard EXPERIMENTAL-grade
benchmark of the field, on a mesh whose elements are bent, and at a Reynolds
number where the flow has structure -- a primary vortex plus two corner eddies --
rather than the Re = 5 of gate G3.

THE EXTRACTION IS THE INTERESTING PART, AND IS WHY THIS IS NOT JUST A RERUN.
Ghia tabulates u along x = 0.5 and v along y = 0.5.  On an affine mesh those are
element interfaces, so the existing comparison scripts read the profile straight
off the tensor-product node lines.  ON A DEFORMED MESH THAT IS FALSE: x = 0.5 is
not a coordinate line, the interfaces through it are curved, and no node sits on
it.  The solution must be evaluated at arbitrary physical points, which means
inverting the element mapping:

    given (x*, y*), find (r, s) with  sum_ij X_ij h_i(r) h_j(s) = x*,  likewise y

by Newton on the 2x2 map Jacobian, then evaluating the field with the same
tensor-product Lagrange basis.  Both meshes are read through this one routine --
the affine mesh included -- so any difference between the two columns is the
solution, not two different extraction methods.

CONFIGURATION follows the repo's established cavity runs (`cavity_steady_ls.py`,
`cavity_ac_pconv.py`): Re = 1000, 6x6 elements, N = 10, driven lid.  Time-accurate
BDF2 marched to a steady state rather than the steady form, because
`cavity_steady_profiles.py` records that w_mass = 0 admits a SPURIOUS fixed point
whose streamlines look right and whose centreline error is 10x worse.  Balanced
weighting w_mom = w_mass = sqrt(dt) throughout.

The deformation again leaves the boundary alone, so both runs solve the same
problem in the same domain with the same lid; only the elements differ.
"""
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
from lssem2d.lgl import diff_matrix, lgl_nodes
from lssem2d.lssem import SolverState
from lssem2d.mesh import build_channel
import lssem2d.solver as S
from vertex_schwarz2d import VertexSchwarzCondensed2D, make_coarse
from pmg_ghia_cavity import snapshot

import argparse

_AP = argparse.ArgumentParser()
_AP.add_argument('--dt', type=float, default=0.25)
_AP.add_argument('--amp', type=float, default=0.10,
                 help='deformation amplitude in units of an element')
_AP.add_argument('--tend', type=float, default=150.0)
_A = _AP.parse_args()

RE, EX, N = 1000.0, 6, 10
AMP = _A.amp
OUT = os.path.join(_R, 'figs',
                   f'curvi_ghia_amp{round(AMP*100):03d}_dt{_A.dt:g}.png')
# dt IS NOT A FREE PARAMETER HERE, and an earlier version of this file claimed it
# was.  At a fixed point the BDF mass and history terms do cancel identically
# (fac1 = sum alpha_m) -- that much is true -- but with balanced weighting
# ls_coeffs returns a_flux = w_mom = sqrt(dt), so what SURVIVES the cancellation
# is sqrt(dt)*N(u) against continuity and vorticity rows of weight 1.  The steady
# functional is therefore
#
#     J = int[ dt*|N(u)|^2 + (div u)^2 + (om + u_y - v_x)^2 ]
#
# which depends on dt explicitly, and so does its minimiser.  Measured on this
# case, both runs converged to a profile drift below 1e-5:
#
#     dt = 0.25  (w_mom = 0.500)   RMS u 1.656e-2, v 1.840e-2
#     dt = 0.10  (w_mom = 0.316)   RMS u 1.475e-2, v 1.630e-2
#
# 11 % apart.  A smaller dt weights momentum LESS against the constraints, so
# div u and the vorticity definition are satisfied more tightly and the answer
# moves toward Ghia.  Any comparison between meshes must therefore hold dt fixed,
# which is why the affine control is re-run at whatever --dt is given rather than
# reused from another run.
DT = _A.dt
MAXSTEP = int(round(_A.tend/DT))
STEADY_TOL = 1e-9
# Checkpoint the profile every 25 time units, whatever dt is, so the drift
# columns are comparable between runs at different step sizes.
CHECK = max(1, int(round(25.0/DT)))
# The preconditioner is rebuilt every REFRESH factory CALLS (one per Newton
# sub-iteration), and its build dominates the step cost here.  Staleness changes
# the iteration count, never the answer -- the linearisation it is built at is
# not the one the answer is defined by.
REFRESH = 60

GH = np.load(os.path.join(_R, 'cavity_re1000_data.npz'))
GHIA_U, GHIA_Y = GH['ghia_u'], GH['ghia_y']
GHIA_XV = np.array([1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594,
                    0.8047, 0.5000, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781,
                    0.0703, 0.0625, 0.0000])
GHIA_V = np.array([0.0000, -0.21388, -0.27669, -0.33714, -0.39188, -0.51550,
                   -0.42665, -0.31966, 0.02526, 0.32235, 0.33075, 0.37095,
                   0.32627, 0.30353, 0.29012, 0.27485, 0.0000])


# --------------------------------------------------- arbitrary-point evaluation

def _lag(nodes, r):
    n = len(nodes)
    h = np.ones(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                h[i] *= (r - nodes[j])/(nodes[i] - nodes[j])
    return h


def _dlag(nodes, r):
    n = len(nodes)
    d = np.zeros(n)
    for i in range(n):
        tot = 0.0
        for k in range(n):
            if k == i:
                continue
            term = 1.0/(nodes[i] - nodes[k])
            for j in range(n):
                if j != i and j != k:
                    term *= (r - nodes[j])/(nodes[i] - nodes[j])
            tot += term
        d[i] = tot
    return d


def coords(m):
    if getattr(m, 'curvilinear', False):
        return m.X, m.Y
    X = np.repeat(m.xnod[:, :, None], m.nterm, axis=2)
    Y = np.repeat(m.ynod[:, None, :], m.nterm, axis=1)
    return X, Y


def invert_map(Xe, Ye, xi, xt, yt, tol=1e-13):
    """Reference coordinates (r, s) of the physical point (xt, yt) in one element.

    Newton on the 2x2 map Jacobian.  Returns None if it leaves the element or
    fails to converge, which is how the caller identifies the owning element.
    """
    r = s = 0.0
    for _ in range(60):
        hr, hs = _lag(xi, r), _lag(xi, s)
        dr, ds = _dlag(xi, r), _dlag(xi, s)
        x = hr @ Xe @ hs
        y = hr @ Ye @ hs
        fx, fy = xt - x, yt - y
        if abs(fx) < tol and abs(fy) < tol:
            break
        J = np.array([[dr @ Xe @ hs, hr @ Xe @ ds],
                      [dr @ Ye @ hs, hr @ Ye @ ds]])
        if abs(np.linalg.det(J)) < 1e-300:
            return None
        d = np.linalg.solve(J, [fx, fy])
        r = float(np.clip(r + d[0], -1.6, 1.6))
        s = float(np.clip(s + d[1], -1.6, 1.6))
    else:
        return None
    if abs(r) > 1 + 1e-7 or abs(s) > 1 + 1e-7:
        return None
    return r, s


def eval_field(m, U, pts, comp):
    """U[..., comp] at arbitrary physical points, on affine or curved meshes."""
    xi = lgl_nodes(m.N)
    X, Y = coords(m)
    lo = np.stack([X.min(axis=(1, 2)), Y.min(axis=(1, 2))], 1)
    hi = np.stack([X.max(axis=(1, 2)), Y.max(axis=(1, 2))], 1)
    out = np.full(len(pts), np.nan)
    for k, (xt, yt) in enumerate(pts):
        pad = 1e-9
        cand = np.where((xt >= lo[:, 0] - pad) & (xt <= hi[:, 0] + pad) &
                        (yt >= lo[:, 1] - pad) & (yt <= hi[:, 1] + pad))[0]
        for e in cand:
            rs = invert_map(X[e], Y[e], xi, xt, yt)
            if rs is None:
                continue
            hr, hs = _lag(xi, rs[0]), _lag(xi, rs[1])
            out[k] = hr @ U[e, :, :, comp] @ hs
            break
    return out


# ------------------------------------------------------------------- the solve

def run(kind, amp=None, verbose=True):
    m = build_channel(1.0, 1.0, EX, EX, N, bcs=(1, 1, 1, 2))
    m.compute_global_indices()
    amp = AMP if amp is None else amp
    if kind == 'deformed':
        # kx = 1, NOT 2.  The deformation carries sin(kx*pi*x), so kx = 2
        # vanishes at x = 0.5 -- Ghia's vertical centreline would have been a
        # NODAL LINE of the deformation, with all 132 of its nodes sitting
        # exactly where the affine mesh put them, and the u(y) comparison
        # would have been extracted from an undeformed line.  kx = 1 vanishes
        # only on the domain boundary and is MAXIMAL at x = 0.5.
        curvi.deform(m, amp=amp, kx=1, ky=1)
    w = np.sqrt(DT)                       # balanced weighting
    st = SolverState(m, diff_matrix(N), nu=1.0/RE, dt=DT, fac1=1.0,
                     w_mom=w, w_mass=w)
    n = N + 1
    cache = {'n': 0, 'pre': None}

    def factory(s_, fu, fv, Mi, pp):
        if cache['pre'] is None or cache['n'] % REFRESH == 0:
            snap = snapshot(s_, fu, fv)
            fu_, fv_ = np.ascontiguousarray(fu), np.ascontiguousarray(fv)
            cache['pre'] = VertexSchwarzCondensed2D(
                snap, fu_, fv_, pin_p=pp,
                coarse=make_coarse(snap, fu_, fv_, Mi, pp))
        cache['n'] += 1
        return cache['pre']
    st.precond_factory = factory

    U = np.zeros((m.nelem, n, n, 4))
    h = [U.copy(), U.copy()]
    t0 = _time.perf_counter()
    step = 0
    # |dU| PER STEP IS NOT THE CONVERGENCE CRITERION, and treating it as one is
    # how the first version of this run stopped at its step cap and still called
    # itself steady.  A small per-step change says nothing on its own: if the
    # remaining decay is geometric with ratio r per step, the change STILL TO
    # COME is |dU|*r/(1-r), which at r = 0.983 is 58 times the per-step value.
    # What the comparison actually depends on is the extracted PROFILE, so that
    # is what is watched: the centreline is re-extracted every CHECK steps and
    # compared with the previous checkpoint.  The run is converged when the
    # profile has stopped moving, not when |dU| looks small.
    drift = []
    prof_prev = None
    for step in range(1, MAXSTEP + 1):
        prev = h[0].copy()
        S.step_bdf(st, h, time=step*DT, max_newton=2, newton_tol=1e-12,
                   newton_factor=0.0, pin_p=True, cgsfac=0.0, cg_tol=1e-11,
                   cg_max_iter=4000, line_search=False)
        d = float(np.abs(h[0] - prev).max())
        if step % CHECK == 0:
            pu = eval_field(m, h[0], [(0.5, y) for y in GHIA_Y], 0)
            pv = eval_field(m, h[0], [(x, 0.5) for x in GHIA_XV], 1)
            pr = np.concatenate([pu, pv])
            dp = np.nan if prof_prev is None else float(np.nanmax(np.abs(pr - prof_prev)))
            drift.append((step, d, dp))
            prof_prev = pr
            if verbose:
                print(f'   {kind:9s} step {step:4d}  t = {step*DT:6.2f}  '
                      f'|dU| = {d:.3e}   profile drift since step '
                      f'{step-CHECK:4d} = {dp:.3e}  '
                      f'({_time.perf_counter()-t0:.0f} s)', flush=True)
        if d < STEADY_TOL:
            break
    if verbose:
        # Say which of the two exits was taken.  "steady" when the run merely ran
        # out of steps is the kind of wording that later gets quoted as if a
        # tolerance had been met.
        why = ('|dU| < tol' if d < STEADY_TOL else f'step cap {MAXSTEP} reached')
        print(f'   {kind:9s} stopped at step {step} (t = {step*DT:.2f}) -- {why}, '
              f'|dU| = {d:.2e}, {_time.perf_counter()-t0:.0f} s', flush=True)
    return m, h[0], step, drift


def profiles(m, U, ny=201):
    yq = np.linspace(0.0, 1.0, ny)
    xq = np.linspace(0.0, 1.0, ny)
    u_c = eval_field(m, U, [(0.5, y) for y in yq], 0)
    v_c = eval_field(m, U, [(x, 0.5) for x in xq], 1)
    u_g = eval_field(m, U, [(0.5, y) for y in GHIA_Y], 0)
    v_g = eval_field(m, U, [(x, 0.5) for x in GHIA_XV], 1)
    return yq, u_c, xq, v_c, u_g, v_g


def main():
    lssem2d.set_backend('numpy')
    print(f'Cavity Re = {RE:.0f}, {EX}x{EX} elements, N = {N} '
          f'({EX*N+1}x{EX*N+1} nodes)')
    print(f'dt = {DT}, balanced weighting, march to t = {MAXSTEP*DT:g} '
          f'({MAXSTEP} steps), profile checked every {CHECK} steps')
    mdef = build_channel(1.0, 1.0, EX, EX, N, bcs=(1, 1, 1, 2))
    mdef.compute_global_indices()
    curvi.deform(mdef, amp=AMP, kx=1, ky=1)
    print(f'deformation amp = {AMP*100:.0f} %: Jacobian {mdef.jacq.min():.5f} to '
          f'{mdef.jacq.max():.5f} (max/min {mdef.jacq.max()/mdef.jacq.min():.3f}), '
          f'min J > 0 so the map does not fold\n')
    res = {}
    for kind in ('affine', 'deformed'):
        m, U, step, drift = run(kind)
        yq, uc, xq, vc, ug, vg = profiles(m, U)
        ru = float(np.sqrt(np.nanmean((ug - GHIA_U)**2)))
        rv = float(np.sqrt(np.nanmean((vg - GHIA_V)**2)))
        res[kind] = dict(m=m, U=U, yq=yq, uc=uc, xq=xq, vc=vc,
                         ug=ug, vg=vg, ru=ru, rv=rv, step=step, drift=drift)
        print(f'   {kind:9s} RMS vs Ghia:  u(y) {ru:.4e}   v(x) {rv:.4e}')
        # WRITE AS SOON AS EACH MESH FINISHES, not after both.  The second run
        # here takes ~40 minutes, and holding the first one's profiles in memory
        # until then means a crash -- or simply wanting to look at them -- loses
        # work that is already complete.
        np.savez(os.path.join(_R, 'scratch',
                              f'curvi_ghia_{kind}_amp{round(AMP*100):03d}'
                              f'_dt{DT:g}.npz'),
                 yq=yq, uc=uc, xq=xq, vc=vc, ug=ug, vg=vg, ru=ru, rv=rv,
                 ghia_u=GHIA_U, ghia_y=GHIA_Y, ghia_xv=GHIA_XV, ghia_v=GHIA_V,
                 U=U, X=coords(m)[0], Y=coords(m)[1], t=step*DT)
        fin = [x for x in drift if not np.isnan(x[2])]
        if fin:
            print(f'   {kind:9s} profile drift over the LAST {CHECK} steps: '
                  f'{fin[-1][2]:.2e}  '
                  f'({100*fin[-1][2]/max(ru, 1e-300):.4f} % of the RMS vs Ghia)')

    a, d = res['affine'], res['deformed']
    print(f'\nprofile difference between the two meshes (max over the centreline):')
    print(f'   u : {np.nanmax(np.abs(a["uc"] - d["uc"])):.3e}')
    print(f'   v : {np.nanmax(np.abs(a["vc"] - d["vc"])):.3e}')
    print(f'\nGhia extrema        u_min {GHIA_U.min():+.4f}   '
          f'v_min {GHIA_V.min():+.4f}   v_max {GHIA_V.max():+.4f}')
    for k in ('affine', 'deformed'):
        r = res[k]
        print(f'{k:12s} extrema  u_min {np.nanmin(r["uc"]):+.4f}   '
              f'v_min {np.nanmin(r["vc"]):+.4f}   v_max {np.nanmax(r["vc"]):+.4f}')
    figure(res)
    np.savez(os.path.join(_R, 'scratch',
                          f'curvi_ghia_amp{round(AMP*100):03d}_dt{DT:g}.npz'),
             **{f'{k}_{f}': res[k][f] for k in res
                for f in ('yq', 'uc', 'xq', 'vc', 'ug', 'vg')})
    return 0


def figure(res):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.2))

    m = res['deformed']['m']
    n = m.nterm
    for el in range(m.nelem):
        for idx in (0, n-1):
            ax[0].plot(m.X[el, idx, :], m.Y[el, idx, :], 'k-', lw=0.9)
            ax[0].plot(m.X[el, :, idx], m.Y[el, :, idx], 'k-', lw=0.9)
    ax[0].axvline(0.5, color='C3', ls='--', lw=1.6)
    ax[0].axhline(0.5, color='C0', ls='--', lw=1.6)
    ax[0].set_aspect('equal'); ax[0].tick_params(labelsize=8)
    ax[0].set_title(f'the mesh, deformed {AMP*100:.0f} %\n'
                    f'{EX}x{EX} elements, $N = {N}$', fontsize=10)

    st = {'affine': ('C7', '--', 2.2), 'deformed': ('C3', '-', 1.8)}
    ax[1].plot(GHIA_U, GHIA_Y, 'ko', ms=8, mfc='none', mew=1.8, zorder=9,
               label='Ghia et al. (1982)')
    for k in ('affine', 'deformed'):
        c, ls, lw = st[k]
        ax[1].plot(res[k]['uc'], res[k]['yq'], color=c, ls=ls, lw=lw,
                   label=f'{k}   RMS {res[k]["ru"]:.2e}')
    ax[1].set_xlabel('$u$'); ax[1].set_ylabel('$y$')
    ax[1].set_title('$u(y)$ on $x = 0.5$', fontsize=11)
    ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3)

    ax[2].plot(GHIA_XV, GHIA_V, 'ko', ms=8, mfc='none', mew=1.8, zorder=9,
               label='Ghia et al. (1982)')
    for k in ('affine', 'deformed'):
        c, ls, lw = st[k]
        ax[2].plot(res[k]['xq'], res[k]['vc'], color=c, ls=ls, lw=lw,
                   label=f'{k}   RMS {res[k]["rv"]:.2e}')
    ax[2].set_xlabel('$x$'); ax[2].set_ylabel('$v$')
    ax[2].set_title('$v(x)$ on $y = 0.5$', fontsize=11)
    ax[2].legend(fontsize=9); ax[2].grid(alpha=0.3)

    fig.suptitle(f'Lid-driven cavity Re = {RE:.0f}, mesh deformed {AMP*100:.0f} %, '
                 f'dt = {DT:g} — against Ghia, Ghia & Shin (1982)\n'
                 'profiles extracted by inverting the element map — the same '
                 'routine for both meshes', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=140)
    print(f'wrote {OUT}')


if __name__ == '__main__':
    sys.exit(main())
