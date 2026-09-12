"""Transient manufactured solution in 2D: the two terms of the least-squares
error model, separated.

THE MODEL TO TEST (ZIGZAG_CURE_RESEARCH.md sec 4.7, 4.9):

    || U_h(T) - U(T) ||  ~  C_2 dt^2  +  Phi(weighting) * ||R_h||

with ||R_h|| the irreducible spatial least-squares residual -- what the discrete
space cannot satisfy no matter how the rows are weighted -- and Phi a constant
that DEPENDS ON THE WEIGHTING.  So the weighting does not change the temporal
ORDER; it changes the spatial accuracy FLOOR of a transient computation, and the
two terms have to be separated to see it.  That is exactly why the existing
evidence could not discriminate:

  * start-up Poiseuille (TEMPORAL_ACCURACY_STUDY.md) gives slope 2.04 for BOTH
    weightings, because its parabolic solution is essentially representable in
    the spectral space, so ||R_h|| ~ 0 and the second term is invisible.  It is
    a null test, not a discriminator -- which the 1D model's control case
    (scratch/model1d_fixedpoint.py) explains: no irreducible residual, no
    pathology, for any weighting and any dt.
  * Orr-Sommerfeld discriminates strongly but measures a growth rate from an
    energy history, not a norm error against a known solution.
  * the Richardson study measures differences between successive dt, so it sees
    an order but not a level.

THIS TEST supplies a time-dependent solution that the space CANNOT represent
exactly, at three polynomial orders, so ||R_h|| is varied by orders of magnitude
while everything else is held fixed.  The prediction is specific:

    both:      error falls as dt^2 while the temporal term dominates, with the
               SAME constant -- the weighting is not a temporal-order effect.
    then:      each curve flattens at a floor proportional to ||R_h||, and the
               legacy floor is several times the balanced one.  Measured so far:
               N=10 (||R_h|| = 4e-7) order 1.86-1.93, weightings agree to 1%;
               N=6 (||R_h|| = 5.4e-3) both flat, legacy floor 4x balanced.

THE SOLUTION.  A time-modulated cellular flow on the unit square, from the
streamfunction psi = g(t) sin^2(pi x) sin^2(pi y), so that u = psi_y, v = -psi_x
is divergence-free by construction and vanishes on all four walls for every t
(no boundary-data bookkeeping, plain no-slip).  It is transcendental, hence not
in the polynomial space at any order.  Full nonlinear Navier-Stokes: the
convective term is carried in the manufactured forcing and Newton is iterated
out, so this exercises the production path rather than a linearised one.

THE FORCING TRAP.  f_known is the source of the WEIGHTED momentum row, so it
must carry a_flux (CHANNEL_VALIDATION.md sec 6); passing the physical f
under-forces the problem by a factor of the weight and silently changes the
answer.  Handled in step() below.

    uv run python scratch/mms2d_temporal.py [--quick]
"""
import argparse
import os
import sys
import time as _time

import numpy as np

_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R)
os.chdir(_R)

from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState, ls_coeffs
from lssem2d.mesh import build_channel
import lssem2d.solver as S

NU = 0.01
TEND = 0.2


# ------------------------------------------------------------ manufactured data

def _fields():
    """Symbolic (u, v, p, omega) and the forcing that makes them exact, lambdified.

    psi = cos(t) sin^2(pi x) sin^2(pi y);  u = psi_y, v = -psi_x;  omega = v_x - u_y.
    Momentum rows carry grad p + nu curl omega = (p_x + nu om_y, p_y - nu om_x),
    matching lssem2d.lssem (a13/a14 and a23/a24)."""
    import sympy as sp
    x, y, t, nu = sp.symbols('x y t nu', real=True)
    g = sp.cos(t)
    psi = g*sp.sin(sp.pi*x)**2*sp.sin(sp.pi*y)**2
    u = sp.diff(psi, y)
    v = -sp.diff(psi, x)
    p = g*sp.cos(sp.pi*x)*sp.cos(sp.pi*y)
    om = sp.diff(v, x) - sp.diff(u, y)
    f1 = sp.diff(u, t) + u*sp.diff(u, x) + v*sp.diff(u, y) + sp.diff(p, x) + nu*sp.diff(om, y)
    f2 = sp.diff(v, t) + u*sp.diff(v, x) + v*sp.diff(v, y) + sp.diff(p, y) - nu*sp.diff(om, x)
    args = (x, y, t, nu)
    L = lambda e: sp.lambdify(args, sp.simplify(e), 'numpy')
    return dict(u=L(u), v=L(v), p=L(p), om=L(om), f1=L(f1), f2=L(f2))


_F = None


def exact(m, t):
    """Nodal exact state (nelem, n, n, 4)."""
    global _F
    if _F is None:
        _F = _fields()
    X = m.xnod[:, :, None] + 0.0*m.ynod[:, None, :]
    Y = m.ynod[:, None, :] + 0.0*m.xnod[:, :, None]
    U = np.empty(X.shape + (4,))
    for i, k in enumerate(('u', 'v', 'p', 'om')):
        U[..., i] = _F[k](X, Y, t, NU)
    return U


def forcing(m, t):
    """Physical (f1, f2) at the nodes -- NOT yet weighted by a_flux."""
    global _F
    if _F is None:
        _F = _fields()
    X = m.xnod[:, :, None] + 0.0*m.ynod[:, None, :]
    Y = m.ynod[:, None, :] + 0.0*m.xnod[:, :, None]
    f = np.zeros(X.shape + (4,))
    f[..., 0] = _F['f1'](X, Y, t, NU)
    f[..., 1] = _F['f2'](X, Y, t, NU)
    return f


# ------------------------------------------------------------------- error norms

def errors(m, U, Ue):
    """Weighted L2 relative errors per field; pressure with its mean removed
    (it is defined up to a constant: the solver pins one node, the exact field
    does not know about that node)."""
    w = m.wq
    tot = w.sum()
    out = {}
    for i, k in enumerate(('u', 'v', 'p', 'om')):
        a, b = U[..., i], Ue[..., i]
        if k == 'p':
            a = a - (a*w).sum()/tot
            b = b - (b*w).sum()/tot
        num = np.sqrt((w*(a - b)**2).sum())
        den = np.sqrt((w*b**2).sum())
        out[k] = num/max(den, 1e-300)
    uv = np.sqrt((w*((U[..., 0] - Ue[..., 0])**2 + (U[..., 1] - Ue[..., 1])**2)).sum())
    uvd = np.sqrt((w*(Ue[..., 0]**2 + Ue[..., 1]**2)).sum())
    out['uv'] = uv/uvd
    return out


def residual_norm(st, m, U, t):
    """|| R_h ||: the least-squares residual of the CONSTRAINT rows at the exact
    (interpolated) state -- the part of the spatial residual that no weighting can
    remove, and the quantity the error model multiplies by dt."""
    from lssem2d.lssem import apply_L
    fu, fv = U[..., 0], U[..., 1]
    R = apply_L(st, U, fu, fv)/m.wq[..., None]          # undo the quadrature weight
    w = m.wq
    return float(np.sqrt((w*(R[..., 2]**2 + R[..., 3]**2)).sum()))


# ------------------------------------------------------------------------ driver

def march(N, E, dt, weighting, tend=TEND, newton=4, cgsfac=1e-12, refresh=20):
    m = build_channel(1.0, 1.0, E, E, N, bcs=(1, 1, 1, 1))
    m.compute_global_indices()
    w = None if weighting == 'legacy' else np.sqrt(dt)
    st = SolverState(m, diff_matrix(N), nu=NU, dt=dt, fac1=1.0, w_mom=w, w_mass=w)
    _, a_flux, _ = ls_coeffs(st)
    # Condensed vertex-patch preconditioner: the solve must reach 1e-12 relative
    # at every step or the solver error contaminates the very quantity being
    # measured, and Jacobi cannot at these orders and time steps.  It changes
    # only the path to the solution, not the solution.
    sys.path.insert(0, os.path.join(_R, 'scratch'))
    from vertex_schwarz2d import VertexSchwarzCondensed2D, make_coarse
    from pmg_ghia_cavity import snapshot
    cache = {'n': 0, 'pre': None}

    def factory(s_, fu, fv, Mi, pp):
        if cache['pre'] is None or cache['n'] % refresh == 0:
            snap = snapshot(s_, fu, fv)
            fu_, fv_ = np.ascontiguousarray(fu), np.ascontiguousarray(fv)
            cache['pre'] = VertexSchwarzCondensed2D(snap, fu_, fv_, pin_p=pp,
                                                   coarse=make_coarse(snap, fu_, fv_, Mi, pp))
        cache['n'] += 1
        return cache['pre']
    st.precond_factory = factory
    nstep = int(round(tend/dt))
    # BDF2 from the first step: both history levels seeded exactly.
    h = [exact(m, 0.0), exact(m, -dt)]
    t0 = _time.perf_counter()
    for n in range(nstep):
        t = (n + 1)*dt
        f = a_flux*forcing(m, t)                        # the weighted-row source
        S.step_bdf(st, h, time=t, max_newton=newton, newton_tol=1e-13,
                   newton_factor=0.0, f_known=f, pin_p=True,
                   cgsfac=cgsfac, cg_tol=1e-16, cg_max_iter=20000,
                   line_search=False)
    Ue = exact(m, nstep*dt)
    e = errors(m, h[0], Ue)
    e['wall'] = _time.perf_counter() - t0
    e['Rh'] = residual_norm(st, m, Ue, nstep*dt)
    return e


def sweep(Ns=(6, 8, 10, 12), dts=(2.5e-2, 1.25e-2, 6.25e-3, 3.125e-3, 1.5625e-3, 7.8125e-4), E=2):
    print(f'transient MMS, nonlinear Navier-Stokes, nu={NU}, T={TEND}, {E}x{E} elements')
    print('psi = cos(t) sin^2(pi x) sin^2(pi y): transcendental, so ||R_h|| > 0 at every order\n')
    res = {}
    for N in Ns:
        print(f'--- N = {N}')
        print(f'{"dt":>9} | {"legacy |u,v|":>12} {"order":>6} {"p":>10} | {"balanced |u,v|":>14} {"order":>6} {"p":>10} | {"ratio":>6}')
        prev = {}
        for dt in dts:
            row = [f'{dt:9.2e} |']
            cur = {}
            for wname in ('legacy', 'balanced'):
                e = march(N, E, dt, wname)
                res[(N, dt, wname)] = e
                cur[wname] = e['uv']
                o = (np.log(prev[wname]/e['uv'])/np.log(2.0)) if wname in prev else np.nan
                row.append(f'{e["uv"]:12.4e} {o:6.2f} {e["p"]:10.3e} |' if wname == 'legacy'
                           else f'{e["uv"]:14.4e} {o:6.2f} {e["p"]:10.3e} |')
            row.append(f'{cur["legacy"]/cur["balanced"]:6.1f}')
            prev = cur
            print(' '.join(row), flush=True)
        print(f'    ||R_h|| (constraint residual of the exact state) = '
              f'{res[(N, dts[-1], "balanced")]["Rh"]:.3e}')
    np.savez('scratch/mms2d_temporal.npz',
             **{f'{N}_{dt:g}_{w}_{q}': res[(N, dt, w)][q]
                for (N, dt, w) in res for q in ('uv', 'p', 'om', 'Rh')})
    return res


def plot(res, Ns, dts, out='figs_fosls_vs_fs/mms2d_temporal.png'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    cols = plt.cm.viridis(np.linspace(0.15, 0.8, len(Ns)))
    for a, wname in zip(ax, ('legacy', 'balanced')):
        for c, N in zip(cols, Ns):
            e = [res[(N, dt, wname)]['uv'] for dt in dts]
            a.loglog(dts, e, 'o-', color=c, label=f'N = {N}')
            rh = res[(N, dts[-1], wname)]['Rh']
            a.axhline(min(e), color=c, ls=':', lw=0.8)
        d = np.array(dts)
        a.loglog(d, e[0]*(d/d[0])**2, 'k--', lw=1, label='$\\Delta t^2$')
        a.set(xlabel='$\\Delta t$', ylabel='relative error in $(u,v)$ at $T$',
              title=f'{wname} weighting')
        a.grid(alpha=.3, which='both')
        a.legend(fontsize=8)
    lo = min(res[k]['uv'] for k in res)*0.5
    hi = max(res[k]['uv'] for k in res)*2
    for a in ax:
        a.set_ylim(lo, hi)
    fig.suptitle('Transient manufactured solution: the spatial residual sets a floor under the legacy weighting')
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print('wrote', out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    a = ap.parse_args()
    Ns = (6, 10) if a.quick else (6, 8, 10, 12)
    dts = ((2.5e-2, 1.25e-2, 6.25e-3) if a.quick else
           (2.5e-2, 1.25e-2, 6.25e-3, 3.125e-3, 1.5625e-3, 7.8125e-4))
    r = sweep(Ns, dts)
    plot(r, Ns, dts)
