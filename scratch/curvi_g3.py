"""Gate G3 — Taylor-Couette flow in an annulus: a curved domain, no forcing.

    uv run python scratch/curvi_g3.py

CURVILINEAR_2D_PLAN.md G3.  G2 puts a manufactured solution on a mesh that has
been bent; the domain is still a square and the forcing is whatever the solution
requires.  This gate removes both crutches.  The domain is an annular sector --
no element edge anywhere is straight, and the boundary the no-slip condition is
imposed on is a circle -- and the exact solution is a genuine solution of the
steady Navier-Stokes equations, so **the right-hand side is zero**.  Nothing is
manufactured to make the answer come out.

THE SOLUTION.  Between cylinders of radius a and b turning at angular velocities
Om_a and Om_b, the steady flow is purely azimuthal:

    u_th(r) = A r + B/r,     A = (Om_b b^2 - Om_a a^2)/(b^2 - a^2),
                             B = (Om_a - Om_b) a^2 b^2/(b^2 - a^2)

with u_r = 0.  Three properties make it a sharp instrument here:

  * **omega = 2A, a constant.**  The viscous term nu*curl(omega) vanishes
    identically, so the balance that remains is convection against the pressure
    gradient, dp/dr = u_th^2/r -- the nonlinearity is doing real work and the
    Newton sub-iterations are being tested, not bypassed.
  * **1/r is not a polynomial**, so the exact solution is outside the discrete
    space at every order and the error is a real approximation error, falling
    exponentially because the solution is analytic.  A solution the space could
    represent exactly would report zero error on a broken metric too.
  * **u_r = 0 exactly.**  The radial velocity is a pure error signal with no
    signal underneath it, and it is measured separately below.  On a curvilinear
    mesh it is also the component most exposed to a metric error, because it is
    the one the mapping mixes into the Cartesian pair the solver stores.

The sector's two radial cuts are given Dirichlet data from the same exact
solution, which is exact rather than approximate because the flow does not
depend on theta.  That avoids needing azimuthal periodicity, which is a
connectivity question and not a curvilinear one.

Reynolds number Om_a a (b - a)/nu = 5: laminar and linearly stable, so the
steady solution is the physical one and the comparison is meaningful.
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
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState
import lssem2d.solver as S
from curvi_g2 import exact_factory

OUT = os.path.join(_R, 'figs', 'curvi_g3.png')
A_IN, B_OUT = 0.5, 1.0
OM_IN, OM_OUT = 1.0, 0.0
NU = 0.05
DT = 1.0e6

_A = (OM_OUT*B_OUT**2 - OM_IN*A_IN**2)/(B_OUT**2 - A_IN**2)
_B = (OM_IN - OM_OUT)*A_IN**2*B_OUT**2/(B_OUT**2 - A_IN**2)


def exact_xy(x, y, t=0.0):
    """(u, v, p, omega) at Cartesian points.  Signature matches `bc.apply_bc`."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    r = np.hypot(x, y)
    ut = _A*r + _B/r
    u = -ut*y/r
    v = ut*x/r
    # dp/dr = u_th^2/r integrates in closed form; the additive constant is
    # irrelevant (the solver pins a node, the error norm removes the mean).
    p = 0.5*_A**2*r**2 + 2*_A*_B*np.log(r) - 0.5*_B**2/r**2
    om = np.full_like(r, 2*_A)
    return u, v, p, om


def exact_state(m):
    U = np.empty(m.X.shape + (4,))
    for i, f in enumerate(exact_xy(m.X, m.Y)):
        U[..., i] = f
    return U


def errors(m, U, Ue):
    w = m.wq
    tot = w.sum()
    out = {}
    for i, k in enumerate(('u', 'v', 'p', 'om')):
        a, b = U[..., i], Ue[..., i]
        if k == 'p':
            a = a - (a*w).sum()/tot
            b = b - (b*w).sum()/tot
        out[k] = float(np.sqrt((w*(a - b)**2).sum())/max(np.sqrt((w*b**2).sum()), 1e-300))
    num = np.sqrt((w*((U[..., 0] - Ue[..., 0])**2 + (U[..., 1] - Ue[..., 1])**2)).sum())
    den = np.sqrt((w*(Ue[..., 0]**2 + Ue[..., 1]**2)).sum())
    out['uv'] = float(num/den)
    # radial velocity: exactly zero in the true solution, so a pure error signal
    r = np.hypot(m.X, m.Y)
    ur = (U[..., 0]*m.X + U[..., 1]*m.Y)/r
    ut = np.hypot(U[..., 0], U[..., 1])
    out['ur'] = float(np.sqrt((w*ur**2).sum()/(w*ut**2).sum()))
    return out


def diagnostics(m, D, U):
    """The two quantities the plan asks for by name, both absolute.

    POINTWISE divergence, not the weak one.  A least-squares formulation drives
    div u into the functional rather than enforcing it, so this is a measured
    quantity with a target (1e-5), not an identity -- and on a curvilinear mesh
    it is also the sharpest single scalar available: the divergence is where a
    metric inconsistency shows up first, because it is the row that sums the two
    derivative directions and so relies on them being mutually consistent rather
    than merely individually plausible.

    OMEGA IN THE CORE.  The analytic vorticity is the constant 2A everywhere, so
    `max |omega - 2A|` over the interior nodes needs no norm and no reference
    scale.  Boundary nodes are excluded because omega is not prescribed there --
    it is whatever the least-squares functional settles on.
    """
    from lssem2d import curvi as _c
    div = _c.ddx(U[..., 0], D, m) + _c.ddy(U[..., 1], D, m)
    core = np.ones(U.shape[:3], bool)
    core[:, 0, :] = core[:, -1, :] = core[:, :, 0] = core[:, :, -1] = False
    return dict(divmax=float(np.abs(div).max()),
                divl2=float(np.sqrt((m.wq*div**2).sum()/m.wq.sum())),
                omcore=float(np.abs(U[..., 3][core] - 2*_A).max()))


def solve(N, E_r=3, E_th=4, dt=DT, newton=1):
    m = curvi.build_annulus(A_IN, B_OUT, E_r, E_th, N, 0.0, np.pi/2,
                            bcs=(1, 1, 1, 1))
    m.compute_global_indices()
    st = SolverState(m, diff_matrix(N), nu=NU, dt=dt, fac1=1.0, w_mom=1.0)
    fac = exact_factory(st)
    st.precond_factory = fac
    _orig = S.pcg_solve

    def _direct(state, b, fu_, fv_, M_inv, mw, pin_p=False, max_iter=5000,
                tol=1e-6, cgsfac=0.0, precond=None):
        return precond(b), 1
    S.pcg_solve = _direct
    Ue = exact_state(m)
    # Started from REST, not from the answer.  Seeding the exact field would let
    # a broken operator sit still and report zero error; from rest the Newton
    # iteration has to find the solution.  The BDF history is the state itself at
    # convergence, so the fixed point is the steady discrete solution regardless.
    U0 = np.zeros_like(Ue)
    h = [U0.copy(), U0.copy()]
    t0 = _time.perf_counter()
    # ONE Newton solve per outer iteration, stopping when the state stops
    # moving.  A fixed count of sub-iterations per step would refactor the QR
    # every one of them -- and the QR, not the operator applications, is what
    # costs at high N.  With exact solves Newton is quadratic, so the break is
    # reached in a handful of iterations and the rest were pure waste.
    nit = 0
    for _ in range(25):
        prev = h[0].copy()
        S.step_bdf(st, h, time=0.0, max_newton=newton, newton_tol=1e-15,
                   newton_factor=0.0, pin_p=True, exact_solution=exact_xy,
                   cgsfac=0.0, cg_tol=1e-30, cg_max_iter=4, line_search=False)
        nit += 1
        if np.abs(h[0] - prev).max() < 1e-13:
            break
    S.pcg_solve = _orig
    e = errors(m, h[0], Ue)
    e.update(diagnostics(m, diff_matrix(N), h[0]))
    e['wall'] = _time.perf_counter() - t0
    e['cond'] = fac.diag.get('cond', float('nan'))
    e['nit'] = nit
    e['m'], e['U'], e['Ue'] = m, h[0], Ue
    return e


def main():
    lssem2d.set_backend('numpy')
    Ns = [4, 5, 6, 7, 8, 9, 10]
    res = []
    print(f'{"N":>3s} {"|u| rel L2":>12s} {"u_r (=0)":>11s} {"p":>11s} '
          f'{"omega":>11s} {"max|div u|":>11s} {"max|om-2A|":>11s} {"it":>3s} {"s":>6s}')
    for N in Ns:
        e = solve(N)
        res.append(e)
        print(f'{N:3d} {e["uv"]:12.3e} {e["ur"]:11.3e} {e["p"]:11.3e} '
              f'{e["om"]:11.3e} {e["divmax"]:11.3e} {e["omcore"]:11.3e} '
              f'{e["nit"]:3d} {e["wall"]:6.1f}')

    x, y = np.asarray(Ns, float), np.log([r['uv'] for r in res])
    ce = np.polyfit(x, y, 1)
    ca = np.polyfit(np.log(x), y, 1)
    r2 = lambda c, xx: 1.0 - (y - np.polyval(c, xx)).var()/y.var()
    b, R2e = -ce[0], r2(ce, x)
    R2a = r2(ca, np.log(x))
    print(f'\nexponential  err ~ exp(-{b:.2f} N)   R2 = {R2e:.4f}')
    print(f'algebraic    err ~ N^-{-ca[0]:.1f}        R2 = {R2a:.4f}')
    dv, oc = res[-1]['divmax'], res[-1]['omcore']
    print(f'\nat N = {Ns[-1]}:  max|div u| = {dv:.2e}  (plan asks < 1e-5)'
          f'   max|omega - 2A| = {oc:.2e}')
    ok = (b > 0.8 and R2e > 0.99 and R2e > R2a and res[-1]['uv'] < 1e-8
          and dv < 1e-5 and oc < 1e-5)
    print(f'\ngate: exponential convergence on a domain with no straight edge, '
          f'against an\n      exact Navier-Stokes solution and zero forcing, with '
          f'pointwise div u and\n      omega - 2A both under 1e-5 -> '
          f'{"PASS" if ok else "FAIL"}')
    figure(Ns, res, b)
    return 0 if ok else 1


def figure(Ns, res, b):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.3))

    e = res[3]
    m, U = e['m'], e['U']
    n = m.nterm
    sp = ax[0].tripcolor(m.X.ravel(), m.Y.ravel(),
                         np.hypot(U[..., 0], U[..., 1]).ravel(), shading='gouraud')
    for el in range(m.nelem):
        for idx in (0, n-1):
            ax[0].plot(m.X[el, idx, :], m.Y[el, idx, :], 'w-', lw=0.6)
            ax[0].plot(m.X[el, :, idx], m.Y[el, :, idx], 'w-', lw=0.6)
    fig.colorbar(sp, ax=ax[0], shrink=0.85)
    ax[0].set_aspect('equal'); ax[0].tick_params(labelsize=8)
    ax[0].set_title(f'$|\\mathbf{{u}}|$, $N = {Ns[3]}$, 3x4 elements', fontsize=10)

    r = np.hypot(m.X, m.Y).ravel()
    ut = ((U[..., 1]*m.X - U[..., 0]*m.Y)/np.hypot(m.X, m.Y)).ravel()
    o = np.argsort(r)
    rr = np.linspace(A_IN, B_OUT, 300)
    ax[1].plot(rr, _A*rr + _B/rr, 'k-', lw=1.6, label='exact  $Ar + B/r$')
    ax[1].plot(r[o], ut[o], 'o', ms=2.5, color='C3', alpha=0.6, label='computed')
    ax[1].set_xlabel('$r$'); ax[1].set_ylabel(r'$u_\theta$')
    ax[1].set_title('azimuthal profile, every node', fontsize=10)
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)

    ax[2].semilogy(Ns, [q['uv'] for q in res], 'o-', color='C0',
                   label=f'$|\\mathbf{{u}}|$  ($b = {b:.2f}$)')
    ax[2].semilogy(Ns, [q['ur'] for q in res], 's-', color='C3',
                   label='$u_r$  (exact value 0)')
    ax[2].semilogy(Ns, [q['p'] for q in res], '^-', color='C1', label='$p$')
    ax[2].semilogy(Ns, [q['divmax'] for q in res], 'v--', color='C2',
                   label=r'$\max|\nabla\cdot\mathbf{u}|$ (abs.)')
    ax[2].axhline(1e-5, color='C7', ls=':', lw=1)
    ax[2].set_xlabel('polynomial order $N$'); ax[2].set_ylabel(r'relative $L^2$ error')
    ax[2].set_title('G3: spectral convergence on a curved domain', fontsize=10)
    ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3, which='both')

    fig.suptitle('Gate G3 — Taylor-Couette: an exact Navier-Stokes solution, '
                 'zero forcing, no straight element edge', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=140)
    print(f'wrote {OUT}')


if __name__ == '__main__':
    sys.exit(main())
