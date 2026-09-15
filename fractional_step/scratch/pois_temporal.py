"""TEMPORAL order of the scheme on startup plane Poiseuille flow.

Why this and not the dt sweep in pois_dt_w1.py: that one marches to STEADY state,
where the BDF mass and history terms cancel identically (fac1 = sum alpha_m) and
the functional actually minimised is

    J = int[ w_mom^2 (N_1^2 + N_2^2) + (div u)^2 + (om + u_y - v_x)^2 ]

With w_mom pinned at 1 that expression contains no dt at all, so the converged
answer is dt-independent BY CONSTRUCTION and refining dt cannot produce a slope.
Legacy's 212,061x spread was a_flux = dt changing the functional, not temporal
error.  Measuring an ORDER needs an unsteady solution.

THE CASE.  Channel y in [-1,1], streamwise-periodic, driven by the body force
f_x = 2*nu that sustains u = 1 - y^2 (a periodic p cannot carry a mean gradient;
see CHANNEL_VALIDATION.md sec 6).  Started from rest the exact solution is

    u(y,t) = (1 - y^2) - sum_n  (4(-1)^n / lam_n^3) cos(lam_n y) exp(-nu lam_n^2 t)
    lam_n  = (2n+1)pi/2,   n = 0, 1, 2, ...

    v = 0,   p = 0,   om = v_x - u_y = 2y - sum_n (4(-1)^n / lam_n^2) sin(lam_n y) e^{...}

u depends on y and t only, so u.grad(u) = 0 identically and this is an exact
solution of the FULL Navier-Stokes equations, not just Stokes.  That is also the
honest limitation of the test: every parallel flow kills the convective term, so
this measures the temporal order of the scheme with the nonlinearity present but
evaluating to zero.  A genuinely nonlinear temporal check needs MMS forcing.

Integration runs from t0 > 0, not from rest, seeded with the exact solution at
t0: the impulsive start is non-smooth in time and would contaminate the order.
The history is seeded at t0-dt as well, so BDF2 is used from the very first step
rather than one BDF1 step polluting the fit.

Weights: w_mom = w_mass = 1, so a_mass = fac1/dt, a_flux = 1, dt_eff = dt.  The
least-squares weight is then FIXED while dt varies, which is what makes the
refinement measure time discretisation alone.  Legacy is swept alongside for
contrast: there a_flux = dt moves the weight with the step.

p-refinement at each dt (following CHANNEL_VALIDATION.md, where fitting only the
finest three dt gave a spurious 1.54): if the curves for different N coincide,
the error is purely temporal and the slope is real.
"""
import os, sys, json
import numpy as np
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import lssem2d
lssem2d.set_backend('numpy')          # numba diverges from numpy at ~4e-06 on
                                      # accumulated states; see pois_dt_w1.py
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState, ls_coeffs
import lssem2d.solver as S

NU = 1.0                              # as in Chan's Stokes case; nu only sets
                                      # the decay scale, the solution is exact
                                      # for any nu since u.grad(u) = 0
LX = 2.0*np.pi
WIDTH = 2.0                           # y in [-1, 1]
EY = 2                                # uniform in y; N carries the resolution
T0, TEND = 0.02, 0.12                 # t0 > 0: skip the non-smooth impulsive start
NMODE = 400                           # series terms; coefficients fall as 1/lam^3
                                      # and the exponential kills n > 6 by t0
DTS = [0.01, 0.005, 0.0025, 0.00125, 0.000625]
ORDERS = [10, 14, 18]
CGSFAC, CGTOL, CGMAX = 1e-8, 1e-14, 20000
NEWTON, NTOL = 10, 1e-13              # iterate sub-iterations out of the answer

_lam = (2*np.arange(NMODE) + 1)*np.pi/2
_c = 4.0*(-1.0)**np.arange(NMODE)/_lam**3


def u_exact(y, t):
    y = np.asarray(y)[..., None]
    return (1.0 - y[..., 0]**2) - np.sum(_c*np.cos(_lam*y)*np.exp(-NU*_lam**2*t), -1)


def om_exact(y, t):
    y = np.asarray(y)[..., None]
    return 2.0*y[..., 0] - np.sum(_c*_lam*np.sin(_lam*y)*np.exp(-NU*_lam**2*t), -1)


def build(N):
    m = build_channel(LX, WIDTH, 1, EY, N, bcs=(0, 0, 1, 1))
    m.periodic_x = LX
    m.compute_global_indices()
    return m


def state_at(m, N, t):
    """Exact (u, v, p, om) at time t on the mesh, y shifted to [-1, 1]."""
    n = N+1
    U = np.zeros((m.nelem, n, n, 4))
    ymid = 0.5*(m.ynod.min() + m.ynod.max())
    for e in range(m.nelem):
        y = m.ynod[e, :] - ymid                    # (n,) nodal y for this element
        U[e, :, :, 0] = u_exact(y, t)[None, :]     # x-independent
        U[e, :, :, 3] = om_exact(y, t)[None, :]
    return U


def run(N, dt, legacy=False):
    m = build(N)
    st = SolverState(m, diff_matrix(N), nu=NU, dt=dt, fac1=1.0,
                     w_mom=None if legacy else 1.0,
                     w_mass=None if legacy else 1.0)
    _, a_flux, _ = ls_coeffs(st)
    f = np.zeros((m.nelem, N+1, N+1, 4))
    f[..., 0] = a_flux*2.0*NU                      # mean gradient as a body force,
                                                   # carrying the row weight
    nsteps = int(round((TEND-T0)/dt))
    assert abs(nsteps*dt - (TEND-T0)) < 1e-12, (dt, nsteps)
    # Seed BOTH levels from the exact solution so step 1 is already BDF2.
    U = state_at(m, N, T0)
    hist = [U, state_at(m, N, T0-dt)]
    pin = (0, (N+1)//2, (N+1)//2)                  # p == 0 exactly here
    for s in range(nsteps):
        U = S.step_bdf(st, hist, time=T0+s*dt, max_newton=NEWTON,
                       newton_tol=NTOL, newton_factor=0.0, f_known=f,
                       pin_p=pin, cgsfac=CGSFAC, cg_tol=CGTOL,
                       cg_max_iter=CGMAX, line_search=False)
        if not np.all(np.isfinite(U)):
            return None
    Uex = state_at(m, N, TEND)
    du = U[..., 0] - Uex[..., 0]
    return dict(N=N, dt=dt, legacy=legacy, a_flux=a_flux, steps=nsteps,
                rms=float(np.sqrt((du**2).mean())), mx=float(np.abs(du).max()),
                umax=float(np.abs(U[..., 0]).max()))


def slope(dts, errs):
    """Least-squares fit of log(err) against log(dt)."""
    d, e = np.asarray(dts, float), np.asarray(errs, float)
    k = e > 0
    if k.sum() < 2:
        return float('nan')
    return float(np.polyfit(np.log(d[k]), np.log(e[k]), 1)[0])


def main():
    print(f"Startup plane Poiseuille, nu = {NU:g}, periodic channel, y in [-1,1]")
    print("exact: u = (1-y^2) - sum 4(-1)^n/lam_n^3 cos(lam_n y) exp(-nu lam_n^2 t)")
    print(f"integrate t = {T0:g} -> {TEND:g} (BDF2 from step 1, history seeded exactly)")
    print(f"cg_tol = {CGTOL:g}, newton to {NTOL:g}\n")

    rows = []
    for legacy in (False, True):
        tag = ('LEGACY  (a_flux = dt)' if legacy else
               'w_mom = w_mass = 1  (a_flux = 1)')
        print(f"--- {tag} ---")
        print(f"{'dt':>10}{'steps':>7}" +
              ''.join(f"{'N='+str(N):>13}" for N in ORDERS))
        tab = {N: [] for N in ORDERS}
        for dt in DTS:
            line = f"{dt:>10g}{int(round((TEND-T0)/dt)):>7}"
            for N in ORDERS:
                r = run(N, dt, legacy)
                rows.append(r)
                if r is None:
                    line += f"{'DIVERGED':>13}"; tab[N].append(np.nan)
                else:
                    line += f"{r['rms']:>13.4e}"; tab[N].append(r['rms'])
            print(line, flush=True)
            with open(f'{SC}/pois_temporal.json', 'w') as fh:
                json.dump([r for r in rows if r], fh, indent=1)
        print(f"{'slope':>10}{'':>7}" +
              ''.join(f"{slope(DTS, tab[N]):>13.3f}" for N in ORDERS))
        # the fit window matters: report the coarse half too, where the spatial
        # floor cannot have been reached yet
        print(f"{'(coarse 3)':>10}{'':>7}" +
              ''.join(f"{slope(DTS[:3], tab[N][:3]):>13.3f}" for N in ORDERS) + "\n",
              flush=True)


if __name__ == '__main__':
    main()
