"""RKW3 / Crank-Nicolson coefficients and stage driver.

Spalart, Moser & Rogers (1991), the standard integrator for spectral channel
DNS.  Per stage k = 0,1,2:

    U^k = U^{k-1} + dt*[ gamma_k N^{k-1} + zeta_k N^{k-2}      (explicit: convection)
                       + alpha_k L^{k-1} + beta_k L^k ]        (implicit: viscous, CN)

Rearranged for the implicit solve, the momentum row carries

    c_k = 1 / (beta_k * dt)

which is the 3D analogue of the 2D solver's a_mass = w_mass*fac1/dt.

WHY THIS SCHEME, honestly accounted (3D_DEVELOPMENT_PLAN.md sec 0.4):

  + AB2 has NO stability interval on the imaginary axis -- it is unstable for
    pure advection at ANY dt, and survives in practice only on viscous damping.
    RKW3 has a genuine interval, CFL ~ sqrt(3) = 1.73.  For DNS on fine grids,
    where convective eigenvalues are nearly imaginary, this is the real argument.
  + 3rd order on convection against AB2's 2nd.
  + 2 storage registers.  At Nz x the 2D footprint that is the binding constraint;
    classical RK4 needs 4.
  + ~13% fewer implicit solves per unit physical time (3 solves per step, but the
    step is ~3.5x larger).

  - a_mass is ~15% WORSE, not better.  1/beta = (4.32, 4.80, 6.00), so the worst
    stage sees c = 6/dt against BDF2's 1.5/dt -- a factor 4 -- and the 3.46x
    larger dt does not quite cancel it: 6/3.46 = 1.73 vs 1.50.  Any claim that
    RK3 relieves the a_mass problem is wrong; it mildly aggravates it.  Budget
    max_k 1/(beta_k*dt) against the measured stability window, NOT 1.5/dt.
"""
from fractions import Fraction as _F

# Explicit (convection) and implicit (viscous, Crank-Nicolson) stage weights.
GAMMA = (8/15, 5/12, 3/4)
ZETA = (0.0, -17/60, -5/12)
ALPHA = (29/96, -3/40, 1/6)
BETA = (37/160, 5/24, 1/6)
NSTAGE = 3

# Exact-arithmetic consistency check, run at import: a_k + b_k == g_k + z_k.
# Mis-transcribing this table is the usual way the scheme silently loses order,
# and it is invisible in a smooth test problem until the convergence study.
_A = (_F(29, 96), _F(-3, 40), _F(1, 6))
_B = (_F(37, 160), _F(5, 24), _F(1, 6))
_G = (_F(8, 15), _F(5, 12), _F(3, 4))
_Z = (_F(0), _F(-17, 60), _F(-5, 12))
for _k in range(NSTAGE):
    assert _A[_k] + _B[_k] == _G[_k] + _Z[_k], f'RKW3 coefficients inconsistent at stage {_k}'


def implicit_coeff(dt, stage):
    """c = 1/(beta_k*dt): the momentum-row mass coefficient for this stage.

    This is what must be compared against the measured a_mass stability window,
    not 1.5/dt.  See the module docstring.
    """
    return 1.0/(BETA[stage]*dt)


def a_mass_worst(dt):
    """Largest c over the three stages -- the number that decides stability."""
    return max(implicit_coeff(dt, k) for k in range(NSTAGE))


def cfl_limit():
    """RKW3 stability limit on the imaginary axis (per full step)."""
    return 3.0**0.5
