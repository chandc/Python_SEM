"""BDF2 time stepping for the low-order FOSLS code.

THE COEFFICIENTS ARE IMPORTED, NOT RE-DERIVED.  `a_mass`, `a_flux` and the
history scale are scalar ROW coefficients with no element dependence, and the
weighting they encode is the paper's thesis.  A low-order code that re-derived
them would be testing a different scheme, and the discrepancy would be silent.
`coeffs()` below calls `lssem2d.lssem.ls_coeffs` through a minimal state object
and gate G6 asserts the result is bit-identical to the spectral path.

THE ROW.  With BDF2, fac1 = 3/2 and the history weights alpha = (2, -1/2):

    w_mass/dt * (fac1 u^{n+1} - 2 u^n + 1/2 u^{n-1})  +  w_mom * N(u^{n+1}) = w_mom f

so a_mass = w_mass*fac1/dt multiplies the new iterate and the KNOWN part,

    hist = (w_mass/dt) (2 u^n - 1/2 u^{n-1}),

moves to the right-hand side alongside the forcing.  Only the two momentum
rows carry it; continuity and the vorticity definition have no time derivative.
"""
import numpy as np

from lssem2d.lssem import ls_coeffs as _spectral_ls_coeffs
from .mesh import NF
from .assemble import assemble_newton, apply_dirichlet, functional
from .solve import solve_direct

BDF2_FAC1 = 1.5
BDF2_ALPHA = (2.0, -0.5)


class _State:
    """Just enough of a SolverState for `ls_coeffs`, so the real one is used."""
    def __init__(self, dt, fac1, w_mom, w_mass):
        self.dt, self.fac1, self.w_mom, self.w_mass = dt, fac1, w_mom, w_mass


def coeffs(dt, nu, weighting='balanced', w_con=1.0, fac1=BDF2_FAC1):
    """((a_mass, a_flux, w_con, nu), hist_scale) from the spectral ls_coeffs.

    THE HISTORY SCALE IS NOT ALWAYS w_mass/dt, and getting it wrong is silent.
    `ls_coeffs`'s own table:

        both None (legacy):  a_mass = fac1,           a_flux = dt,    hist = 1
        w_mom only:          a_mass = fac1/dt,        a_flux = w_mom, hist = 1/dt
        both set:            a_mass = w_mass*fac1/dt, a_flux = w_mom, hist = w_mass/dt

    Legacy does NOT pre-multiply the row by 1/dt -- the whole row is scaled by
    dt instead, which is why a_flux = dt there.  Using w_mass/dt for legacy
    makes the history term dt times too large, and the symptom is a scheme that
    DIVERGES as dt is refined: measured errors of 1.5e2, 6.8e2, 4.8e4, 6.2e5
    for dt = 0.08 down to 0.01.  That looks exactly like a dramatic
    confirmation of the paper's small-dt thesis and is nothing of the kind.
    """
    w = {'balanced': np.sqrt(dt), 'legacy': None, 'unit': 1.0}[weighting]
    a_mass, a_flux, _ = _spectral_ls_coeffs(_State(dt, fac1, w, w))
    hist = 1.0 if w is None else w/dt
    return (a_mass, a_flux, w_con, nu), hist


def history_rhs(mesh, hist_scale, Un, Unm1, f=None):
    """Nodal row right-hand side: the BDF2 history plus any forcing.

    `hist_scale` is w_mass/dt (1.0 for the legacy weighting, where the row is
    not pre-multiplied).  Rows 0 and 1 get forcing only.
    """
    r = np.zeros((mesh.nnode, NF)) if f is None else np.array(f, float)
    a0, a1 = BDF2_ALPHA
    for fld in (0, 1):
        r[:, 2 + fld] += hist_scale*(a0*Un[fld::NF] + a1*Unm1[fld::NF])
    return r


def step(mesh, Un, Unm1, coef, hist_scale, fixed, values, f=None,
         tol=1e-11, maxit=20):
    """One BDF2 step by Gauss-Newton.  Returns (U, iterations, functional)."""
    rhs = history_rhs(mesh, hist_scale, Un, Unm1, f)
    U = Un.copy()
    U[fixed] = values
    for k in range(maxit):
        A, b = assemble_newton(mesh, U, coef, rhs)
        Ac, bc = apply_dirichlet(A, b, fixed, np.zeros(len(fixed)))
        dU = solve_direct(Ac, bc)
        U += dU
        if np.abs(dU).max() < tol:
            break
    return U, k + 1, functional(mesh, U, coef, rhs)
