"""Manufactured solutions, with the forcing derived symbolically.

Hand-differentiating the VVP residual is exactly the kind of step that produces
a sign error which every convergence study then quietly absorbs into its rate.
`sympy` removes that risk: the forcing is generated from the same symbolic
expressions the exact solution is evaluated from, so the two cannot disagree.
"""
import numpy as np
import sympy as sp

X, Y = sp.symbols('x y', real=True)


def taylor_like(nu=1.0, pscale=1.0):
    """A divergence-free velocity from a streamfunction, zero on the unit square.

    psi = sin^2(pi x) sin^2(pi y) gives u = psi_y, v = -psi_x, so div u = 0
    identically and u = v = 0 on the whole boundary -- the no-slip condition
    under which the ADN complementing condition FAILS for this functional and
    BG predict vorticity and pressure one order below velocity.
    """
    psi = sp.sin(sp.pi*X)**2 * sp.sin(sp.pi*Y)**2
    u = sp.diff(psi, Y)
    v = -sp.diff(psi, X)
    # PRESSURE SCALE.  With pscale = 1 the viscous term nu*curl(omega) exceeds
    # grad p by a factor ~50 in rms, so the momentum row hardly constrains p at
    # all and its RELATIVE error converges far too slowly to read a rate from.
    # That is a defect of the test case, not of the method: a manufactured
    # solution intended to measure a pressure rate must put the pressure
    # gradient on the same scale as the other terms in the row it appears in.
    p = pscale*sp.cos(sp.pi*X)*sp.cos(sp.pi*Y)   # zero mean on the unit square
    om = sp.diff(v, X) - sp.diff(u, Y)           # so that omega + u_y - v_x = 0
    return u, v, p, om, nu


def rows(u, v, p, om, nu, coef, convect=True):
    """The four FOSLS residual rows as symbolic expressions.

    Matches `element_q1` exactly: continuity, vorticity, momentum x, momentum y,
    with the NONLINEAR convection (the steady solve linearises about the current
    iterate, but the manufactured forcing must use the true residual).
    """
    a_mass, a_flux, w_con, _ = coef
    r0 = w_con*(sp.diff(u, X) + sp.diff(v, Y))
    r1 = om + sp.diff(u, Y) - sp.diff(v, X)
    cx = (u*sp.diff(u, X) + v*sp.diff(u, Y)) if convect else 0
    cy = (u*sp.diff(v, X) + v*sp.diff(v, Y)) if convect else 0
    nx = cx + sp.diff(p, X) + nu*sp.diff(om, Y)
    ny = cy + sp.diff(p, Y) - nu*sp.diff(om, X)
    r2 = a_mass*u + a_flux*nx
    r3 = a_mass*v + a_flux*ny
    return [sp.simplify(r) for r in (r0, r1, r2, r3)]


def lambdify_all(exprs):
    return [sp.lambdify((X, Y), e, 'numpy') for e in exprs]
