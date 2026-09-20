"""Q1 element matrices for the 2D velocity-vorticity-pressure FOSLS system.

THE ROWS, and they must match `lssem2d.lssem` exactly or this code tests a
different scheme.  With fields U = (u, v, p, omega) and the functional

    J = int[ w_mom^2 (N_1^2 + N_2^2) + w_con^2 (div u)^2 + (omega + u_y - v_x)^2 ]

the four residual rows carried here are

    R0  continuity   w_con * (u_x + v_y)
    R1  vorticity    omega + u_y - v_x
    R2  momentum x   a_mass*u + a_flux*( conv_x + p_x + nu*omega_y )   - hist_u
    R3  momentum y   a_mass*v + a_flux*( conv_y + p_y - nu*omega_x )   - hist_v

with N = (u.grad)u + grad p + nu curl(omega) - f, and curl of a scalar omega in
2D being (omega_y, -omega_x).  `a_mass`, `a_flux` and `w_con` come from
`lssem2d.lssem.ls_coeffs`; they are scalar ROW coefficients with no element
dependence, which is why importing them is correct and re-deriving them would
be a silent divergence from the paper's scheme.

CONVECTION is linearised about (fu, fv) in the same Newton form the spectral
code uses -- `apply_L` computes `fu*u_x + u*fu_x`, i.e.

    conv_x(dU) = fu*du_x + fv*du_y + du*fu_x + dv*fu_y
    conv_y(dU) = fu*dv_x + fv*dv_y + du*fv_x + dv*fv_y

Passing fu = u/2, fv = v/2 recovers the true nonlinear residual, exactly as
`newton_step` does.

TWO PATHS, deliberately.  `element_matrix` integrates with 2x2 Gauss and works
on any quadrilateral.  `element_matrix_affine` uses the closed form, valid only
when the map is affine (parallelogram, hence rectangle).  Gate G3 requires them
to agree to machine precision on a parallelogram -- the one configuration where
both are exact, and therefore the only place the Gauss path can be checked
against ground truth before being used where none exists.

DOF ORDERING is node-major: dof `4*i + f` is field `f` at local node `i`, with
nodes counter-clockwise from the lower-left of the reference square.
"""
import numpy as np

NF = 4                      # u, v, p, omega
U_, V_, P_, W_ = 0, 1, 2, 3
NN = 4                      # nodes per Q1 element

# reference square [-1,1]^2, nodes counter-clockwise from lower-left
_XI = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
def gauss_rule(n):
    """Tensor-product Gauss-Legendre, n points per direction, exact to 2n-1."""
    g, w = np.polynomial.legendre.leggauss(n)
    return (np.array([[a, b] for a in g for b in g]),
            np.array([x*y for x in w for y in w]))


# DEFAULT 3x3, NOT THE USUAL 2x2 -- and this is a measured decision.
#
# Textbook practice for Q1 is 2x2 Gauss, exact to degree 3, which suffices when
# the integrand is a product of two shape-function derivatives.  It does NOT
# suffice here.  The convection term carries a bilinear coefficient, so
# `a_flux*fu*dN/dx` is degree 2 in each reference coordinate and its SQUARE is
# degree 4.  Measured on a rectangle with a bilinear (fu, fv):
#
#     fu = fv = 0        |2x2 - 6x6|/|6x6| = 4.3e-16    (2x2 exact)
#     fu, fv bilinear    |2x2 - 6x6|/|6x6| = 4.7e-04    (2x2 NOT exact)
#                        |3x3 - 6x6|/|6x6| = 7.6e-16    (3x3 exact)
#
# Worse than the accuracy loss: 2x2 leaves the element matrix RANK DEFICIENT by
# one beyond its structural null space -- 3 zero modes against the 2 that 3x3,
# 4x4, 6x6 and 8x8 all agree on.  That extra mode is under-integration
# hourglassing, of exactly the kind this route was chosen to avoid, and it is
# invisible unless the eigenvalues are looked at.
_GP, _GW = gauss_rule(3)


def shape(xi, eta):
    """Q1 shape functions and their reference derivatives at (xi, eta)."""
    s, t = _XI[:, 0], _XI[:, 1]
    N = 0.25*(1.0 + s*xi)*(1.0 + t*eta)
    dNdxi = 0.25*s*(1.0 + t*eta)
    dNdeta = 0.25*t*(1.0 + s*xi)
    return N, dNdxi, dNdeta


def geometry(xy, xi, eta):
    """Jacobian, its determinant and the physical shape-function gradients."""
    N, dNdxi, dNdeta = shape(xi, eta)
    J = np.array([[dNdxi @ xy[:, 0], dNdeta @ xy[:, 0]],
                  [dNdxi @ xy[:, 1], dNdeta @ xy[:, 1]]])
    detJ = J[0, 0]*J[1, 1] - J[0, 1]*J[1, 0]
    if detJ <= 0.0:
        raise ValueError(f'non-positive Jacobian {detJ:.3e}: element inverted '
                         f'or degenerate')
    Jinv = np.array([[J[1, 1], -J[0, 1]], [-J[1, 0], J[0, 0]]])/detJ
    dNdx = Jinv[0, 0]*dNdxi + Jinv[1, 0]*dNdeta
    dNdy = Jinv[0, 1]*dNdxi + Jinv[1, 1]*dNdeta
    return N, dNdx, dNdy, detJ


def _B(N, dNdx, dNdy, fu, fv, dfudx, dfudy, dfvdx, dfvdy, coef):
    """The 4 x 16 matrix with (L U)_row = B[row] @ U_elem at one point.

    Built column by column over (node, field) so that every entry traces back
    to one term of one row above.  Clearer than a clever assembly, and this is
    the piece that a sign error hides in.
    """
    a_mass, a_flux, w_con, nu = coef
    B = np.zeros((NF, NN*NF))
    for i in range(NN):
        n, nx, ny = N[i], dNdx[i], dNdy[i]
        cu, cv, cp, cw = 4*i + U_, 4*i + V_, 4*i + P_, 4*i + W_
        # R0  continuity: w_con (u_x + v_y)
        B[0, cu] += w_con*nx
        B[0, cv] += w_con*ny
        # R1  vorticity: omega + u_y - v_x
        B[1, cw] += n
        B[1, cu] += ny
        B[1, cv] += -nx
        # R2  momentum x: a_mass u + a_flux (fu u_x + fv u_y + u fu_x + v fu_y
        #                                    + p_x + nu omega_y)
        B[2, cu] += a_mass*n + a_flux*(fu*nx + fv*ny + n*dfudx)
        B[2, cv] += a_flux*(n*dfudy)
        B[2, cp] += a_flux*nx
        B[2, cw] += a_flux*nu*ny
        # R3  momentum y: a_mass v + a_flux (fu v_x + fv v_y + u fv_x + v fv_y
        #                                    + p_y - nu omega_x)
        B[3, cv] += a_mass*n + a_flux*(fu*nx + fv*ny + n*dfvdy)
        B[3, cu] += a_flux*(n*dfvdx)
        B[3, cp] += a_flux*ny
        B[3, cw] += -a_flux*nu*nx
    return B


def element_matrix(xy, flin, coef, rhs=None, gauss=_GP, gw=_GW):
    """Element L^T L (16x16) and L^T f (16,) by quadrature.

    xy    : (4,2) node coordinates
    flin  : (4,2) linearisation velocities (fu, fv) at the nodes
    coef  : (a_mass, a_flux, w_con, nu)
    rhs   : (4,4) nodal values of the row right-hand side, or None.  Row 2 and
            3 carry the BDF history term; rows 0 and 1 are homogeneous.
    """
    xy = np.asarray(xy, float); flin = np.asarray(flin, float)
    A = np.zeros((NN*NF, NN*NF)); b = np.zeros(NN*NF)
    for (xi, eta), w in zip(gauss, gw):
        N, dNdx, dNdy, detJ = geometry(xy, xi, eta)
        fu, fv = N @ flin[:, 0], N @ flin[:, 1]
        dfudx, dfudy = dNdx @ flin[:, 0], dNdy @ flin[:, 0]
        dfvdx, dfvdy = dNdx @ flin[:, 1], dNdy @ flin[:, 1]
        B = _B(N, dNdx, dNdy, fu, fv, dfudx, dfudy, dfvdx, dfvdy, coef)
        jw = w*detJ
        A += jw*(B.T @ B)
        if rhs is not None:
            b += jw*(B.T @ (N @ np.asarray(rhs, float)))
    return A, b


def is_affine(xy, tol=1e-12):
    """A Q1 map is affine exactly when the quad is a parallelogram."""
    xy = np.asarray(xy, float)
    return bool(np.all(np.abs((xy[0] - xy[1]) + (xy[2] - xy[3])) < tol))


def element_matrix_affine(xy, flin, coef, rhs=None):
    """Closed form, valid ONLY for an affine map (parallelogram).

    On a parallelogram the Jacobian is CONSTANT, so the map contributes no
    rational factor and every entry of B^T B is a polynomial.  Its degree is
    set by the highest-order term:

        no convection      dN/dx * dN/dx          degree 2  -> 2x2 suffices
        with convection    (fu * dN/dx)^2         degree 4  -> needs 3x3

    An earlier version of this docstring claimed degree 2 unconditionally and
    was WRONG: it overlooked that the linearisation coefficient fu is itself a
    bilinear field.  Measurement settled it -- see the note on `_GP`.

    So the closed form is implemented as a degree-5-exact tensor-product rule
    (3-point Gauss-Legendre per direction), which integrates the degree-4
    integrand exactly.  Gate G3 requires it to agree with the default rule to
    machine precision on a parallelogram; since the two use different point
    sets, that is a genuine test of the geometry and assembly code and not a
    tautology.
    """
    if not is_affine(xy):
        raise ValueError('element_matrix_affine requires a parallelogram; '
                         'use element_matrix (2x2 Gauss) for a general quad')
    gp, gw = gauss_rule(5)          # deliberately finer than the default 3x3
    return element_matrix(xy, flin, coef, rhs, gauss=gp, gw=gw)
