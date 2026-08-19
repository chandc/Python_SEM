"""Per-mode VVP operator L and its adjoint, for one Fourier wavenumber k_z.

NEW CODE.  Imports lssem2d.operators for the (x,y) derivatives -- reuse, not
modification.  lssem2d is not touched anywhere in this module.

THE SYSTEM.  7 unknowns per mode, U = (u, v, w, ox, oy, oz, p), and 8 residual
rows.  With d/dz -> i*k_z (3d_vvp_fourier_expansion.md sec 2):

  0 continuity      u_x + v_y + i k w
  1 vorticity x     w_y - i k v - ox
  2 vorticity y     i k u - w_x - oy
  3 vorticity z     v_x - u_y - oz
  4 momentum x      c u + p_x + nu (oz_y - i k oy)
  5 momentum y      c v + p_y + nu (i k ox - oz_x)
  6 momentum z      c w + i k p + nu (oy_x - ox_y)
  7 vorticity div   ox_x + oy_y + i k oz

c is the implicit mass coefficient.  Under RKW3/Crank-Nicolson it is
1/(beta_k*dt), NOT fac1/dt -- see lssem3d.timestep and 3D_DEVELOPMENT_PLAN.md
sec 0.4.  Convection is explicit and lives in the right-hand side, so it does
not appear here at all; that is what keeps the modes decoupled, and it is also
why this operator is Stokes-like rather than the linearised Navier-Stokes
operator whose stability was characterised in 2D.

COMPLEX INSIDE, REAL OUTSIDE.  The residuals are written in complex form because
that is how the equations read and how they are checked.  The public entry
points take and return SPLIT-REAL arrays (..., 14) so that L^T L is symmetric
rather than Hermitian and the existing verified real-valued CG, Jacobi and
adjoint machinery apply unchanged -- 3D_DEVELOPMENT_PLAN.md sec 1.2.  The
split/join pair is trivial and is tested as a round trip.
"""
import numpy as np
from .deriv import ddx as dUdx, ddy as dUdy, ddxT as DxT, ddyT as DyT

NVAR = 7                 # u v w ox oy oz p          (complex)
NROW = 8                 # residual rows             (complex)
NVAR_R = 2*NVAR          # split-real unknowns
NROW_R = 2*NROW          # split-real rows

U_, V_, W_, OX_, OY_, OZ_, P_ = range(NVAR)


# ----------------------------------------------------------- real <-> complex

def to_complex(Ur):
    """(..., 14, nmode) real -> (..., 7, nmode) complex.

    LAYOUT: the field axis is -2 and z-modes are -1 throughout this module, per
    3D_DEVELOPMENT_PLAN.md sec 1.1.  Slicing the LAST axis instead -- the
    natural-looking `Ur[..., :NVAR]` -- silently splits the mode axis and was a
    real bug here.
    """
    return Ur[..., :NVAR, :] + 1j*Ur[..., NVAR:, :]


def to_real(Uc):
    """(..., 7, nmode) complex -> (..., 14, nmode) real."""
    return np.concatenate([Uc.real, Uc.imag], axis=-2)


# ------------------------------------------------------------------ operator

def apply_L0_complex(U, D, facx, facy, kz, nu, c):
    """UNWEIGHTED residual rows: the raw differential operator L0.

    apply_LT_complex is the exact transpose of THIS, not of the weighted form
    below -- see the note on the weighting convention there.
    """
    g = lambda f: (dUdx(U[..., f, :], D, facx), dUdy(U[..., f, :], D, facy))
    ux, uy = g(U_)
    vx, vy = g(V_)
    wx, wy = g(W_)
    oxx, oxy = g(OX_)
    oyx, oyy = g(OY_)
    ozx, ozy = g(OZ_)
    px, py = g(P_)
    u, v, w = U[..., U_, :], U[..., V_, :], U[..., W_, :]
    ox, oy, oz = U[..., OX_, :], U[..., OY_, :], U[..., OZ_, :]
    p = U[..., P_, :]
    ik = 1j*kz

    R = np.empty(U.shape[:-2] + (NROW, U.shape[-1]), dtype=complex)
    R[..., 0, :] = ux + vy + ik*w
    R[..., 1, :] = wy - ik*v - ox
    R[..., 2, :] = ik*u - wx - oy
    R[..., 3, :] = vx - uy - oz
    R[..., 4, :] = c*u + px + nu*(ozy - ik*oy)
    R[..., 5, :] = c*v + py + nu*(ik*ox - ozx)
    R[..., 6, :] = c*w + ik*p + nu*(oyx - oxy)
    R[..., 7, :] = oxx + oyy + ik*oz
    return R


def apply_LT_complex(R, D, facx, facy, kz, nu, c):
    """Adjoint of apply_L_complex: 8 complex rows -> 7 complex fields.

    TWO RULES, and both were wrong in the first draft:

      * The (x,y) derivatives are NOT anti-self-adjoint.  lssem2d/operators.py
        supplies genuine transposes DxT/DyT, used with the SAME sign as the
        forward term -- not `-dUdx`.  (An earlier version here assumed
        anti-self-adjointness and failed the adjoint test by ~70%.)
      * In the split-real form the transpose of the real 2Nx2N block matrix
        [[M_re, -M_im], [M_im, M_re]] corresponds to the complex HERMITIAN
        adjoint: transpose the spatial operator AND conjugate the scalar.  So
        (i*k) -> -(i*k), while real coefficients c and nu are unchanged.

    Mechanically: for every forward term `s * Op(U_f)` appearing in row r, add
    `conj(s) * Op^T(R_r)` to field f.
    """
    dxT = lambda f: DxT(f, D, facx)
    dyT = lambda f: DyT(f, D, facy)
    mik = -1j*kz                                     # conjugate of i*k
    r0, r1, r2, r3, r4, r5, r6, r7 = (R[..., i, :] for i in range(NROW))

    C = np.empty(R.shape[:-2] + (NVAR, R.shape[-1]), dtype=complex)
    C[..., U_, :] = dxT(r0) + mik*r2 - dyT(r3) + c*r4
    C[..., V_, :] = dyT(r0) - mik*r1 + dxT(r3) + c*r5
    C[..., W_, :] = mik*r0 + dyT(r1) - dxT(r2) + c*r6
    C[..., OX_, :] = -r1 + nu*mik*r5 - nu*dyT(r6) + dxT(r7)
    C[..., OY_, :] = -r2 - nu*mik*r4 + nu*dxT(r6) + dyT(r7)
    C[..., OZ_, :] = -r3 + nu*dyT(r4) - nu*dxT(r5) + mik*r7
    C[..., P_, :] = dxT(r4) + dyT(r5) + mik*r6
    return C


def apply_L_complex(U, D, facx, facy, kz, nu, c, wq=None):
    """Quadrature-WEIGHTED rows: W * L0, matching lssem2d's convention.

    THE CONVENTION, and it is not the obvious one.  lssem2d's apply_L multiplies
    every row by wq = jac*w_i*w_j, while its apply_LT is the UNWEIGHTED
    transpose.  The pair is built so that

        apply_LT(apply_L(x))  =  L0^T W L0 x

    which is the normal operator for the least-squares functional J = int R^2 dOmega
    -- symmetric because W is diagonal.  Neither routine is the plain adjoint of
    the other, and testing them as if they were is a mistake (this module did
    exactly that until the weights were added: without W the operator minimises
    the unweighted NODAL sum of squares, which is a different functional on a
    GLL grid, where the node spacing varies as 1/N^2 near element edges).

    wq is (nelem, n, n); pass None only for the unweighted operator in tests.
    """
    R = apply_L0_complex(U, D, facx, facy, kz, nu, c)
    return R if wq is None else R*wq[..., None, None]


# --------------------------------------------------------- split-real facade

def apply_L(Ur, D, facx, facy, kz, nu, c, wq=None):
    """(..., 14, nmode) real -> (..., 16, nmode) real.  Weighted if wq given."""
    return to_real(apply_L_complex(to_complex(Ur), D, facx, facy, kz, nu, c, wq))


def apply_LT(Rr, D, facx, facy, kz, nu, c):
    """(..., 16, nmode) real -> (..., 14, nmode) real."""
    Rc = Rr[..., :NROW, :] + 1j*Rr[..., NROW:, :]
    return to_real(apply_LT_complex(Rc, D, facx, facy, kz, nu, c))


def reduces_to_2d(kz):
    """At k_z = 0 the system splits: (u, v, oz, p) is exactly the 2D VVP system
    and (w, ox, oy) decouples from it.  3D_DEVELOPMENT_PLAN.md Stage 1 turns on
    this, and it is asserted in the tests rather than assumed."""
    return kz == 0.0
