"""Explicit convective term u.grad u, dealiased by the 3/2 rule in z.

NEW CODE.  Imports lssem2d.operators for the (x,y) derivatives; lssem2d is not
modified.

WHERE THIS SITS.  Convection is explicit -- it is the RKW3 half of the
RKW3/Crank-Nicolson scheme and lives entirely in the right-hand side.  That is
what keeps the Fourier modes decoupled: a linearised-implicit treatment would
make the coefficients z-dependent, which is a convolution in k_z and recouples
every mode (3d_vvp_fourier_expansion.md sec 5).

THE PIPELINE, and why the derivatives are taken first.

    1. d/dx, d/dy      in MODE space, via the 2D differentiation matrix
    2. d/dz            in MODE space, multiply by i*k_z
    3. -> padded physical      (3/2 rule)
    4. form the products       u*du/dx + v*du/dy + w*du/dz
    5. -> modes, truncate

Derivatives commute with the z-transform (D acts on (i,j) only, i*k_z on the
trailing axis), so taking them in mode space is exact and costs no extra
transforms.  Only the PRODUCTS need the padded grid, because only they are
quadratic and therefore only they alias.

DEALIASING IS NOT OPTIONAL HERE.  In (x,y) the GLL quadrature is exact to degree
2N-1 and the 2D code carries no dealiasing.  In z the Fourier basis makes the
aliasing exact and unavoidable: a product of modes k1, k2 lands at k1+k2, which
folds back into the resolved band whenever k1+k2 > Nz/2.  At Re_tau = 180 that
shows up as an energy pile-up at high k_z.  lssem3d/tests carries the check with
a negative control.
"""
import numpy as np
from .deriv import ddx as dUdx, ddy as dUdy
from .fourier import dealias_forward, dealias_backward
from . import device as DEV
from . import operator as OP


def convective(Uh, D, facx, facy, kz, nz):
    """u.grad u for the three momentum components, in mode space.

    Uh  (nelem, n, n, 7, nmode) complex; only u, v, w are read.  Note the var
        axis is SECOND TO LAST and z-modes are LAST, per the layout in
        3D_DEVELOPMENT_PLAN.md sec 1.1 -- so fields are indexed `Uh[..., f, :]`,
        not `Uh[..., f]`.  Getting that wrong silently selects a single mode
        instead of a single field.
    kz  (nmode,) wavenumbers, broadcast over the trailing axis.
    nz  physical z-size.

    Returns (nelem, n, n, 3, nmode) complex: (N_x, N_y, N_z), UNSIGNED -- the
    caller fixes the sign when assembling the right-hand side.
    """
    ik = 1j*kz
    comp = (OP.U_, OP.V_, OP.W_)

    # 1-2. derivatives in mode space (exact; commute with the z-transform)
    dxm = [dUdx(Uh[..., f, :], D, facx) for f in comp]
    dym = [dUdy(Uh[..., f, :], D, facy) for f in comp]
    dzm = [ik*Uh[..., f, :] for f in comp]

    # 3. to the padded physical grid -- only the products alias, but every
    #    factor of a product must live on the same grid
    up = [dealias_forward(Uh[..., f, :], nz) for f in comp]
    dxp = [dealias_forward(a, nz) for a in dxm]
    dyp = [dealias_forward(a, nz) for a in dym]
    dzp = [dealias_forward(a, nz) for a in dzm]

    # 4-5. products on the padded grid, then back and truncate
    out = DEV.empty_complex(tuple(Uh.shape[:-2]) + (3, Uh.shape[-1]), Uh)
    for c in range(3):
        Np = up[0]*dxp[c] + up[1]*dyp[c] + up[2]*dzp[c]
        out[..., c, :] = dealias_backward(Np, nz)
    return out


def cfl(U_phys, D, facx, facy, lz, nz, dt):
    """Convective CFL for the explicit stage.

    U_phys is PHYSICAL (..., n, n, >=3).  The (x,y) spacing is taken from the
    minimum GLL node gap, which is the O(1/N^2) spacing near element edges --
    the correct and restrictive measure for a spectral element, not h/N.
    """
    # n from axis 1, NOT shape[-2] -- with the (nelem, n, n, var, mode) layout
    # shape[-2] is NVAR = 7, so the original made the reference spacing
    # independent of the polynomial order, which is how it was spotted (CFL came
    # out identical at N = 6, 8 and 10).  Fourth instance of this layout bug.
    assert U_phys.ndim == 5 and U_phys.shape[1] == U_phys.shape[2], \
        f'expected (nelem, n, n, var, mode), got {U_phys.shape}'
    n = U_phys.shape[1]
    # minimum node spacing on the reference element, mapped by facx/facy
    xg = np.polynomial.legendre.leggauss(max(n - 1, 2))[0]
    dref = np.min(np.diff(np.sort(np.concatenate(([-1.0], xg, [1.0])))))
    dx = dref/np.max(np.abs(facx))
    dy = dref/np.max(np.abs(facy))
    dz = lz/nz
    # [..., f, :] not [..., f]: the field axis is -2 and z is -1.  The wrong
    # form selects a MODE and is silent -- it even returns a plausible number
    # for field 0, which is how it survived its own unit test.  Third instance
    # of this bug in this module; see 3D_DEVELOPMENT_PLAN.md sec 1.1.
    umax = np.abs(U_phys[..., OP.U_, :]).max()
    vmax = np.abs(U_phys[..., OP.V_, :]).max()
    wmax = np.abs(U_phys[..., OP.W_, :]).max()
    return float(dt*(umax/dx + vmax/dy + wmax/dz))


def max_dt_for_cfl(U_phys, D, facx, facy, lz, nz, target):
    """Largest dt meeting `target` (RKW3 is stable to ~sqrt(3); see timestep)."""
    unit = cfl(U_phys, D, facx, facy, lz, nz, 1.0)
    return float('inf') if unit == 0.0 else target/unit
