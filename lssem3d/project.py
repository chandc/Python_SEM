"""Fractional-step (projection) substeps — FRACTIONAL_STEP_PLAN.md sec 1.2.

Incremental, ROTATIONAL pressure correction, applied once per RKW3/CN substage
so the time scheme is unchanged from the least-squares path.  Per substage k
with c_k = 1/(beta_k*dt) = timestep.implicit_coeff(dt, k):

    r        = u^{k-1} + dt*[gamma_k N^{k-1} + zeta_k N^{k-2}
                             + alpha_k nu grad^2 u^{k-1}] - beta_k dt grad p^{k-1}
    (c_k - nu grad^2) uhat = c_k r
    grad^2 phi             = c_k div uhat
    u^k      = uhat - beta_k dt grad phi
    p^k      = p^{k-1} + phi - nu div uhat

THE LAST TERM IS NOT DECORATION.  Without -nu*div(uhat) the scheme is only
O(dt) in pressure and carries a numerical boundary layer at walls; with it,
O(dt^2) in both.  It is the difference between matching the LS path's verified
order 2.00 and not, which Gate 1 measures directly.

STATE LAYOUT.  Velocity and pressure are held as COMPLEX arrays
(nelem, n, n, F, nk) with F = 3 and 1, because the divergence and gradient mix
real and imaginary parts through i*k_z.  The Helmholtz solves are real, so they
run on the split-real view and the halves are solved together.
"""
import numpy as np

from . import device as DEV
from . import deriv as DV
from . import helmholtz as HH
from . import solver3d as S3


def _split(Uc):
    """complex (..., F, nk) -> real (..., 2F, nk), real half then imaginary."""
    xp = DEV.xp(Uc)
    return xp.concatenate([Uc.real, Uc.imag], axis=-2)


def _join(Ur):
    F = Ur.shape[-2]//2
    return Ur[..., :F, :] + 1j*Ur[..., F:, :]


def divergence(Uc, D, facx, facy, kz):
    """div u for complex velocity (nelem, n, n, 3, nk) -> (..., 1, nk)."""
    return (DV.ddx(Uc[..., 0:1, :], D, facx)
            + DV.ddy(Uc[..., 1:2, :], D, facy)
            + 1j*kz*Uc[..., 2:3, :])


def gradient(pc, D, facx, facy, kz):
    """grad p for complex (nelem, n, n, 1, nk) -> (..., 3, nk)."""
    xp = DEV.xp(pc)
    return xp.concatenate([DV.ddx(pc, D, facx),
                           DV.ddy(pc, D, facy),
                           1j*kz*pc], axis=-2)


def laplacian_rhs(Uc, D, facx, facy, kz, nu):
    """nu*grad^2 u, strong form, for the explicit viscous term."""
    return nu*(DV.ddx(DV.ddx(Uc, D, facx), D, facx)
               + DV.ddy(DV.ddy(Uc, D, facy), D, facy)
               - (kz**2)*Uc)


def substage(s, Uc, pc, Nk, Nprev, k, dt):
    """One RKW3/CN substage.  Returns (u^k, p^k, iterations)."""
    from . import timestep as T
    ck = T.implicit_coeff(dt, k)
    nu, kz = s['nu'], s['kz']
    D, fx, fy, wq, m = s['D'], s['m'].facx, s['m'].facy, s['m'].wq, s['m']

    # (a) explicit assembly
    r = Uc + dt*(T.GAMMA[k]*Nk + T.ZETA[k]*Nprev
                 + T.ALPHA[k]*laplacian_rhs(Uc, D, fx, fy, kz, nu))
    r = r - (dt*T.BETA[k])*gradient(pc, D, fx, fy, kz)

    # (b) velocity Helmholtz, weak form: (c_k + nu kz^2) M + nu K
    lam_u = ck + nu*(kz**2)
    bu = s['wq3']*_split(ck*r)
    bu = S3.gs(m, bu)*s['mask_u']
    uhat, it_u = HH.solve(bu, D, fx, fy, wq, lam_u, nu, m, s['mask_u'],
                          s['Mu'], tol=s['tol'])
    uhat = _join(uhat)

    # (c) pressure Poisson: kz^2 M + K, natural (Neumann) walls, pinned at kz=0
    div = divergence(uhat, D, fx, fy, kz)
    bp = -S3.gs(m, s['wq1']*_split(ck*div))*s['mask_p']
    phi, it_p = HH.solve(bp, D, fx, fy, wq, kz**2, 1.0, m, s['mask_p'],
                         s['Mp'], tol=s['tol'])
    phi = _join(phi)

    # (d) projection and (e) rotational pressure update
    Uc = uhat - (dt*T.BETA[k])*gradient(phi, D, fx, fy, kz)
    pc = pc + phi - nu*div
    return Uc, pc, it_u + it_p


def build_masks(mesh, nk, nz, nfield_c, wall=False):
    """Split-real mask for `nfield_c` complex fields.

    Two constraints, both required and both easy to forget:
      * at a REAL mode (k = 0, and Nyquist when nz is even) the imaginary half
        is unphysical -- irfft discards it -- so it is prescribed, not solved.
      * `wall` freezes the velocity on wall edges (Dirichlet).  The PRESSURE
        correction takes no wall condition at all: homogeneous Neumann is the
        NATURAL boundary condition of the weak form, so imposing nothing is
        imposing dphi/dn = 0.
    """
    from . import bc as BC
    from .bc import real_mode_columns
    n = mesh.N + 1
    F = 2*nfield_c
    mask = np.ones((mesh.nelem, n, n, F, nk))
    for k in real_mode_columns(nk, nz):
        mask[..., nfield_c:, k] = 0.0
    if wall:
        for e in range(mesh.nelem):
            for code, idx in BC._edges(mesh, e):
                if code in BC.WALLISH:
                    mask[idx] = 0.0
    return mask
