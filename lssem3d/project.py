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


def visc_weak(Uc, D, facx, facy, wq, kz, nu):
    """nu*(K + kz^2 M) u -- the WEAK viscous operator, UNASSEMBLED.

    This must be the SAME operator the implicit side inverts.  Crank-Nicolson
    only cancels if the explicit and implicit halves are the same discrete
    operator; using a strong-form nu*grad^2 (two pointwise differentiations)
    against a weak implicit K leaves a residue that does NOT vanish with dt and
    accumulates every step.  Measured: Gate 1 came out at order -2.08, the
    error GROWING as dt shrank, which is the signature of exactly that.
    """
    return HH.apply(Uc, D, facx, facy, wq, nu*kz**2, nu, mesh=None, mask=None)


def _solve_dg(b, D, fx, fy, wq, kz, mesh, mask, M, tol, check_every):
    """PCG against the D.G operator.  Same recurrence as helmholtz.solve; only
    the operator differs, so it is written out rather than parameterised."""
    xp = DEV.xp(b)
    mw = DEV.to_device(S3.multiplicity_weight(mesh, tuple(b.shape)), b)
    A = lambda v: HH.apply_dg(v, D, fx, fy, wq, kz, mesh, mask)
    if check_every is None:
        check_every = 1 if xp is np else 10
    nk_ = b.shape[-1]
    M_ = b.size//nk_
    if DEV.is_cupy(b):
        ones = S3._ones_row(M_, b)
        dot = lambda a, c: (ones @ (a*c*mw).reshape(M_, nk_)).reshape(-1)
    else:
        dot = lambda a, c: xp.sum(a*c*mw, axis=(0, 1, 2, 3))
    x = DEV.zeros_like(b)
    r = b - A(x)
    z = M(r)
    p = DEV.clone(z)
    rz = dot(r, z)
    bn = xp.sqrt(dot(b, b))
    target = xp.maximum(tol*bn, 1e-300)
    one = xp.ones_like(rz)
    it = 0
    for it in range(1, 4001):
        Ap = A(p)
        den = dot(p, Ap)
        al = xp.where(abs(den) > 1e-300, rz/xp.where(den == 0, one, den),
                      0.0*one)
        x = x + al*p
        r = r - al*Ap
        if it % check_every == 0 or it == 4000:
            if bool(xp.all(xp.sqrt(dot(r, r)) < target)):
                break
        z = M(r)
        rzn = dot(r, z)
        be = xp.where(abs(rz) > 1e-300, rzn/xp.where(rz == 0, one, rz), 0.0*one)
        p = z + be*p
        rz = rzn
    rt = xp.sqrt(dot(b - A(x), b - A(x)))
    return x, it, float(xp.max(rt/xp.maximum(bn, 1e-300)))


def substage(s, Uc, pc, Nk, Nprev, k, dt):
    """One RKW3/CN substage.  Returns (u^k, p^k, iterations)."""
    from . import timestep as T
    ck = T.implicit_coeff(dt, k)
    nu, m = s['nu'], s['m']
    # DEVICE ARRAYS, not the mesh's host copies.  s['m'].facx and friends are
    # numpy whatever the backend; pulling them here would hand host arrays to a
    # cupy kernel, which fails with a bare "Unsupported type numpy.ndarray"
    # from inside an ElementwiseKernel -- the same trap PMG falls into.
    D, fx, fy, wq, kz = s['Dg'], s['fxg'], s['fyg'], s['wqg'], s['kzg']

    # (a) explicit assembly, ENTIRELY IN WEAK FORM so the CN halves match.
    #     Multiply the momentum equation by M throughout: the mass and
    #     convective terms pick up wq, the viscous term is the same weak
    #     operator the implicit side inverts, and alpha_k*c_k*dt = alpha_k/beta_k.
    w = s['wq3']
    r = w*(ck*(Uc + dt*(T.GAMMA[k]*Nk + T.ZETA[k]*Nprev)))
    r = r - (T.ALPHA[k]/T.BETA[k])*visc_weak(Uc, D, fx, fy, wq, kz, nu)
    # INCREMENTAL vs NON-INCREMENTAL.  The incremental form carries p forward
    # and re-enters it here; the pressure-free form drops both, trading second
    # order in pressure for the removal of that feedback loop.  If a growing
    # mode is the incremental pressure feeding back through an inconsistent
    # wall condition, it disappears when this term does.
    if s.get('incremental', True):
        r = r - w*gradient(pc, D, fx, fy, kz)

    # (b) velocity Helmholtz, weak form: (c_k + nu kz^2) M + nu K
    #
    # KIM-MOIN WALL CORRECTION.  Imposing uhat = 0 at a wall is what leaves the
    # tangential slip -beta_k*dt*dphi/dt in u^k -- the classic O(dt) boundary
    # layer, and the inconsistency the incremental pressure loop amplifies into
    # the instability measured on Gate 1 (sigma 9.316 -> 9.944 over 40 steps).
    # But the slip is KNOWN: the projection subtracts beta_k*dt*grad(phi), so
    # prescribing uhat|wall = beta_k*dt*grad(phi) makes u^k = 0 there exactly.
    # phi is lagged by one substage, which is what makes this explicit and
    # cheap.
    lam_u = ck + nu*(kz**2)
    bu = S3.gs(m, _split(r))
    ubc = s.get('ubc')
    if ubc is not None:
        # inhomogeneous Dirichlet by lifting: solve for the correction with
        # homogeneous BCs against a right-hand side that already carries A*ubc
        bu = bu - HH.apply(ubc, D, fx, fy, wq, lam_u, nu, m, None)
    bu = bu*s['mask_u']
    uhat, it_u, res_u = HH.solve(bu, D, fx, fy, wq, lam_u, nu, m, s['mask_u'],
                                 s['Mu'], tol=s['tol'],
                                 check_every=s.get('check_every'))
    if ubc is not None:
        uhat = uhat + ubc
    uhat = _join(uhat)

    # (c) pressure Poisson: kz^2 M + K, natural (Neumann) walls, pinned at kz=0
    div = divergence(uhat, D, fx, fy, kz)
    bp = -S3.gs(m, s['wq1']*_split(ck*div))*s['mask_p']
    # COMPATIBILITY AT kz = 0.  That mode has all-Neumann walls and periodic x,
    # so its operator is singular with the constant in its null space, and
    # A x = b is solvable only if <1, b> = 0.  Analytically it is -- the
    # integral of div(uhat) is the flux through a closed boundary where
    # uhat = 0 -- but discretely it is not, and PINNING a dof does not fix an
    # incompatible right-hand side: it returns the least-squares answer to an
    # inconsistent system, with the defect dumped near the pinned node and
    # accumulated into p every substage.  Project it out instead.
    v = s['null_kz0']
    if v is not None:
        num = float((bp[..., 0:1, 0:1]*v*s['mw1']).sum())
        bp = bp.copy()
        bp[..., 0:1, 0:1] -= (num/s['null_norm'])*v
    # CONSISTENT PROJECTION when asked for: invert D.G, the same operators the
    # update uses, so the divergence cancels exactly rather than only weakly.
    if s.get('dg_pressure'):
        phi, it_p, res_p = _solve_dg(bp, D, fx, fy, wq, kz, m, s['mask_p'],
                                     s['Mp'], s['tol'],
                                     s.get('check_every'))
    else:
        phi, it_p, res_p = HH.solve(bp, D, fx, fy, wq, kz**2, 1.0, m,
                                    s['mask_p'], s['Mp'], tol=s['tol'],
                                    check_every=s.get('check_every'))
    phi = _join(phi)

    # (d) projection and (e) rotational pressure update
    Uc = uhat - (dt*T.BETA[k])*gradient(phi, D, fx, fy, kz)
    if s.get('incremental', True):
        pc = pc + phi - nu*div
    else:
        pc = phi - nu*div
    # wall value for the NEXT substage's intermediate velocity
    if s.get('wall_u') is not None:
        gp = _split((dt*T.BETA[k])*gradient(phi, D, fx, fy, kz))
        s['ubc'] = gp*s['wall_u']
    return Uc, pc, (it_u, res_u, it_p, res_p)


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


def wall_indicator(mesh, nk, nz, nfield_c):
    """1 on wall velocity dofs, 0 elsewhere -- where uhat is PRESCRIBED.

    Not simply 1 - mask: the mask also zeroes the imaginary half at real modes,
    which is a different constraint and must stay at zero, not be given a wall
    value.
    """
    from . import bc as BC
    from .bc import real_mode_columns
    n = mesh.N + 1
    F = 2*nfield_c
    ind = np.zeros((mesh.nelem, n, n, F, nk))
    for e in range(mesh.nelem):
        for code, idx in BC._edges(mesh, e):
            if code in BC.WALLISH:
                ind[idx] = 1.0
    for k in real_mode_columns(nk, nz):
        ind[..., nfield_c:, k] = 0.0
    return ind


# Jameson's four-stage RK, the convection integrator of the Kim-Moin scheme.
JAMESON = (0.25, 1.0/3.0, 0.5, 1.0)


def step_kim_moin(s, Uc, phi_prev, dt, pc=None):
    """One FULL step: RK convection -> one CN viscous solve -> one projection.

    THE STRUCTURE IS THE POINT.  substage() above projects inside every RKW3
    substage with beta_k*dt, and measures ~1.6 order at walls.  The Kim-Moin
    correction it uses,

        uhat|wall = u^{n+1}|wall + dt * grad(phi^{n-1})|wall,

    is an extrapolation IN TIME OVER A UNIFORM STEP.  The SMR weights are
    beta = (0.2315, 0.2083, 0.1667) and sum to 0.606, not 1, so applied per
    substage the correction is scaled to the wrong interval -- right in form,
    wrong in magnitude.  Kim & Moin apply it once per step, over the whole dt,
    and report second order.

    So this is the reference's own sequence:
      1. Jameson's four-stage RK advances CONVECTION ONLY, no pressure;
      2. ONE Crank-Nicolson viscous solve,
             (2/dt) M u* + Avisc u*  =  (2/dt) M u^p - Avisc u^n,
         with the Kim-Moin wall value on u*;
      3. ONE projection, grad^2 phi = div(u*)/dt, u^{n+1} = u* - dt grad(phi).

    Pressure-free: phi is an auxiliary variable, not the physical pressure, and
    is carried only to supply the next step's wall value.  The incremental form
    was measured UNSTABLE at walls (sigma 9.316 -> 9.944 over 40 steps).
    """
    from . import timestep as T
    m, nu = s['m'], s['nu']
    D, fx, fy, wq, kz = s['Dg'], s['fxg'], s['fyg'], s['wqg'], s['kzg']
    xp = DEV.xp(Uc)

    # 1. convection only, low-storage: each stage evaluates H at the previous
    #    stage and adds to u^n
    un = Uc
    u = un
    for a in JAMESON:
        from . import convect as CV
        H = -CV.convective(u, D, fx, fy, kz, s['nz'])
        if s.get('force') is not None:
            H = H + s['force']
        u = un + (dt*a)*H
    up = u

    # 2. Crank-Nicolson viscous, ONE solve over the whole dt
    lam = 2.0/dt + nu*(kz**2)
    r = s['wq3']*((2.0/dt)*up) - visc_weak(un, D, fx, fy, wq, kz, nu)
    # INCREMENTAL PRESSURE, when a pressure is carried.  The pressure-free form
    # is FIRST order, and with convection active that shows as first-order
    # excess dissipation: the TGV balance -dE/dt / 2nuOmega measured 1.2302,
    # 1.1153, 1.0578 as dt halved -- clean O(dt), against a least-squares
    # reference holding 0.999.  Feeding grad(p) back makes it second order.
    #
    # This is safe here and was NOT safe per-substage at walls: the loop that
    # ran sigma to 9.944 amplified an inconsistent WALL pressure condition, and
    # a periodic domain has none.  Pass pc=None to keep the pressure-free form.
    if pc is not None:
        r = r - s['wq3']*gradient(pc, D, fx, fy, kz)
    bu = S3.gs(m, _split(r))
    ubc = None
    if s.get('wall_u') is not None and phi_prev is not None:
        # Kim-Moin: uhat carries the slip the projection is about to remove,
        # over the FULL step
        ubc = _split(dt*gradient(phi_prev, D, fx, fy, kz))*s['wall_u']
        bu = bu - HH.apply(ubc, D, fx, fy, wq, lam, nu, m, None)
    bu = bu*s['mask_u']
    ustar, it_u, res_u = HH.solve(bu, D, fx, fy, wq, lam, nu, m, s['mask_u'],
                                  s['Mu'], tol=s['tol'],
                                  check_every=s.get('check_every'))
    if ubc is not None:
        ustar = ustar + ubc
    ustar = _join(ustar)

    # 3. projection
    div = divergence(ustar, D, fx, fy, kz)
    bp = -S3.gs(m, s['wq1']*_split(div/dt))*s['mask_p']
    v = s.get('null_kz0')
    if v is not None:
        num = float((bp[..., 0:1, 0:1]*v*s['mw1']).sum())
        bp = bp.copy()
        bp[..., 0:1, 0:1] -= (num/s['null_norm'])*v
    # CONSISTENT PROJECTION when asked for: invert D.G, the same operators the
    # update uses, so the divergence cancels exactly rather than only weakly.
    if s.get('dg_pressure'):
        phi, it_p, res_p = _solve_dg(bp, D, fx, fy, wq, kz, m, s['mask_p'],
                                     s['Mp'], s['tol'],
                                     s.get('check_every'))
    else:
        phi, it_p, res_p = HH.solve(bp, D, fx, fy, wq, kz**2, 1.0, m,
                                    s['mask_p'], s['Mp'], tol=s['tol'],
                                    check_every=s.get('check_every'))
    phi = _join(phi)
    Uc = ustar - dt*gradient(phi, D, fx, fy, kz)
    if pc is not None:
        pc = pc + phi - nu*div          # rotational form
    return Uc, phi, (it_u, res_u, it_p, res_p), pc
