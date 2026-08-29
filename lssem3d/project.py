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
    if s.get('modepool') is not None:
        uhat, it_u, res_u = s['modepool'].solve('u', k, bu)
    else:
        uhat, it_u, res_u = HH.solve(bu, D, fx, fy, wq, lam_u, nu, m,
                                     s['mask_u'], s['Mu'], tol=s['tol'],
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
    if s.get('consistent_p'):
        # P_N-P_N consistent projection: E = G^T M^{-1} G, weak divergence
        # zeroed identically.  bp/null handling happens inside.
        Uc, phi, it_p, res_p = project_consistent(s, uhat, dt*T.BETA[k])
    elif s.get('dg_pressure'):
        phi, it_p, res_p = _solve_dg(bp, D, fx, fy, wq, kz, m, s['mask_p'],
                                     s['Mp'], s['tol'],
                                     s.get('check_every'))
        phi = _join(phi)
        Uc = uhat - (dt*T.BETA[k])*gradient(phi, D, fx, fy, kz)
    elif s.get('modepool') is not None:
        phi, it_p, res_p = s['modepool'].solve('p', 0, bp)
        phi = _join(phi)
        Uc = uhat - (dt*T.BETA[k])*gradient(phi, D, fx, fy, kz)
    else:
        phi, it_p, res_p = HH.solve(bp, D, fx, fy, wq, kz**2, 1.0, m,
                                    s['mask_p'], s['Mp'], tol=s['tol'],
                                    check_every=s.get('check_every'))
        phi = _join(phi)
        Uc = uhat - (dt*T.BETA[k])*gradient(phi, D, fx, fy, kz)

    # (e) rotational pressure update
    if s.get('incremental', True):
        pc = pc + phi - nu*div
    else:
        pc = phi - nu*div
    # wall value for the NEXT substage's intermediate velocity
    if s.get('wall_u') is not None:
        gp = _split((dt*T.BETA[k])*gradient(phi, D, fx, fy, kz))
        s['ubc'] = gp*s['wall_u']
        if s.get('ubc_in') is not None:
            # static inflow profile rides on top of the wall correction
            s['ubc'] = s['ubc'] + s['ubc_in']
    elif s.get('ubc_in') is not None:
        s['ubc'] = s['ubc_in']
    return Uc, pc, (it_u, res_u, it_p, res_p)


def apply_E(ph, D, fx, fy, wq3, kz, mesh, mask_p, mask_u, Mginv):
    """The CONSISTENT pressure operator  E = G^T M^{-1} G  (P_N-P_N).

    G   : weak gradient, pressure -> velocity space: gs(wq * strong_grad),
          then the velocity mask (essential BCs live in the velocity space).
    M   : assembled diagonal GLL mass, Mginv = 1/gs(wq) per node.
    G^T : its exact adjoint in the multiplicity-weighted inner product --
          the weak divergence gs(Dx^T(wq vx) + Dy^T(wq vy)) - wq ikz^* vz.

    Symmetric PSD BY CONSTRUCTION, which is the gate apply_dg failed (2.8e-2
    asymmetry): strong-div o strong-grad with no mass weighting is NOT an
    adjoint pair on C0 elements.  Solving E phi = (1/dt) G^T uhat and updating
    u = uhat - dt M^{-1} G phi zeroes the WEAK divergence identically -- the
    K-vs-G^T M^{-1} G mismatch that set the divergence floor, destabilised the
    Kim-Moin stage pressure, and grew to 22% in the tripping channel is gone
    in the norm the projection controls.

    ph: split-real (nelem, n, n, 2, nk).  Returns the same shape, masked.
    """
    phc = _join(ph)
    g = gradient(phc, D, fx, fy, kz)                  # strong grad, elementwise
    v = S3.gs(mesh, _split(wq3*g))*mask_u             # G ph, in velocity space
    v = v*Mginv                                       # M^{-1}
    vc = _join(v)
    z = (DV.ddxT(wq3*vc[..., 0:1, :], D, fx)
         + DV.ddyT(wq3*vc[..., 1:2, :], D, fy)
         - 1j*kz*(wq3*vc[..., 2:3, :]))              # (ikz)^* = -ikz
    return S3.gs(mesh, _split(z))*mask_p


def project_consistent(s, uhat_c, dtc):
    """Consistent projection: solve E phi = (1/dtc) G^T uhat, correct uhat.

    Returns (u_corrected_complex, phi_complex, iters, res).  The pressure
    preconditioner s['Mp'] (built for K) is spectrally close enough to E to
    precondition it; iteration counts run ~2x the K solve.
    """
    m = s['m']
    D, fx, fy, kz, wq3 = s['Dg'], s['fxg'], s['fyg'], s['kzg'], s['wq3']
    if 'Mginv' not in s:
        s['Mginv'] = 1.0/S3.gs(m, wq3 + DEV.zeros_like(wq3))
    mask_p, mask_u, Mginv = s['mask_p'], s['mask_u'], s['Mginv']
    # RHS from the VELOCITY-SPACE PROJECTION of uhat.  With the Kim-Moin
    # wall value active, uhat carries prescribed nonzero wall dofs; the
    # correction M^{-1} G phi is masked to the interior space and can never
    # cancel their flux, so G^T of the raw uhat has a component OUTSIDE
    # range(E) and CG diverges (measured: res 4e18 whenever s['ubc'] is set,
    # converges when it is None -- the A/B crash in one line).  Masking uhat
    # first keeps b in range(E) up to the purged constant.
    uv = _join(_split(uhat_c)*mask_u)
    b = (DV.ddxT(wq3*uv[..., 0:1, :], D, fx)
         + DV.ddyT(wq3*uv[..., 1:2, :], D, fy)
         - 1j*kz*(wq3*uv[..., 2:3, :]))
    b = S3.gs(m, _split(b))*mask_p*(1.0/dtc)
    v = s.get('null_kz0')
    if v is not None:
        num = float((b[..., 0:1, 0:1]*v*s['mw1']).sum())
        b = b.copy()
        b[..., 0:1, 0:1] -= (num/s['null_norm'])*v
    A = lambda p: apply_E(p, D, fx, fy, wq3, kz, m, mask_p, mask_u, Mginv)
    # E's null vector is the PURE CONSTANT at kz=0 -- valid only on an
    # UNPINNED pressure mask.  A pinned dof (the K-path convention) rotates
    # the null vector into an unknown direction and CG amplifies it to 1e15.
    # Purge the constant from every preconditioned residual.
    nb = s.get('null_basis_kz0')      # list of (nelem,n,n), mw-orthonormal
    # PURGE AT EVERY REAL FOURIER MODE.  At k=0 the z-gradient is identically
    # zero; at the NYQUIST mode (even nz) it is imaginary and the mask kills
    # it.  Either way E degenerates to the same singular 2-D operator with
    # the same kernel -- the Nyquist lane diverged with PHYSICAL amplitude
    # (bn 2.5e-2) once the cascade filled it, serial and pooled alike.
    real_lanes = s.get('purge_lanes', (0,))
    def purge(z):
        if nb is not None:
            z = z.copy()
            mw0 = s['mw1'][..., 0, 0]
            for kl in real_lanes:
                for q in nb:
                    num = (z[..., 0, kl]*q*mw0).sum()
                    z[..., 0, kl] -= num*q
            return z
        if v is None:
            return z
        num = (z[..., 0:1, 0:1]*v*s['mw1']).sum()
        z = z.copy()
        z[..., 0:1, 0:1] -= (num/s['null_norm'])*v
        return z
    b = purge(b)
    sub = None
    if hasattr(s['Mp'], 'subset'):
        def _mkA(idx):
            kz_a, mp_a, mu_a = kz[idx], mask_p[..., idx], mask_u[..., idx]
            return lambda p_: apply_E(p_, D, fx, fy, wq3, kz_a, m, mp_a,
                                      mu_a, Mginv)
        sub = (_mkA, s['Mp'].subset)
    pool = s.get('modepool')
    if pool is not None and getattr(pool, 'with_e', False):
        ph, it, res = pool.solve('e', 0, b)
    else:
        ph, it, res = _pcg(A, b, s['Mp'], m, s.get('tol_p', s['tol']),
                           s.get('check_every'), purge=purge, subset=sub)
    phc = _join(ph)
    corr = _join(S3.gs(m, _split(wq3*gradient(phc, D, fx, fy, kz)))
                 * mask_u * Mginv)
    return uhat_c - dtc*corr, phc, it, res


def _pcg(A, b, M, mesh, tol, check_every, purge=None, subset=None,
         bmax_global=None):
    """PCG in the multiplicity-weighted inner product; operator passed in.

    `purge`, when given, removes known null-space components from each
    preconditioned residual -- required for singular operators whose
    preconditioner does not respect the null space."""
    xp = DEV.xp(b)
    mw = DEV.to_device(S3.multiplicity_weight(mesh, tuple(b.shape)), b)
    if check_every is None:
        check_every = 1 if xp is np else 10
    nk_ = b.shape[-1]; M_ = b.size//nk_
    if DEV.is_cupy(b):
        ones = S3._ones_row(M_, b)
        dot = lambda a, c: (ones @ (a*c*mw).reshape(M_, nk_)).reshape(-1)
    else:
        dot = lambda a, c: xp.sum(a*c*mw, axis=(0, 1, 2, 3))
    x = DEV.zeros_like(b)
    bn = xp.sqrt(dot(b, b))
    # DEAD-LANE GUARD.  A mode whose RHS is at roundoff (e.g. the Nyquist
    # lane after a kernel purge) must be excluded, not iterated: its CG
    # alpha = rz/den is 0/0 and amplifies roundoff into that lane -- step 1
    # looked sane (|u| unchanged) while the dead lanes carried 1e15 of junk
    # that step 2's convection mixed into everything (1e120).  Zero the lane
    # and mark it converged.
    # a WORKER solving a block of modes must judge deadness against the
    # GLOBAL bmax: its local max can be orders smaller, and roundoff lanes
    # then pass as alive and diverge (measured: 4000-iter zombie solves).
    bmax = xp.max(bn) if bmax_global is None else max(float(xp.max(bn)),
                                                     float(bmax_global))
    # 1e-8: the roundoff floor SCALES WITH PROBLEM SIZE -- 1e-13 failed on
    # the tiny grid (floor ~2e-12) and the recalibrated 1e-10 failed at 88^3
    # (floor 1.06e-10, Nyquist lane amplified to 2e17).  1e-8 is 100x the
    # measured production floor and still ~8 orders below any physical mode;
    # a lane parked dead carries unsolved-pressure error of its own tiny size.
    dead = bn < 1e-8*xp.maximum(bmax, 1e-300)
    if bool(xp.any(dead)):
        b = b*(~dead).astype(b.dtype)
    r = b - A(x)
    z = M(r)
    if purge is not None:
        z = purge(z)
    p = DEV.clone(z)
    rz = dot(r, z)
    target = xp.where(dead, xp.inf, xp.maximum(tol*bn, 1e-300))
    one = xp.ones_like(rz)
    it = 0
    # MODE-ADAPTIVE FREEZING (numpy path, subset callbacks provided).  Same
    # rationale as helmholtz.solve: per-mode systems are independent, and
    # iterating converged (or dead) lanes to the worst lane's count is the
    # max-vs-sum waste.  subset = (makeA, makeM): rebuild operator and
    # preconditioner on an index subset.
    # no freezing when a purge is active: compaction moves lane positions
    # under the purge closure's fixed indices, and a singular lane iterated
    # without its purge diverges.  E-solve counts are small; the loss is minor.
    can_c = (xp is np) and subset is not None and purge is None
    active = None
    xfull = x
    A_a, M_a, tgt, mw_a = A, M, target, mw
    nk_full = b.shape[-1]
    def dot_a(a, c):
        return xp.sum(a*c*mw_a, axis=(0, 1, 2, 3))
    for it in range(1, 4001):
        Ap = A_a(p)
        den = dot_a(p, Ap)
        al = xp.where(abs(den) > 1e-300, rz/xp.where(den == 0, one, den),
                      0.0*one)
        x = x + al*p
        r = r - al*Ap
        if it % check_every == 0 or it == 4000:
            res_v = xp.sqrt(dot_a(r, r))
            conv = res_v < tgt
            if bool(xp.all(conv)):
                break
            if can_c and int(conv.sum()) >= max(2, len(conv)//3):
                keep = np.flatnonzero(~np.asarray(conv))
                cur = (np.arange(nk_full) if active is None else active)
                if active is None:
                    xfull = x.copy()
                else:
                    xfull[..., cur] = x
                active = cur[keep]
                x = xfull[..., active].copy()
                r, p = r[..., keep].copy(), p[..., keep].copy()
                mw_a = mw[..., active]
                A_a = subset[0](active)
                M_a = subset[1](active)
                tgt = target[active]
                rz, one = rz[keep], one[keep]

        z = M_a(r)
        if purge is not None:
            z = purge(z)
        rzn = dot_a(r, z)
        be = xp.where(abs(rz) > 1e-300, rzn/xp.where(rz == 0, one, rz),
                      0.0*one)
        p = z + be*p
        rz = rzn
    if active is not None:
        xfull[..., active] = x
        x = xfull
    rt = xp.sqrt(dot(b - A(x), b - A(x)))
    rel = rt/xp.maximum(bn, 1e-300)
    # LAST-LINE DEFENCE: a diverged lane is zeroed, never returned.  Garbage
    # in one lane poisoned a full run (and its checkpoint) at t=1.197; a
    # zeroed lane merely skips one mode's pressure for one solve.
    blown = rel > 1.0
    if bool(xp.any(blown)):
        for kbad in [int(k) for k in xp.flatnonzero(blown)]:
            print(f'_pcg: lane k={kbad} diverged '
                  f'(bn={float(bn[kbad]):.3e}, rel={float(rel[kbad]):.2e}) '
                  f'-- ZEROED', flush=True)
        x = x*(~blown).astype(x.dtype)
        rel = xp.where(blown, 0.0, rel)
    return x, it, float(xp.max(rel))


def build_masks(mesh, nk, nz, nfield_c, wall=False,
                outflow_p=False):
    """Split-real mask for `nfield_c` complex fields.

    Two constraints, both required and both easy to forget:
      * at a REAL mode (k = 0, and Nyquist when nz is even) the imaginary half
        is unphysical -- irfft discards it -- so it is prescribed, not solved.
      * `wall` freezes the velocity on wall edges (Dirichlet).  The PRESSURE
        correction takes no wall condition at all: homogeneous Neumann is the
        NATURAL boundary condition of the weak form, so imposing nothing is
        imposing dphi/dn = 0.
      * `outflow_p` (edge code 4) is the OPEN boundary of the projection
        path: the VELOCITY stays FREE there -- the weak form's natural
        condition is zero viscous traction -- while the PRESSURE mask is
        zeroed, imposing the homogeneous Dirichlet phi = 0 that anchors the
        pressure.  Together: the classical do-nothing outflow (p = 0,
        nu du/dn = 0).  With an outflow the pressure Poisson is NONSINGULAR;
        callers must disable the null-space projection.
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
    if outflow_p:
        for e in range(mesh.nelem):
            for code, idx in BC._edges(mesh, e):
                if code == OUTFLOW:
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


# Edge code for an OPEN (outflow) boundary: WALLISH (1,2,3) are Dirichlet
# velocity edges; 4 leaves the velocity free and Dirichlets the pressure.
OUTFLOW = 4

# Jameson's four-stage RK, the convection integrator of the Kim-Moin scheme.
JAMESON = (0.25, 1.0/3.0, 0.5, 1.0)


def step_kim_moin(s, Uc, phi_prev, dt, pc=None, skew=True):
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
    # LAGGED PRESSURE IN THE STAGES.  With convection alone, the sweep drifts
    # off the divergence-free manifold by dt*grad(p); the skew form conserves
    # TOTAL energy, so the energy in that gradient mode -- (1/2) dt^2 |grad p|^2
    # per step, 23.6% of 2*nu*Omega at TGV Re=800, dt=0.00567 -- is skimmed
    # from the physical field, and the projection can only discard it
    # (measured: predicted 0.2365 vs 0.2362, halves exactly with dt).
    # Subtracting the LAGGED pressure gradient inside each stage is the
    # interior analogue of the Kim-Moin wall correction (2.10): the surviving
    # drift is dt*grad(p - p_lag) = O(dt^2), and the loss goes as its SQUARE.
    # pc then accumulates the projection increments: pc^n = pc^{n-1} + phi^n.
    from . import convect as CV
    # FRESH PRESSURE, NOT ACCUMULATED.  pc = pc + phi has no cleaning
    # mechanism: the projection's divergence floor leaks a little high-k junk
    # into phi every step, the accumulator keeps all of it, and the stages
    # differentiate it back into the momentum equation -- a positive feedback
    # that blew up at t = 5.11 (Om 657 -> 686 in 10 steps, balance 0.68,
    # energy CREATED) after ~900 steps.  Solving grad^2 p = div N(u^n) fresh
    # each step is memoryless and bounded by construction; the source reuses
    # stage 1's convective evaluation, so the extra cost is one Poisson solve.
    H0 = -CV.convective(un, D, fx, fy, kz, s['nz'], skew=skew)
    gp = None
    if s.get('consistent_p') and pc is not None:
        # E-CONSISTENT STAGE PRESSURE.  Solve E p = G^T N(u^n) and force the
        # stages with the WEAK gradient M^{-1} G p -- then the sweep's weak
        # divergence production cancels EXACTLY:  G^T(N - M^{-1}G p) =
        # G^T N - E p = 0.  The strong-gradient variant left the K-vs-DG
        # mismatch at full-dt amplitude and destabilised at t = 5.1.
        if 'Mginv' not in s:
            s['Mginv'] = 1.0/S3.gs(m, s['wq3'] + DEV.zeros_like(s['wq3']))
        bfp = (DV.ddxT(s['wq3']*H0[..., 0:1, :], D, fx)
               + DV.ddyT(s['wq3']*H0[..., 1:2, :], D, fy)
               - 1j*kz*(s['wq3']*H0[..., 2:3, :]))
        bfp = S3.gs(m, _split(bfp))*s['mask_p']
        v = s.get('null_kz0')
        if v is not None:
            num = float((bfp[..., 0:1, 0:1]*v*s['mw1']).sum())
            bfp = bfp.copy()
            bfp[..., 0:1, 0:1] -= (num/s['null_norm'])*v
        A = lambda p_: apply_E(p_, D, fx, fy, s['wq3'], kz, m, s['mask_p'],
                               s['mask_u'], s['Mginv'])
        nb = s.get('null_basis_kz0')
        real_lanes = s.get('purge_lanes', (0,))
        def _purge(z):
            if nb is not None:
                z = z.copy()
                mw0 = s['mw1'][..., 0, 0]
                for kl in real_lanes:
                    for q in nb:
                        nz_ = (z[..., 0, kl]*q*mw0).sum()
                        z[..., 0, kl] -= nz_*q
                return z
            if v is None:
                return z
            nz_ = (z[..., 0:1, 0:1]*v*s['mw1']).sum()
            z = z.copy()
            z[..., 0:1, 0:1] -= (nz_/s['null_norm'])*v
            return z
        bfp = _purge(bfp)
        subst = None
        if hasattr(s['Mp'], 'subset'):
            def _mkA2(idx):
                kz_a = kz[idx]
                mp_a = s['mask_p'][..., idx]
                mu_a = s['mask_u'][..., idx]
                return lambda p_: apply_E(p_, D, fx, fy, s['wq3'], kz_a, m,
                                          mp_a, mu_a, s['Mginv'])
            subst = (_mkA2, s['Mp'].subset)
        pool = s.get('modepool')
        if pool is not None and getattr(pool, 'with_e', False):
            pn, it_fp, res_fp = pool.solve('e', 0, bfp)
        else:
            pn, it_fp, res_fp = _pcg(A, bfp, s['Mp'], m,
                                     s.get('tol_p', s['tol']),
                                     s.get('check_every'), purge=_purge,
                                     subset=subst)
        s['_dbg_stage_p'] = (it_fp, res_fp)
        pc = _join(pn)
        gp = _join(S3.gs(m, _split(s['wq3']*gradient(pc, D, fx, fy, kz)))
                   * s['mask_u'] * s['Mginv'])
    elif pc is not None:
        bfp = -S3.gs(m, s['wq1']*_split(divergence(H0, D, fx, fy, kz)))              * s['mask_p']
        v = s.get('null_kz0')
        if v is not None:
            num = float((bfp[..., 0:1, 0:1]*v*s['mw1']).sum())
            bfp = bfp.copy()
            bfp[..., 0:1, 0:1] -= (num/s['null_norm'])*v
        pn, it_fp, _ = HH.solve(bfp, D, fx, fy, wq, kz**2, 1.0, m,
                                s['mask_p'], s['Mp'], tol=s['tol'],
                                check_every=s.get('check_every'))
        pc = _join(pn)
        gp = gradient(pc, D, fx, fy, kz)
    for ist, a in enumerate(JAMESON):
        # SKEW-SYMMETRIC: Kim & Moin, "following Horiuti's recommendation
        # ... to control aliasing errors"; x-y is not dealiased here, and the
        # advective form killed the substage TGV run at t = 9.32.
        H = H0 if ist == 0 else -CV.convective(u, D, fx, fy, kz, s['nz'],
                                               skew=skew)
        if gp is not None:
            H = H - gp
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
    # (No pressure term here: the lagged gradient is applied inside the
    # Jameson stages above, where the drift it must cancel is produced.
    # A CN-side pressure term was measured to recover nothing -- the energy
    # is skimmed during the sweep, before this solve runs.)
    bu = S3.gs(m, _split(r))
    ubc = None
    if s.get('wall_u') is not None and phi_prev is not None:
        # Kim-Moin: uhat carries the slip the projection is about to remove,
        # over the FULL step
        ubc = _split(dt*gradient(phi_prev, D, fx, fy, kz))*s['wall_u']
        bu = bu - HH.apply(ubc, D, fx, fy, wq, lam, nu, m, None)
    bu = bu*s['mask_u']
    pool = s.get('modepool')
    if pool is not None and getattr(pool, 'with_e', False):
        ustar, it_u, res_u = pool.solve('u', 0, bu)
    else:
        ustar, it_u, res_u = HH.solve(bu, D, fx, fy, wq, lam, nu, m,
                                      s['mask_u'], s['Mu'], tol=s['tol'],
                                      check_every=s.get('check_every'))
    if ubc is not None:
        ustar = ustar + ubc
    ustar = _join(ustar)

    # 3. projection
    if s.get('consistent_p'):
        Uc2, phi, it_p, res_p = project_consistent(s, ustar, dt)
        return Uc2, phi, (it_u, res_u, it_p, res_p), pc
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
    # pc carries this step's freshly solved pressure out for diagnostics
    # only; nothing accumulates.
    return Uc, phi, (it_u, res_u, it_p, res_p), pc
