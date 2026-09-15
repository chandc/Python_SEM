"""3D Stokes decay: an EXACT unsteady solution to measure temporal accuracy on.

Rig only; `stokes3d_run.py` drives it.

WHY THIS REPLACES THE RICHARDSON TEST.  Measuring temporal order by
self-convergence has two weaknesses, and this case removes both:

  * it needs no reference solution -- but it can only measure the rate at which a
    scheme approaches ITS OWN limit, never whether that limit is right.  A scheme
    converging beautifully to the wrong equations looks perfect.
  * Stokes decay has an ANALYTIC rate, so the error is ABSOLUTE.  Both the rate
    (how fast) and the destination (toward what) fall out of one measurement.

It is also already validated in 2D against Chan (1996) Fig. 1
(`figs/chan_fig1_pref.png`, sigma = 9.313955 measured vs 9.313740 analytic), so a
disagreement here is a 3D defect rather than a new and unvetted benchmark.

AND IT IS EXACTLY THE REGIME THE AC QUESTION LIVES IN: unsteady, with a
decaying pressure, and -- because Stokes drops convection entirely -- perfectly
LINEAR.  The 2D harness always carried u.grad u and had to keep the amplitude
small to approximate the Stokes limit; here convection is switched off, so the
amplitude is irrelevant and the reference rate is exact rather than approached.

THE MODES.  For horizontal dependence e^{i(alpha x + k_z z)} the Laplacian sees
only k^2 = alpha^2 + k_z^2, so the wall-normal eigenproblem is the 2D one with
`a` -> k, and

    sigma = nu * (k^2 + beta^2),     beta = slowest root of the no-slip system

Two cases, deliberately:

  KZ0  alpha = 1, k_z = 0  -- the 2D mode embedded in the 3D code.  sigma =
       9.313740, the Chan figure's number, so this cross-checks against both the
       analytic rate AND the 2D solver.
  SPAN alpha = 0, k_z != 0 -- no x-dependence at all.  Only v, w and omega_x are
       non-zero, i.e. exactly the components every k_z = 0 test is blind to, and
       every i*k_z term in the operator is exercised.
"""
import numpy as np

from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem3d import (operator as OP, solver3d as S3, bc as BC, fourier as FR,
                     timestep as T, parallel as PAR, deriv as DV)
from stokes_ic import slowest_mode, f_and_derivs

LX, LY = 2.0*np.pi, 2.0        # y in [0,2] on the mesh, shifted to [-1,1]
NU = 1.0
SIGMA_2D = 9.3137399           # alpha=1, k_z=0; Chan reports 9.313316


def setup(N=8, ex=2, ey=4, nz=8, lz=2.0*np.pi):
    m = build_channel(LX, LY, ex, ey, N, bcs=(0, 0, 1, 1))
    m.periodic_x = LX
    m.compute_global_indices()
    nk = nz//2 + 1
    mask = BC.build_mask(m, nk, pin_p=False, nz=nz)
    BC.pin_dof(m, mask, OP.P_, 0)            # pressure constant, k_z = 0 only
    BC.pin_dof(m, mask, OP.NVAR + OP.P_, 0)  # ALL copies (periodic seam)
    n = N+1
    X = np.empty((m.nelem, n, n)); Y = np.empty_like(X)
    for e in range(m.nelem):
        X[e] = m.xnod[e][:, None]
        Y[e] = m.ynod[e][None, :] - 1.0      # shift to the half-height-1 domain
    return dict(m=m, D=diff_matrix(N), N=N, nz=nz, nk=nk, lz=lz, nu=NU,
                kz=FR.wavenumbers(nz, lz), mask=mask, X=X, Y=Y)


def sigma_exact(alpha, kzval):
    """nu*(k^2 + beta^2) with beta from the no-slip eigenproblem at k."""
    k = np.hypot(alpha, kzval)
    b1, c, _ = slowest_mode(a=k)
    return NU*(k*k + b1*b1), b1, c, k


def initial_state(s, mode='kz0', kmode=1, amp=1.0e-3):
    """Exact Stokes eigenmode.  Convection is off, so amp only sets the scale."""
    m, N, nz, nk = s['m'], s['N'], s['nz'], s['nk']
    X, Y = s['X'], s['Y']
    Uc = np.zeros((m.nelem, N+1, N+1, OP.NVAR, nk), dtype=complex)

    if mode == 'kz0':
        alpha, kzv, col = 1.0, 0.0, 0
    else:
        alpha, kzv, col = 0.0, float(s['kz'][kmode]), kmode
    sig, b1, c, k = sigma_exact(alpha, kzv)
    f, f1, f2 = f_and_derivs(Y, b1, c, a=k)

    A, B, P, Q = c
    if mode == 'kz0':
        # psi = f(y) cos(alpha x):  u = f' cos, v = alpha f sin, om_z = (a^2 f - f'') cos
        Uc[..., OP.U_, 0] = (f1*np.cos(alpha*X))*nz
        Uc[..., OP.V_, 0] = (alpha*f*np.sin(alpha*X))*nz
        Uc[..., OP.OZ_, 0] = ((alpha*alpha*f - f2)*np.cos(alpha*X))*nz
        # PRESSURE IS NOT ZERO.  From -sigma u = -p_x + nu grad^2 u the
        # hyperbolic terms cancel and p = sigma*(A sinh + B cosh)*sin(alpha x).
        # Starting from p = 0 leaves AC to establish the pressure at the rate
        # div u / kappa_p per step, which for kappa_p = a_mass is far too slow
        # over half a decay time -- it was worth 12.5% in sigma.
        Uc[..., OP.P_, 0] = (sig*(A*np.sinh(k*Y) + B*np.cosh(k*Y))
                             * np.sin(alpha*X))*nz
    else:
        # same construction with z in place of x:
        #   w = f' cos(k z),  v = k f sin(k z),  om_x = (f'' - k^2 f) cos(k z)
        # cos(kz) -> coefficient nz/2 ; sin(kz) -> -i nz/2
        c_cos, c_sin = 0.5*nz, -0.5j*nz
        Uc[..., OP.W_, col] = f1*c_cos
        Uc[..., OP.V_, col] = (k*f)*c_sin
        Uc[..., OP.OX_, col] = (f2 - k*k*f)*c_cos
        Uc[..., OP.P_, col] = (sig*(A*np.sinh(k*Y) + B*np.cosh(k*Y)))*c_sin

    U = np.concatenate([Uc.real, Uc.imag], axis=-2)
    scale = amp/max(np.abs(U).max(), 1e-300)
    return U*scale, dict(sigma=sig, beta=b1, k=k, alpha=alpha, kz=kzv)


def energy(s, U):
    """E = 1/2 integral (u^2+v^2+w^2) dV, quadrature in (x,y), uniform in z."""
    Up = FR.to_physical(OP.to_complex(U), s['nz'])
    e = sum(Up[..., f, :]**2 for f in (OP.U_, OP.V_, OP.W_))
    wq = s['m'].wq[..., None]
    return 0.5*float(np.sum(e*wq))*(s['lz']/s['nz'])


def stage(s, U, k, dt, kap, nsub=1, Minv=None, tol=1e-12, max_iter=40000,
          sub_tol=1e-13, workers=None, rw=None):
    """One RKW3/CN stage of the STOKES problem -- no convection, no body force.

    Dropping u.grad u makes the problem exactly linear, so the analytic decay
    rate is exact rather than an amplitude-dependent approximation.
    """
    m, D, kz, mask, nu = s['m'], s['D'], s['kz'], s['mask'], s['nu']
    c = T.implicit_coeff(dt, k)
    Uc = OP.to_complex(U)
    R0 = OP.apply_L0_complex(Uc, D, m.facx, m.facy, kz, nu, 0.0, kap)
    Lk = -R0[..., 4:7, :]

    fc = np.zeros(Uc.shape[:-2] + (OP.NROW, Uc.shape[-1]), dtype=complex)
    for row, fld in ((4, OP.U_), (5, OP.V_), (6, OP.W_)):
        fc[..., row, :] = c*(Uc[..., fld, :]
                             + dt*T.ALPHA[k]*Lk[..., row-4, :])
    wqR = m.wq[..., None, None]
    # f must carry the SAME row weighting as the operator, or the defect
    # correction solves a different problem than apply_L represents.
    rwR = (1.0 if rw is None
           else np.asarray(rw).reshape((1,)*(len(fc.shape)-2) + (len(rw), 1)))
    p_prev = Uc[..., OP.P_, :]
    Uit, its = U, 0
    for _ in range(max(1, nsub)):
        fc[..., 0, :] = kap*p_prev
        f = np.concatenate([fc.real, fc.imag], axis=-2)
        r = OP.apply_LT(
            OP.apply_L(Uit, D, m.facx, m.facy, kz, nu, c, m.wq, kap, rw)
            - f*wqR*np.concatenate([rwR, rwR], axis=-2) if rw is not None else
            OP.apply_L(Uit, D, m.facx, m.facy, kz, nu, c, m.wq, kap) - f*wqR,
            D, m.facx, m.facy, kz, nu, c, kap)
        b = -S3.gs(m, r)*mask
        dU, it, _ = PAR.pcg(b, D, m.facx, m.facy, kz, nu, c, mesh=m, mask=mask,
                            M_inv=None if Minv is None else Minv[k], tol=tol,
                            max_iter=max_iter, wq=m.wq, kap=kap,
                            workers=workers, rw=rw)
        Uit = Uit + dU
        its += it
        p_new = OP.to_complex(Uit)[..., OP.P_, :]
        dp = np.abs(p_new - p_prev).max()/max(np.abs(p_new).max(), 1e-30)
        p_prev = p_new
        if kap == 0.0 or dp < sub_tol:
            break
    return Uit, its


def step(s, U, dt, kap, rowweight=False, **kw):
    tot = 0
    for k in range(T.NSTAGE):
        rw = OP.momentum_row_weights(T.implicit_coeff(dt, k)) if rowweight else None
        U, it = stage(s, U, k, dt, kap, rw=rw, **kw)
        tot += it
    return U, tot


def make_precond(s, dt, kap, rowweight=False):
    shape = (s['m'].nelem, s['N']+1, s['N']+1, OP.NVAR_R, s['nk'])
    out = []
    for k in range(T.NSTAGE):
        cc = T.implicit_coeff(dt, k)
        rw = OP.momentum_row_weights(cc) if rowweight else None
        out.append(S3.jacobi_inverse(S3.jacobi_diagonal_analytic(
            shape, s['D'], s['m'].facx, s['m'].facy, s['kz'], s['nu'], cc,
            s['m'], s['mask'], s['m'].wq, kap, rw=rw), s['mask']))
    return out


def measure_sigma(s, U0, dt, kap, nsub=1, tend=0.1, half=True, **kw):
    """Integrate and fit ln(E/E0) = -2 sigma t.

    Fitted over the SECOND HALF by default, past any startup transient -- the
    same convention as the validated 2D harness.
    """
    Minv = make_precond(s, dt, kap, rowweight=kw.get('rowweight', False))
    U = U0.copy()
    nstep = max(6, int(round(tend/dt)))
    ts, Es = [0.0], [energy(s, U)]
    tot = 0
    for i in range(nstep):
        U, it = step(s, U, dt, kap, Minv=Minv, nsub=nsub, **kw)
        tot += it
        if not np.all(np.isfinite(U)):
            return dict(status='BLEWUP', dt=dt, step=i+1)
        ts.append((i+1)*dt)
        Es.append(energy(s, U))
    ts, Es = np.array(ts), np.array(Es)
    k0 = len(ts)//2 if half else 1
    slope = np.polyfit(ts[k0:], np.log(Es[k0:]/Es[0]), 1)[0]
    return dict(status='ok', dt=dt, sigma=-0.5*slope, cg=tot, nstep=nstep,
                ts=ts, Es=Es)
