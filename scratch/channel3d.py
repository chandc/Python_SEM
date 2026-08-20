"""STAGE 5 rig: laminar 3D channel, Poiseuille + a decaying z-perturbation.

Importable setup/stepper for the Stage 5 sweeps.  No __main__ side effects --
`channel3d_stage5.py` drives it (scratch scripts that run work at import get
re-run by every importer).

GEOMETRY.  Walls in y (no-slip), periodic in x via SEM connectivity
(`mesh.periodic_x`, the same mechanism the 2D periodic-Poiseuille runs use),
periodic in z via Fourier.  bcs=(0,0,1,1): W/E free -- the seam is merged by
`compute_global_indices`, so periodicity arrives through the gather-scatter and
needs no boundary handling of its own.

BASE FLOW.  u = 6y(1-y), sustained by a constant body force f_x = 12*nu, which
is the exact balance 0 = f + nu*u'' for u'' = -12.  Poiseuille is exactly
representable in this basis, which is the whole point of the control case: its
least-squares residual is ~0, and 3D_DEVELOPMENT_PLAN.md sec 0.2 says that is
precisely the condition under which the a_mass instability stays hidden.  So the
laminar case is expected to look healthy and proves nothing on its own -- it is
a control, and the perturbed case is the measurement.

PERTURBATION.  Streamwise-independent rolls from a streamfunction in (y,z),

    psi = A sin^2(pi y) cos(k z)
    v' =  d(psi)/dz = -A k sin^2(pi y) sin(k z)
    w' = -d(psi)/dy = -A pi sin(2 pi y) cos(k z)

which is divergence-free by construction (d v'/dy + d w'/dz = 0 identically) and
vanishes at both walls.  Chosen over a random field because it is analytic,
reproducible, and genuinely three-dimensional -- it puts energy in w and in the
transverse vorticities, the components every k_z = 0 test is blind to.

BODY-FORCE AND STATE CONVENTION.  `fourier.to_modes` is an unnormalised rfft, so
a physical constant C has mode-0 coefficient C*nz.  Both the force and the
initial condition use that convention; getting it wrong scales the flow by nz
and is not otherwise visible.
"""
import numpy as np

from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem3d import (operator as OP, solver3d as S3, bc as BC, convect as CV,
                     fourier as FR, timestep as T, parallel as PAR, deriv as DV)

LX, LY = 2.0*np.pi, 1.0
KPERT = 1                         # z-mode INDEX of the perturbation (not a wavenumber)


def setup(N=8, ex=4, ey=4, nz=16, re=180.0, lz=2.0*np.pi, pin_kz0_only=True):
    """Mesh, mask, wavenumbers.  Pressure is pinned at ONE node and, by default,
    only in the k_z = 0 mode: at k_z != 0 the pressure is already determined, so
    pinning it there adds an inconsistent constraint the least-squares system
    would silently absorb."""
    m = build_channel(LX, LY, ex, ey, N, bcs=(0, 0, 1, 1))
    m.periodic_x = LX
    m.compute_global_indices()
    nk = nz//2 + 1
    mask = BC.build_mask(m, nk, pin_p=False, nz=nz)
    BC.pin_dof(m, mask, OP.P_, 0)            # ALL copies (periodic seam)
    BC.pin_dof(m, mask, OP.NVAR + OP.P_, 0)
    if not pin_kz0_only:
        mask[0, 0, 0, OP.P_, :] = 0.0
        mask[0, 0, 0, OP.NVAR + OP.P_, :] = 0.0
    nu = 1.0/re
    return dict(m=m, D=diff_matrix(N), N=N, nz=nz, nk=nk, lz=lz, nu=nu, re=re,
                kz=FR.wavenumbers(nz, lz), mask=mask, fx=12.0*nu,
                Y=_ynodes(m, N))


def _ynodes(m, N):
    Y = np.empty((m.nelem, N+1, N+1))
    for e in range(m.nelem):
        Y[e] = m.ynod[e][None, :]
    return Y


def poiseuille(y):
    return 6.0*y*(1.0 - y)


def initial_state(s, amp=0.0):
    """Poiseuille (mode 0) plus the roll perturbation (mode KPERT).

    Vorticity is set by DISCRETE differentiation of the velocity, not from the
    analytic curl: the state must satisfy the discrete vorticity-definition rows,
    and an analytically-exact curl would leave a non-zero residual in exactly
    those rows at t = 0.
    """
    m, N, nz, nk = s['m'], s['N'], s['nz'], s['nk']
    Uc = np.zeros((m.nelem, N+1, N+1, OP.NVAR, nk), dtype=complex)
    Y = s['Y']
    Uc[..., OP.U_, 0] = poiseuille(Y)*nz

    if amp:
        k = s['kz'][KPERT]
        # cos(kz) -> mode coefficient nz/2 ; sin(kz) -> -i*nz/2
        c_cos = 0.5*nz
        c_sin = -0.5j*nz
        Uc[..., OP.V_, KPERT] = (-amp*k*np.sin(np.pi*Y)**2)*c_sin
        Uc[..., OP.W_, KPERT] = (-amp*np.pi*np.sin(2*np.pi*Y))*c_cos

    _set_vorticity(s, Uc)
    return np.concatenate([Uc.real, Uc.imag], axis=-2)


def _set_vorticity(s, Uc):
    """omega = curl u, using the same discrete derivatives as the operator."""
    m, D, kz = s['m'], s['D'], s['kz']
    u, v, w = Uc[..., OP.U_, :], Uc[..., OP.V_, :], Uc[..., OP.W_, :]
    ux = lambda q: DV.ddx(q, D, m.facx)
    uy = lambda q: DV.ddy(q, D, m.facy)
    ik = 1j*kz
    Uc[..., OP.OX_, :] = uy(w) - ik*v
    Uc[..., OP.OY_, :] = ik*u - ux(w)
    Uc[..., OP.OZ_, :] = ux(v) - uy(u)


def _fw(rw, nrow2):
    """Row weights broadcast onto the split-real f array (real rows then imag)."""
    if rw is None:
        return 1.0
    return np.concatenate([rw, rw]).reshape((1, 1, 1, nrow2, 1))


def stage(s, U, Nprev, k, dt, kap, workers=None, tol=1e-9, max_iter=3000,
          Minv=None, nsub=1, sub_tol=1e-10, rw=None):
    """One RKW3/CN stage with AC and the body force.

    The body force is a constant, so it rides in the EXPLICIT term: per stage the
    weights are gamma_k + zeta_k, which sum to 1 over the three stages, giving
    exactly dt*f per step.  Putting it in the implicit term instead would weight
    it by alpha+beta and quietly rescale the driving.

    SUB-ITERATIONS (nsub > 1) -- dual time stepping, which is what makes AC
    legitimate for an UNSTEADY problem.  The continuity row solves

        kappa_p*p + div u = kappa_p*p_prev   =>   div u = -kappa_p*(p - p_prev)

    so with nsub = 1 and p_prev taken from the previous time level, div u is
    driven not to zero but to -kappa_p*dp/dt*O(dt), and since kappa_p ~ 1/dt
    that error is O(1) in dt -- it does NOT vanish under time refinement.  At a
    steady state p = p_prev and it is exact, which is why the M2 cavity gate
    could never see this.

    Sub-iterating fixes it at its root: p_prev is refreshed from the previous
    SUB-ITERATE, and on convergence p = p_prev makes the AC term vanish
    identically, recovering div u = 0 at the current time level.

    Only the continuity row is refreshed.  The momentum rows encode the physical
    time derivative and the explicit terms at the START of the stage, so they
    are built once, outside the loop -- updating them would move the time level
    the stage is solving for.  Convection is explicit, so the system is linear
    and each sub-iteration is one solve, warm-started from the last.
    """
    m, D, kz, mask, nu, nz = s['m'], s['D'], s['kz'], s['mask'], s['nu'], s['nz']
    c = T.implicit_coeff(dt, k)
    Uc = OP.to_complex(U)
    Nk = -CV.convective(Uc, D, m.facx, m.facy, kz, nz)
    Nk[..., 0, 0] += s['fx']*nz                    # body force, mode 0, physical
    R0 = OP.apply_L0_complex(Uc, D, m.facx, m.facy, kz, nu, 0.0, kap)
    Lk = -R0[..., 4:7, :]

    # momentum rows: fixed for the whole stage
    fc = np.zeros(Uc.shape[:-2] + (OP.NROW, Uc.shape[-1]), dtype=complex)
    for row, fld in ((4, OP.U_), (5, OP.V_), (6, OP.W_)):
        i = row - 4
        fc[..., row, :] = c*(Uc[..., fld, :] + dt*(
            T.GAMMA[k]*Nk[..., i, :] + T.ZETA[k]*Nprev[..., i, :]
            + T.ALPHA[k]*Lk[..., i, :]))

    wqR = m.wq[..., None, None]
    p_prev = Uc[..., OP.P_, :]          # sub-iterate 0: previous time level
    Uit, its, nit = U, 0, 0
    for _ in range(max(1, nsub)):
        fc[..., 0, :] = kap*p_prev
        f = np.concatenate([fc.real, fc.imag], axis=-2)
        # f must carry the SAME row weighting as the operator, or the defect
        # correction solves a different problem than apply_L represents.
        r = OP.apply_LT(
            OP.apply_L(Uit, D, m.facx, m.facy, kz, nu, c, m.wq, kap, rw)
            - f*wqR*_fw(rw, f.shape[-2]),
            D, m.facx, m.facy, kz, nu, c, kap)
        b = -S3.gs(m, r)*mask
        dU, it, _ = PAR.pcg(b, D, m.facx, m.facy, kz, nu, c, mesh=m, mask=mask,
                            M_inv=None if Minv is None else Minv[k], tol=tol,
                            max_iter=max_iter, wq=m.wq, kap=kap, workers=workers,
                        rw=rw)
        Uit = Uit + dU
        its += it
        nit += 1
        p_new = OP.to_complex(Uit)[..., OP.P_, :]
        dp = np.abs(p_new - p_prev).max()/max(np.abs(p_new).max(), 1e-30)
        p_prev = p_new
        if kap == 0.0 or dp < sub_tol:   # AC off => nothing to sub-iterate
            break
    return Uit, Nk, its


def make_precond(s, dt, kap, rowweight=False):
    shape = (s['m'].nelem, s['N']+1, s['N']+1, OP.NVAR_R, s['nk'])
    out = []
    for k in range(T.NSTAGE):
        cc = T.implicit_coeff(dt, k)
        rw = OP.momentum_row_weights(cc) if rowweight else None
        d = S3.jacobi_diagonal(shape, s['D'], s['m'].facx, s['m'].facy, s['kz'],
                               s['nu'], cc, s['m'],
                               s['mask'], s['m'].wq, kap, rw=rw)
        # jacobi_inverse, not 1/max(d, 1e-30): the clamp puts 1e30 on every
        # PRESCRIBED dof (diagonal exactly 0) and survives only because the
        # masked residual happens to be exactly zero.
        out.append(S3.jacobi_inverse(d, s['mask']))
    return out


def step(s, U, Nprev, dt, kap, rowweight=False, **kw):
    its = 0
    for k in range(T.NSTAGE):
        rw = OP.momentum_row_weights(T.implicit_coeff(dt, k)) if rowweight else None
        U, Nprev, it = stage(s, U, Nprev, k, dt, kap, rw=rw, **kw)
        its = max(its, it)
    return U, Nprev, its


def divergence(s, U):
    """rms |div u| = |u_x + v_y + i k w|.  The quantity AC is suspected of
    corrupting when it is used without sub-iterations."""
    m, D, kz = s['m'], s['D'], s['kz']
    Uc = OP.to_complex(U)
    d = (DV.ddx(Uc[..., OP.U_, :], D, m.facx)
         + DV.ddy(Uc[..., OP.V_, :], D, m.facy)
         + 1j*kz*Uc[..., OP.W_, :])
    return float(np.sqrt(np.mean(np.abs(d)**2)))


# ------------------------------------------------------------- diagnostics

def perturbation_energy(s, U):
    """Energy in every mode except k_z = 0 -- i.e. the 3D content.

    This is the quantity with a known answer: for laminar plane Poiseuille well
    below Re_crit ~ 5772, it must DECAY.  Growth means either a real instability
    (not expected here) or the numerics."""
    Uc = OP.to_complex(U)
    e = 0.0
    for f in (OP.U_, OP.V_, OP.W_):
        e += float(np.sum(np.abs(Uc[..., f, 1:])**2))
    return e


def mean_profile_error(s, U):
    """max |u_mean(y) - 6y(1-y)|, from the k_z = 0 mode."""
    Uc = OP.to_complex(U)
    u = Uc[..., OP.U_, 0].real/s['nz']
    return float(np.abs(u - poiseuille(s['Y'])).max())


def cfl(s, U, dt):
    Uphys = FR.to_physical(OP.to_complex(U), s['nz'])
    return float(CV.cfl(Uphys, s['D'], s['m'].facx, s['m'].facy, s['lz'],
                        s['nz'], dt))
