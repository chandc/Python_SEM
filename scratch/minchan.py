"""Minimal-channel (Jiménez–Moin) rig at Re_tau = 180 — the M7 rehearsal.

    uv run --quiet python scratch/minchan.py check      # laminar + forcing gates
    uv run --quiet python scratch/minchan.py price      # time one step, cost the run
    uv run --quiet python scratch/minchan.py run [n]    # the real thing

NEW FILE.  `channel3d.py` is a Stage 5 LAMINAR rig and is not touched: its
module-level LX, LY and its `poiseuille` initial state are wrong for this case in
three separate ways (below).  Its `stage`, `step` and `make_precond` take the
setup dict `s` and never touch those constants, so they are REUSED unchanged --
the same arrangement every other driver here uses.

THREE THINGS CHANGE FROM STAGE 5, and each was a measured gap, not a preference:

 1. REYNOLDS NUMBER.  channel3d runs u = 6y(1-y), nu = 1/180, delta = 0.5, which
    gives u_tau = 0.183 and **Re_tau = 16.4** -- its `re=180` is a BULK Reynolds
    number.  Here the standard wall normalisation is used instead:

        delta = 1,  u_tau = 1,  nu = 1/Re_tau = 1/180,  f_x = u_tau^2/delta = 1

    so the channel is y in [0, 2] and one wall unit is exactly 1/180.  Against
    channel3d's f_x = 12*nu = 0.0667 that is a **120x** change in forcing, and it
    has never been run at this magnitude -- hence the `check` gate.

 2. BOX SIZE.  channel3d's Lx = Lz = 2*pi is Lx+ = Lz+ = 2262: a full channel.
    Jiménez & Moin (1991) put the minimal unit at Lx+ ~ 350, Lz+ ~ 100, with the
    SPANWISE width the binding constraint -- below Lz+ ~ 100 the near-wall cycle
    cannot sustain itself and the flow relaminarises.

    This box is deliberately ABOVE that threshold, not on it:

        Lx = pi       -> Lx+ = 565
        Lz = 0.34*pi  -> Lz+ = 192

    Sitting exactly at the minimum invites a relaminarisation that would be
    indistinguishable from a solver bug.  Still ~16x cheaper than full M7
    (4*pi x 2 x 4*pi/3).

 3. THE TRIP.  Plane Poiseuille is LINEARLY STABLE here (critical Re ~ 5772 on
    centreline/half-height; this is ~3300), so no infinitesimal disturbance
    grows and transition must be bypassed with a finite-amplitude one.
    channel3d's roll perturbation is streamwise-INDEPENDENT, which is right for
    a decay measurement and useless here: by Squire's theorem such a disturbance
    cannot sustain itself.  The trip below is genuinely 3-D -- rolls to build
    streaks, plus x-dependent noise to break the symmetry that would otherwise
    let them decay.

The initial mean profile is Reichardt's law of the wall, not the parabola: at
f_x = 1 the LAMINAR solution has centreline u = 90, and starting there would
spend the whole run shedding a transient.
"""
import json
import os
import sys
import time

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
SC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SC)
sys.path.insert(0, ROOT)
sys.path.insert(0, SC)

import numpy as np
from lssem2d.lgl import diff_matrix
from lssem2d.mesh import build_channel

from lssem3d import (backend, operator as OP, solver3d as S3, bc as BC,
                     fourier as FR, timestep as T, convect as CV, deriv as DV,
                     device as DEV)
import channel3d as C
from minchan_stats import PlaneStats

RE_TAU = 180.0
DELTA = 1.0                        # half-height; channel is y in [0, 2]
LX = np.pi                         # Lx+ = 565
LZ = 0.34*np.pi                    # Lz+ = 192
FX = 1.0                           # u_tau^2 / delta, the CPG body force
KAPPA, B_LOG = 0.41, 7.8           # Reichardt constants


def setup(N=8, ex=6, ey=18, nz=32):
    """Minimal box, walls in y, periodic x (SEM seam) and z (Fourier)."""
    m = build_channel(LX, 2.0*DELTA, ex, ey, N, bcs=(0, 0, 1, 1))
    m.periodic_x = LX
    m.compute_global_indices()
    nk = nz//2 + 1
    mask = BC.build_mask(m, nk, pin_p=False, nz=nz)
    BC.pin_dof(m, mask, OP.P_, 0)              # every copy on the periodic seam
    BC.pin_dof(m, mask, OP.NVAR + OP.P_, 0)
    nu = 1.0/RE_TAU                            # u_tau = delta = 1
    X = np.empty((m.nelem, N+1, N+1)); Y = np.empty_like(X)
    for e in range(m.nelem):
        X[e] = m.xnod[e][:, None]
        Y[e] = m.ynod[e][None, :]
    return dict(m=m, D=diff_matrix(N), N=N, nz=nz, nk=nk, lz=LZ, nu=nu,
                re=RE_TAU, kz=FR.wavenumbers(nz, LZ), mask=mask, fx=FX,
                X=X, Y=Y, zpl=(LZ/nz)*np.arange(nz))


def reichardt(yp):
    """Smooth law of the wall, valid from the wall through the log layer."""
    return ((1.0/KAPPA)*np.log(1.0 + KAPPA*yp)
            + B_LOG*(1.0 - np.exp(-yp/11.0) - (yp/11.0)*np.exp(-0.33*yp)))


def mean_profile(Y):
    """u+(y+) using distance to the NEAREST wall, so it is symmetric."""
    d = np.minimum(Y, 2.0*DELTA - Y)
    return reichardt(np.maximum(d, 1e-12)*RE_TAU)


def initial_state(s, amp_roll=1.0, amp_noise=0.3, seed=0):
    """Turbulent mean profile + a 3-D finite-amplitude trip.

    `amp_roll` IS THE PEAK |v'| IN WALL UNITS, not a streamfunction amplitude.
    That distinction cost a CFL violation: the streamfunction gives
    max|v'| = A*k_z with k_z = 2*pi/L_z = 5.88, so a "0.7" amplitude produced
    v' = 4.1 u_tau -- about six times what developed channel turbulence carries
    (v'_rms ~ 1 u_tau).  Measured on the first real state: |v| = 5.86 at y+ =
    160, and since h_y is 4.7x finer than h_x the v/h_y term DOMINATES the CFL,
    pushing it to 1.79 against the RKW3 limit of 1.732.

    Normalising by k_z makes the knob mean what it says and puts the trip at the
    physical scale, which is the right fix rather than shrinking dt.
    """
    m, N, nz, nk = s['m'], s['N'], s['nz'], s['nk']
    X, Y = s['X'][..., None], s['Y'][..., None]
    Z = s['zpl'].reshape(1, 1, 1, -1)
    rng = np.random.default_rng(seed)

    P = np.zeros(X.shape[:3] + (OP.NVAR, nz))
    P[..., OP.U_, :] = mean_profile(Y)

    # Streamwise rolls -> streaks.  Divergence-free by construction from
    # psi = A sin^2(pi y/2) cos(kz z), so this adds no continuity residual.
    kzr = 2.0*np.pi/LZ
    A = amp_roll/kzr                       # so that peak |v'| == amp_roll
    env = np.sin(np.pi*Y/(2.0*DELTA))**2
    P[..., OP.V_, :] += -A*kzr*env*np.sin(kzr*Z)
    P[..., OP.W_, :] += -A*(np.pi/(2.0*DELTA))*np.sin(np.pi*Y/DELTA)*np.cos(kzr*Z)

    # x-dependent noise.  WITHOUT THIS THE TRIP CANNOT WORK: the rolls above are
    # streamwise-independent, and by Squire's theorem such a disturbance decays.
    #
    # BUILT AS THE CURL OF A VECTOR POTENTIAL, so it is DIVERGENCE-FREE by
    # construction.  Independent randn on u, v, w is NOT solenoidal, and that
    # cost a run: it injected relative divergence of 2.0e-01, the solver reduced
    # it only to 1.1e-01 and then stalled there -- against 5e-09 on the
    # Stokes-decay case that reproduces the analytic rate to 8 significant
    # figures.  Seven orders of magnitude, constant over t = 0.1 ... 1.0.
    #
    # LSSEM penalises div u as a weighted ROW; it does not project the state
    # onto the solenoidal manifold, so divergence put into the INITIAL CONDITION
    # is not removed.  It has to be absent to begin with.
    #
    # A = wall-damped random potential; u' = curl A vanishes at the walls because
    # A and its tangential derivatives do (the envelope is sin^2, so A ~ y^2).
    wall = np.sin(np.pi*Y/(2.0*DELTA))**2
    A = [wall*rng.standard_normal(P[..., 0, :].shape) for _ in range(3)]
    Ah = [FR.to_modes(a)[..., :nk] for a in A]
    ikz = 1j*s['kz']
    dx = lambda q: DV.ddx(q, s['D'], m.facx)
    dy = lambda q: DV.ddy(q, s['D'], m.facy)
    cx = FR.to_physical(dy(Ah[2]) - ikz*Ah[1], nz)
    cy = FR.to_physical(ikz*Ah[0] - dx(Ah[2]), nz)
    cz = FR.to_physical(dx(Ah[1]) - dy(Ah[0]), nz)
    scale = amp_noise/max(np.abs(cx).max(), np.abs(cy).max(), np.abs(cz).max(), 1e-30)
    P[..., OP.U_, :] += scale*cx
    P[..., OP.V_, :] += scale*cy
    P[..., OP.W_, :] += scale*cz

    Uc = FR.to_modes(P)[..., :nk]
    C._set_vorticity(s, Uc)                    # discrete curl, not analytic
    U = np.concatenate([Uc.real, Uc.imag], axis=-2)
    return U*s['mask']                          # enforce no-slip exactly at t=0


# ------------------------------------------------------------------ diagnostics

def to_physical(s, U):
    return FR.to_physical(OP.to_complex(U), s['nz'])


def u_tau(s, U):
    """Friction velocity from the mean wall shear, averaged over both walls.

    The whole run is judged on this: u_tau must hold near 1.0 (it is what the
    forcing prescribes) and NOT decay to the laminar value, which is what
    relaminarisation looks like.
    """
    m, D = s['m'], s['D']
    Uc = OP.to_complex(U)
    du = DV.ddy(Uc[..., OP.U_, :], D, m.facy)[..., 0].real/s['nz']
    Y = s['Y']
    lo = np.abs(Y - Y.min()) < 1e-12
    hi = np.abs(Y - Y.max()) < 1e-12
    tw = 0.5*(np.abs(du[lo]).mean() + np.abs(du[hi]).mean())*s['nu']
    return float(np.sqrt(max(tw, 0.0)))


def bulk(s, U):
    Uc = OP.to_complex(U)
    return float(Uc[..., OP.U_, 0].real.mean()/s['nz'])


def rms_w(s, U):
    """Spanwise rms -- zero in any 2-D state, so it is the 3-D liveness check."""
    P = to_physical(s, U)
    return float(np.sqrt(np.mean(P[..., OP.W_, :]**2)))


def momentum_budget(s, U):
    """The x-momentum balance, integrated over the domain.

        V dU_b/dt  =  f_x*V  +  nu*Int lap(u)  -  Int (u.grad u)_x

    THE LAST TERM IS NOT ZERO, and assuming it was cost a day of chasing a
    phantom.  `Int (u.grad u)_x = 0` requires the flow to be EXACTLY
    divergence-free; in a least-squares formulation it never is -- continuity is
    a weighted row, not a constraint (3D_STATUS.md L5).  Dropping it made the
    balance miss by ~18%, which looked like an 18% error in the body force and
    then like an 18% error in u_tau, and was neither.

    Returns (f_x*V, viscous, convective, div_measure).  `convective` is therefore
    a direct DIAGNOSTIC OF DIVERGENCE ERROR: it would vanish for a solenoidal
    field, so its size relative to f_x*V says how far the state is from one.
    """
    m, D, kz, nz, nu = s['m'], s['D'], s['kz'], s['nz'], s['nu']
    Uc = OP.to_complex(U)
    u = Uc[..., OP.U_, :]
    wq3 = m.wq[..., None]
    dz = LZ/nz
    vol = float(m.wq.sum())*LZ
    vint = lambda phys: float((phys*wq3).sum()*dz)

    uxx = DV.ddx(DV.ddx(u, D, m.facx), D, m.facx)
    uyy = DV.ddy(DV.ddy(u, D, m.facy), D, m.facy)
    visc = nu*vint(FR.to_physical(uxx + uyy - (kz**2)*u, nz))
    conv = vint(FR.to_physical(CV.convective(Uc, D, m.facx, m.facy, kz, nz)[..., 0, :], nz))

    d = (DV.ddx(u, D, m.facx) + DV.ddy(Uc[..., V_ if False else OP.V_, :], D, m.facy)
         + 1j*kz*Uc[..., OP.W_, :])
    divrms = float(np.sqrt(np.mean(FR.to_physical(d, nz)**2)))
    return FX*vol, visc, conv, divrms


def energy(s, U):
    """(kinetic energy, dissipation nu*Int|grad u|^2) -- both TOTAL, not per volume.

    Dissipation from grad u, NOT from the stored omega.  They agree here to
    0.05% (checked), but omega is an independent unknown in VVP that satisfies
    omega = curl u only to the least-squares residual, and bc.py leaves it
    entirely free on the walls -- where enstrophy is largest.
    """
    m, D, kz, nz, nu = s['m'], s['D'], s['kz'], s['nz'], s['nu']
    Uc = OP.to_complex(U)
    wq3 = m.wq[..., None]; dz = LZ/nz
    vint = lambda phys: float((phys*wq3).sum()*dz)
    P = FR.to_physical(Uc, nz)
    ke = 0.5*sum(vint(P[..., f, :]**2) for f in (OP.U_, OP.V_, OP.W_))
    g2 = 0.0
    for f in (OP.U_, OP.V_, OP.W_):
        c = Uc[..., f, :]
        for q in (DV.ddx(c, D, m.facx), DV.ddy(c, D, m.facy), 1j*kz*c):
            g2 += vint(FR.to_physical(q, nz)**2)
    return ke, nu*g2


def cfl(s, U, dt):
    """Convective CFL on the finest spacing in each direction.

    GLL points cluster at element edges, so the limiting spacing is the first
    GLL gap, NOT h/N.  Using the nominal spacing would under-report CFL by ~3x
    at N = 8 and is how a run gets launched above its stability limit.
    """
    from lssem2d.lgl import lgl_nodes
    P = to_physical(s, U)
    m, N = s['m'], s['N']
    xi = lgl_nodes(N)[0] if isinstance(lgl_nodes(N), tuple) else lgl_nodes(N)
    gap = float(np.min(np.diff(np.sort(np.asarray(xi)))))/2.0   # ref elem [-1,1]
    hx = (LX/(m.nelem//12))*gap if False else (LX/6)*gap        # 6 elems in x
    hy = (2.0*DELTA/18)*gap
    hz = LZ/s['nz']
    umax = np.abs(P[..., OP.U_, :]).max()
    vmax = np.abs(P[..., OP.V_, :]).max()
    wmax = np.abs(P[..., OP.W_, :]).max()
    return float(dt*(umax/hx + vmax/hy + wmax/hz))


def wall_units(s, ex=6, ey=18):
    wu = DELTA/RE_TAU
    return dict(wall_unit=wu, Lx_plus=LX/wu, Lz_plus=LZ/wu,
                dx_plus=(LX/(ex*s['N']))/wu, dz_plus=(LZ/s['nz'])/wu,
                dof=s['m'].nelem*(s['N']+1)**2*OP.NVAR_R*s['nk'])


# ---------------------------------------------------------------------- driver

def to_device(s, U):
    """Move the whole problem onto the accelerator, once, before stepping.

    WHY THIS EXISTS.  Phase 3 made the operator, FFTs and convection dispatch,
    but the DRIVER still handed NumPy in -- so every call crossed the bus.
    Measured on one step with CG capped at 40/stage: **6768 host->device and
    1128 device->host transfers**, and the full step ran 636.8 s against numba's
    76.8 s.  8x SLOWER while every unit test passed, which is the failure mode
    GPU_PORT_PLAN.md sec 1 warned about: "the trap is thinking this is port the
    matvec".

    The mesh is SHALLOW-COPIED before facx/facy are replaced, so the original
    stays usable by the NumPy path -- a run is entirely one or the other, and
    mutating the shared mesh would silently couple them.
    """
    import copy
    # BACKEND-AGNOSTIC.  This was written against torch and imported it
    # directly, so a cupy run failed at `import torch` with no torch installed
    # -- the backend was selected, the state was never moved, and the operator
    # met NumPy.  Both accelerated backends move arrays the same way.
    name = backend.get_backend()
    if name == 'cupy':
        import cupy as cp
        t = lambda a: cp.asarray(np.ascontiguousarray(a, dtype=np.float64))
    elif name in ('torch', 'cuda'):
        # the fused-CUDA backend also carries torch tensors, so it takes the
        # torch path; without this branch `dev` including 'cuda' would raise.
        import torch
        from lssem3d import kernels_torch as KT
        dev = KT.device()
        t = lambda a: torch.as_tensor(
            np.ascontiguousarray(a, dtype=np.float64), device=dev)
    else:
        raise ValueError(f'to_device called with backend {name!r}; '
                         'only cupy and torch have devices')
    m2 = copy.copy(s['m'])
    m2.facx, m2.facy = t(s['m'].facx), t(s['m'].facy)
    m2.wq = t(s['m'].wq)
    s2 = dict(s, m=m2, D=t(s['D']), kz=t(s['kz']), mask=t(s['mask']))
    return s2, t(U)


def _precond(s_host, dt, like=None):
    """Build the Jacobi preconditioner ON THE HOST, then move it.

    `jacobi_diagonal_analytic` is closed-form NumPy and is evaluated ONCE per
    dt, not per iteration, so there is nothing to gain from porting it -- and
    handing it tensors just fails (`can't convert cuda:0 device type tensor to
    numpy`).  Build from the host setup, move the result to wherever the state
    lives.
    """
    Minv = C.make_precond(s_host, dt, 0.0, rowweight=True)
    return [DEV.to_device(q, like) for q in Minv] if like is not None else Minv


def advance(s, U, Nprev, dt, Minv, tol=1e-6, max_iter=20000, **kw):
    """One RKW3/CN step.  `tol` = 1e-6 is the measured policy (3D_STATUS.md
    sec 7F): error unchanged to within 1%, ~40% fewer iterations than 1e-12.
    1e-3 was swept and REJECTED there -- 19-238x the error.

    **kw forwards to `channel3d.stage`, notably `warm=` for the warm start."""
    return C.step(s, U, Nprev, dt, 0.0, rowweight=True, Minv=Minv, tol=tol,
                  max_iter=max_iter, **kw)


def check(N=8, ex=6, ey=18, nz=32):
    """Gates that must pass BEFORE a multi-hour run is launched."""
    s = setup(N, ex, ey, nz)
    wu = wall_units(s, ex, ey)
    print('minimal channel, Re_tau = %g' % RE_TAU)
    print('  box     Lx+ = %.0f   Lz+ = %.0f   (Jimenez-Moin minimum ~350 x ~100)'
          % (wu['Lx_plus'], wu['Lz_plus']))
    print('  grid    dx+ = %.1f   dz+ = %.1f   (KMM 17.7, 5.9)   dof = %.2fM'
          % (wu['dx_plus'], wu['dz_plus'], wu['dof']/1e6))
    Y = np.sort(np.unique(s['Y']))
    print('  wall    y1+ = %.2f   (want < 1)' % (Y[1]*RE_TAU))

    # GATE 1 -- the forcing is 120x Stage 5's and has never been run.  A laminar
    # state under this force must give u_tau = 1 to the extent the profile is
    # resolved; more importantly it must not blow up.
    U0 = initial_state(s, amp_roll=0.0, amp_noise=0.0)
    print('\n  laminar control (no trip):')
    print('    u_tau  = %.4f   (forcing prescribes 1.0)' % u_tau(s, U0))
    print('    U_bulk = %.3f   rms w = %.2e  (must be ~0 with no trip)'
          % (bulk(s, U0), rms_w(s, U0)))

    # GATE 2 -- the tripped field is genuinely 3-D and satisfies no-slip.
    U = initial_state(s)
    P = to_physical(s, U)
    lo = np.abs(s['Y'] - s['Y'].min()) < 1e-12
    print('\n  tripped initial field:')
    print('    u_tau  = %.4f   U_bulk = %.3f   rms w = %.4f'
          % (u_tau(s, U), bulk(s, U), rms_w(s, U)))
    print('    max|u| = %.2f   no-slip residual at wall = %.2e'
          % (np.abs(P[..., OP.U_, :]).max(),
             np.abs(P[lo][..., :3, :]).max()))
    return s, U


def price(N=8, ex=6, ey=18, nz=32, dt=None):
    """Time ONE step on the real grid, then cost the run.  GPU_PORT_PLAN.md
    Phase 6: never commit a long run to an extrapolated table."""
    s, U = check(N, ex, ey, nz)
    if dt is None:
        dt = 2.0e-3
    print('\n  cfl(dt=%g) = %.3f   (RKW3 limit sqrt(3) = 1.732)' % (dt, cfl(s, U, dt)))
    # MOVE THE PROBLEM TO THE DEVICE, exactly as run() does.  price() did not,
    # so with LSSEM3D_BACKEND=cupy the operator dispatched to the CuPy kernels
    # while the state was still NumPy -- "Unsupported type numpy.ndarray" from
    # inside an ElementwiseKernel.  On the torch path it merely crossed the bus
    # on every call, which is the 8x slowdown to_device was written to stop.
    s_host = s
    if backend.get_backend() in ('torch', 'cupy'):
        s, U = to_device(s, U)
        print('  state moved to device (%s)' % backend.get_backend())
    Minv = _precond(s_host, dt, like=U)
    # allocate the convective history WHERE THE STATE IS.  np.zeros here left
    # a host array meeting a device Nk in the RK combination, one line deeper
    # than the operator -- the same "Unsupported type numpy.ndarray", with the
    # traceback pointing at arithmetic rather than at the allocation.
    _xp = DEV.xp(U)
    Nprev = _xp.zeros(tuple(OP.to_complex(U).shape[:-2]) + (3, s['nk']),
                      dtype=complex)
    t0 = time.perf_counter(); U1, Nprev, it = advance(s, U, Nprev, dt, Minv)
    t1 = time.perf_counter() - t0
    t0 = time.perf_counter(); U1, Nprev, it = advance(s, U1, Nprev, dt, Minv)
    t2 = time.perf_counter() - t0
    print('  step 1: %.1f s (%d CG)   step 2: %.1f s (%d CG)   backend=%s'
          % (t1, it, t2, it, backend.get_backend()))
    for T_end in (20.0, 100.0):
        n = int(T_end/dt)
        print('    t=%5.0f  (%6d steps)  ->  %6.1f h' % (T_end, n, n*t2/3600))
    return s, U1, dt, t2


def _atomic_savez(path, **kw):
    """Write to a temp name, then rename.  Rename is atomic, so a crash mid-write
    cannot corrupt the last good checkpoint -- and an rsync running against the
    directory never sees a half-written file."""
    tmp = path + '.tmp'
    np.savez(tmp, **kw)
    os.replace(tmp + '.npz' if not tmp.endswith('.npz') else tmp, path)


def run(out='.', nstep=20000, dt=1.0e-3, every=100, backend_name=None,
        resume=None, N=8, ex=6, ey=18, nz=32):
    """The production minimal-channel run.

    OUTPUT ALL GOES TO `out`, which is a BIND MOUNT in the container -- anything
    written elsewhere dies with `--rm` (GPU_PORT_PLAN.md sec 4).

    THE RUN IS JUDGED ON u_tau.  The forcing prescribes u_tau = 1; if it decays
    toward the laminar value the near-wall cycle has died.  That is not
    necessarily a bug -- the minimal unit is intermittent by design (Jimenez &
    Moin 1991) -- but it must be visible while it happens, not discovered
    afterwards, so it is logged every step and warned on.
    """
    import json
    os.makedirs(out, exist_ok=True)
    name = backend_name or os.environ.get('LSSEM3D_BACKEND', 'numba')
    backend.set_backend(name)

    s = setup(N, ex, ey, nz)
    # 'cupy' BELONGS HERE.  Omitting it selected the cupy KERNELS while leaving
    # `dev` False, so to_device() was never called, the state stayed NumPy, and
    # every cupy kernel rejected it with "Unsupported type <class
    # 'numpy.ndarray'>".  The backend appeared to be selected and the run could
    # not take a single step.
    dev = name in ('torch', 'cuda', 'cupy')
    step0 = 0
    if resume:
        z = np.load(resume)
        U, step0 = z['U'], int(z['step'])
        Nprev = z['Nprev_re'] + 1j*z['Nprev_im']
        print(f'resumed from {resume} at step {step0}', flush=True)
    else:
        U = initial_state(s)
        Nprev = np.zeros(OP.to_complex(U).shape[:-2] + (3, s['nk']), dtype=complex)

    Minv_host = _precond(s, dt)
    if dev:
        s_run, U = to_device(s, U)
        Minv = [DEV.to_device(q, U) for q in Minv_host]
        # BACKEND-AGNOSTIC, for the same reason to_device() itself is.  This line
        # used to call torch.as_tensor directly, so a cupy run moved the state
        # and the preconditioner to the device and left Nprev on the host --
        # every kernel then met a mixed pair and cupy raised
        # "TypeError: Unsupported type <class 'numpy.ndarray'>".  DEV.to_device
        # dispatches on `like` and handles both backends, complex included.
        Nprev = DEV.to_device(Nprev, U)
    else:
        s_run, Minv = s, Minv_host

    wu = wall_units(s, ex, ey)
    cfg = dict(N=N, ex=ex, ey=ey, nz=nz, dt=dt, nstep=nstep, backend=name,
               re_tau=RE_TAU, Lx=LX, Lz=LZ, fx=FX, tol=1e-6, **wu)
    json.dump({k: float(v) if isinstance(v, (int, float, np.floating)) else v
               for k, v in cfg.items()}, open(f'{out}/config.json', 'w'), indent=1)

    log = open(f'{out}/run.log', 'a')
    hdr = (f'# minimal channel Re_tau={RE_TAU:g}  {ex}x{ey} N={N} Nz={nz}  '
           f'dt={dt:g}  backend={name}  Lx+={wu["Lx_plus"]:.0f} Lz+={wu["Lz_plus"]:.0f}')
    print(hdr, flush=True); log.write(hdr + '\n'); log.flush()

    # PROFILE STATISTICS.  The scalar log alone cannot demonstrate DNS -- the
    # deliverable is U+(y+) and the u/v/w rms and <uv> profiles against KMM.
    # Accumulated on the same cadence as the diagnostics and checkpointed with
    # the state, so a restart continues the average instead of resetting it.
    stats = PlaneStats(s, nz)
    if resume:
        stats.load(z)

    hist, t0 = [], time.perf_counter()
    for i in range(step0, nstep):
        U, Nprev, it = advance(s_run, U, Nprev, dt, Minv)
        if not bool(np.all(np.isfinite(DEV.to_host(U)))):
            log.write('BLEWUP\n'); log.close()
            raise SystemExit('non-finite state -- aborting')
        if (i + 1) % 10 == 0 or i == step0:
            Uh = DEV.to_host(U)
            ut, ub, rw_ = u_tau(s, Uh), bulk(s, Uh), rms_w(s, Uh)
            stats.accumulate(Uh, t=(i + 1)*dt, utau=ut)
            cf = cfl(s, Uh, dt)
            ke, eps = energy(s, Uh)
            Pf, visc, conv, divrms = momentum_budget(s, Uh)
            line = (f't={(i+1)*dt:8.3f} u_tau={ut:.4f} U_b={ub:6.3f} '
                    f"rms_w={rw_:.4f} E={ke:9.2f} eps={eps:8.2f} "
                    f"div={divrms:.2e} conv={conv:7.3f} "
                    f"CFL={cf:.2f} CG={it} [{time.perf_counter()-t0:.0f}s]")
            if cf > 1.732:
                line += '  ** CFL ABOVE THE RKW3 LIMIT **'
            if ut < 0.5:
                line += '  ** u_tau COLLAPSING -- possible relaminarisation **'
            print(line, flush=True); log.write(line + '\n'); log.flush()
            hist.append(((i+1)*dt, ut, ub, rw_, cf, it, ke, eps,
                         Pf, visc, conv, divrms))
        if (i + 1) % every == 0:
            Uh = DEV.to_host(U)
            Nh = DEV.to_host(Nprev)
            # The running statistics ride WITH the checkpoint, so a restart
            # resumes the average rather than silently starting a new one.
            _atomic_savez(f'{out}/checkpoint_{i+1:07d}.npz', U=Uh, step=i+1,
                          Nprev_re=Nh.real, Nprev_im=Nh.imag, t=(i+1)*dt,
                          **stats.state())
            _atomic_savez(f'{out}/diag.npz', hist=np.array(hist))
            stats.save(f'{out}/stats.npz', s['nu'], dt, (i+1)*dt)
    log.write(f'done, {nstep} steps, {time.perf_counter()-t0:.0f}s\n'); log.close()


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else 'check'
    if what == 'check':
        backend.set_backend(os.environ.get('LSSEM3D_BACKEND', 'numba')); check()
    elif what == 'price':
        backend.set_backend(os.environ.get('LSSEM3D_BACKEND', 'numba')); price()
    elif what == 'run':
        kw = {}
        for a in sys.argv[2:]:
            k, v = a.split('=')
            kw[k] = v if k in ('out', 'resume', 'backend_name') else (
                float(v) if '.' in v or 'e' in v.lower() else int(v))
        run(**kw)
    else:
        raise SystemExit(f'unknown mode {what!r}: check | price | run')


if __name__ == '__main__':
    main()
