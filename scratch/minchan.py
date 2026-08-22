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


def initial_state(s, amp_roll=0.7, amp_noise=0.3, seed=0):
    """Turbulent mean profile + a 3-D finite-amplitude trip.

    Amplitudes are in wall units (u_tau = 1).  Perturbations in a turbulent
    channel are O(u_tau), so O(1) here is the physical scale, not a large
    disturbance -- but it is far above anything that would decay linearly.
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
    env = np.sin(np.pi*Y/(2.0*DELTA))**2
    P[..., OP.V_, :] += -amp_roll*kzr*env*np.sin(kzr*Z)
    P[..., OP.W_, :] += -amp_roll*(np.pi/(2.0*DELTA))*np.sin(np.pi*Y/DELTA)*np.cos(kzr*Z)

    # x-dependent noise.  WITHOUT THIS THE TRIP CANNOT WORK: the rolls above are
    # streamwise-independent, and by Squire's theorem such a disturbance decays.
    # Damped at the walls so the no-slip condition is not violated at t = 0.
    wall = np.sin(np.pi*Y/(2.0*DELTA))**2
    for f in (OP.U_, OP.V_, OP.W_):
        P[..., f, :] += amp_noise*wall*rng.standard_normal(P[..., f, :].shape)

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
    import torch
    from lssem3d import kernels_torch as KT
    dev = KT.device()
    t = lambda a: torch.as_tensor(np.ascontiguousarray(a, dtype=np.float64),
                                  device=dev)
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
    Minv = _precond(s, dt)
    Nprev = np.zeros(OP.to_complex(U).shape[:-2] + (3, s['nk']), dtype=complex)
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


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else 'check'
    backend.set_backend(os.environ.get('LSSEM3D_BACKEND', 'numba'))
    if what == 'check':
        check()
    elif what == 'price':
        price()
    else:
        raise SystemExit('run mode not wired yet -- see GPU_PORT_PLAN.md Phase 4')


if __name__ == '__main__':
    main()
