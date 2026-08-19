"""Does the a_mass instability exist in the STOKES-like operator?  Feasibility
gate for 3D_DEVELOPMENT_PLAN.md sec 0.2/0.4.

    uv run --quiet python scratch/stokes_amass_probe.py <a_mass> <off|half|match> [channel|cavity]

THE QUESTION.  With explicit convection (the 3D plan's choice), u.grad u moves to
the right-hand side and the operator inside the least-squares functional loses
its convective terms entirely -- it becomes Stokes-like.  Every a_mass threshold
measured in this repo (6.05/12.1 on the BFS, 60 on the channel, AC holding to
300) was measured with convection INSIDE the functional.  Plan sec 0.1 says those
numbers must be re-measured; this is that measurement, and it decides whether
explicit convection is viable at all:

    RKW3/CN needs a_mass_worst = 6/dt, and CFL at Re_tau=180 implies
    dt ~ 1e-3..1e-2, i.e. a_mass ~ 600..6000.

If the Stokes operator is stable there, explicit convection is viable and the 3D
plan proceeds.  If it still fails near 300, the plan must switch to semi-implicit
convection BEFORE more code is written against the wrong architecture.

HOW, without touching lssem2d.  Setting the linearisation velocities to zero
makes apply_L's convective terms vanish identically, leaving exactly the
Stokes-like operator.  Done by patching the module-level names this script
imports and the SolverState instance -- lssem2d itself is not modified, which
3D_DEVELOPMENT_PLAN.md Stage 1 depends on.

The test problem is the plane channel with a parabolic inlet.  Poiseuille is a
Stokes solution (u.grad u = 0 identically for parallel flow), so the exact
answer is unchanged by dropping convection and the run is a genuine test rather
than a different problem.

Fixed 200-step budget: every divergence measured in this project appeared within
33-71 steps, so this window catches the failure mode cheaply.
"""
import os, sys, time
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
sys.path.insert(0, SC)
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
import lssem2d
lssem2d.set_backend('numpy')
from fgrid import load
from lssem2d.lgl import diff_matrix, lgl_weights
from lssem2d.lssem import SolverState
import lssem2d.solver as S
import lssem2d.bc as BC

GRID = 'grids/channel_L12_12x2_N10_grid.dat'
RE, NSTEP, NSUB = 100.0, 200, 5

# --- make the operator Stokes-like, without modifying lssem2d -----------------
_L, _LT = S.apply_L, S.apply_LT


def _zero_lin_L(st, U, fu, fv):
    return _L(st, U, np.zeros_like(fu), np.zeros_like(fv))


def _zero_lin_LT(st, su, fu, fv):
    return _LT(st, su, np.zeros_like(fu), np.zeros_like(fv))


S.apply_L, S.apply_LT = _zero_lin_L, _zero_lin_LT


def run_cavity(a_mass, kspec):
    """CLOSED domain: lid-driven cavity, no outflow boundary anywhere.

    This is the configuration the 3D target case actually resembles -- a
    Re_tau=180 channel is periodic in x and z with walls in y, so it has NO
    outflow plane.  Every a_mass threshold in this repo was measured WITH an
    outflow, and ARTIFICIAL_COMPRESSIBILITY.md sec 5.1 already showed the closed
    cavity converging at a_mass = 30 where the BFS diverges at 12.1.  The open
    question is how far that exemption extends: 3D needs 600-6000.
    """
    from lssem2d.mesh import build_channel
    EX, N = 6, 10
    n = N+1
    mesh = build_channel(1.0, 1.0, EX, EX, N, bcs=(1, 1, 1, 2))
    dt = 1.5/a_mass
    st = SolverState(mesh, diff_matrix(N), nu=1.0/1000.0, dt=dt, fac1=1.0,
                     w_mom=1.0, w_mass=1.0)
    _upd = st.update_linearisation
    st.update_linearisation = lambda fu, fv: _upd(np.zeros_like(fu),
                                                  np.zeros_like(fv))
    kap = {'off': None, 'half': a_mass/2.0, 'match': a_mass}.get(kspec)
    if kap is None and kspec != 'off':
        kap = float(kspec)
    st.dtau_p = None if kap is None else 1.0/kap
    U = np.zeros((mesh.nelem, n, n, 4)); hist = [U.copy()]
    t0 = time.perf_counter(); status = 'ok'
    for s in range(NSTEP):
        U = S.step_bdf(st, hist, time=(s+1)*dt, max_newton=NSUB,
                       newton_tol=1e-13, newton_factor=1e-6, pin_p=True,
                       cgsfac=1e-3, cg_tol=1e-8, cg_max_iter=200000,
                       line_search=True)
        if not np.all(np.isfinite(U)):
            status = f'NaN@{s+1}'; break
        if np.abs(U[..., 0]).max() > 20.0:
            status = f'BLEWUP@{s+1}'; break
    ok = np.all(np.isfinite(U))
    return dict(a_mass=a_mass, dt=dt, kappa_p=(0.0 if kap is None else kap),
                status=status, steps=s+1,
                maxu=float(np.abs(U[..., 0]).max()) if ok else float('nan'),
                l2_u=float('nan'), wall=time.perf_counter()-t0)


def run(a_mass, kspec):
    m, _, _ = load(GRID)
    D, w, n = diff_matrix(m.N), lgl_weights(m.N), m.N+1
    dt = 1.5/a_mass                                  # BDF2 fac1 = 1.5
    st = SolverState(m, D, nu=1.0/RE, dt=dt, fac1=1.0, w_mom=1.0, w_mass=1.0)
    # zero the stored linearisation gradients too -- apply_L reads dfu_dx etc.
    _upd = st.update_linearisation
    st.update_linearisation = lambda fu, fv: _upd(np.zeros_like(fu),
                                                  np.zeros_like(fv))
    kap = {'off': None, 'half': a_mass/2.0, 'match': a_mass}.get(kspec)
    if kap is None and kspec not in ('off',):
        kap = float(kspec)
    st.dtau_p = None if kap is None else 1.0/kap

    xmax = m.xnod.max()
    out = [e for e in range(m.nelem) if abs(m.xnod[e, -1]-xmax) < 1e-9]
    OB = BC.apply_bc

    def bc2(mesh, U, **kw):                          # P+Z outlet, as elsewhere
        U = OB(mesh, U, **kw)
        for e in out:
            U[e, -1, :, 3] = -(D[-1, :-1] @ U[e, :-1, :, 3])/D[-1, -1]
        return U

    st.get_global_mask(pin_p=False)
    for e in out:
        st._global_mask[e, -1, :, 3] = 0.0
    S.apply_bc = bc2
    inl = lambda x, y, t: 6.0*y*(1.0-y)
    U = np.zeros((m.nelem, n, n, 4)); hist = [U.copy()]
    t0 = time.perf_counter(); status = 'ok'; lin_checked = False
    try:
        for s in range(NSTEP):
            U = S.step_bdf(st, hist, time=(s+1)*dt, max_newton=NSUB,
                           newton_tol=1e-13, newton_factor=1e-6,
                           custom_inlet=inl, pin_p=False, cgsfac=1e-3,
                           cg_tol=1e-8, cg_max_iter=200000, line_search=True)
            if not lin_checked:      # prove the patch took effect, once
                assert np.abs(st.dfu_dx).max() == 0.0 and np.abs(st.dfv_dy).max() == 0.0, \
                    'linearisation not zeroed -- this is NOT the Stokes operator'
                lin_checked = True
            if not np.all(np.isfinite(U)):
                status = f'NaN@{s+1}'; break
            if np.abs(U[..., 0]).max() > 20.0:
                status = f'BLEWUP@{s+1}'; break
    finally:
        S.apply_bc = OB
    ok = np.all(np.isfinite(U))
    maxu = float(np.abs(U[..., 0]).max()) if ok else float('nan')
    # Poiseuille is the exact Stokes answer, so L2 error is meaningful
    err = np.nan
    if ok:
        e2 = a2 = 0.0
        for e in range(m.nelem):
            ye = m.ynod[e][None, :]*np.ones((n, 1))
            wq = np.outer(w, w)*0.25*m.hx[e]*m.hy[e]
            e2 += np.sum((U[e, :, :, 0]-6.0*ye*(1.0-ye))**2*wq); a2 += wq.sum()
        err = float(np.sqrt(e2/a2))
    return dict(a_mass=a_mass, dt=dt, kappa_p=(0.0 if kap is None else kap),
                status=status, steps=s+1, maxu=maxu, l2_u=err,
                wall=time.perf_counter()-t0)


if __name__ == '__main__':
    dom = sys.argv[3] if len(sys.argv) > 3 else 'channel'
    r = (run_cavity if dom == 'cavity' else run)(float(sys.argv[1]), sys.argv[2])
    print(f'{dom:>9}', end='')
    print(f"{r['a_mass']:>9.0f}{r['dt']:>10.5f}{r['kappa_p']:>9.4g}"
          f"{r['status']:>13}{r['steps']:>7}{r['maxu']:>9.4f}"
          f"{r['l2_u']:>11.3e}{r['wall']:>8.0f}s", flush=True)
