"""Artificial compressibility on the Ghia Re=1000 lid-driven cavity.

    uv run --quiet python scratch/cavity_ac.py <dt> <off|match|half|VALUE> [w_mass]

WHY THE CAVITY IS THE RIGHT TEST FOR THIS.  It removes the two variables that
confounded every earlier AC run:

  * NO OUTFLOW.  The domain is closed, so the whole outflow-BC question -- free
    vs p=0 vs P+Z, the soft modes, the second attractor -- simply does not arise.
  * GENUINELY NONLINEAR with a real residual.  Unlike plane Poiseuille and the
    periodic channel (parallel flows, u.grad u == 0, R ~ 0, where the weighting
    provably cannot matter), the cavity recirculates and has lid corner
    singularities, so R is substantial.  That is the regime where a_mass and
    kappa_p actually bite.

And it has a hard benchmark: Ghia, Ghia & Shin (1982), u(y) on the vertical
centreline x = 0.5, 17 tabulated points (reference in cavity_re1000_data.npz).

a_mass = w_mass*fac1/dt = 1.5/dt with w_mom = w_mass = 1 (time-accurate).
kappa_p: 'off' = no AC, 'match' = a_mass, 'half' = a_mass/2 (the value that
worked best on the BFS), or an explicit number.

Sub-iterations matter: the AC term only vanishes at sub-iteration convergence,
so max_newton = 5 with a tight newton_factor.
"""
import os, sys, time
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
sys.path.insert(0, SC)
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
import lssem2d
lssem2d.set_backend('numpy')
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState
import lssem2d.solver as S

RE = 1000.0
EX, N = 6, 10                     # 6x6 elements, order 10 -> 17424 dof
MAXSTEP, STEADY = 3000, 1.0e-9
PRINT_EVERY = 25                  # progress trace + checkpoint; see run()
STALL_N = 10                      # |dU| flat over this many steps => fixed point
GH = np.load('cavity_re1000_data.npz')


def lagrange(xn, xq):
    n = len(xn); w = np.ones(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                w[i] /= (xn[i]-xn[j])
    dd = xq-xn
    if np.any(np.abs(dd) < 1e-13):
        L = np.zeros(n); L[np.argmin(np.abs(dd))] = 1.0; return L
    num = w/dd
    return num/num.sum()


def centreline_u(mesh, U, n):
    ys, us = [], []
    for e in range(mesh.nelem):
        xs = mesh.xnod[e]
        if xs[0]-1e-9 <= 0.5 <= xs[-1]+1e-9:
            L = lagrange(xs, 0.5)
            for j in range(n):
                ys.append(mesh.ynod[e, j]); us.append(np.dot(L, U[e, :, j, 0]))
    o = np.argsort(ys); ys, us = np.array(ys)[o], np.array(us)[o]
    k = np.concatenate(([True], np.diff(ys) > 1e-9))
    return ys[k], us[k]


def run(dt, kspec, w_mass=1.0, ic=None, N=N):
    """w_mass = 0 is the PURE STEADY FORM.  ls_coeffs returns s = w_mass/dt = 0,
    so a_mass and hist_scale both vanish and the functional collapses to

        J = int[ w_mom^2 (N_1^2 + N_2^2) + (div u)^2 + (om + u_y - v_x)^2 ]

    with no time-derivative term at all.  dt is then decorative -- it survives
    only in the `time` argument used to evaluate boundary conditions, which are
    steady here -- so each "step" is simply a fresh Newton sweep on the steady
    system and the |dU| test measures Newton convergence, not a physical
    transient.  Line search matters much more in this mode: there is no mass term
    damping the update, so an undamped Newton step from rest at Re = 1000 can
    overshoot badly."""
    n = N+1
    mesh = build_channel(1.0, 1.0, EX, EX, N, bcs=(1, 1, 1, 2))
    st = SolverState(mesh, diff_matrix(N), nu=1.0/RE, dt=dt, fac1=1.0,
                     w_mom=1.0, w_mass=w_mass)
    a_mass = w_mass*1.5/dt                           # BDF2 fac1 = 1.5
    if kspec == 'off':
        kap = None
    elif kspec in ('match', 'half'):
        kap = a_mass if kspec == 'match' else a_mass/2.0
        if kap == 0.0:                               # w_mass = 0 => a_mass = 0
            raise SystemExit(f"kspec '{kspec}' is meaningless at w_mass = 0: it "
                             f"scales kappa_p off a_mass, which is 0 here. "
                             f"Pass 'off' or an explicit kappa_p.")
    else:
        kap = float(kspec)
    st.dtau_p = None if kap is None else 1.0/kap
    # In steady mode the loop index is a Newton sweep, not a time level, so
    # report the step rather than a fictitious t.
    stamp = (lambda s: f'step{s+1}') if w_mass == 0.0 else \
            (lambda s: f't={(s+1)*dt:.2f}')
    # w_mass goes in the FILENAME, not just the payload: the steady run shares dt
    # and kspec with a transient one and would otherwise overwrite it.  A
    # filename collision has already destroyed one completed run in this repo
    # (GARTLING_VALIDATION.md sec 11).
    tag = '' if w_mass == 1.0 else f'_wm{w_mass:g}'
    tag += '' if ic is None else '_restart'
    tag += '' if N == 10 else f'_N{N}'          # N=10 keeps the historical names
    out = f'{SC}/cavity_ac_dt{dt:g}_{kspec}{tag}.npz'

    def save(U, status, steps, dU, trace):
        """Checkpoint.  This used to run only after the loop, so killing a stalled
        run threw the field away and the whole thing had to be re-solved -- which
        is the one thing the save-every-run rule exists to prevent."""
        ok = np.all(np.isfinite(U))
        rms = umax = np.nan
        if ok:
            ys, us = centreline_u(mesh, U, n)
            rms = float(np.sqrt(np.mean((np.interp(GH['ghia_y'], ys, us)
                                         - GH['ghia_u'])**2)))
            umax = float(np.abs(U[..., 0]).max())
        np.savez(out, U=U, xnod=mesh.xnod, ynod=mesh.ynod, dt=dt, N=N,
                 kappa_p=(0.0 if kap is None else kap), a_mass=a_mass,
                 w_mass=w_mass, status=status, steps=steps, dU=dU, rms=rms,
                 trace=np.array(trace))
        return rms, umax

    # ic = path to an npz with a compatible U.  Used to ask whether a fixed point
    # is REACHABLE from rest as opposed to merely STABLE once you are on it --
    # the two came apart for the steady form (w_mass = 0), which converges from
    # rest to a spurious oscillatory state.
    if ic is None:
        U = np.zeros((mesh.nelem, n, n, 4))
    else:
        U = np.load(ic, allow_pickle=True)['U'].copy()
        assert U.shape == (mesh.nelem, n, n, 4), \
            f'ic {ic} has shape {U.shape}, mesh wants {(mesh.nelem, n, n, 4)}'
        print(f'    restart from {ic}', flush=True)
    hist = [U.copy()]
    t0 = time.perf_counter(); status = 'CAP'; dU = np.nan
    # Trace the convergence, don't just report the last value.  This script used
    # to print nothing until it finished, so a run that stalled looked identical
    # to one making progress for as long as it took -- unacceptable for the
    # steady form, where whether Newton converges from rest at all is the
    # question being asked.
    trace = []
    for s in range(MAXSTEP):
        Up = hist[0].copy()
        U = S.step_bdf(st, hist, time=(s+1)*dt, max_newton=5, newton_tol=1e-13,
                       newton_factor=1e-6, pin_p=True, cgsfac=1e-3,
                       cg_tol=1e-8, cg_max_iter=60000, line_search=True)
        if not np.all(np.isfinite(U)):
            status = f'NaN@{stamp(s)}'; break
        if np.abs(U[..., 0]).max() > 5.0:
            status = f'BLEWUP@{stamp(s)}'; break
        dU = float(np.abs(U-Up).max())
        trace.append((s+1, dU, float(np.abs(U[..., 0]).max())))
        if s < 5 or (s+1) % PRINT_EVERY == 0:
            print(f'    {stamp(s):>10}  |dU| = {dU:.3e}  max|u| = '
                  f'{trace[-1][2]:.4f}  {time.perf_counter()-t0:6.0f}s',
                  flush=True)
            save(U, status, s+1, dU, trace)
        if s > 3 and dU < STEADY:
            status = 'conv'; break
        # STAGNATION EXIT.  |dU| cannot go below the solver's own floor -- set by
        # cg_tol and max_newton, measured at 6.9e-08 for the steady form here --
        # so a STEADY threshold under that floor is unreachable and the run
        # burns to MAXSTEP having converged at step ~50.  That exact trap has
        # cost this project hours twice.  Detect the fixed point directly:
        # |dU| stopped changing => nothing more is happening, whatever the
        # threshold says.
        if len(trace) > STALL_N:
            recent = [t[1] for t in trace[-STALL_N:]]
            if max(recent) - min(recent) <= 1e-3*max(recent):
                status = f'stalled@{stamp(s)}|dU|={dU:.3e}'; break
    rms, umax = save(U, status, s+1, dU, trace)
    return dict(dt=dt, w_mass=w_mass, a_mass=a_mass,
                kap=(0.0 if kap is None else kap), status=status, steps=s+1,
                dU=dU, rms=rms, umax=umax, wall=time.perf_counter()-t0)


if __name__ == '__main__':
    dt = float(sys.argv[1]); kspec = sys.argv[2]
    w_mass = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    ic = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] != '-' else None
    Nord = int(sys.argv[5]) if len(sys.argv) > 5 else N
    r = run(dt, kspec, w_mass, ic, Nord)
    print(f"{kspec:>7}{r['dt']:>8g}{r['w_mass']:>8g}{r['a_mass']:>8.4g}"
          f"{r['kap']:>9.4g}{r['status']:>14}{r['steps']:>7}{r['wall']:>7.0f}s"
          f"{r['umax']:>8.4f}{r['dU']:>11.2e}{r['rms']:>11.4e}", flush=True)
