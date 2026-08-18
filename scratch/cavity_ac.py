"""Artificial compressibility on the Ghia Re=1000 lid-driven cavity.

    uv run --quiet python scratch/cavity_ac.py <dt> <off|match|half|VALUE>

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


def run(dt, kspec):
    n = N+1
    mesh = build_channel(1.0, 1.0, EX, EX, N, bcs=(1, 1, 1, 2))
    st = SolverState(mesh, diff_matrix(N), nu=1.0/RE, dt=dt, fac1=1.0,
                     w_mom=1.0, w_mass=1.0)
    a_mass = 1.5/dt                                  # BDF2 fac1 = 1.5
    if kspec == 'off':
        kap = None
    elif kspec == 'match':
        kap = a_mass
    elif kspec == 'half':
        kap = a_mass/2.0
    else:
        kap = float(kspec)
    st.dtau_p = None if kap is None else 1.0/kap
    U = np.zeros((mesh.nelem, n, n, 4)); hist = [U.copy()]
    t0 = time.perf_counter(); status = 'CAP'; dU = np.nan
    for s in range(MAXSTEP):
        Up = hist[0].copy()
        U = S.step_bdf(st, hist, time=(s+1)*dt, max_newton=5, newton_tol=1e-13,
                       newton_factor=1e-6, pin_p=True, cgsfac=1e-3,
                       cg_tol=1e-8, cg_max_iter=60000, line_search=True)
        if not np.all(np.isfinite(U)):
            status = f'NaN@{(s+1)*dt:.2f}'; break
        if np.abs(U[..., 0]).max() > 5.0:
            status = f'BLEWUP@{(s+1)*dt:.2f}'; break
        dU = float(np.abs(U-Up).max())
        if s > 3 and dU < STEADY:
            status = 'conv'; break
    ok = np.all(np.isfinite(U))
    rms = umax = np.nan
    if ok:
        ys, us = centreline_u(mesh, U, n)
        ui = np.interp(GH['ghia_y'], ys, us)
        rms = float(np.sqrt(np.mean((ui-GH['ghia_u'])**2)))
        umax = float(np.abs(U[..., 0]).max())
    np.savez(f'{SC}/cavity_ac_dt{dt:g}_{kspec}.npz', U=U, xnod=mesh.xnod,
             ynod=mesh.ynod, dt=dt, kappa_p=(0.0 if kap is None else kap),
             a_mass=a_mass, status=status, steps=s+1, dU=dU, rms=rms)
    return dict(dt=dt, a_mass=a_mass, kap=(0.0 if kap is None else kap),
                status=status, steps=s+1, dU=dU, rms=rms, umax=umax,
                wall=time.perf_counter()-t0)


if __name__ == '__main__':
    dt = float(sys.argv[1]); kspec = sys.argv[2]
    r = run(dt, kspec)
    print(f"{kspec:>7}{r['dt']:>8g}{r['a_mass']:>8.4g}{r['kap']:>9.4g}"
          f"{r['status']:>13}{r['steps']:>7}{r['wall']:>7.0f}s{r['umax']:>8.4f}"
          f"{r['dU']:>11.2e}{r['rms']:>11.4e}", flush=True)
