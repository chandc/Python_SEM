"""Are the spurious steady states (sec 5.3) real, or a line-search artefact?

    uv run --quiet python scratch/cavity_steady_ls.py <on|off> <rest|restart>

WHY.  newton_step's backtracking loop runs at most max_backtrack = 25 halvings
and then takes the step ANYWAY, with no failure signal:

    for _ in range(max_backtrack):
        if _ls_merit(... U + alpha*dU ...) <= (1 - 1e-4*alpha)*J_ref: break
        alpha *= 0.5
    U_new = U + alpha*dU              # <- taken even if nothing was accepted

0.5**25 = 2.98e-08.  Measured on the steady cavity, alpha collapses to exactly
that at sweep 26 and the state stops moving: |dU| = alpha*|step| ~ 2.98e-08*2.3
= 6.9e-08, constant, which cavity_ac.py's stagnation test then reports as
convergence.  So the "converged steady fixed point" of sec 5.3 may be nothing but
a stalled line search.

This script settles it by running the same problem with line_search=False, from
rest and from the converged transient field, and recording alpha every sweep.

    scratch/cavity_steadyls_{on|off}_{rest|restart}.npz
"""
import os, sys, time
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
import lssem2d
lssem2d.set_backend('numpy')
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState
import lssem2d.solver as S

RE, EX, N = 1000.0, 6, 10
MAXSWEEP = 400
ALPHA_MIN = 1e-6            # below this the line search has effectively stalled
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


def rms_vs_ghia(mesh, U, n):
    ys, us = [], []
    for e in range(mesh.nelem):
        xs = mesh.xnod[e]
        if xs[0]-1e-9 <= 0.5 <= xs[-1]+1e-9:
            L = lagrange(xs, 0.5)
            for j in range(n):
                ys.append(mesh.ynod[e, j]); us.append(np.dot(L, U[e, :, j, 0]))
    o = np.argsort(ys); ys, us = np.array(ys)[o], np.array(us)[o]
    k = np.concatenate(([True], np.diff(ys) > 1e-9)); ys, us = ys[k], us[k]
    return float(np.sqrt(np.mean((np.interp(GH['ghia_y'], ys, us)
                                  - GH['ghia_u'])**2)))


def run(ls, start):
    n = N+1
    mesh = build_channel(1.0, 1.0, EX, EX, N, bcs=(1, 1, 1, 2))
    st = SolverState(mesh, diff_matrix(N), nu=1.0/RE, dt=1.0, fac1=1.0,
                     w_mom=1.0, w_mass=0.0)          # a_mass = 0: steady form
    st.dtau_p = None
    if start == 'rest':
        U = np.zeros((mesh.nelem, n, n, 4))
    else:
        U = np.load(f'{SC}/cavity_ac_dt0.05_match.npz',
                    allow_pickle=True)['U'].copy()
    hist = [U.copy()]
    T = []                                            # sweep, alpha, |dU|, max|u|
    t0 = time.perf_counter(); status = 'CAP'
    print(f"{'sweep':>6}{'alpha':>12}{'|dU|':>12}{'max|u|':>10}", flush=True)
    for s in range(MAXSWEEP):
        Up = hist[0].copy()
        U = S.step_bdf(st, hist, time=(s+1), max_newton=5, newton_tol=1e-13,
                       newton_factor=1e-6, pin_p=True, cgsfac=1e-3,
                       cg_tol=1e-8, cg_max_iter=60000, line_search=ls)
        a = float(getattr(st, '_last_alpha', 1.0))
        if not np.all(np.isfinite(U)):
            status = f'NaN@{s+1}'; break
        um = float(np.abs(U[..., 0]).max())
        d = float(np.abs(U-Up).max())
        T.append((s+1, a, d, um))
        if s < 6 or (s+1) % 10 == 0:
            print(f'{s+1:>6}{a:>12.3e}{d:>12.3e}{um:>10.4f}', flush=True)
        if um > 20.0:
            status = f'BLEWUP@{s+1}'; break
        if ls and a < ALPHA_MIN:
            # Report the stall as a STALL, never as convergence.  Conflating the
            # two is what produced the sec 5.3 claim in the first place.
            status = f'LS_STALL@{s+1}(alpha={a:.2e})'; break
        if s > 3 and d < 1e-10:
            status = 'conv'; break
    ok = np.all(np.isfinite(U))
    rms = rms_vs_ghia(mesh, U, n) if ok else np.nan
    np.savez(f'{SC}/cavity_steadyls_{"on" if ls else "off"}_{start}.npz',
             U=U, xnod=mesh.xnod, ynod=mesh.ynod, trace=np.array(T),
             line_search=ls, start=start, status=status, rms=rms,
             sweeps=len(T), wall=time.perf_counter()-t0)
    return status, rms, len(T), time.perf_counter()-t0


if __name__ == '__main__':
    ls = sys.argv[1] == 'on'
    start = sys.argv[2]
    st_, rms, nsw, w = run(ls, start)
    print(f'\nline_search={"on" if ls else "off":<3} start={start:<8} '
          f'{st_:<28} sweeps={nsw:<4} wall={w:6.0f}s  RMS u={rms:.4e}',
          flush=True)
