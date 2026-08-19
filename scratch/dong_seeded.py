"""Basin probe: is the exact Poiseuille solution a fixed point of the Dong
outlet at the dt where the COLD START blows up (0.1, 0.05)?

OUTFLOW_BC_STUDY.md sec 6 found free outflow's small-dt failure is basin
capture, not instability: seeded with the exact field, every dt holds it
bit-exactly.  Exact Poiseuille zeroes the Dong rows too (p = 0, du/dx = 0,
v = 0 at the exit), so if seeded runs hold, Dong's dt = 0.1 blow-up is the
same cold-start basin story; if they drift, the boundary rows themselves are
destabilising.

    uv run --quiet python scratch/dong_seeded.py
"""
import os, sys, time
SC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SC)
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import numpy as np
import lssem2d
lssem2d.set_backend('numpy')
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState
import lssem2d.solver as S

RE = 100.0


def seeded(dt, D0=0.0, nsteps=300):
    m = build_channel(L_x=10.0, L_y=1.0, E_x=10, E_y=2, N=8, bcs=(3, 6, 1, 1))
    D = diff_matrix(8); n = 9
    st = SolverState(m, D, nu=1.0 / RE, dt=dt, fac1=1.0, w_mom=1.0, w_mass=1.0)
    st.obc_D0 = D0
    U = np.zeros((m.nelem, n, n, 4))
    for e in range(m.nelem):
        ye = m.ynod[e][None, :]
        xe = m.xnod[e][:, None]
        U[e, :, :, 0] = 6.0 * ye * (1.0 - ye)
        U[e, :, :, 2] = 0.12 * (10.0 - xe)      # dp/dx = -12 nu, p(out) = 0
        U[e, :, :, 3] = 12.0 * ye - 6.0
    U0 = U.copy()
    inl = lambda x, y, t: 6.0 * np.asarray(y) * (1.0 - np.asarray(y))
    h = [U.copy()]
    status = 'held'
    for s in range(nsteps):
        U = S.step_bdf(st, h, time=(s + 1) * dt, max_newton=1,
                       newton_tol=1e-13, newton_factor=0.0, custom_inlet=inl,
                       pin_p=False, cgsfac=1e-8, cg_tol=1e-10,
                       cg_max_iter=200000)
        if not np.all(np.isfinite(U)) or np.abs(U[..., 0]).max() > 20.0:
            status = f'BLEWUP@step{s + 1}'
            break
    drift = float(np.abs(U - U0).max()) if np.all(np.isfinite(U)) else np.nan
    np.savez(f'{SC}/dong_seeded_dt{dt:g}_D0{D0:g}.npz', U=U, status=status)
    return status, drift


if __name__ == '__main__':
    print('Seeded-with-exact-solution probe, Dong outlet, [0,10] 10x2 N=8, '
          '300 steps, nsub=1:\n')
    print(f"{'dt':>6}{'D0':>5}{'status':>15}{'max drift |U-U0|':>19}")
    for dt in (0.1, 0.05):
        for D0 in (0.0, 1.0):
            t0 = time.perf_counter()
            stat, drift = seeded(dt, D0)
            print(f'{dt:>6g}{D0:>5g}{stat:>15}{drift:>19.3e}'
                  f'   ({time.perf_counter() - t0:.0f}s)', flush=True)
