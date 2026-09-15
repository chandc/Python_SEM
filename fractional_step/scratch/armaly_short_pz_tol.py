"""CONTROL: does the linear-solve tolerance explain why the ER-1.94 short/P+Z run
fabricates a reattachment at x_r/S = 5.174, where the earlier cnos short/P+Z run
correctly showed the bubble running off the outlet?

Everything is held at armaly_run.py's settings EXCEPT (cgsfac, cg_tol), which are
set to the values the earlier bfs_outflow_ic.py path used: 1e-8 / 1e-10.

Saved to armaly_short_pz_cgs<...>_tol<...>.npz -- never overwrites armaly_short_pz.npz.
"""
import os, sys, time
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
sys.path.insert(0, SC)
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
import lssem2d
lssem2d.set_backend('numpy')
from fgrid import load
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState
import lssem2d.solver as S
import lssem2d.bc as BC
from armaly_run import GRIDS, NU, S_STEP, inlet_profile, reattach

OB = BC.apply_bc


def run(domain, cgsfac, tol, dt=1.0, cap=1500, wall=3000.0):
    m, _, _ = load(GRIDS[domain]); N = m.N; n = N+1
    D = diff_matrix(N)
    xmax = m.xnod.max()
    out = [e for e in range(m.nelem) if abs(m.xnod[e, -1]-xmax) < 1e-9]

    def bc2(mesh, U, **kw):
        U = OB(mesh, U, **kw)
        for e in out:
            U[e, -1, :, 3] = -(D[-1, :-1] @ U[e, :-1, :, 3])/D[-1, -1]
        return U

    st = SolverState(m, D, nu=NU, dt=dt, fac1=1.0, w_mom=1.0, w_mass=1.0)
    st.get_global_mask(pin_p=False)
    for e in out:
        st._global_mask[e, -1, :, 3] = 0.0
    S.apply_bc = bc2
    inl = lambda x, y, t: inlet_profile(y)
    U = np.zeros((m.nelem, n, n, 4)); h = [U.copy()]
    t0 = time.perf_counter(); status = 'CAP'; d = np.nan
    try:
        for s in range(cap):
            prev = h[0].copy()
            U = S.step_bdf(st, h, time=s*dt, max_newton=1, newton_tol=1e-12,
                           newton_factor=0.0, custom_inlet=inl, pin_p=False,
                           cgsfac=cgsfac, cg_tol=tol, cg_max_iter=300000)
            if not np.all(np.isfinite(U)):
                status = 'NaN'; break
            d = float(np.abs(U-prev).max())
            if np.abs(U[..., 0]).max() > 20.0:
                status = 'BLEWUP'; break
            if d < 1e-11:
                status = 'conv'; break
            if time.perf_counter()-t0 > wall:
                status = 'WALLCAP'; break
    finally:
        S.apply_bc = OB
    tag = f'{SC}/armaly_{domain}_pz_cgs{cgsfac:g}_tol{tol:g}.npz'
    np.savez(tag, U=U, xnod=m.xnod, ynod=m.ynod, hy=m.hy, N=N, nu=NU, dt=dt,
             status=status, steps=s+1, dU=d, cgsfac=cgsfac, cg_tol=tol)
    ok = np.all(np.isfinite(U))
    xr = reattach(U, m.xnod, m.ynod, m.hy, N) if ok else np.nan
    # wall shear at the outlet -- the diagnostic that separates the two behaviours
    tw_out = np.nan
    if ok:
        for e in out:
            if m.ynod[e, 0] < 0.01:
                tw_out = float(np.dot(D[0, :], U[e, -1, :, 0])*(2.0/m.hy[e]))
    print(f"  {domain}/P+Z cgsfac={cgsfac:g} tol={tol:g}: {status:>8} {s+1:>5} steps "
          f"|dU|={d:.3e}  max|u|={(np.abs(U[...,0]).max() if ok else np.nan):.4f}  "
          f"du/dy_outlet={tw_out:+.5f}  x_r/S={(xr/S_STEP if np.isfinite(xr) else np.nan):.3f}  "
          f"{time.perf_counter()-t0:.0f}s -> {os.path.basename(tag)}", flush=True)


if __name__ == '__main__':
    print("CONTROL: ER 1.94 short/P+Z, only (cgsfac, cg_tol) varied.")
    print("baseline armaly_short_pz.npz was cgsfac=1e-3 tol=1e-6 -> x_r/S = 5.174, "
          "du/dy_outlet = +1.41057\n")
    run('short', 1e-8, 1e-10)
