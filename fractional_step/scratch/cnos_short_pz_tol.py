"""Second half of the 2x2 control: the EARLY cnos ER-2.00 short/P+Z case, rerun at
the LOOSE tolerance that armaly_run.py uses (cgsfac=1e-3, cg_tol=1e-6).

If loose CG is what fabricates a reattachment, this run -- same grid, same nu, same
IC as bfs_pz_state.npz, only the solve tolerance changed -- should turn its outlet
wall shear from -1.783 to positive and invent an x_r near the exit.
"""
import os, sys, time
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
sys.path.insert(0, SC)
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
import lssem2d
lssem2d.set_backend('numpy')
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState
import lssem2d.solver as S
import lssem2d.bc as BC
from bfs_outflow_ic import build, reattach

OB = BC.apply_bc
RE = 389.0


def run(cgsfac, tol, dt=1.0, cap=1500, wall=3000.0):
    m, n, pin = build(); N = m.N
    D = diff_matrix(N)
    xmax = m.xnod.max()
    out = [e for e in range(m.nelem) if abs(m.xnod[e, -1]-xmax) < 1e-9]

    def bc2(mesh, U, **kw):
        U = OB(mesh, U, **kw)
        for e in out:
            U[e, -1, :, 3] = -(D[-1, :-1] @ U[e, :-1, :, 3])/D[-1, -1]
        return U

    st = SolverState(m, D, nu=1.0/RE, dt=dt, fac1=1.0, w_mom=1.0, w_mass=1.0)
    st.get_global_mask(pin_p=False)
    for e in out:
        st._global_mask[e, -1, :, 2] = 0.0
        st._global_mask[e, -1, :, 3] = 0.0
    S.apply_bc = bc2
    inlet = lambda x, y, t: 6.0*((y-0.5)/0.5)*(1.0-(y-0.5)/0.5)
    U = np.zeros((m.nelem, n, n, 4)); h = [U.copy()]
    t0 = time.perf_counter(); status = 'CAP'; d = np.nan
    try:
        for s in range(cap):
            prev = h[0].copy()
            U = S.step_bdf(st, h, time=s*dt, max_newton=1, newton_tol=1e-12,
                           newton_factor=0.0, custom_inlet=inlet, pin_p=False,
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
    tag = f'{SC}/cnos_short_pz_cgs{cgsfac:g}_tol{tol:g}.npz'
    np.savez(tag, U=U, xnod=m.xnod, ynod=m.ynod, hy=m.hy, N=N, nu=1.0/RE, dt=dt,
             status=status, steps=s+1, dU=d, cgsfac=cgsfac, cg_tol=tol)
    ok = np.all(np.isfinite(U))
    xr = reattach(U, m, D) if ok else np.nan
    tw_out = np.nan
    for e in out:
        if m.bc[e, 2] == 1:
            tw_out = float(np.dot(D[0, :], U[e, -1, :, 0])*(2.0/m.hy[e]))
    print(f"  cnos short/P+Z cgsfac={cgsfac:g} tol={tol:g}: {status:>8} {s+1:>5} steps "
          f"|dU|={d:.3e}  max|u|={(np.abs(U[...,0]).max() if ok else np.nan):.4f}  "
          f"du/dy_outlet={tw_out:+.5f}  x_r/S={(xr/0.5 if np.isfinite(xr) else np.nan):.3f}  "
          f"{time.perf_counter()-t0:.0f}s -> {os.path.basename(tag)}", flush=True)


if __name__ == '__main__':
    print("CONTROL: cnos ER 2.00 short/P+Z, loose tolerance.")
    print("baseline bfs_pz_state.npz was cgsfac=1e-8 tol=1e-10 -> no x_r in domain, "
          "du/dy_outlet = -1.78285\n")
    run(1e-3, 1e-6)
