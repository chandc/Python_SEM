"""HOW does small dt fail at w_mom = w_mass = 1?  Instrumented, not just capped.

pois_dt_w1.py converged cleanly at dt = 5, 2, 1 (prof err 3.6e-06 .. 6.7e-06,
dp = 1.20000 to six figures) and then sat on dt = 0.5 for 54 minutes without
reaching the steady criterion.  That reproduces POISEUILLE_DT_STUDY.md sec 4,
where the same configuration ran 600 steps to dp = 15.22 and did not converge.
A wall-clock cap only tells us it failed; this prints the trajectory so we can
see WHICH failure it is:

    stall      |dU|/dt flattens above the tolerance, fields stay sane
    drift      dp walks away steadily -- converging to the wrong state
    oscillate  |dU|/dt cycles, no trend
    blow-up    |dU|/dt and max|u| grow without bound

The a priori block ratio (pois_blockratio.py) rules OUT pressure
under-weighting: at w = 1 the p/u diagonal ratio holds 0.75 -> 0.47 from dt = 5
to dt = 0.05, against legacy's 1256x collapse.  So this is an ITERATION failure,
not an ill-posed minimiser -- which is testable: max_newton = 1 makes successive
time steps ONE continuous Newton iteration (see step_bdf's ls_memory note), so
if sub-iterating each step fixes it, that identifies the cause exactly.
Hence the nsub = 5 row at the same dt.
"""
import os, sys, time, json
import numpy as np
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import lssem2d
lssem2d.set_backend('numpy')
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix, lgl_weights
from lssem2d.lssem import SolverState
import lssem2d.solver as S

LX, LY, RE = 10.0, 1.0, 100.0
NU = 1.0*LY/RE
DP_EXACT = 1.2
N, EX, EY = 8, 10, 2
RATE_TOL, T_MIN = 1.0e-9, 300.0
T_CAP, WALL_CAP = 600.0, 900.0        # 2x the physical time we require, 15 min
CGSFAC, CGTOL, CGMAX = 1e-8, 1e-10, 300000
u_exact = lambda y: 6.0*y*(1.0-y)

CASES = [(0.5, 1), (0.5, 5), (0.25, 1), (0.1, 1), (0.05, 1)]


def run(dt, nsub, report):
    mesh = build_channel(LX, LY, EX, EY, N, bcs=(3, 4, 1, 1))
    n = N+1
    pin = next((e, 0, 0) for e in range(mesh.nelem)
               if mesh.bc[e, 0] == 3 and mesh.bc[e, 2] == 1)
    for e in range(mesh.nelem):
        if mesh.bc[e, 1] == 4:
            mesh.bc[e, 1] = 0
    st = SolverState(mesh, diff_matrix(N), nu=NU, dt=dt, fac1=1.0,
                     w_mom=1.0, w_mass=1.0)
    inlet = lambda x, y, t: u_exact(y)
    w = lgl_weights(N); xn, hy = mesh.xnod, mesh.hy
    xmax, xmin = xn.max(), xn.min()

    def pbar(U, edge):
        tot = a = 0.0
        for e in range(mesh.nelem):
            xe = xn[e, 0] if edge == 'in' else xn[e, -1]
            ref = xmin if edge == 'in' else xmax
            if abs(xe-ref) < 1e-9:
                i = 0 if edge == 'in' else -1
                tot += np.sum(w*U[e, i, :, 2])*(hy[e]/2); a += hy[e]
        return tot/a

    U = np.zeros((mesh.nelem, n, n, 4)); hist = [U]
    t0 = time.perf_counter(); traj = []; status = 'nocon'
    nmax = int(T_CAP/dt)
    for s in range(nmax):
        Up = hist[0].copy()
        U = S.step_bdf(st, hist, time=s*dt, max_newton=nsub,
                       newton_tol=1e-12 if nsub > 1 else 1e-12,
                       newton_factor=0.0, custom_inlet=inlet, pin_p=pin,
                       cgsfac=CGSFAC, cg_tol=CGTOL, cg_max_iter=CGMAX)
        if not np.all(np.isfinite(U)):
            status = 'BLEW UP'; break
        rate = float(np.max(np.abs(U-Up))/dt)
        t = (s+1)*dt
        if s % report == 0 or s == nmax-1:
            dp = float(pbar(U, 'in')-pbar(U, 'out'))
            traj.append((t, rate, dp, float(np.abs(U[..., 0]).max())))
            print(f"    t={t:>7.1f}  step={s+1:>6}  |dU|/dt={rate:>10.3e}"
                  f"  dp={dp:>11.5f}  max|u|={np.abs(U[...,0]).max():>9.4f}",
                  flush=True)
        if t >= T_MIN and rate < RATE_TOL:
            status = 'conv'; break
        if time.perf_counter()-t0 > WALL_CAP:
            status = 'WALLCAP'; break
    else:
        status = 'TCAP'
    wall = time.perf_counter()-t0
    dp = float(pbar(U, 'in')-pbar(U, 'out')) if np.all(np.isfinite(U)) else float('nan')
    return dict(dt=dt, nsub=nsub, status=status, steps=s+1, t_end=(s+1)*dt,
                rate=rate, dp=dp, wall=wall, traj=traj)


print(f"Poiseuille Re={RE:g}, order {N}, {EX}x{EY}, w_mom = w_mass = 1")
print(f"tight solve, steady test |dU|/dt < {RATE_TOL:g} after t >= {T_MIN:g}")
print(f"caps: t <= {T_CAP:g}, wall <= {WALL_CAP:g}s.   exact dp = {DP_EXACT}\n")
out = []
for dt, nsub in CASES:
    print(f"=== dt = {dt:g}, nsub = {nsub} "
          f"(a_mass = {1.5/dt:.2f}, a_flux = 1, dt_eff = {dt:g}) ===", flush=True)
    r = run(dt, nsub, report=max(1, int(25.0/dt)))     # report every 25 time units
    out.append(r)
    print(f"  -> {r['status']}  after {r['steps']} steps (t = {r['t_end']:g}), "
          f"dp = {r['dp']:.5f}, |dU|/dt = {r['rate']:.3e}, {r['wall']:.0f}s\n",
          flush=True)
    with open(f'{SC}/pois_dt_small.json', 'w') as fh:
        json.dump(out, fh, indent=1)

print(f"{'dt':>6}{'nsub':>6}{'status':>9}{'steps':>7}{'t_end':>8}"
      f"{'|dU|/dt':>12}{'dp':>11}{'dp err':>10}")
for r in out:
    print(f"{r['dt']:>6g}{r['nsub']:>6}{r['status']:>9}{r['steps']:>7}"
          f"{r['t_end']:>8g}{r['rate']:>12.3e}{r['dp']:>11.5f}"
          f"{abs(r['dp']-DP_EXACT)/DP_EXACT:>10.2e}")
