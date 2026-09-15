"""The small-dt sweep with the outflow boundary REMOVED and nothing else changed.

pois_outflow_test.py showed the dt = 0.5 period-2 cycle is manufactured at the
outflow: the step-to-step change is 9.2 in the last element and 4.5e-04 four
elements upstream -- 20,000x -- and a periodic channel at the same dt and the
same weights converges monotonically.  But that comparison also changed the
geometry (2pi x 2, 1x2 elements), so it was not controlled.

This is the controlled version.  IDENTICAL mesh to the Poiseuille runs --
10 x 1 domain, 10 x 2 elements, order 8, nu = 0.01, same exact solution
u = 6y(1-y) -- with exactly one difference:

    Poiseuille   inlet (bc 3) + FREE outflow (bc 0), pressure pinned at inlet
    here         streamwise PERIODIC, driven by the body force that sustains
                 the same parabola:  u'' = -12  =>  f_x = 12*nu = 0.12
                 (this is the study's own dp/dx = 0.12), pressure pinned once

Same operator, same weights, same dt, same resolution, same exact answer.  If
the small-dt cycle vanishes here, the instability attributed in
POISEUILLE_DT_STUDY.md sec 4 to the momentum weight is an outflow-BC artifact.

The discriminator does not need full convergence: a 2-cycle locks |dU| to a
CONSTANT while |U(k)-U(k-2)| collapses, whereas healthy convergence drives both
down together.  Both are printed.
"""
import os, sys, time, json
import numpy as np
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import lssem2d
lssem2d.set_backend('numpy')
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState, ls_coeffs
import lssem2d.solver as S

LX, LY, NU = 10.0, 1.0, 0.01
N, EX, EY = 8, 10, 2
FX = 12.0*NU                      # sustains u = 6y(1-y):  0 = f + nu*u'', u'' = -12
T_CAP, WALL_CAP = 150.0, 500.0
RATE_TOL = 1.0e-9
CGSFAC, CGTOL, CGMAX = 1e-8, 1e-10, 300000
u_exact = lambda y: 6.0*y*(1.0-y)
DTS = [1.0, 0.5, 0.25, 0.1, 0.05]


def run(dt):
    m = build_channel(LX, LY, EX, EY, N, bcs=(0, 0, 1, 1))
    m.periodic_x = LX
    m.compute_global_indices()
    n = N+1
    st = SolverState(m, diff_matrix(N), nu=NU, dt=dt, fac1=1.0,
                     w_mom=1.0, w_mass=1.0)
    _, a_flux, _ = ls_coeffs(st)
    f = np.zeros((m.nelem, n, n, 4))
    f[..., 0] = a_flux*FX                       # body force carries the row weight
    pin = (0, n//2, n//2)
    U = np.zeros((m.nelem, n, n, 4)); hist = [U]
    hold = {}; t0 = time.perf_counter(); status = 'TCAP'; traj = []
    nmax = int(T_CAP/dt)
    rep = max(1, int(25.0/dt))
    for s in range(nmax):
        U = S.step_bdf(st, hist, time=s*dt, max_newton=1, newton_tol=1e-12,
                       newton_factor=0.0, f_known=f, pin_p=pin,
                       cgsfac=CGSFAC, cg_tol=CGTOL, cg_max_iter=CGMAX)
        if not np.all(np.isfinite(U)):
            status = 'BLEWUP'; break
        hold[s] = U.copy()
        d1 = np.abs(U-hold[s-1]).max() if s >= 1 else np.nan
        d2 = np.abs(U-hold[s-2]).max() if s >= 2 else np.nan
        hold.pop(s-3, None)
        if s % rep == 0 or s == nmax-1:
            traj.append(((s+1)*dt, float(d1), float(d2)))
            print(f"    t={(s+1)*dt:>7.1f} step={s+1:>6}  |dU|={d1:>11.4e}"
                  f"  |U(k)-U(k-2)|={d2:>11.4e}  max|u|={np.abs(U[...,0]).max():>8.4f}",
                  flush=True)
        if s >= 1 and d1/dt < RATE_TOL:
            status = 'conv'; break
        if time.perf_counter()-t0 > WALL_CAP:
            status = 'WALLCAP'; break
    # error against the exact parabola
    prof = float(np.sqrt(np.mean((U[..., 0] - u_exact(m.ynod)[:, None, :])**2))/1.5)
    return dict(dt=dt, status=status, steps=s+1, t_end=(s+1)*dt,
                d1=float(d1), d2=float(d2), rate=float(d1)/dt, prof=prof,
                wall=time.perf_counter()-t0, traj=traj)


print("PERIODIC channel -- identical mesh/order/nu/exact solution to the")
print(f"Poiseuille runs, outflow boundary removed.  f_x = 12*nu = {FX:g}")
print(f"w_mom = w_mass = 1.  caps: t <= {T_CAP:g}, wall <= {WALL_CAP:g}s\n")
out = []
for dt in DTS:
    print(f"=== dt = {dt:g}  (a_mass = {1.5/dt:.2f}, a_flux = 1) ===", flush=True)
    r = run(dt)
    out.append(r)
    print(f"  -> {r['status']}  {r['steps']} steps (t={r['t_end']:g}), "
          f"|dU|={r['d1']:.3e}, |dU|/dt={r['rate']:.3e}, prof err={r['prof']:.3e}, "
          f"{r['wall']:.0f}s\n", flush=True)
    with open(f'{SC}/pois_periodic_dt.json', 'w') as fh:
        json.dump(out, fh, indent=1)

print(f"{'dt':>6}{'a_mass':>8}{'status':>9}{'steps':>7}{'t_end':>8}"
      f"{'|dU|':>12}{'|dU|/dt':>12}{'prof err':>12}")
for r in out:
    print(f"{r['dt']:>6g}{1.5/r['dt']:>8.2f}{r['status']:>9}{r['steps']:>7}"
          f"{r['t_end']:>8g}{r['d1']:>12.3e}{r['rate']:>12.3e}{r['prof']:>12.3e}")
print("\nA 2-cycle pins |dU| to a constant while |U(k)-U(k-2)| collapses.")
print("Healthy convergence drives both down together.")
