"""Poiseuille Re=100 with the CORRECTED w_mom semantics (w multiplies only N(u)).

Target configuration: dt = 0.5, w_mom = 1.0, Jacobi.

    a_mass = fac1/dt = 3.0     a_flux = w_mom = 1.0     hist = 1/dt = 2.0

a_flux = 1 means the momentum residual at steady state is exactly N(u), carrying
the same weight as the continuity and vorticity rows.  If the decoupling works,
this should recover the analytic answers at a dt where LEGACY does not:

    exact:  dp over L=10 = 1.20,  outlet pressure CONSTANT across the channel
    legacy at dt=0.5:  dp = 1.23704 (3.1% high), outlet spread 0.266 (22% of dp)
    legacy at dt=1.0:  dp = 1.19999,             outlet spread 1.50e-03

Jacobi, not p-MG: measured earlier, MG costs more matvecs than it saves here.
"""
import os, sys, time
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix, lgl_weights
from lssem2d.lssem import SolverState, ls_coeffs
import lssem2d.solver as S

SC = os.path.dirname(os.path.abspath(__file__))
LX, LY, RE = 10.0, 1.0, 100.0
NU = 1.0*LY/RE
DP_EXACT = 12.0*NU*1.0/LY**2*LX             # 1.20
N, EX, EY = 8, 10, 2
RATE_TOL, T_MIN, MAXSTEP = 1.0e-9, 300.0, 40000
u_exact = lambda y: 6.0*y*(1.0-y)

CASES = [(0.5, 1.0), (0.5, None), (1.0, None), (0.1, 1.0), (0.05, 1.0)]


def run(dt, w_mom):
    mesh = build_channel(LX, LY, EX, EY, N, bcs=(3, 4, 1, 1))
    n = N+1
    pin = next((e, 0, 0) for e in range(mesh.nelem)
               if mesh.bc[e, 0] == 3 and mesh.bc[e, 2] == 1)
    for e in range(mesh.nelem):
        if mesh.bc[e, 1] == 4:
            mesh.bc[e, 1] = 0                 # FREE outflow
    st = SolverState(mesh, diff_matrix(N), nu=NU, dt=dt, fac1=1.0, w_mom=w_mom)
    inlet = lambda x, y, t: u_exact(y)
    U = np.zeros((mesh.nelem, n, n, 4)); hist = [U]
    t0 = time.perf_counter()
    for s in range(MAXSTEP):
        Up = hist[0].copy()
        U = S.step_bdf(st, hist, time=s*dt, max_newton=1, newton_tol=1e-12,
                       newton_factor=0.0, custom_inlet=inlet, pin_p=pin,
                       cgsfac=1e-3, cg_max_iter=40000, verbose=False)
        if not np.all(np.isfinite(U)):
            return None
        if (s+1)*dt >= T_MIN and np.max(np.abs(U-Up))/dt < RATE_TOL:
            break
    wall = time.perf_counter()-t0
    w = lgl_weights(N); xn, yn, hy = mesh.xnod, mesh.ynod, mesh.hy
    xmax = xn.max()
    ys, ps, us = [], [], []
    for e in range(mesh.nelem):
        if abs(xn[e, -1]-xmax) < 1e-9:
            for j in range(n):
                ys.append(yn[e, j]); ps.append(U[e, -1, j, 2]); us.append(U[e, -1, j, 0])
    o = np.argsort(ys)
    ys, ps, us = np.array(ys)[o], np.array(ps)[o], np.array(us)[o]
    k = np.concatenate(([True], np.diff(ys) > 1e-12))
    ys, ps, us = ys[k], ps[k], us[k]

    def pbar(edge):
        tot = a = 0.0
        for e in range(mesh.nelem):
            xe = xn[e, 0] if edge == 'in' else xn[e, -1]
            ref = xn.min() if edge == 'in' else xmax
            if abs(xe-ref) < 1e-9:
                i = 0 if edge == 'in' else -1
                tot += np.sum(w*U[e, i, :, 2])*(hy[e]/2); a += hy[e]
        return tot/a
    dp = float(pbar('in')-pbar('out'))
    am, af, hs = ls_coeffs(st)
    np.savez(f'{SC}/new_dt{dt:g}_w{"legacy" if w_mom is None else w_mom:}.npz',
             y=ys, p=ps, u=us)
    return dict(dt=dt, w=w_mom, a_mass=am, a_flux=af, hist=hs, steps=s+1,
                prof=float(np.sqrt(np.mean((us-u_exact(ys))**2))/1.5),
                dp=dp, dp_err=abs(dp-DP_EXACT)/DP_EXACT,
                spread=float(ps.max()-ps.min()), wall=wall, y=ys, p=ps)


print(f"Poiseuille Re={RE:g}, order {N}, {EX}x{EY}, Jacobi, CORRECTED w_mom semantics")
print(f"exact:  dp = {DP_EXACT},  outlet pressure constant across the channel\n")
print(f"{'dt':>6}{'w_mom':>8}{'a_mass':>8}{'a_flux':>8}{'steps':>7}"
      f"{'|u-u_ex|/Umax':>15}{'dp':>11}{'dp err':>10}{'p_out spread':>14}{'wall s':>8}")
best = None
for dt, w in CASES:
    r = run(dt, w)
    if r is None:
        print(f"{dt:>6}{str(w):>8}   DIVERGED"); continue
    lab = 'legacy' if w is None else f'{w:g}'
    print(f"{dt:>6}{lab:>8}{r['a_mass']:>8.2f}{r['a_flux']:>8.2f}{r['steps']:>7}"
          f"{r['prof']:>15.3e}{r['dp']:>11.5f}{r['dp_err']:>10.2e}"
          f"{r['spread']:>14.3e}{r['wall']:>8.1f}")
    if dt == 0.5 and w == 1.0:
        best = r

if best is not None:
    print(f"\nOUTLET PRESSURE PROFILE  dt=0.5, w_mom=1.0   (exact: constant at "
          f"{-DP_EXACT:.5f})")
    print(f"  {'y':>8}{'p':>14}{'p - mean':>14}")
    for yy, pp in zip(best['y'], best['p']):
        if abs(yy*10 - round(yy*10)) < 0.35 or yy in (best['y'][0], best['y'][-1]):
            print(f"  {yy:>8.4f}{pp:>14.6f}{pp-best['p'].mean():>14.3e}")
    print(f"  mean = {best['p'].mean():.6f}   spread = {best['spread']:.3e}"
          f"   = {100*best['spread']/DP_EXACT:.4f}% of dp")
