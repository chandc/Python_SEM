"""Where between dt = 1 (converges) and dt = 0.5 (does not) does it break?

pois_dt_w1.py converged at dt = 5, 2, 1 and failed at 0.5, all at
w_mom = w_mass = 1.  Nothing so far says 0.5 is special -- it is just the first
value below 1 that was tried.  The only thing varying monotonically is

    a_mass = fac1/dt  =  0.30, 0.75, 1.50, 3.00, 6.00, 15.0, 30.0
    for dt =              5     2     1     0.5   0.25  0.1   0.05

with a_flux pinned at 1, so every dt < 1 is legacy's row scaled by 1/dt: the
momentum equation weighted 1/dt times more heavily against the constraints.
dt = 1 is where (a_mass, a_flux) = (1.5, 1) coincides with legacy exactly.

Two possibilities, and they imply different things:

  SHARP   a stability threshold in (0.5, 1).  Then there is a critical weight
          ratio and it is worth finding, because it bounds the usable range.
  GRADUAL convergence just gets slower until it misses the budget.  Then
          "fails" is a statement about the iteration count, not the method,
          and the fix is more steps or better globalisation.

Distinguished by dp and the final rate, not by the converged/not flag: a
threshold shows dp jumping to O(10), slow convergence shows dp near 1.2 with
the rate still falling.
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

LX, LY, NU, DP_EXACT = 10.0, 1.0, 0.01, 1.2
N, EX, EY = 8, 10, 2
RATE_TOL, T_MIN, WALL_CAP = 1.0e-9, 300.0, 700.0
CGSFAC, CGTOL, CGMAX = 1e-8, 1e-10, 300000
u_exact = lambda y: 6.0*y*(1.0-y)
DTS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.55, 0.5]


def run(dt):
    m = build_channel(LX, LY, EX, EY, N, bcs=(3, 4, 1, 1))
    n = N+1
    pin = next((e, 0, 0) for e in range(m.nelem)
               if m.bc[e, 0] == 3 and m.bc[e, 2] == 1)
    for e in range(m.nelem):
        if m.bc[e, 1] == 4:
            m.bc[e, 1] = 0
    st = SolverState(m, diff_matrix(N), nu=NU, dt=dt, fac1=1.0,
                     w_mom=1.0, w_mass=1.0)
    inlet = lambda x, y, t: u_exact(y)
    w = lgl_weights(N); xn, hy = m.xnod, m.hy
    xmax, xmin = xn.max(), xn.min()

    def pbar(U, edge):
        tot = a = 0.0
        for e in range(m.nelem):
            xe = xn[e, 0] if edge == 'in' else xn[e, -1]
            ref = xmin if edge == 'in' else xmax
            if abs(xe-ref) < 1e-9:
                i = 0 if edge == 'in' else -1
                tot += np.sum(w*U[e, i, :, 2])*(hy[e]/2); a += hy[e]
        return tot/a

    U = np.zeros((m.nelem, n, n, 4)); hist = [U]
    t0 = time.perf_counter(); status = 'nocon'
    nmax = int(2*T_MIN/dt)
    r10 = None
    for s in range(nmax):
        Up = hist[0].copy()
        U = S.step_bdf(st, hist, time=s*dt, max_newton=1, newton_tol=1e-12,
                       newton_factor=0.0, custom_inlet=inlet, pin_p=pin,
                       cgsfac=CGSFAC, cg_tol=CGTOL, cg_max_iter=CGMAX)
        if not np.all(np.isfinite(U)):
            status = 'BLEWUP'; break
        rate = float(np.max(np.abs(U-Up))/dt)
        if abs((s+1)*dt - 10.0) < dt:
            r10 = rate                       # early rate, for slow-vs-broken
        if (s+1)*dt >= T_MIN and rate < RATE_TOL:
            status = 'conv'; break
        if time.perf_counter()-t0 > WALL_CAP:
            status = 'WALLCAP'; break
    else:
        status = 'TCAP'
    fin = np.all(np.isfinite(U))
    dp = float(pbar(U, 'in')-pbar(U, 'out')) if fin else float('nan')
    ys, us = [], []
    if fin:
        for e in range(m.nelem):
            if abs(xn[e, -1]-xmax) < 1e-9:
                for j in range(n):
                    ys.append(m.ynod[e, j]); us.append(U[e, -1, j, 0])
        o = np.argsort(ys); ys, us = np.array(ys)[o], np.array(us)[o]
        k = np.concatenate(([True], np.diff(ys) > 1e-12)); ys, us = ys[k], us[k]
        prof = float(np.sqrt(np.mean((us-u_exact(ys))**2))/1.5)
    else:
        prof = float('nan')
    return dict(dt=dt, a_mass=1.5/dt, status=status, steps=s+1,
                t_end=(s+1)*dt, rate=rate, rate_t10=r10, dp=dp, prof=prof,
                wall=time.perf_counter()-t0)


print("Bracketing the small-dt failure, w_mom = w_mass = 1, a_flux = 1")
print(f"steady: |dU|/dt < {RATE_TOL:g} after t >= {T_MIN:g};  caps t <= {2*T_MIN:g}, "
      f"wall <= {WALL_CAP:g}s;  exact dp = {DP_EXACT}\n")
hdr = (f"{'dt':>6}{'a_mass':>8}{'status':>9}{'steps':>7}{'t_end':>8}"
       f"{'rate@t=10':>12}{'final rate':>12}{'prof err':>12}{'dp':>11}{'wall s':>8}")
print(hdr); print('-'*len(hdr))
out = []
for dt in DTS:
    r = run(dt)
    out.append(r)
    print(f"{r['dt']:>6g}{r['a_mass']:>8.2f}{r['status']:>9}{r['steps']:>7}"
          f"{r['t_end']:>8g}"
          f"{(r['rate_t10'] if r['rate_t10'] else float('nan')):>12.3e}"
          f"{r['rate']:>12.3e}{r['prof']:>12.3e}{r['dp']:>11.5f}{r['wall']:>8.0f}",
          flush=True)
    with open(f'{SC}/pois_dt_bracket.json', 'w') as fh:
        json.dump(out, fh, indent=1)
