"""Does decoupling the LS weight from dt actually remove the dt sensitivity?

Same Poiseuille control case, swept over dt twice:
    w_mom = None  -> legacy, the momentum weight IS dt
    w_mom = 1.0   -> decoupled, momentum weighted 1 regardless of dt

If the decoupling works, the second sweep should be flat in dt: the converged
steady state should no longer care what time step was used to reach it.
"""
import sys, os, time, json
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix, lgl_weights
from lssem2d.lssem import SolverState
import lssem2d.solver as S

SC = os.path.dirname(os.path.abspath(__file__))
LX, LY, RE, UMEAN = 10.0, 1.0, 100.0, 1.0
NU = UMEAN*LY/RE
DP_EXACT = 12.0*NU*UMEAN/LY**2*LX          # 1.20
N, EX, EY = 8, 10, 2
RATE_TOL, T_MIN, MAXSTEP = 1.0e-9, 300.0, 20000
CGSFAC, NITCGS = 1.0e-3, 40000
DTS = [0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
u_exact = lambda y: 6.0*y*(1.0-y)


def run(dt, w_mom):
    mesh = build_channel(LX, LY, EX, EY, N, bcs=(3, 4, 1, 1))
    n = N+1
    pin = next((e, 0, 0) for e in range(mesh.nelem)
               if mesh.bc[e, 0] == 3 and mesh.bc[e, 2] == 1)
    for e in range(mesh.nelem):
        if mesh.bc[e, 1] == 4:
            mesh.bc[e, 1] = 0
    st = SolverState(mesh, diff_matrix(N), nu=NU, dt=dt, fac1=1.0, w_mom=w_mom)
    inlet = lambda x, y, t: u_exact(y)
    U = np.zeros((mesh.nelem, n, n, 4)); hist = [U]
    t0 = time.perf_counter()
    for s in range(MAXSTEP):
        Up = hist[0].copy()
        U = S.step_bdf(st, hist, time=s*dt, max_newton=1, newton_tol=1e-12,
                       newton_factor=0.0, custom_inlet=inlet, pin_p=pin,
                       cgsfac=CGSFAC, cg_max_iter=NITCGS, verbose=False)
        if not np.all(np.isfinite(U)):
            return None
        if (s+1)*dt >= T_MIN and np.max(np.abs(U-Up))/dt < RATE_TOL:
            break
    wall = time.perf_counter()-t0
    w = lgl_weights(N); xn, yn, hy = mesh.xnod, mesh.ynod, mesh.hy
    xmax = xn.max()
    ys, us = [], []
    for e in range(mesh.nelem):
        if abs(xn[e, -1]-xmax) < 1e-9:
            for j in range(n):
                ys.append(yn[e, j]); us.append(U[e, -1, j, 0])
    ys, us = np.array(ys), np.array(us)
    o = np.argsort(ys); ys, us = ys[o], us[o]
    k = np.concatenate(([True], np.diff(ys) > 1e-12)); ys, us = ys[k], us[k]
    prof = float(np.sqrt(np.mean((us-u_exact(ys))**2))/1.5)

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
    return dict(dt=dt, w_mom=w_mom, steps=s+1, prof=prof, dp=dp,
                dp_err=abs(dp-DP_EXACT)/DP_EXACT, wall=wall)


out = []
print(f"Poiseuille control, Re={RE:g}, order {N}, {EX}x{EY} elements, exact dp = {DP_EXACT}\n")
for w_mom, lab in ((None, 'LEGACY   w_mom = None  (momentum weight = dt)'),
                   (1.0,  'DECOUPLED w_mom = 1.0  (momentum weight = 1)')):
    print(f"=== {lab} ===")
    print(f"{'dt':>6}{'steps':>7}{'|u-u_ex|/Umax':>15}{'dp':>11}{'dp err':>11}{'wall s':>9}")
    for dt in DTS:
        r = run(dt, w_mom)
        if r is None:
            print(f"{dt:>6}   DIVERGED"); continue
        out.append(r)
        print(f"{dt:>6}{r['steps']:>7}{r['prof']:>15.3e}{r['dp']:>11.5f}"
              f"{r['dp_err']:>11.2e}{r['wall']:>9.1f}")
    sub = [r for r in out if r['w_mom'] == w_mom]
    if len(sub) > 1:
        p = np.array([r['prof'] for r in sub])
        print(f"   spread over dt: profile error max/min = {p.max()/p.min():.1f}x\n")
json.dump(out, open(f'{SC}/poiseuille_wmom.json', 'w'), indent=1)
print(f"saved {SC}/poiseuille_wmom.json")
