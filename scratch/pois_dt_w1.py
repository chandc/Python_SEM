"""Poiseuille Re=100: dt sweep at w_mom = w_mass = 1.0 (the 1/dt momentum row).

    a_mass = w_mass*fac1/dt = fac1/dt     a_flux = w_mom = 1     hist = 1/dt

so the momentum row is  (fac1*u - sum_m alpha_m u^{n-m})/dt + N(u), the momentum
weight is 1 at EVERY dt, and dt_eff = dt*w_mom/w_mass = dt exactly.  This is the
configuration POISEUILLE_DT_STUDY.md calls "time-accurate": the least-squares
weighting is held fixed while only the physical step varies, which is the one
sweep that separates temporal error from the dt-as-weight effect.

Legacy for contrast has a_flux = dt, so its weight and its step move together --
that is what produced the 1875x (tight: 212,061x) spread over dt.  Here the
weight is pinned, so whatever spread remains is NOT the pressure-underweighting
mechanism.

Prediction to be tested (POISEUILLE_DT_STUDY.md sec 4): at fixed weight 1 the
iteration destabilises as dt falls -- dt=0.5, w_mass=w_mom=1 was measured at
600 steps, no convergence, dp = 15.22.  Every converged run in that table had
dt_eff = 1.

TIGHT linear solve throughout (cgsfac=1e-8, cg_tol=1e-10) per
STEADY_FORM_STUDY.md sec 6: at the default tol=1e-6 floor the answers are
solver-limited, not discretisation-limited.

Control rows: legacy at dt=1 (tight) should reproduce prof err 4.65e-06,
dp = 1.200000 from that same table.
"""
import os, sys, time, json
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
import lssem2d
# NUMPY, deliberately.  numba is 2.3x faster and passes the operator-level parity
# gates at 1e-16, but that is ONE apply.  Over 300 accumulated steps with a tight
# CG the two backends settle on different fixed points: this exact control case
# gives prof err 4.6471e-06 on numpy (= the published tight value to every digit)
# and 8.4673e-06 on numba, independent of newton_tol.  A 1.8x discrepancy at 4e-06
# is the same size as the signal this sweep is measuring, so numba is not usable
# here.  Cost: ~0.44 s/step instead of 0.18.
lssem2d.set_backend('numpy')
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix, lgl_weights
from lssem2d.lssem import SolverState, ls_coeffs
import lssem2d.solver as S

SC = os.path.dirname(os.path.abspath(__file__))
LX, LY, RE = 10.0, 1.0, 100.0
NU = 1.0*LY/RE
DP_EXACT = 12.0*NU*1.0/LY**2*LX                  # 1.20
N, EX, EY = 8, 10, 2
RATE_TOL, T_MIN = 1.0e-9, 300.0                  # three viscous times, dt-normalised rate
MAXSTEP, WALL_CAP = 40000, 9000.0                # per-case caps; dt=5 hung 50 min once.
                                                 # T_MIN/dt steps dominates: dt=0.05
                                                 # is 6000 steps ~= 100 min on numpy.
CGSFAC, CGTOL, CGMAX = 1e-8, 1e-10, 300000
u_exact = lambda y: 6.0*y*(1.0-y)

# (dt, w_mom, w_mass).  None/None = legacy, carried as the control.  Ordered
# cheapest-first: small dt needs T_MIN/dt steps, so dt=0.05 alone is ~6000.
CASES = [(1.0, None, None),                      # control vs the published table
         (1.0, 1.0, 1.0), (2.0, 1.0, 1.0), (5.0, 1.0, 1.0),
         (0.5, 1.0, 1.0), (0.25, 1.0, 1.0), (0.1, 1.0, 1.0), (0.05, 1.0, 1.0)]


def run(dt, w_mom, w_mass):
    mesh = build_channel(LX, LY, EX, EY, N, bcs=(3, 4, 1, 1))
    n = N+1
    pin = next((e, 0, 0) for e in range(mesh.nelem)
               if mesh.bc[e, 0] == 3 and mesh.bc[e, 2] == 1)
    for e in range(mesh.nelem):
        if mesh.bc[e, 1] == 4:
            mesh.bc[e, 1] = 0                    # FREE outflow
    st = SolverState(mesh, diff_matrix(N), nu=NU, dt=dt, fac1=1.0,
                     w_mom=w_mom, w_mass=w_mass)
    inlet = lambda x, y, t: u_exact(y)
    U = np.zeros((mesh.nelem, n, n, 4)); hist = [U]
    t0 = time.perf_counter(); conv, blew, capped = False, False, False
    trace = []                                   # (t, rate) -- stall vs divergence
    for s in range(MAXSTEP):
        Up = hist[0].copy()
        U = S.step_bdf(st, hist, time=s*dt, max_newton=1, newton_tol=1e-12,
                       newton_factor=0.0, custom_inlet=inlet, pin_p=pin,
                       cgsfac=CGSFAC, cg_tol=CGTOL, cg_max_iter=CGMAX,
                       verbose=False)
        if not np.all(np.isfinite(U)):
            blew = True; break
        rate = float(np.max(np.abs(U-Up))/dt)
        if s % 25 == 0 or (s+1)*dt >= T_MIN:
            trace.append(((s+1)*dt, rate))
        if (s+1)*dt >= T_MIN and rate < RATE_TOL:
            conv = True; break
        if time.perf_counter()-t0 > WALL_CAP:
            capped = True; break
    wall = time.perf_counter()-t0
    am, af, hs = ls_coeffs(st)
    base = dict(dt=dt, w_mom=w_mom, w_mass=w_mass, a_mass=am, a_flux=af,
                hist=hs, dt_eff=dt*(1.0 if w_mom is None else w_mom) /
                (1.0 if w_mass is None else w_mass),
                steps=s+1, t_end=(s+1)*dt, conv=conv, blew=blew, capped=capped,
                rate=None if blew else rate, wall=wall, trace=trace[-6:])
    if blew:
        return base

    # outlet plane: p(y) must be a flat line at -DP_EXACT, u(y) the parabola
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
    tag = 'legacy' if w_mom is None else f'w{w_mom:g}m{w_mass:g}'
    np.savez(f'{SC}/dtsweep_dt{dt:g}_{tag}.npz', y=ys, p=ps, u=us)
    base.update(prof=float(np.sqrt(np.mean((us-u_exact(ys))**2))/1.5), dp=dp,
                dp_err=abs(dp-DP_EXACT)/DP_EXACT,
                spread=float(ps.max()-ps.min()))
    return base


print(f"Poiseuille Re={RE:g}, order {N}, {EX}x{EY}, parabolic inlet, FREE outflow")
print(f"tight solve: cgsfac={CGSFAC:g}, cg_tol={CGTOL:g}   "
      f"steady: |dU|/dt < {RATE_TOL:g} after t >= {T_MIN:g}")
print(f"exact:  dp = {DP_EXACT:g},  outlet p flat at {-DP_EXACT:g},  u = 6y(1-y)\n")
hdr = (f"{'dt':>6}{'w_mom':>7}{'w_mass':>8}{'a_mass':>8}{'a_flux':>8}{'dt_eff':>8}"
       f"{'steps':>7}{'t_end':>8}{'status':>9}{'prof err':>12}{'dp':>11}"
       f"{'dp err':>10}{'p spread':>11}{'wall s':>9}")
print(hdr); print('-'*len(hdr))
rows = []
for dt, wf, wm in CASES:
    r = run(dt, wf, wm)
    rows.append(r)
    lab_f = 'legacy' if wf is None else f'{wf:g}'
    lab_m = 'legacy' if wm is None else f'{wm:g}'
    stat = ('DIVERGED' if r['blew'] else 'conv' if r['conv']
            else 'WALLCAP' if r['capped'] else 'nocon')
    head = (f"{dt:>6g}{lab_f:>7}{lab_m:>8}{r['a_mass']:>8.2f}{r['a_flux']:>8.2f}"
            f"{r['dt_eff']:>8.3g}{r['steps']:>7}{r['t_end']:>8g}{stat:>9}")
    if r['blew']:
        print(head + f"{'-':>12}{'-':>11}{'-':>10}{'-':>11}{r['wall']:>9.1f}",
              flush=True)
    else:
        print(head + f"{r['prof']:>12.3e}{r['dp']:>11.5f}{r['dp_err']:>10.2e}"
                     f"{r['spread']:>11.3e}{r['wall']:>9.1f}", flush=True)
    with open(f'{SC}/pois_dt_w1.json', 'w') as fh:
        json.dump(rows, fh, indent=1)

ok = [r for r in rows if r.get('conv') and r['w_mom'] is not None]
if len(ok) > 1:
    p = [r['prof'] for r in ok]
    print(f"\nspread in profile error over CONVERGED w=1 runs: {max(p)/min(p):.4g}x"
          f"   (legacy tight sweep: 212,061x)")
    b = min(ok, key=lambda r: r['prof'])
    print(f"best: dt = {b['dt']:g}   prof err = {b['prof']:.3e}   dp = {b['dp']:.6f}")
bad = [r for r in rows if not r['conv']]
print(f"\nnot converged: {[r['dt'] for r in bad] or 'none'}")
for r in bad:                                    # stalling or blowing up?
    print(f"  dt={r['dt']:g}  last (t, |dU|/dt): "
          + '  '.join(f"({t:g}, {q:.2e})" for t, q in r['trace']))
