"""Zero-gradient omega at the outflow: small dt, and the developing-flow stress test.

d(om)/dx = 0 imposed algebraically from the GLL derivative row at the outlet
plane, with u, v and p left completely FREE there.  Confirmed at dt = 1, 0.5,
0.25: bit-exact fixed points, dp = 1.20000 to ~2e-07 as a genuine prediction,
and 83x better than free outflow at dt = 1.

Two gaps this closes.

SMALL dt.  Free outflow degrades catastrophically -- |dU| 9.2 (dt=0.5), 45
(0.25), 354+ (0.1), with dp reaching 537 against an exact 1.2.  a_mass = fac1/dt
reaches 30 at dt = 0.05, the most lopsided the momentum row gets, so if the BC
has a limit this is where it shows.  Run to a genuine fixed point rather than a
fixed step count: 600 steps is t = 30 at dt = 0.05 against t = 600 at dt = 1.

DEVELOPING flow.  d(om)/dx = 0 is exact only for FULLY DEVELOPED flow, and the
control case is developed everywhere by construction, so it cannot expose that
assumption.  The uniform-inlet variant (POISEUILLE_DT_STUDY.md's `develop`) is
genuinely undeveloped at the exit; its dp is legitimately ~1.6 from the entrance
loss, so there is no analytic target and the two BCs are judged against each
other.  A large split means the BC is imposing developedness the flow lacks --
which would rule it out for the BFS, whose short-domain outflow sits inside a
recirculation.
"""
import os, sys, time
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
import lssem2d
lssem2d.set_backend('numpy')
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix, lgl_weights
from lssem2d.lssem import SolverState
import lssem2d.solver as S
import lssem2d.bc as BC

N, EX, EY = 8, 10, 2
NU = 0.01
ue = lambda y: 6.0*y*(1.0-y)
D = diff_matrix(N)
wq = lgl_weights(N)
OB = BC.apply_bc


def run(dt, mode, inlet_kind='para', wall=1200.0, tmax=600.0):
    m = build_channel(10., 1., EX, EY, N, bcs=(3, 4, 1, 1))
    n = N+1
    pin = next((e, 0, 0) for e in range(m.nelem)
               if m.bc[e, 0] == 3 and m.bc[e, 2] == 1)
    for e in range(m.nelem):
        if m.bc[e, 1] == 4:
            m.bc[e, 1] = 0                      # u, v, p, om all free at the outlet
    xn = m.xnod; xmax = xn.max(); xmin = xn.min()
    out = [e for e in range(m.nelem) if abs(xn[e, -1]-xmax) < 1e-9]

    def bc2(mesh, U, **kw):
        U = OB(mesh, U, **kw)
        for e in out:                            # (D om)_N = 0 along each y-row
            U[e, -1, :, 3] = -(D[-1, :-1] @ U[e, :-1, :, 3])/D[-1, -1]
        return U

    st = SolverState(m, D, nu=NU, dt=dt, fac1=1.0, w_mom=1.0, w_mass=1.0)
    if mode == 'zerograd':
        st.get_global_mask(pin_p=pin)            # populate cache, then edit
        for e in out:
            st._global_mask[e, -1, :, 3] = 0.0
        S.apply_bc = bc2
    inl = ((lambda x, y, t: ue(y)) if inlet_kind == 'para'
           else (lambda x, y, t: np.ones_like(y)))
    t0 = time.perf_counter(); status = 'TCAP'; d = np.nan
    try:
        U = np.zeros((m.nelem, n, n, 4)); h = [U.copy()]
        for s in range(int(tmax/dt)):
            prev = h[0].copy()
            U = S.step_bdf(st, h, time=s*dt, max_newton=1, newton_tol=1e-12,
                           newton_factor=0.0, custom_inlet=inl, pin_p=pin,
                           cgsfac=1e-8, cg_tol=1e-10, cg_max_iter=300000)
            if not np.all(np.isfinite(U)):
                return None
            d = float(np.abs(U-prev).max())
            if d < 1e-13:
                status = 'conv'; break
            if time.perf_counter()-t0 > wall:
                status = 'WALLCAP'; break
    finally:
        S.apply_bc = OB

    def pbar(edge):
        tot = a = 0.0
        for e in range(m.nelem):
            xe = xn[e, 0] if edge == 'in' else xn[e, -1]
            ref = xmin if edge == 'in' else xmax
            if abs(xe-ref) < 1e-9:
                i = 0 if edge == 'in' else -1
                tot += np.sum(wq*U[e, i, :, 2])*(m.hy[e]/2); a += m.hy[e]
        return tot/a
    prof = float(np.sqrt(np.mean((U[..., 0]-ue(m.ynod)[:, None, :])**2))/1.5)
    return (status, s+1, (s+1)*dt, d, float(pbar('in')-pbar('out')), prof,
            time.perf_counter()-t0)


def show(dt, mode, r, exact_dp=1.2):
    lab = {'free': 'free', 'zerograd': 'ZERO-GRAD'}[mode]
    if r is None:
        print(f"{dt:>7g}{1.5/dt:>8.1f}{lab:>12}   DIVERGED (non-finite)", flush=True)
        return
    e = f"{abs(r[4]-exact_dp)/exact_dp:>10.2e}" if exact_dp else f"{'-':>10}"
    print(f"{dt:>7g}{1.5/dt:>8.1f}{lab:>12}{r[0]:>9}{r[1]:>7}{r[2]:>8g}"
          f"{r[3]:>11.3e}{r[4]:>11.5f}{e}{r[5]:>11.3e}{r[6]:>8.0f}", flush=True)


HDR = (f"{'dt':>7}{'a_mass':>8}{'outlet om':>12}{'status':>9}{'steps':>7}"
       f"{'t_end':>8}{'|dU|':>11}{'dp':>11}{'dp err':>10}{'prof err':>11}{'wall s':>8}")

print("A. SMALL dt, control (parabolic inlet), cold start, to |dU| < 1e-13\n")
print(HDR)
for dt in (0.1, 0.05):
    show(dt, 'zerograd', run(dt, 'zerograd', wall=1500.0))
print("   (free outflow at these dt: |dU| = 354-437, dp = 147-537 -- see"
      " pois_omega_bc2.log / pois_omega_smalldt.log)")

print("\nB. DEVELOPING flow (uniform inlet). d(om)/dx = 0 is NOT exact here.")
print("   dp is genuinely ~1.6 (entrance loss): judge the two against EACH OTHER.\n")
print(HDR)
for dt in (1.0, 0.5):
    for mode in ('free', 'zerograd'):
        show(dt, mode, run(dt, mode, inlet_kind='unif', wall=600.0, tmax=400.0),
             exact_dp=None)
