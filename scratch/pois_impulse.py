"""Is the cold-start failure an IMPULSIVE-START pressure transient scaling as 1/dt?

THE HYPOTHESIS.  A cold start has U = 0 while the inlet immediately demands
u = 6y(1-y).  Accelerating fluid from rest to O(1) within one step of size dt
needs du/dt ~ 1/dt, and in an incompressible formulation that acceleration is
delivered by the pressure gradient -- so the first steps must carry p ~ O(1/dt).
As dt falls that impulse grows without bound, and it has to be absorbed at the
boundaries.  A boundary constraining p and omega can absorb it; free outflow,
constraining nothing, cannot.

This fits what the other two hypotheses could not:
  - SEEDED starts are bit-exact fixed points at every dt down to 0.01
    (a_mass = 150) -- no acceleration, no impulse, no problem.
  - more admissible conditions lower the dt threshold (0 -> 0.9, 1 -> 0.25,
    2 -> ~0.075) -- more capacity to absorb the impulse.
  - the BFS, whose outlet sits in REVERSED flow, is worse still.
  - it is invisible to the block-ratio and divergence diagnostics, which measure
    neither transient nor pressure.

PREDICTION, falsifiable: max|p| at step 1 from a cold start scales as 1/dt --
slope -1 on a log-log fit.  Flat, or any slope near 0, refutes it.

Control: the same measurement from the EXACT seed, where the hypothesis says
there is no impulse at all and max|p| should be dt-independent.

Only 3 steps per case, so this is seconds per dt, not minutes.
"""
import os, sys
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
import lssem2d
lssem2d.set_backend('numpy')
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState
import lssem2d.solver as S
import lssem2d.bc as BC

N, EX, EY = 8, 10, 2
NU = 0.01
ue = lambda y: 6.0*y*(1.0-y)
D = diff_matrix(N)
OB = BC.apply_bc
DTS = [2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01, 0.005]
NSTEP = 3


def run(dt, seed, pz):
    m = build_channel(10., 1., EX, EY, N, bcs=(3, 4, 1, 1))
    n = N+1
    ipin = next((e, 0, 0) for e in range(m.nelem)
                if m.bc[e, 0] == 3 and m.bc[e, 2] == 1)
    for e in range(m.nelem):
        if m.bc[e, 1] == 4 and not pz:
            m.bc[e, 1] = 0
    xn = m.xnod; xmax = xn.max(); xmin = xn.min()
    out = [e for e in range(m.nelem) if abs(xn[e, -1]-xmax) < 1e-9]
    pin = False if pz else ipin

    def bc2(mesh, U, **kw):
        U = OB(mesh, U, **kw)
        for e in out:
            U[e, -1, :, 3] = -(D[-1, :-1] @ U[e, :-1, :, 3])/D[-1, -1]
        return U

    st = SolverState(m, D, nu=NU, dt=dt, fac1=1.0, w_mom=1.0, w_mass=1.0)
    if pz:
        st.get_global_mask(pin_p=pin)
        for e in out:
            st._global_mask[e, -1, :, 3] = 0.0
        S.apply_bc = bc2
    try:
        U = np.zeros((m.nelem, n, n, 4))
        y = m.ynod[:, None, :]
        if seed == 'exact':
            U[..., 0] = np.broadcast_to(ue(y), (m.nelem, n, n))
            U[..., 3] = np.broadcast_to(-(6.0-12.0*y), (m.nelem, n, n))
            U[..., 2] = -0.12*(xn[:, :, None]-xmin) + (1.2 if pz else 0.0)
        h = [U.copy()]
        pk = []
        for s in range(NSTEP):
            U = S.step_bdf(st, h, time=s*dt, max_newton=1, newton_tol=1e-12,
                           newton_factor=0.0, custom_inlet=lambda x, y, t: ue(y),
                           pin_p=pin, cgsfac=1e-8, cg_tol=1e-10,
                           cg_max_iter=300000)
            if not np.all(np.isfinite(U)):
                pk.append(np.nan); break
            pk.append(float(np.abs(U[..., 2]).max()))
    finally:
        S.apply_bc = OB
    np.savez(f'{SC}/impulse_dt{dt:g}_{seed}_{"pz" if pz else "free"}.npz',
             U=U, xnod=m.xnod, ynod=m.ynod, hy=m.hy, N=N, dt=dt, peaks=pk)
    return pk


def fit(dts, vals):
    d, v = np.asarray(dts, float), np.asarray(vals, float)
    k = np.isfinite(v) & (v > 0)
    if k.sum() < 2:
        return float('nan')
    return float(np.polyfit(np.log(d[k]), np.log(v[k]), 1)[0])


for pz in (False, True):
    lab = 'P+Z (two conditions)' if pz else 'free outflow (none)'
    print(f"\n=== {lab} ===")
    print(f"{'dt':>8}{'1/dt':>9}{'COLD max|p| step 1':>21}{'step 2':>11}{'step 3':>11}"
          f"{'EXACT-seed max|p| s1':>23}")
    cold1, ex1 = [], []
    for dt in DTS:
        c = run(dt, 'cold', pz)
        e = run(dt, 'exact', pz)
        cold1.append(c[0] if c else np.nan); ex1.append(e[0] if e else np.nan)
        g = lambda a, i: (f"{a[i]:.4e}" if len(a) > i and np.isfinite(a[i]) else "   NaN   ")
        print(f"{dt:>8g}{1.0/dt:>9.1f}{g(c,0):>21}{g(c,1):>11}{g(c,2):>11}"
              f"{g(e,0):>23}", flush=True)
    print(f"  log-log slope, COLD  max|p| step 1 vs dt : {fit(DTS, cold1):+.3f}"
          "   (hypothesis: -1)")
    print(f"  log-log slope, EXACT max|p| step 1 vs dt : {fit(DTS, ex1):+.3f}"
          "   (hypothesis:  0)")
