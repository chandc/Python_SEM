"""Does the cold-start transient violate incompressibility MORE at small dt?

THE HYPOTHESIS.  The momentum rows carry a_mass = fac1/dt on the time-derivative
term.  The continuity and vorticity rows carry weight 1 and contain NO dt.  So
the ratio

    "u must not change much"   :   "u must be divergence-free"
             a_mass = fac1/dt   :   1

grows as 1/dt -- 1.5 at dt = 1, but 30 at dt = 0.05.  From a cold start (U = 0,
inlet immediately demanding u = 6y(1-y), field maximally non-solenoidal) the
least-squares compromise at each step then favours "stay near the previous
state" over "satisfy the constraints", by a factor that grows as dt shrinks.
The transient should therefore be LESS incompressible at small dt, and can wander
somewhere the iteration cannot recover from.

This reframes the puzzle.  Diagonal dominance does grow as dt falls -- that part
of the intuition is right -- but it is dominance of the term that RESISTS CHANGE,
not of the terms that enforce the physics.

PREDICTION, falsifiable: peak rms(div u) over the cold-start transient grows
systematically as dt falls, and the growth tracks where the runs fail
(free outflow dies below dt = 0.9; one BC below 0.25; two below 0.1).

If instead rms(div u) is flat or falls with dt, the hypothesis is wrong and the
small-dt failure is something else.

Compared at matched PHYSICAL time, not matched step count, since the whole point
is the path through the transient.
"""
import os, sys, time
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
import lssem2d
lssem2d.set_backend('numpy')
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState
from lssem2d.operators import dUdx, dUdy
import lssem2d.solver as S
import lssem2d.bc as BC

N, EX, EY = 8, 10, 2
NU = 0.01
ue = lambda y: 6.0*y*(1.0-y)
D = diff_matrix(N)
OB = BC.apply_bc
TEND = 50.0
PROBE = (0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0)


def run(dt, pz=False):
    m = build_channel(10., 1., EX, EY, N, bcs=(3, 4, 1, 1))
    n = N+1
    ipin = next((e, 0, 0) for e in range(m.nelem)
                if m.bc[e, 0] == 3 and m.bc[e, 2] == 1)
    for e in range(m.nelem):
        if m.bc[e, 1] == 4 and not pz:
            m.bc[e, 1] = 0
    xmax = m.xnod.max()
    out = [e for e in range(m.nelem) if abs(m.xnod[e, -1]-xmax) < 1e-9]
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
            st._global_mask[e, -1, :, 2] = 0.0
            st._global_mask[e, -1, :, 3] = 0.0
        S.apply_bc = bc2
    div_at = {}
    peak = 0.0
    try:
        U = np.zeros((m.nelem, n, n, 4)); h = [U.copy()]
        nst = int(round(TEND/dt))
        for s in range(nst):
            U = S.step_bdf(st, h, time=s*dt, max_newton=1, newton_tol=1e-12,
                           newton_factor=0.0, custom_inlet=lambda x, y, t: ue(y),
                           pin_p=pin, cgsfac=1e-8, cg_tol=1e-10,
                           cg_max_iter=300000)
            if not np.all(np.isfinite(U)):
                return None, None, None
            dv = (dUdx(np.ascontiguousarray(U[..., 0]), D, m.facx) +
                  dUdy(np.ascontiguousarray(U[..., 1]), D, m.facy))
            r = float(np.sqrt((dv**2).mean()))
            peak = max(peak, r)
            t = (s+1)*dt
            for p in PROBE:
                if p not in div_at and t >= p - 0.5*dt:
                    div_at[p] = r
    finally:
        S.apply_bc = OB
    return div_at, peak, float(np.abs(U[..., 0]).max())


print("rms(div u) along the COLD-START transient, matched physical time.")
print("Hypothesis: the mass term outweighs the constraint rows by fac1/dt,")
print("so the transient should be LESS incompressible as dt falls.\n")
print(f"{'dt':>7}{'a_mass':>8}{'outlet':>8}" +
      ''.join(f"{'t='+str(p):>11}" for p in PROBE) + f"{'PEAK':>11}{'max|u|':>9}")
for dt in (1.0, 0.5, 0.25, 0.1, 0.05):
    d, pk, mu = run(dt)
    if d is None:
        print(f"{dt:>7g}{1.5/dt:>8.1f}{'free':>8}   NON-FINITE", flush=True); continue
    print(f"{dt:>7g}{1.5/dt:>8.1f}{'free':>8}" +
          ''.join(f"{d.get(p, float('nan')):>11.3e}" for p in PROBE) +
          f"{pk:>11.3e}{mu:>9.3f}", flush=True)
print()
for dt in (0.5, 0.05):
    d, pk, mu = run(dt, pz=True)
    if d is None:
        print(f"{dt:>7g}{1.5/dt:>8.1f}{'P+Z':>8}   NON-FINITE", flush=True); continue
    print(f"{dt:>7g}{1.5/dt:>8.1f}{'P+Z':>8}" +
          ''.join(f"{d.get(p, float('nan')):>11.3e}" for p in PROBE) +
          f"{pk:>11.3e}{mu:>9.3f}", flush=True)
print("\nSupported if PEAK grows monotonically as dt falls; refuted if flat or falling.")
