"""Is the dt = 0.5 period-2 cycle caused by the FREE OUTFLOW boundary?

At dt = 0.5, w_mom = w_mass = 1 the iteration converges to a period-2 orbit:
|U(k)-U(k-1)| = 9.1974 forever while |U(k)-U(k-2)| ~ 1e-04 and falling.  The
largest change sits at elem 18/19, node (8,0) and (8,8), field omega -- the
OUTLET PLANE at both walls, i.e. exactly where the free outflow meets no-slip.
omega is the one field the free outflow constrains not at all.

Three variants, differing ONLY in what happens at the outflow:

  A  free      bc_E = 0, pressure pinned at the inlet corner.  The published
                configuration; dp and the outlet p(y) are both predictions.
  B  p = 0     bc_E = 4, Dirichlet pressure on the outlet plane, inlet pin
                REMOVED (pinning as well would over-determine the level).
                Still nothing imposed on u, v, omega there.
  C  periodic  no outflow boundary at all.  Streamwise-periodic channel driven
                by the body force f_x = 2*nu*a_flux that sustains u = 1-y^2,
                the CHANNEL_VALIDATION.md sec 6 configuration.  Same operator,
                same weights, same dt -- but the corner does not exist.

If A cycles and C does not, the outflow BC is the cause and the dt threshold is
a property of the test case, not of the w_mom = 1 weighting.  If C cycles too,
the BC is exonerated and the problem is in the scheme at a_mass/a_flux = 3.

Also maps WHERE the oscillation lives in A: if the amplitude collapses with
distance from the outlet, that is corroboration independent of the C result.
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
from lssem2d.lssem import SolverState, ls_coeffs
import lssem2d.solver as S

LX, LY, NU = 10.0, 1.0, 0.01
N, EX, EY = 8, 10, 2
DT = 0.5
NSTEP = 200
CGSFAC, CGTOL, CGMAX = 1e-8, 1e-10, 300000
u_exact = lambda y: 6.0*y*(1.0-y)


def channel(free):
    m = build_channel(LX, LY, EX, EY, N, bcs=(3, 4, 1, 1))
    pin = next((e, 0, 0) for e in range(m.nelem)
               if m.bc[e, 0] == 3 and m.bc[e, 2] == 1)
    if free:
        for e in range(m.nelem):
            if m.bc[e, 1] == 4:
                m.bc[e, 1] = 0
        return m, pin
    return m, False                     # keep bc 4 (p=0), drop the inlet pin


def periodic():
    m = build_channel(2.0*np.pi, 2.0, 1, 2, N, bcs=(0, 0, 1, 1))
    m.periodic_x = 2.0*np.pi
    m.compute_global_indices()
    return m


def march(tag, m, pin, f=None, inlet=None, nstep=NSTEP):
    st = SolverState(m, diff_matrix(N), nu=NU, dt=DT, fac1=1.0,
                     w_mom=1.0, w_mass=1.0)
    n = N+1
    U = np.zeros((m.nelem, n, n, 4)); hist = [U]
    if f is not None:                   # body force carries the row weight
        _, a_flux, _ = ls_coeffs(st)
        f = f*a_flux
    hold = {}
    print(f"--- {tag} ---", flush=True)
    print(f"  {'step':>6}{'|U(k)-U(k-1)|':>16}{'|U(k)-U(k-2)|':>16}"
          f"{'max|u|':>10}  where max|dU| sits", flush=True)
    for s in range(nstep):
        U = S.step_bdf(st, hist, time=s*DT, max_newton=1, newton_tol=1e-12,
                       newton_factor=0.0, f_known=f, custom_inlet=inlet,
                       pin_p=pin, cgsfac=CGSFAC, cg_tol=CGTOL,
                       cg_max_iter=CGMAX)
        if not np.all(np.isfinite(U)):
            print("  NON-FINITE"); return None
        hold[s] = U.copy()
        if s >= 2 and (s % 50 == 0 or s >= nstep-3):
            d1 = np.abs(U-hold[s-1]); d2 = np.abs(U-hold[s-2]).max()
            k = np.unravel_index(d1.argmax(), d1.shape)
            print(f"  {s+1:>6}{d1.max():>16.4e}{d2:>16.4e}"
                  f"{np.abs(U[...,0]).max():>10.4f}"
                  f"  elem{k[0]} node({k[1]},{k[2]}) field={'uvpw'[k[3]]}",
                  flush=True)
        hold.pop(s-3, None)
    d1 = np.abs(U-hold[nstep-2])
    return U, d1


print(f"Outflow-BC test at dt = {DT:g}, w_mom = w_mass = 1 "
      f"(a_mass = {1.5/DT:.1f}, a_flux = 1), {NSTEP} steps\n")

mA, pinA = channel(free=True)
inlet = lambda x, y, t: u_exact(y)
rA = march('A  free outflow (bc 0), inlet pin', mA, pinA, inlet=inlet)

mB, pinB = channel(free=False)
rB = march('B  outlet p = 0 (bc 4), no pin', mB, pinB, inlet=inlet)

mC = periodic()
fC = np.zeros((mC.nelem, N+1, N+1, 4)); fC[..., 0] = 2.0*NU
rC = march('C  periodic channel, body force, NO outflow at all', mC,
           (0, (N+1)//2, (N+1)//2), f=fC)

# where does the oscillation live in A?
if rA is not None:
    U, d1 = rA
    print("\nA: amplitude of the step-to-step change by streamwise element column")
    print(f"  {'x range':>16}{'max|dU| there':>16}")
    xn = mA.xnod
    cols = {}
    for e in range(mA.nelem):
        key = round(float(xn[e, 0]), 6)
        cols[key] = max(cols.get(key, 0.0), float(d1[e].max()))
    for x in sorted(cols):
        print(f"  {x:>7.2f} - {x+LX/EX:<6.2f}{cols[x]:>16.4e}")
