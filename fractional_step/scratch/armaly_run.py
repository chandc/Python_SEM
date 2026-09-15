"""BFS at Armaly's ACTUAL specifications: expansion ratio 1.94, Re = 389 his way.

Run as:  python armaly_run.py short|long [free|pz|both]

WHY THIS DIFFERS FROM THE cnos RUNS.  Everything so far used cnos_{short,long}_grid,
which is Chan (1996)'s idealisation: expansion ratio 2.0, inlet height 0.5, step
0.5.  Armaly's rig (JFM 127, 1983, sec. 2.1) is

    inlet height  h = 5.2 mm      outlet H = 10.1 mm      step S = 4.9 mm
    expansion ratio 1 : 1.94      so S/h = 0.942, NOT 1

and the armaly_{short,long}_grid files reproduce that: y in [0, 1.94], inlet
y in [0.94, 1.94], h = 1.0, S = 0.94.

REYNOLDS NUMBER -- the trap.  Armaly (p.478) defines

    Re = V*D/nu,   V = 2/3 * u_max,inlet = AVERAGE inlet velocity
                   D = hydraulic diameter of the INLET channel = 2h

On the cnos grid h = 0.5 so D = 1.0 = the code's length unit and nu = 1/389 is
right -- "the two factors of two cancel".  On the ARMALY grid h = 1.0, so
D = 2.0 and matching Re = 389 needs

    nu = 2h/Re = 2.0/389 = 5.141388e-03

Using nu = 1/389 here would be Re = 778, twice Armaly's.  The Fortran study
records exactly this error ("our Re=389 != Armaly's physical Re=389").

REATTACHMENT is reported as x/S with S = 0.94, since Armaly normalises by STEP
height.  His measured value at Re = 389 is x1/S = 8.0 +/- 0.7
(reference/armaly_fig4_x1_measured.csv).

DOMAIN LENGTH.  Armaly's own computation (p.486) used an exit at L = 4*X_R with
dU/dx = dV/dx = 0, "sufficient to make the reattachment length independent of the
length of the calculation domain".  With x_R ~ 8*S = 7.5:  long L = 17 gives
L/X_R = 2.3, short L = 5 gives 0.67.  Both are under his rule; the long one is
the closer.

State saved to armaly_<domain>_<bc>.npz -- never re-solve to answer a follow-up.
"""
import os, sys, time
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
sys.path.insert(0, SC)
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
import lssem2d
lssem2d.set_backend('numpy')
from fgrid import load
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState
import lssem2d.solver as S
import lssem2d.bc as BC

# NOTE: the F90 armaly_{short,long}_grid files have a SYMMETRY top wall (bc=5).
# Armaly's rig is a closed channel -- his fig 2(b) shows non-zero vorticity at the
# top wall, i.e. NO-SLIP -- and running the symmetry grids gives x_r/S = 18 against
# his 8.0, because a slip top removes the wall friction that decelerates the jet.
# The Fortran study records the same mistake in its own sec 10.3.  These grids are
# regenerated with a no-slip top by scratch/mesh_armaly_er194.py.
GRIDS = {'short': 'grids/armaly_er194_short_grid.dat',
         'long':  'grids/armaly_er194_long_grid.dat'}
RE = 389.0
H_IN, Y_STEP = 1.0, 0.94          # inlet height, step height (= y of the step edge)
NU = 2.0*H_IN/RE                  # Armaly's D = 2h convention
S_STEP = Y_STEP                   # reattachment normalised by STEP height
OB = BC.apply_bc


def inlet_profile(y):
    """parabolic on the inlet channel, u_avg = 1 so u_max = 1.5"""
    eta = (np.asarray(y)-Y_STEP)/H_IN
    return np.where((eta >= 0.0) & (eta <= 1.0), 6.0*eta*(1.0-eta), 0.0)


def reattach(U, xn, yn, hy, N):
    D = diff_matrix(N); n = N+1; xs, tw = [], []
    for e in range(U.shape[0]):
        if yn[e, 0] > 0.01 or xn[e, 0] < -1e-9:      # bottom wall, downstream of step
            continue
        for i in range(n):
            xs.append(xn[e, i])
            tw.append(np.dot(D[0, :], U[e, i, :, 0])*(2.0/hy[e]))
    o = np.argsort(xs); xs, tw = np.array(xs)[o], np.array(tw)[o]
    for k in range(len(xs)-1):
        if tw[k] < 0 and tw[k+1] > 0 and xs[k] > 0.05:
            return xs[k]-tw[k]*(xs[k+1]-xs[k])/(tw[k+1]-tw[k])
    return float('nan')


def run(domain, pz, dt=1.0, cap=1200, wall=2400.0, cgsfac=1e-3, tol=1e-6):
    m, _, _ = load(GRIDS[domain]); N = m.N; n = N+1
    ipin = next((e, n-1, 0) for e in range(m.nelem)
                if m.bc[e, 1] == 4 and m.bc[e, 2] == 1)
    for e in range(m.nelem):
        if m.bc[e, 1] == 4 and not pz:
            m.bc[e, 1] = 0
    D = diff_matrix(N)
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
            st._global_mask[e, -1, :, 3] = 0.0
        S.apply_bc = bc2
    inl = lambda x, y, t: inlet_profile(y)
    U = np.zeros((m.nelem, n, n, 4)); h = [U.copy()]
    t0 = time.perf_counter(); status = 'CAP'; d = np.nan
    try:
        for s in range(cap):
            prev = h[0].copy()
            U = S.step_bdf(st, h, time=s*dt, max_newton=1, newton_tol=1e-12,
                           newton_factor=0.0, custom_inlet=inl, pin_p=pin,
                           cgsfac=cgsfac, cg_tol=tol, cg_max_iter=200000)
            if not np.all(np.isfinite(U)):
                status = 'NaN'; break
            d = float(np.abs(U-prev).max())
            if np.abs(U[..., 0]).max() > 20.0:
                status = 'BLEWUP'; break
            if d < 1e-11:
                status = 'conv'; break
            if time.perf_counter()-t0 > wall:
                status = 'WALLCAP'; break
    finally:
        S.apply_bc = OB
    ok = np.all(np.isfinite(U))
    np.savez(f'{SC}/armaly_{domain}_{"pz" if pz else "free"}.npz',
             U=U, xnod=m.xnod, ynod=m.ynod, hy=m.hy, N=N, nu=NU, dt=dt,
             status=status, steps=s+1, dU=d)
    xr = reattach(U, m.xnod, m.ynod, m.hy, N) if ok else np.nan
    print(f"  {domain:>5} / {'P+Z' if pz else 'free':>4}: {status:>8}  {s+1:>5} steps  "
          f"|dU| = {d:.3e}  max|u| = {(np.abs(U[...,0]).max() if ok else np.nan):.4f}  "
          f"x_r = {xr:.4f}  x_r/S = {(xr/S_STEP if np.isfinite(xr) else np.nan):.3f}  "
          f"{time.perf_counter()-t0:.0f}s", flush=True)
    return xr


if __name__ == '__main__':
    dom = sys.argv[1] if len(sys.argv) > 1 else 'long'
    which = sys.argv[2] if len(sys.argv) > 2 else 'both'
    print(f"ARMALY specification: ER = 1.94, h = {H_IN}, S = {S_STEP}, "
          f"Re = {RE:g} (D = 2h), nu = {NU:.6e}")
    print(f"Armaly measured x1/S = 8.0 +/- 0.7 at this Re;  his own computation gives 7.0\n")
    for pz in ((False, True) if which == 'both' else (which == 'pz',)):
        run(dom, pz)
