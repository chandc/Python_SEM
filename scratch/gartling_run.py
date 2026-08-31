"""Gartling (1990) BFS at Re = 800 with the P+Z outlet -- Chan & Mittal figs 3-6.

    python gartling_run.py steady   <NX> <N>
    python gartling_run.py unsteady <NX> <N> [dt] [tmax]

WHAT CHAN & MITTAL REPORT (CTR Proc. Summer Program 1996, pp.352-354):

  fig 3  u, v, omega at x = 7 and x = 15, against Gartling's benchmark, on the
         11 x 4 grid at 5th, 6th and 7th order.  "All except the vertical
         velocity profile at the axial location of 7 show an excellent
         agreement with the benchmark data of Gartling."
  fig 4  wall vorticity on both walls.  "Along the lower wall, UniFlo predicts a
         reattachment length of 6.1, whereas along the upper wall, it predicts a
         separation at the streamwise location of 4.8 and a reattachment at the
         streamwise location of 10.5."
  fig 5  UNSTEADY on the SAME 11 x 4 grid, 6th order, from a stagnant start.
         "final state is a temporally periodic flow" -- and the paper's point is
         that this periodicity is a NUMERICAL ARTIFACT: "numerical error that
         develops in a small region can grow over time and contaminate the
         entire flowfield ... the transient flow predicted above is a numerical
         artifact."
  fig 6  the same unsteady problem with 18 streamwise elements: "the initial
         transient flow features decay rapidly in time and the flow evolves
         asymptotically towards a steady state", agreeing with Gresho et al.

So figs 5 and 6 are a PAIR: the same physics on two grids, where the coarse one
invents a limit cycle and the finer one does not.  Reproducing that contrast is
the point, not reproducing either run alone.

VISCOSITY -- the trap, again.  Chan says "the Reynolds number based on the step
height and mean velocity is 800".  Taken literally with S = 0.5 and mean inlet
velocity 1 that is nu = 0.5/800 = 6.25e-4.  But the standard Gartling benchmark,
and the value that reproduces his own quoted reattachment of 6.1, is

    nu = 1/800 = 1.25e-3

i.e. Re built on the inlet hydraulic diameter 2h = 1, not on S = 0.5.  Check:
x_r = 6.1 with S = 0.5 is x_r/S = 12.2, and Armaly's curve gives x1/S ~ 12 at
Re = 800 in that same convention.  With nu = 6.25e-4 the case would be Re = 1600
and the bubble far longer.  This is the same convention slip already documented
for the Armaly runs in armaly_run.py.

FORMULATION.  Steady runs use the pure steady form w_mass = 0 (momentum row is
exactly w_mom*N(U), no time derivative), the loose solve, and line_search=True
-- the configuration STEADY_FORM_STUDY.md sec 9 recommends and warns must not be
tightened.  Unsteady runs use w_mom = w_mass = 1 (dt_eff = dt) with BDF2 and
sub-iterations, matching Chan's own "sub-iterations are required at each time
step" and his 2nd-order backward differencing.

Every run is saved to a unique gartling_*.npz.  Never re-solve to answer a
follow-up.
"""
import os, sys, time
SC = os.path.dirname(os.path.abspath(__file__))
# Derived from __file__, not hardcoded: this module is imported by scripts that
# run on other machines (the T3 sweep runs in a container on the DGX Spark), and
# a hardcoded '/Users/danielchan/...' chdir kills them at import with
# FileNotFoundError before a single line of solver code executes.
ROOT = os.path.dirname(SC)
sys.path.insert(0, ROOT)
sys.path.insert(0, SC)
os.chdir(ROOT)
import numpy as np
import lssem2d
lssem2d.set_backend('numpy')
from fgrid import load
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState
import lssem2d.solver as S
import lssem2d.bc as BC

NU = 1.25e-3                     # = 1/800; see the module docstring
S_STEP = 0.5
SNAPS = (10.0, 20.0, 30.0, 50.0, 80.0, 100.0, 140.0)     # Chan fig 5
OB = BC.apply_bc


def grid_path(NX, N):
    return f'grids/gartling_nx{NX}_N{N}_grid.dat'


def inlet_profile(y):
    """parabolic on the upper half only; u_max = 1.5 at y = 0.25, mean = 1"""
    y = np.asarray(y, dtype=float)
    return np.where((y >= -1e-12) & (y <= 0.5+1e-12), 24.0*y*(0.5-y), 0.0)


def wall_shear(U, m, D):
    """(x, du/dy) along the bottom wall and along the top wall."""
    lo_x, lo_t, up_x, up_t = [], [], [], []
    n = m.N+1
    for e in range(m.nelem):
        if m.bc[e, 2] == 1:                       # bottom wall, j = 0
            for i in range(n):
                lo_x.append(m.xnod[e, i])
                lo_t.append(float(np.dot(D[0, :], U[e, i, :, 0]))*(2.0/m.hy[e]))
        if m.bc[e, 3] == 1:                       # top wall, j = N
            for i in range(n):
                up_x.append(m.xnod[e, i])
                up_t.append(float(np.dot(D[-1, :], U[e, i, :, 0]))*(2.0/m.hy[e]))
    o = np.argsort(lo_x); lo_x = np.array(lo_x)[o]; lo_t = np.array(lo_t)[o]
    o = np.argsort(up_x); up_x = np.array(up_x)[o]; up_t = np.array(up_t)[o]
    return lo_x, lo_t, up_x, up_t


def crossings(x, t, rising=True, xmin=0.05):
    """x locations where t changes sign (rising = negative -> positive)"""
    out = []
    for k in range(len(x)-1):
        if x[k] < xmin:
            continue
        a, b = t[k], t[k+1]
        if (a < 0 <= b) if rising else (a > 0 >= b):
            out.append(x[k] - a*(x[k+1]-x[k])/(b-a))
    return out


def features(U, m, D):
    """Chan's three numbers: lower reattachment, upper separation, upper reattachment."""
    lx, lt, ux, ut = wall_shear(U, m, D)
    lo_re = crossings(lx, lt, rising=True)
    # top wall: attached forward flow has du/dy < 0, so separation is du/dy going
    # negative -> positive, reattachment is positive -> negative
    up_sep = crossings(ux, ut, rising=True)
    up_re = crossings(ux, ut, rising=False)
    return (lo_re[0] if lo_re else np.nan,
            up_sep[0] if up_sep else np.nan,
            up_re[0] if up_re else np.nan)


def build(NX, N, steady, outlet='pz', wmom=1.0, wmass=1.0):
    m, _, _ = load(grid_path(NX, N))
    D = diff_matrix(m.N)
    n = m.N+1
    xmax = m.xnod.max()
    out = [e for e in range(m.nelem) if abs(m.xnod[e, -1]-xmax) < 1e-9]
    pin = False
    if outlet == 'free':
        # Chan's own condition: the outflow plane is left as unknowns.  Drop the
        # p = 0 flag and pin the pressure at one corner instead.
        pin = next((e, n-1, 0) for e in out if m.bc[e, 2] == 1)
        for e in out:
            m.bc[e, 1] = 0

    def bc2(mesh, U, **kw):
        U = OB(mesh, U, **kw)
        for e in out:                              # dw/dx = 0 at the outlet
            U[e, -1, :, 3] = -(D[-1, :-1] @ U[e, :-1, :, 3])/D[-1, -1]
        return U

    st = SolverState(m, D, nu=NU, dt=1.0, fac1=1.0,
                     w_mom=wmom, w_mass=(0.0 if steady else wmass))
    st.get_global_mask(pin_p=pin)
    if outlet == 'pz':
        for e in out:
            st._global_mask[e, -1, :, 3] = 0.0     # omega is determined, not free
    return m, D, st, (bc2 if outlet == 'pz' else OB), pin


def run_steady(NX, N, cap=300, wmom=1.0):
    m, D, st, bc2, pin = build(NX, N, steady=True, wmom=wmom)
    S.apply_bc = bc2
    n = m.N+1
    inl = lambda x, y, t: inlet_profile(y)
    U = np.zeros((m.nelem, n, n, 4)); h = [U.copy()]
    t0 = time.perf_counter(); status = 'CAP'; d = np.nan
    try:
        for s in range(cap):
            prev = h[0].copy()
            U = S.step_bdf(st, h, time=0.0, max_newton=1, newton_tol=1e-12,
                           newton_factor=0.0, custom_inlet=inl, pin_p=False,
                           cgsfac=1e-3, cg_tol=1e-6, cg_max_iter=200000,
                           line_search=True)
            if not np.all(np.isfinite(U)):
                status = 'NaN'; break
            d = float(np.abs(U-prev).max())
            if np.abs(U[..., 0]).max() > 20.0:
                status = 'BLEWUP'; break
            if d < 1e-10:
                status = 'conv'; break
    finally:
        S.apply_bc = OB
    lo, us, ur = features(U, m, D)
    tag = (f'{SC}/gartling_steady_nx{NX}_N{N}.npz' if wmom == 1.0
           else f'{SC}/gartling_steady_nx{NX}_N{N}_wm{wmom:g}.npz')
    np.savez(tag, U=U, xnod=m.xnod, ynod=m.ynod, hy=m.hy, hx=m.hx, N=m.N,
             nu=NU, status=status, iters=s+1, dU=d, NX=NX,
             lo_reatt=lo, up_sep=us, up_reatt=ur)
    print(f"  STEADY nx{NX} N{N}: {status:>7} {s+1:>4} it  |dU| = {d:.3e}  "
          f"max|u| = {np.abs(U[...,0]).max():.4f}", flush=True)
    print(f"      lower reattach = {lo:.3f}  (Chan 6.1)   upper sep = {us:.3f} "
          f"(4.8)   upper reattach = {ur:.3f} (10.5)   {time.perf_counter()-t0:.0f}s",
          flush=True)


def run_unsteady(NX, N, dt=0.1, tmax=140.0, nsub=5, outlet='pz',
                 ic='stagnant', ramp=0.0, wmom=1.0, wmass=1.0, dtau_p=None):
    """ic='stagnant' is Chan's own start.  ic='steady' restarts from the converged
    steady field: the time stepper MUST hold that fixed point, so if it does not,
    the failure is in the time stepping and not in the violent startup transient.
    ramp>0 turns the inlet on smoothly over `ramp` time units instead of
    impulsively -- a deviation from Chan, used only to isolate the impulsive start."""
    m, D, st, bc2, pin = build(NX, N, steady=False, outlet=outlet,
                               wmom=wmom, wmass=wmass)
    st.dt = dt
    # artificial compressibility on the continuity row (lssem.ls_pseudo_p).
    # 'match' sets kappa_p = a_mass, the balance that makes the continuity row
    # scale with the momentum row at every dt.  Needs converged sub-iterations.
    a_mass_ = wmass*1.5/dt
    if dtau_p == 'match':
        dtau_p = 1.0/a_mass_
    elif dtau_p is not None:
        dtau_p = float(dtau_p)
    st.dtau_p = dtau_p
    print(f"      a_mass = {a_mass_:.4g}, a_flux = {wmom:g}, "
          f"kappa_p = {0.0 if dtau_p is None else 1.0/dtau_p:.4g}", flush=True)
    S.apply_bc = bc2
    n = m.N+1
    dt_eff_ = dt*wmom/wmass
    if ramp > 0:
        inl = lambda x, y, t: inlet_profile(y)*min(t/ramp, 1.0)
    else:
        inl = lambda x, y, t: inlet_profile(y)
    if ic.startswith('file:'):
        # continuation from an arbitrary saved field -- used to change ONE
        # parameter on an already-converged state and watch where it moves.
        U = np.load(ic[5:], allow_pickle=True)['U'].copy()
        print(f"      restart from {os.path.basename(ic[5:])}, max|u| = "
              f"{np.abs(U[...,0]).max():.6f}", flush=True)
    elif ic == 'steady':
        U = np.load(f'{SC}/gartling_steady_nx{NX}_N{N}.npz')['U'].copy()
        print(f"      restart from converged steady field, max|u| = "
              f"{np.abs(U[...,0]).max():.6f}", flush=True)
    else:
        U = np.zeros((m.nelem, n, n, 4))           # stagnant start, as Chan
    h = [U.copy()]
    # PHYSICAL time per step is dt_eff = dt*w_mom/w_mass, not dt (ls_coeffs:
    # "w_mass/w_mom ... a rescaling of time").  tmax is a PHYSICAL time, so the
    # step count and every reported t must use dt_eff.  Using dt here silently
    # overstated elapsed time by 1/(w_mom/w_mass) in the w_mom=0.1, w_mass=1 runs.
    dt_eff = dt*wmom/wmass
    nsteps = int(round(tmax/dt_eff))
    hist = np.zeros((nsteps+1, 6))                 # t, max|u|, max|v|, lo, usep, ure
    snaps, snap_t = [], []
    t0 = time.perf_counter(); status = 'ok'; nfilled = 0
    try:
        for s in range(nsteps):
            tnow = (s+1)*dt_eff
            U = S.step_bdf(st, h, time=tnow, max_newton=nsub, newton_tol=1e-12,
                           newton_factor=(1e-6 if st.dtau_p is not None else 1e-4),
                           custom_inlet=inl, pin_p=pin,
                           cgsfac=1e-3, cg_tol=1e-6, cg_max_iter=200000,
                           line_search=True)
            if not np.all(np.isfinite(U)):
                status = f'NaN@t={tnow:.2f}'; break
            if np.abs(U[..., 0]).max() > 20.0:
                status = f'BLEWUP@t={tnow:.2f}'; break
            lo, us_, ur = features(U, m, D)
            hist[s+1] = (tnow, np.abs(U[..., 0]).max(), np.abs(U[..., 1]).max(),
                         lo, us_, ur)
            nfilled = s+1
            for T in SNAPS:
                if abs(tnow-T) < 0.5*dt:
                    snaps.append(U.copy()); snap_t.append(tnow)
            if (s+1) % 100 == 0:
                print(f"      t = {tnow:7.2f}  max|u| = {hist[s+1,1]:.4f}  "
                      f"max|v| = {hist[s+1,2]:.5f}  lo_reatt = {lo:.3f}  "
                      f"{time.perf_counter()-t0:.0f}s", flush=True)
    finally:
        S.apply_bc = OB
    hh = hist[1:nfilled+1]                         # ONLY filled rows; row 0 is the zero IC
    # tmax MUST be in the name: two runs identical except for how long they ran
    # are different results, and omitting it silently overwrote the fig-5 run
    # (t=140) with a dt-sweep run (t=10) that shared every other parameter.
    # ic may be "file:/some/path.npz" for a continuation; that path must NOT go
    # into the filename verbatim -- its slashes and colon make an unwritable name
    # and np.savez then fails AFTER the whole run has been computed.
    ic_tag = 'restart' if ic.startswith('file:') else ic
    tag = (f'{SC}/gartling_unsteady_nx{NX}_N{N}_dt{dt:g}_T{tmax:g}_nsub{nsub}'
           f'_{outlet}_{ic_tag}{("_ramp%g" % ramp) if ramp else ""}'
           f'_wm{wmom:g}_ws{wmass:g}'
           f'{"" if dtau_p is None else "_ac%g" % (1.0/dtau_p)}.npz')
    np.savez(tag, U=U, xnod=m.xnod, ynod=m.ynod, hy=m.hy, hx=m.hx, N=m.N,
             nu=NU, status=status, dt=dt, tmax=tmax, NX=NX, nsub=nsub,
             outlet=outlet, ic=ic, ramp=ramp, wmom=wmom, wmass=wmass,
             dt_eff=dt*wmom/wmass, kappa_p=(0.0 if dtau_p is None else 1.0/dtau_p),
             hist=hh,
             snaps=np.array(snaps), snap_t=np.array(snap_t))
    treach = hh[-1, 0] if nfilled else 0.0
    print(f"  UNSTEADY nx{NX} N{N} dt={dt:g} nsub={nsub} {outlet} ic={ic}"
          f"{' ramp=%g' % ramp if ramp else ''} wmom={wmom:g} wmass={wmass:g}"
          f" [dt_eff={dt*wmom/wmass:g}]: {status}  "
          f"reached t = {treach:.2f}  {len(snaps)} snapshots  "
          f"{time.perf_counter()-t0:.0f}s -> {os.path.basename(tag)}", flush=True)
    if nfilled and treach > 20.0:
        tail = hh[hh[:, 0] > treach-20.0, 2]
        p2p = tail.max()-tail.min()
        print(f"      last 20 t.u. of max|v|: min {tail.min():.6f} max {tail.max():.6f} "
              f"peak-to-peak {p2p:.3e}", flush=True)
        if status == 'ok':
            print(f"      -> {'PERIODIC (fig 5)' if p2p > 1e-4 else 'STEADY (fig 6)'}",
                  flush=True)
        else:
            print("      -> NO VERDICT: the run did not survive to tmax", flush=True)


if __name__ == '__main__':
    mode = sys.argv[1]
    # NX may be a plain element count ("11") or a grid tag ("13g" = the 13-column
    # grid graded 2:1 as measured off Chan's own skeleton), so keep it as a string
    # unless it is purely numeric.
    NX = sys.argv[2]
    NX = int(NX) if NX.isdigit() else NX
    N = int(sys.argv[3])
    print(f"GARTLING Re = 800, nu = {NU:g}, domain [0,17] x [-0.5,0.5], "
          f"grid {NX} x 4, order {N}, P+Z outlet")
    if mode == 'steady':
        run_steady(NX, N,
                   cap=(int(sys.argv[5]) if len(sys.argv) > 5 else 300),
                   wmom=(float(sys.argv[4]) if len(sys.argv) > 4 else 1.0))
    else:
        dt = float(sys.argv[4]) if len(sys.argv) > 4 else 0.1
        tmax = float(sys.argv[5]) if len(sys.argv) > 5 else 140.0
        nsub = int(sys.argv[6]) if len(sys.argv) > 6 else 5
        outlet = sys.argv[7] if len(sys.argv) > 7 else 'pz'
        ic = sys.argv[8] if len(sys.argv) > 8 else 'stagnant'
        ramp = float(sys.argv[9]) if len(sys.argv) > 9 else 0.0
        wmom = float(sys.argv[10]) if len(sys.argv) > 10 else 1.0
        wmass = float(sys.argv[11]) if len(sys.argv) > 11 else 1.0
        dtau_p = sys.argv[12] if len(sys.argv) > 12 else None
        if dtau_p in ('none', 'None', ''):
            dtau_p = None
        run_unsteady(NX, N, dt, tmax, nsub, outlet, ic, ramp, wmom, wmass, dtau_p)
