"""Does the Poiseuille outflow result transfer to the SHORT-domain BFS?

OUTFLOW_BC_STUDY.md establishes, on Poiseuille:
  - free outflow supplies ZERO admissible boundary conditions and is what breaks
    the cold start; the correct answer stays a stable fixed point regardless;
  - the number of admissible conditions sets how far down dt the COLD START can
    reach (0 -> 0.9, 1 -> 0.25, 2 -> 0.1);
  - so the conditions widen the BASIN rather than curing an instability.

Every BFS result in WEIGHT_VS_TIMESTEP_STUDY.md and STEADY_FORM_STUDY.md uses
the same free outflow (`bc[e,1] == 4` overridden to 0), and STEADY_FORM_STUDY.md
sec 8 independently found the short domain has TWO converged states, reaching the
physical one only when seeded from the long-domain solution -- "a basin problem,
not a hopeless one".  That is the same phenomenon, described before its cause
was known.  This tests whether it is the same cause.

Two knobs, crossed:

  IC        cold   U = 0
            para   the LOCAL fully developed parabola everywhere -- inlet channel
                   u = 6*eta*(1-eta), eta = (y-0.5)/0.5;  downstream u = 3y(1-y),
                   which carries the same mass flux (0.5).  Discontinuous at the
                   step, and deliberately so: it assumes only "fully developed
                   somewhere", not the answer.
            devc   the smooth blended IC bfs_steady.py already uses, for contrast

  OUTFLOW   free   nothing imposed (what every BFS result to date used)
            P+Z    p = 0 on the outlet plane AND d(omega)/dx = 0 -- the full
                   admissible pair, with the bc=4 mask bug fixed

If `para` or `P+Z` reaches a different converged state than `cold`+`free`, the
short-domain two-state finding is an outflow/basin artifact and the BFS
conclusions that rest on it need re-examining.

Re = 389, nu = 1/Re, w_mom = w_mass = 1, dt = 1 (the recommended operating point).
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
from lssem2d.lgl import diff_matrix, lgl_weights
from lssem2d.lssem import SolverState, apply_L
import lssem2d.solver as S
import lssem2d.bc as BC

GRID = '/Users/danielchan/Dropbox/F90_SEM/pmg_clean/cnos_short_grid.dat'
RE, LB = 389.0, 1.0
OB = BC.apply_bc


def build():
    m, _, _ = load(GRID)
    n = m.N+1
    pin = next((e, n-1, 0) for e in range(m.nelem)
               if m.bc[e, 1] == 4 and m.bc[e, 2] == 1)
    return m, n, pin


def ic_para(m, n):
    """Local fully-developed parabola everywhere.  Same mass flux both sides."""
    U = np.zeros((m.nelem, n, n, 4))
    for e in range(m.nelem):
        for i in range(n):
            x = m.xnod[e, i]
            for j in range(n):
                y = m.ynod[e, j]
                if x < 0.0:                      # inlet channel, y in [0.5, 1]
                    eta = (y-0.5)/0.5
                    U[e, i, j, 0] = 6.0*eta*(1.0-eta)
                    U[e, i, j, 3] = -12.0*(1.0-2.0*eta)
                else:                            # expansion, y in [0, 1]
                    U[e, i, j, 0] = 3.0*y*(1.0-y)
                    U[e, i, j, 3] = -3.0*(1.0-2.0*y)
    return U


def ic_devc(m, n):
    """The smooth blended IC from bfs_steady.py."""
    U = np.zeros((m.nelem, n, n, 4))
    ud = lambda y: 3.0*y*(1.0-y); dud = lambda y: 3.0-6.0*y
    def us(y):
        if y <= 0.5: return 0.0
        e = 2.0*y-1.0; return 6.0*e*(1.0-e)
    def dus(y):
        if y <= 0.5: return 0.0
        e = 2.0*y-1.0; return 12.0*(1.0-2.0*e)
    def G(y):
        I = 1.5*y*y-y**3
        if y > 0.5:
            e = 2.0*y-1.0; I -= 0.5*(3.0*e*e-2.0*e**3)
        return I
    for e in range(m.nelem):
        for i in range(n):
            x = m.xnod[e, i]
            if x <= 0.0:
                t = sp = spp = 0.0
            else:
                t = min(x/LB, 1.0); sp = (6.0*t-6.0*t*t)/LB; spp = (6.0-12.0*t)/LB**2
                if x >= LB: sp = spp = 0.0
            sv = 3.0*t*t-2.0*t**3
            for j in range(n):
                y = m.ynod[e, j]
                if x < 0.0:
                    eta = (y-0.5)/0.5
                    U[e, i, j, 0] = 6.0*eta*(1.0-eta)
                    U[e, i, j, 3] = -12.0*(1.0-2.0*eta)
                else:
                    U[e, i, j, 0] = (1.0-sv)*us(y)+sv*ud(y)
                    U[e, i, j, 1] = -sp*G(y)
                    U[e, i, j, 3] = -spp*G(y)-((1.0-sv)*dus(y)+sv*dud(y))
    return U


def merit(st, U):
    f = np.ascontiguousarray(U[..., 0]/2.0); g = np.ascontiguousarray(U[..., 1]/2.0)
    st.update_linearisation(f, g)
    r = apply_L(st, U, f, g)
    return float(np.sum(r*r/st.mesh.wq[..., None]))


def reattach(U, m, D):
    """First x > 0 on the bottom wall where the wall shear turns positive."""
    pts = []
    for e in range(m.nelem):
        if m.bc[e, 2] != 1:                      # bottom wall only
            continue
        for i in range(m.N+1):
            x = m.xnod[e, i]
            if x <= 0.0:
                continue
            dudy = float(np.dot(D[0, :], U[e, i, :, 0]))*(2.0/m.hy[e])
            pts.append((x, dudy))
    pts.sort()
    for k in range(1, len(pts)):
        if pts[k-1][1] < 0.0 <= pts[k][1]:
            x0, s0 = pts[k-1]; x1, s1 = pts[k]
            return x0 + (x1-x0)*(-s0)/(s1-s0)
    return float('nan')


def run(ic, pz, dt=1.0, cap=400, wall=600.0):
    m, n, pin = build(); N = m.N
    D = diff_matrix(N)
    for e in range(m.nelem):
        if m.bc[e, 1] == 4 and not pz:
            m.bc[e, 1] = 0                       # free outflow
    xmax = m.xnod.max()
    out = [e for e in range(m.nelem) if abs(m.xnod[e, -1]-xmax) < 1e-9]
    use_pin = False if pz else pin

    def bc2(mesh, U, **kw):
        U = OB(mesh, U, **kw)
        for e in out:
            U[e, -1, :, 3] = -(D[-1, :-1] @ U[e, :-1, :, 3])/D[-1, -1]
        return U

    st = SolverState(m, D, nu=1.0/RE, dt=dt, fac1=1.0, w_mom=1.0, w_mass=1.0)
    if pz:
        st.get_global_mask(pin_p=use_pin)
        for e in out:
            st._global_mask[e, -1, :, 2] = 0.0
            st._global_mask[e, -1, :, 3] = 0.0
        S.apply_bc = bc2
    inlet = lambda x, y, t: 6.0*((y-0.5)/0.5)*(1.0-(y-0.5)/0.5)
    U = {'cold': lambda: np.zeros((m.nelem, n, n, 4)),
         'para': lambda: ic_para(m, n),
         'devc': lambda: ic_devc(m, n)}[ic]()
    J0 = merit(st, U)
    t0 = time.perf_counter(); status = 'CAP'; d = np.nan
    try:
        h = [U.copy()]
        for s in range(cap):
            prev = h[0].copy()
            U = S.step_bdf(st, h, time=s*dt, max_newton=1, newton_tol=1e-12,
                           newton_factor=0.0, custom_inlet=inlet, pin_p=use_pin,
                           cgsfac=1e-8, cg_tol=1e-10, cg_max_iter=300000)
            if not np.all(np.isfinite(U)):
                status = 'NaN'; break
            d = float(np.abs(U-prev).max())
            if np.abs(U[..., 0]).max() > 20.0:
                status = 'BLEWUP'; break
            if d < 1e-12:
                status = 'conv'; break
            if time.perf_counter()-t0 > wall:
                status = 'WALLCAP'; break
    finally:
        S.apply_bc = OB
    ok = np.all(np.isfinite(U))
    return (status, s+1, d, (merit(st, U) if ok else np.nan), J0,
            float(np.abs(U[..., 0]).max()) if ok else np.nan,
            (reattach(U, m, D) if ok else np.nan), time.perf_counter()-t0)


def main():
    print("SHORT-domain BFS, Re = 389, w_mom = w_mass = 1, dt = 1, cold vs parabolic IC,")
    print("free outflow vs the full admissible pair (p = 0 and d(omega)/dx = 0).\n")
    hdr = (f"{'IC':>6}{'outflow':>9}{'status':>9}{'steps':>7}{'|dU|':>11}"
           f"{'J start':>11}{'J end':>11}{'max|u|':>9}{'x_r':>9}{'wall s':>8}")
    print(hdr)
    for ic in ('cold', 'para', 'devc'):
        for pz, lab in ((False, 'free'), (True, 'P+Z')):
            r = run(ic, pz)
            print(f"{ic:>6}{lab:>9}{r[0]:>9}{r[1]:>7}{r[2]:>11.3e}{r[4]:>11.3e}"
                  f"{r[3]:>11.3e}{r[5]:>9.3f}{r[6]:>9.3f}{r[7]:>8.0f}", flush=True)
    print("\nDifferent converged states between rows => the short-domain two-state")
    print("finding (STEADY_FORM_STUDY.md sec 8) is an outflow/basin artifact.")


# Guard added 2026-08-12 -- WITHOUT IT, `from bfs_outflow_ic import ic_para`
# re-runs the whole six-case sweep before the importer does anything.  That cost
# ~15 min in bfs_plot_pz.py.  Third time this bit: see pois_temporal.py and
# pois_option4.py.  Any scratch script meant to be importable needs this.
if __name__ == '__main__':
    main()
