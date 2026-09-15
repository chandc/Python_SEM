"""dt from 0.1 down to 0.01: where does the basin ladder bottom out, and does the
fixed point stay stable all the way down?

Two separate questions, and they must not be conflated:

  COLD START  -- P+Z (two admissible conditions, the most the ADN count allows)
                 converges at dt = 0.1 and fails at 0.05.  Where exactly, and
                 does anything survive below?

  SEEDED      -- at dt = 0.1 the exact solution is a bit-exact fixed point even
                 with FREE outflow, so there is no instability there.  Does that
                 hold down to dt = 0.01, where a_mass = fac1/dt = 150?  If it
                 does, the "small dt is unstable" reading is dead for good and
                 the whole effect is basin size.

a_mass over this range: 15 (dt=0.1), 20, 30, 60, 150 (dt=0.01).  The velocity
block gets steadily MORE diagonally dominant, which is exactly why an instability
reading was suspect in the first place.

Seeded runs need the pressure datum to match the BC: free outflow pins p at the
inlet (p = -0.12 x), P+Z sets p = 0 at the outlet (p = -0.12 x + 1.2).
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
DTS = (0.1, 0.075, 0.05, 0.025, 0.01)


def run(dt, seed, pz, nst, wall=300.0):
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
            st._global_mask[e, -1, :, 2] = 0.0
            st._global_mask[e, -1, :, 3] = 0.0
        S.apply_bc = bc2
    t0 = time.perf_counter(); status = 'TCAP'; d = np.nan
    try:
        U = np.zeros((m.nelem, n, n, 4))
        y = m.ynod[:, None, :]
        if seed == 'exact':
            U[..., 0] = np.broadcast_to(ue(y), (m.nelem, n, n))
            U[..., 3] = np.broadcast_to(-(6.0-12.0*y), (m.nelem, n, n))
            U[..., 2] = -0.12*(xn[:, :, None]-xmin) + (1.2 if pz else 0.0)
        h = [U.copy()]
        for s in range(nst):
            prev = h[0].copy()
            U = S.step_bdf(st, h, time=s*dt, max_newton=1, newton_tol=1e-12,
                           newton_factor=0.0, custom_inlet=lambda x, y, t: ue(y),
                           pin_p=pin, cgsfac=1e-8, cg_tol=1e-10,
                           cg_max_iter=300000)
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
    return (status, s+1, d, float(pbar('in')-pbar('out')), prof,
            time.perf_counter()-t0)


def show(dt, lab, r):
    if r is None:
        print(f"{dt:>7g}{1.5/dt:>8.1f}{lab:>16}   NON-FINITE", flush=True); return
    print(f"{dt:>7g}{1.5/dt:>8.1f}{lab:>16}{r[0]:>9}{r[1]:>7}{r[2]:>11.3e}"
          f"{r[3]:>12.5f}{abs(r[3]-1.2)/1.2:>10.2e}{r[4]:>11.3e}{r[5]:>8.0f}",
          flush=True)


HDR = (f"{'dt':>7}{'a_mass':>8}{'case':>16}{'status':>9}{'steps':>7}{'|dU|':>11}"
       f"{'dp':>12}{'dp err':>10}{'prof err':>11}{'wall s':>8}")

print("A. COLD START with P+Z (two conditions -- the ADN maximum).")
print("   Converges at dt = 0.1, fails at 0.05.  Where does it bottom out?\n")
print(HDR)
for dt in DTS:
    show(dt, 'P+Z cold', run(dt, 'cold', True, nst=int(60.0/dt)))

print("\nB. SEEDED with the exact solution.  Is the fixed point still stable?")
print("   If |dU| stays 0 down to dt = 0.01 (a_mass = 150), there is no")
print("   small-dt instability at all and the effect is entirely basin size.\n")
print(HDR)
for dt in DTS:
    show(dt, 'free exact', run(dt, 'exact', False, nst=300))
print()
for dt in DTS:
    show(dt, 'P+Z exact', run(dt, 'exact', True, nst=300))
