"""Fine dt scan, 0.01 to 0.1, with the two-condition outlet (p = 0 and dw/dx = 0).

Run as:  python pois_pz_dtscan.py 0.1 0.09 0.08     (dt values as argv)
so several instances can cover disjoint dt in parallel.

The coarse scan left the threshold ambiguous.  dt = 0.075 was WALL-capped at
300 s with |dU| = 0.337 and profile error already 3.7e-03 -- marginal, possibly
just slow -- while 0.05 and below failed outright (|dU| = 504, 586, 1322).
This gives every case a 900 s / t = 60 budget and records the |dU| trend at four
points through the run, so "converging slowly" and "diverging" are separable
rather than both landing in a WALLCAP bucket.

Cold start throughout: the seeded fixed point is already known to be bit-exact
at every dt down to 0.01, so nothing here is about stability -- it is about how
far down the COLD-START basin reaches with two admissible conditions.

The bc = 4 mask bug is fixed in the library as of 2026-08-13, so pressure needs
no per-script patch now; only the omega row does.
"""
import os, sys, time
SC = os.path.dirname(os.path.abspath(__file__))
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
WALL, TEND = 900.0, 60.0


def run(dt):
    m = build_channel(10., 1., EX, EY, N, bcs=(3, 4, 1, 1))   # bc_E = 4 KEPT: p = 0
    n = N+1
    xn = m.xnod; xmax = xn.max(); xmin = xn.min()
    out = [e for e in range(m.nelem) if abs(xn[e, -1]-xmax) < 1e-9]

    def bc2(mesh, U, **kw):
        U = OB(mesh, U, **kw)
        for e in out:
            U[e, -1, :, 3] = -(D[-1, :-1] @ U[e, :-1, :, 3])/D[-1, -1]
        return U

    st = SolverState(m, D, nu=NU, dt=dt, fac1=1.0, w_mom=1.0, w_mass=1.0)
    st.get_global_mask(pin_p=False)
    for e in out:
        st._global_mask[e, -1, :, 3] = 0.0        # omega; p handled by bc = 4 now
    S.apply_bc = bc2
    nst = int(round(TEND/dt))
    marks = {int(nst*f): None for f in (0.25, 0.5, 0.75)}
    t0 = time.perf_counter(); status = 'TCAP'; d = np.nan
    try:
        U = np.zeros((m.nelem, n, n, 4)); h = [U.copy()]
        for s in range(nst):
            prev = h[0].copy()
            U = S.step_bdf(st, h, time=s*dt, max_newton=1, newton_tol=1e-12,
                           newton_factor=0.0, custom_inlet=lambda x, y, t: ue(y),
                           pin_p=False, cgsfac=1e-8, cg_tol=1e-10,
                           cg_max_iter=300000)
            if not np.all(np.isfinite(U)):
                status = 'NaN'; break
            d = float(np.abs(U-prev).max())
            if s in marks:
                marks[s] = d
            if d < 1e-13:
                status = 'conv'; break
            if time.perf_counter()-t0 > WALL:
                status = 'WALLCAP'; break
    finally:
        S.apply_bc = OB
    ok = np.all(np.isfinite(U))
    # ALWAYS persist the field, under a name carrying the distinguishing
    # parameters -- re-solving to answer a follow-up question wastes real time.
    np.savez(f'{SC}/pzscan_dt{dt:g}.npz', U=U, xnod=m.xnod, ynod=m.ynod,
             hy=m.hy, N=N, dt=dt, status=status, steps=s+1, dU=d)

    def pbar(edge):
        tot = a = 0.0
        for e in range(m.nelem):
            xe = xn[e, 0] if edge == 'in' else xn[e, -1]
            ref = xmin if edge == 'in' else xmax
            if abs(xe-ref) < 1e-9:
                i = 0 if edge == 'in' else -1
                tot += np.sum(wq*U[e, i, :, 2])*(m.hy[e]/2); a += m.hy[e]
        return tot/a
    dp = float(pbar('in')-pbar('out')) if ok else np.nan
    prof = float(np.sqrt(np.mean((U[..., 0]-ue(m.ynod)[:, None, :])**2))/1.5) if ok else np.nan
    tr = [marks[k] for k in sorted(marks)]
    pout = max(np.abs(U[e, -1, :, 2]).max() for e in out) if ok else np.nan
    return (status, s+1, (s+1)*dt, d, dp, prof, tr, pout, time.perf_counter()-t0)


DTS = [float(a) for a in sys.argv[1:]] or [0.1]
print(f"P+Z cold start, dt = {', '.join(f'{d:g}' for d in DTS)}   "
      f"(caps: t <= {TEND:g}, wall <= {WALL:g}s)", flush=True)
print(f"{'dt':>7}{'a_mass':>8}{'status':>9}{'steps':>7}{'t_end':>8}{'|dU| final':>12}"
      f"{'|dU| @25/50/75%':>34}{'dp':>11}{'prof err':>11}{'p_out':>10}{'wall s':>8}",
      flush=True)
for dt in DTS:
    r = run(dt)
    tr = "  ".join(f"{v:.2e}" if v is not None else "   --   " for v in r[6])
    print(f"{dt:>7g}{1.5/dt:>8.1f}{r[0]:>9}{r[1]:>7}{r[2]:>8.1f}{r[3]:>12.3e}"
          f"{tr:>34}{r[4]:>11.5f}{r[5]:>11.3e}{r[7]:>10.2e}{r[8]:>8.0f}", flush=True)
