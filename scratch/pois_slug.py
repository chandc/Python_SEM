"""SLUG-flow IC: does removing the acceleration-from-rest extend the small-dt range?

Run as:  python pois_slug.py 0.1 0.05 0.01     (dt values as argv, for parallelism)

THE POINT.  Three initial conditions, ranked by how much they ask of the first
steps:

  cold   U = 0.  The inlet demands u = 6y(1-y) immediately, so the fluid must be
         ACCELERATED FROM REST.  Cold-start max|p| at step 1 runs 10-40x the
         steady 1.2 (pois_impulse.py).
  slug   u = 1 everywhere, v = p = omega = 0.  SAME MASS FLUX as the parabola
         (both integrate to 1) and already moving at O(1), so no acceleration --
         but still not the solution: it violates no-slip at the walls and has the
         wrong profile, so a viscous transient remains.
  exact  the parabola.  A bit-exact fixed point at every dt down to 0.01, with
         max|p| = 1.2000 exactly -- no transient at all.

If the acceleration impulse is what shrinks the cold-start basin, SLUG should
converge at dt where COLD fails, and its step-1 max|p| should be far smaller.
If slug fails at the same dt as cold, the impulse is not the mechanism and what
matters is simply having ANY transient.

Everything saved to pois_slug_dt<dt>_<ic>_<bc>.npz -- no re-solving to answer a
follow-up.
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
WALL, TEND = 700.0, 60.0


def run(dt, ic, pz):
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

    U = np.zeros((m.nelem, n, n, 4))
    y = m.ynod[:, None, :]
    if ic == 'slug':
        U[..., 0] = 1.0          # uniform: same flux as 6y(1-y), omega = 0
    elif ic == 'exact':
        U[..., 0] = np.broadcast_to(ue(y), (m.nelem, n, n))
        U[..., 3] = np.broadcast_to(-(6.0-12.0*y), (m.nelem, n, n))
        U[..., 2] = -0.12*(xn[:, :, None]-xmin) + (1.2 if pz else 0.0)

    t0 = time.perf_counter(); status = 'TCAP'; d = np.nan; p1 = np.nan
    try:
        h = [U.copy()]
        for s in range(int(round(TEND/dt))):
            prev = h[0].copy()
            U = S.step_bdf(st, h, time=s*dt, max_newton=1, newton_tol=1e-12,
                           newton_factor=0.0, custom_inlet=lambda x, y, t: ue(y),
                           pin_p=pin, cgsfac=1e-8, cg_tol=1e-10,
                           cg_max_iter=300000)
            if not np.all(np.isfinite(U)):
                status = 'NaN'; break
            if s == 0:
                p1 = float(np.abs(U[..., 2]).max())    # the impulse metric
            d = float(np.abs(U-prev).max())
            if d < 1e-13:
                status = 'conv'; break
            if time.perf_counter()-t0 > WALL:
                status = 'WALLCAP'; break
    finally:
        S.apply_bc = OB
    ok = np.all(np.isfinite(U))
    np.savez(f'{SC}/pois_slug_dt{dt:g}_{ic}_{"pz" if pz else "free"}.npz',
             U=U, xnod=m.xnod, ynod=m.ynod, hy=m.hy, N=N, dt=dt, ic=ic,
             pz=pz, status=status, steps=s+1, dU=d, p_step1=p1)

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
    return status, s+1, d, dp, prof, p1, time.perf_counter()-t0


DTS = [float(a) for a in sys.argv[1:]] or [0.1]
print(f"dt = {', '.join(f'{d:g}' for d in DTS)}   ICs: cold / slug / exact   "
      f"outlet: P+Z", flush=True)
print(f"{'dt':>7}{'IC':>7}{'status':>9}{'steps':>7}{'|dU|':>11}{'dp':>11}"
      f"{'dp err':>10}{'prof err':>11}{'max|p| s1':>11}{'wall s':>8}", flush=True)
for dt in DTS:
    for ic in ('cold', 'slug', 'exact'):
        r = run(dt, ic, True)
        print(f"{dt:>7g}{ic:>7}{r[0]:>9}{r[1]:>7}{r[2]:>11.3e}{r[3]:>11.5f}"
              f"{abs(r[3]-1.2)/1.2:>10.2e}{r[4]:>11.3e}{r[5]:>11.4e}{r[6]:>8.0f}",
              flush=True)
    print(flush=True)
