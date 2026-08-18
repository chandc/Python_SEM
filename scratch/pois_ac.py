"""Artificial compressibility on the continuity row, tested on 2D channel flow.

    uv run --quiet python scratch/pois_ac.py [dt] [dtau_p|none|match]

Plane Poiseuille, [0,4] x [0,1], 4x2 elements N=10, Re = 100, P+Z outlet
(p = 0 with dw/dx = 0).  The exact solution is representable in the discrete
space, so this is a check with no discretisation error to hide behind:

    u = 6y(1-y)   v = 0   om = 12y-6   dp = 12L/Re = 0.48

WHAT THIS TESTS.  Two things, in order of importance.

1. AC MUST NOT BREAK THE EXACT SOLUTION.  ls_pseudo_p perturbs the converged
   state by O(kappa_p * R).  Poiseuille drives R to ~0, so a correct
   implementation must return the exact answer whether AC is on or off.  If it
   does not, the term is wrong -- not merely inaccurate.

2. Does AC move the a_mass stability limit?  a_mass = w_mass*fac1/dt = 1.5/dt
   here, so the dt sweep spans a_mass = 1.5 .. 60, straddling the 6.05-12.1
   threshold measured on the Gartling BFS.  'match' sets dtau_p = 1/a_mass, the
   balance that makes the continuity row scale with the momentum row.

SUB-ITERATIONS MATTER.  The AC term vanishes only at sub-iteration convergence,
so max_newton is 8 with a tight newton_factor.  With max_newton = 1 it would
degenerate into a physical-time term and lose the accuracy guarantee.
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
from lssem2d.lssem import SolverState
import lssem2d.solver as S
import lssem2d.bc as BC

GRID = 'grids/poiseuille_f90_4x2_N10_grid.dat'
RE, L = 100.0, 4.0
NU = 1.0/RE
DP_EXACT = 12.0*L/RE
OB = BC.apply_bc


def inlet_profile(y, kind='parabolic'):
    """kind='parabolic' -> fully developed, u = 6y(1-y), the exact solution is
    then representable everywhere and the residual is ~0.
    kind='uniform'   -> slug inlet u = 1 (same mass flux).  The flow must now
    DEVELOP, so the exact solution is not representable, the residual is not
    zero, and the least-squares weighting has something to trade off -- the
    precondition the a_mass mechanism needs (GARTLING_VALIDATION.md sec 8).
    The wall BC is applied after the inlet in bc.apply_bc (W then S,N), so the
    corner nodes end at u = 0: the inlet singularity is resolved to no-slip."""
    y = np.asarray(y, dtype=float)
    if kind == 'uniform':
        return np.ones_like(y)
    return 6.0*y*(1.0-y)


def diagnostics(U, m, D, w):
    n = m.N+1
    eu = eom = div2 = area = 0.0; vmax = 0.0
    for e in range(m.nelem):
        u = U[e, :, :, 0]; v = U[e, :, :, 1]; om = U[e, :, :, 3]
        ux = (D @ u)*(2.0/m.hx[e]); vy = (v @ D.T)*(2.0/m.hy[e])
        wq = np.outer(w, w)*0.25*m.hx[e]*m.hy[e]
        div2 += np.sum((ux+vy)**2*wq); area += wq.sum()
        ye = m.ynod[e][None, :]*np.ones((n, 1))
        eu += np.sum((u-6.0*ye*(1.0-ye))**2*wq)
        eom += np.sum((om-(12.0*ye-6.0))**2*wq)
        vmax = max(vmax, np.abs(v).max())

    def pm(xt, i):
        num = den = 0.0
        for e in range(m.nelem):
            if abs(m.xnod[e, i]-xt) < 1e-9:
                num += np.sum(U[e, i, :, 2]*w)*(0.5*m.hy[e]); den += m.hy[e]
        return num/den
    dp = pm(m.xnod.min(), 0) - pm(m.xnod.max(), n-1)
    # Reference must come from the GRID, not a module constant: the original
    # short channel was L = 4, this one is L = 12, and hard-coding 12*4/RE made
    # dp_err wrong by 3x on any other grid.
    Lgrid = float(m.xnod.max() - m.xnod.min())
    dp_fd = 12.0*Lgrid/RE                 # fully-developed part
    # A UNIFORM inlet flow is still developing, so dp_fd is a lower bound: the
    # entrance adds an incremental drop K(inf)*rho*u^2/2 with K(inf) ~ 0.674 for
    # parallel plates, i.e. ~0.34 here.  Quoted for orientation only -- for a
    # developing flow the defensible check is agreement ACROSS dt, not against
    # an analytic value.
    dp_dev = dp_fd + 0.674*0.5
    return dict(maxu=np.abs(U[..., 0]).max(), maxv=vmax,
                rms_div=np.sqrt(div2/area), l2_u=np.sqrt(eu/area),
                l2_om=np.sqrt(eom/area), dp=dp, L=Lgrid, dp_fd=dp_fd,
                dp_dev=dp_dev, dp_err=(dp/dp_fd-1)*100)


def run(dt, dtau_p, cap=4000, nsub=8, tmax=200.0,
        grid=None, inlet='parabolic'):
    m, _, _ = load(grid or GRID)
    D = diff_matrix(m.N); w = lgl_weights(m.N); n = m.N+1
    xmax = m.xnod.max()
    out = [e for e in range(m.nelem) if abs(m.xnod[e, -1]-xmax) < 1e-9]

    def bc2(mesh, U, **kw):
        U = OB(mesh, U, **kw)
        for e in out:                                  # Z: dw/dx = 0
            U[e, -1, :, 3] = -(D[-1, :-1] @ U[e, :-1, :, 3])/D[-1, -1]
        return U

    st = SolverState(m, D, nu=NU, dt=dt, fac1=1.0, w_mom=1.0, w_mass=1.0)
    a_mass = 1.0*1.5/dt                                # BDF2 fac1 = 1.5
    if dtau_p == 'match':
        dtau_p = 1.0/a_mass
    st.dtau_p = dtau_p
    st.get_global_mask(pin_p=False)
    for e in out:
        st._global_mask[e, -1, :, 3] = 0.0             # P+Z: omega determined
    S.apply_bc = bc2
    inl = lambda x, y, t: inlet_profile(y, inlet)
    U = np.zeros((m.nelem, n, n, 4)); h = [U.copy()]
    nsteps = min(cap, int(round(tmax/dt)))
    t0 = time.perf_counter(); status = 'ok'; d = np.nan
    try:
        for s in range(nsteps):
            prev = h[0].copy()
            U = S.step_bdf(st, h, time=(s+1)*dt, max_newton=nsub,
                           newton_tol=1e-13, newton_factor=1e-6,
                           custom_inlet=inl, pin_p=False,
                           cgsfac=1e-3, cg_tol=1e-8, cg_max_iter=200000,
                           line_search=True)
            if not np.all(np.isfinite(U)):
                status = f'NaN@{(s+1)*dt:.2f}'; break
            if np.abs(U[..., 0]).max() > 20.0:
                status = f'BLEWUP@{(s+1)*dt:.2f}'; break
            d = float(np.abs(U-prev).max())
            if d < 1e-12:
                status = 'conv'; break
    finally:
        S.apply_bc = OB
    ok = np.all(np.isfinite(U))
    g = diagnostics(U, m, D, w) if ok else None
    # ALWAYS save the field under a unique name -- re-solving to answer a
    # follow-up is what this avoids.
    tagk = 'acoff' if dtau_p is None else f'ac{1.0/dtau_p:g}'
    np.savez(f'{SC}/chan_{inlet}_dt{dt:g}_{tagk}.npz', U=U, xnod=m.xnod,
             ynod=m.ynod, hx=m.hx, hy=m.hy, N=m.N, dt=dt, inlet=inlet,
             kappa_p=(0.0 if dtau_p is None else 1.0/dtau_p), status=status)
    return dict(dt=dt, a_mass=a_mass, dtau_p=dtau_p, kappa_p=(0.0 if dtau_p is None
                else 1.0/dtau_p), status=status, steps=s+1, dU=d,
                wall=time.perf_counter()-t0, **(g or {}))


if __name__ == '__main__':
    DTS = [1.0, 0.5, 0.25, 0.1, 0.05, 0.025]
    print(f'Poiseuille, Re = {RE:g}, P+Z outlet, w_mom = w_mass = 1, nsub = 8.')
    print(f'Exact: u = 6y(1-y), v = 0, om = 12y-6, dp = {DP_EXACT:.5f}')
    print(f'a_mass = 1.5/dt.  Gartling threshold: stable <= 6.05, divergent >= 12.1.\n')
    hdr = (f"{'AC':>8}{'dt':>7}{'a_mass':>8}{'kappa_p':>9}{'status':>13}{'steps':>7}"
           f"{'max|u|':>9}{'max|v|':>10}{'rms div':>10}{'L2 u':>10}{'dp err':>10}")
    print(hdr); print('-'*len(hdr))
    for tag, dtp in (('off', None), ('match', 'match')):
        for dt in DTS:
            r = run(dt, dtp)
            if 'maxu' not in r:
                print(f"{tag:>8}{dt:>7g}{r['a_mass']:>8.3g}{r['kappa_p']:>9.3g}"
                      f"{r['status']:>13}{r['steps']:>7}"); continue
            print(f"{tag:>8}{dt:>7g}{r['a_mass']:>8.3g}{r['kappa_p']:>9.3g}"
                  f"{r['status']:>13}{r['steps']:>7}{r['maxu']:>9.4f}"
                  f"{r['maxv']:>10.2e}{r['rms_div']:>10.2e}{r['l2_u']:>10.2e}"
                  f"{r['dp_err']:>9.3f}%", flush=True)
        print()
    print(f"{'exact':>8}{'':>7}{'':>8}{'':>9}{'':>13}{'':>7}{1.5:>9.4f}"
          f"{0.0:>10.2e}{0.0:>10.2e}{0.0:>10.2e}{0.0:>9.3f}%")
