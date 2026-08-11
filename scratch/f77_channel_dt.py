"""Reproduce the 1996 F77 solver setup on plane-channel (Poiseuille) flow.

Configuration copied from reference/tj_channel_1996.f, not chosen by us:

    legacy weights   w_mom = w_mass = None   ->  a_mass = fac1, a_flux = dt
    dtau = 1.0                               ->  kappa = dt/dtau = dt, i.e. the
                                                 bare `dt*u` in su(ij,1) and the
                                                 (dt+fac1) transpose coefficient
    nsub = 2         max_newton=2, newton_tol=0, newton_factor=0
                     (the F77 has NO Newton convergence test at all)
    no line search   (the F77 has no globalisation)
    cgsfac = 0.01    1% RELATIVE CG test
    cg_tol = 1e-14   absolute floor effectively OFF -- this is what the new
                     cg_tol argument unlocks; it was pinned at 1e-6 before
    nitcgs = 1000
    Jacobi preconditioner (dge builds the exact diagonal; pcg_solve's default)

Exact answers for Poiseuille Re=100, so all three are checkable:
    u(y) = 6y(1-y),  U0 = 1.5
    dp/dx = -12*nu*U_mean/H^2  ->  dp = 1.2 over L = 10
    outlet pressure CONSTANT across the channel

Swept against the pre-cg_tol default (1e-6) so the effect of the floor is
visible, and against dtau off, so the pseudo-time term's contribution is too.
"""
import os, sys, time
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix, lgl_weights
from lssem2d.lssem import SolverState, ls_coeffs, ls_pseudo
import lssem2d.solver as S

LX, LY, RE = 10.0, 1.0, 100.0
NU = 1.0*LY/RE
DP_EXACT = 12.0*NU*1.0/LY**2*LX          # 1.2
N, EX, EY = 8, 10, 2
DT = float(os.environ.get('DTV','1.0'))
NST, TOL, TRIP = 4000, 1e-11, 50.0
u_ex = lambda y: 6.0*y*(1.0-y)


def run(dtau, cg_tol, cgsfac=0.01, nsub=2, label=''):
    m = build_channel(LX, LY, EX, EY, N, bcs=(3, 4, 1, 1))
    n = N+1
    # F77 pins ONE pressure node: mid-height, west edge of element 1, and sets
    # it to zero.  Ours masks the increment only, so the level is inherited --
    # immaterial here because the level is a null mode and the inlet pin fixes
    # the datum for the dp measurement anyway.
    pin = next((e, 0, 0) for e in range(m.nelem)
               if m.bc[e, 0] == 3 and m.bc[e, 2] == 1)
    for e in range(m.nelem):
        if m.bc[e, 1] == 4:
            m.bc[e, 1] = 0                # free outflow
    st = SolverState(m, diff_matrix(N), nu=NU, dt=DT, fac1=1.0, dtau=dtau)
    inlet = lambda x, y, t: u_ex(y)

    U = np.zeros((m.nelem, n, n, 4)); hist = [U]
    t0 = time.perf_counter(); status = 'cap'; s = 0
    for s in range(NST):
        Up = hist[0].copy()
        U = S.step_bdf(st, hist, time=s*DT, max_newton=nsub,
                       newton_tol=0.0, newton_factor=0.0,
                       custom_inlet=inlet, pin_p=pin,
                       cgsfac=cgsfac, cg_tol=cg_tol, cg_max_iter=1000,
                       line_search=False)
        if not np.all(np.isfinite(U)):
            status = 'NaN'; break
        if np.abs(U[..., 0]).max() > TRIP:
            status = 'diverged'; break
        if s > 2 and np.max(np.abs(U-Up)) < TOL:
            status = 'steady'; break
    wall = time.perf_counter()-t0
    if status not in ('steady', 'cap'):
        return dict(label=label, status=status, steps=s+1, wall=wall)

    w = lgl_weights(N); xn, yn, hy = m.xnod, m.ynod, m.hy; xmax = xn.max()
    ys, ps, us = [], [], []
    for e in range(m.nelem):
        if abs(xn[e, -1]-xmax) < 1e-9:
            for j in range(n):
                ys.append(yn[e, j]); ps.append(U[e, -1, j, 2]); us.append(U[e, -1, j, 0])
    o = np.argsort(ys); ys, ps, us = np.array(ys)[o], np.array(ps)[o], np.array(us)[o]
    k = np.concatenate(([True], np.diff(ys) > 1e-12)); ys, ps, us = ys[k], ps[k], us[k]

    def pbar(edge):
        tot = a = 0.0
        for e in range(m.nelem):
            xe = xn[e, 0] if edge == 'in' else xn[e, -1]
            ref = xn.min() if edge == 'in' else xmax
            if abs(xe-ref) < 1e-9:
                i = 0 if edge == 'in' else -1
                tot += np.sum(w*U[e, i, :, 2])*(hy[e]/2); a += hy[e]
        return tot/a

    dp = float(pbar('in')-pbar('out'))
    return dict(label=label, status=status, steps=s+1, wall=wall,
                kappa=ls_pseudo(st), coef=ls_coeffs(st),
                prof=float(np.sqrt(np.mean((us-u_ex(ys))**2))/1.5),
                dp=dp, dperr=abs(dp-DP_EXACT)/DP_EXACT,
                spread=float(ps.max()-ps.min()),
                y=ys, p=ps, u=us)


print("Poiseuille Re=100, order 8, 10x2, free outflow + inlet pin")
print(f"exact:  u = 6y(1-y),  dp = {DP_EXACT},  outlet p constant\n")
print(f"{'configuration':<40}{'status':>9}{'steps':>7}{'kappa':>8}"
      f"{'prof err':>11}{'dp':>10}{'dp err':>10}{'p_out spread':>14}{'wall':>7}")

CASES = [
    ('F77 setup (dtau=1, cgsfac.01, tol1e-14)', 1.0,  1e-14, 0.01, 2),
    ('  ... but the OLD pinned tol = 1e-6',     1.0,  1e-6,  0.01, 2),
    ('  ... F77 tolerances, dtau OFF',          None, 1e-14, 0.01, 2),
    ('  ... dtau OFF and old tol (pre-today)',  None, 1e-6,  0.01, 2),
    ('F77 setup, nsub = 5',                     1.0,  1e-14, 0.01, 5),
]
out = {}
for label, dtau, ctol, cgs, nsub in CASES:
    r = run(dtau, ctol, cgs, nsub, label)
    out[label] = r
    if 'prof' not in r:
        print(f"{label:<40}{r['status']:>9}{r['steps']:>7}")
    else:
        print(f"{label:<40}{r['status']:>9}{r['steps']:>7}{r['kappa']:>8.3f}"
              f"{r['prof']:>11.3e}{r['dp']:>10.6f}{r['dperr']:>9.2%}"
              f"{r['spread']:>14.3e}{r['wall']:>7.1f}")
    sys.stdout.flush()

np.savez_compressed(f'{SC}/f77_channel_dt{os.environ.get("DTV","1.0")}.npz',
                    **{f'{k}_{a}': out[k][a] for k in out for a in ('y', 'p', 'u')
                       if a in out[k]})
