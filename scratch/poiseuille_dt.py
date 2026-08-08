"""Impact of dt on plane Poiseuille, Re = U_mean*H/nu = 100.

WHY THIS IS THE RIGHT CONTROL
-----------------------------
The exact solution lies inside the discrete space:
    u = 6 y (1-y)   degree 2        v = 0
    p = p0 - Gx     degree 1        om = -du/dy, degree 1
and the convective terms vanish identically (u_x = 0, v = 0).  So the
least-squares residual can be driven to ZERO, and a zero residual is minimal
under ANY weighting.  dt should therefore have no effect whatsoever.  Note this
is NOT fixed by using a coarse or low-order mesh -- degree 2 is exact for any
N >= 2, and the rectangular geometry mapping is affine, so the residual stays
zero.  To make dt matter, the SOLUTION has to leave the polynomial space.

VARIANTS
    control  parabolic inlet, order 8      -> residual ~ 0, expect NO dt effect
    develop  UNIFORM inlet, order 8        -> entrance region is non-polynomial
    coarse   UNIFORM inlet, order 4, half the elements  -> larger residual

Analytic reference (H = 1, U_mean = 1, nu = 0.01):
    u(y)    = 6 y (1-y),  U_max = 1.5
    dp/dx   = -12 nu U_mean / H^2 = -0.12
    dp over L = 10        = 1.20

Pressure is pinned at the INLET (lower-left corner), outflow is FREE, so the
pressure drop is a prediction, not an imposed boundary value.
"""
import sys, os, time, json
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix, lgl_weights
from lssem2d.lssem import SolverState
import lssem2d.solver as S

SC = os.path.dirname(os.path.abspath(__file__))
LX, LY = 10.0, 1.0
RE, UMEAN = 100.0, 1.0
NU = UMEAN*LY/RE                      # Re = U_mean H / nu  ->  nu = 0.01
DPDX_EXACT = -12.0*NU*UMEAN/LY**2     # -0.12
DP_EXACT = -DPDX_EXACT*LX             # 1.20
TOL, NITCGS, CGSFAC = 1.0e-6, 40000, 1.0e-3
# Convergence must NOT be judged on the per-step change: that scales with dt, so a
# small-dt run stops while still far from steady state.  Use the RATE |dU|/dt, and
# require a minimum physical time -- the viscous time here is H^2/nu = 100, so the
# flow needs t >> 100 to develop from rest regardless of dt.
RATE_TOL, T_MIN, MAXSTEP = 1.0e-9, 300.0, 20000

VARIANTS = {                          # name: (N, EX, EY, inlet kind)
    'control': (8, 10, 2, 'parabolic'),
    'develop': (8, 10, 2, 'uniform'),
    'coarse':  (4,  5, 1, 'uniform'),
}
DTS = [0.0, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]

u_exact = lambda y: 6.0*y*(1.0-y)


def run(name, dt):
    N, EX, EY, inlet_kind = VARIANTS[name]
    mesh = build_channel(LX, LY, EX, EY, N, bcs=(3, 4, 1, 1))   # W inlet, E outflow, S/N wall
    n = N+1
    # pressure pin: inlet plane, lower-left corner
    pin = None
    for e in range(mesh.nelem):
        if mesh.bc[e, 0] == 3 and mesh.bc[e, 2] == 1:
            pin = (e, 0, 0); break
    # FREE outflow: downgrade bc 4 -> 0 so nothing is imposed (bc==4 would set p=0
    # along the whole outflow plane, which would fight the inlet pin)
    outl = [e for e in range(mesh.nelem) if mesh.bc[e, 1] == 4]
    for e in outl:
        mesh.bc[e, 1] = 0
    st = SolverState(mesh, diff_matrix(N), nu=NU, dt=dt, fac1=1.0)
    if inlet_kind == 'parabolic':
        inlet = lambda x, y, t: u_exact(y)
    else:
        inlet = lambda x, y, t: np.full_like(np.asarray(y, dtype=float), UMEAN)

    U = np.zeros((mesh.nelem, n, n, 4))
    t0 = time.perf_counter()
    steps = 0
    if dt == 0.0:
        # PURE STEADY FORM.  This CANNOT go through step_bdf: su_history is built
        # unconditionally from the BDF alphas while apply_L zeroes f1 when dt==0,
        # so the residual becomes N(u)wq - (2u_n - 0.5u_{n-1})wq and the fixed
        # point solves N(u) = 1.5u -- a spurious reaction term.  Drive newton_step
        # directly with su_history = 0, which gives the correct residual N(u) = 0.
        from lssem2d.assembly import gather_scatter
        from lssem2d.solver import compute_jacobi, newton_step
        mult = gather_scatter(mesh, np.ones_like(U))
        mw = 1.0/np.where(mult < 1e-10, 1.0, mult)
        zero_hist = np.zeros_like(U)
        for it in range(400):
            Up = U.copy()
            fu = np.ascontiguousarray(U[..., 0]); fv = np.ascontiguousarray(U[..., 1])
            st.update_linearisation(fu, fv)
            Mi = compute_jacobi(st, fu, fv, pin_p=pin)
            U, _dU, _it = newton_step(st, U, zero_hist, Mi, mw, custom_inlet=inlet,
                                      pin_p=pin, cgsfac=CGSFAC, cg_max_iter=NITCGS)
            steps = it+1
            if not np.all(np.isfinite(U)):
                return dict(name=name, dt=dt, ok=False)
            if it > 2 and np.max(np.abs(U-Up)) < 1.0e-11:
                break
    else:
        hist = [U]
        for s in range(MAXSTEP):
            Up = hist[0].copy()
            U = S.step_bdf(st, hist, time=s*dt, max_newton=1, newton_tol=1e-12,
                           newton_factor=0.0, custom_inlet=inlet, pin_p=pin,
                           cgsfac=CGSFAC, cg_max_iter=NITCGS, verbose=False)
            steps = s+1
            if not np.all(np.isfinite(U)):
                return dict(name=name, dt=dt, ok=False)
            rate = np.max(np.abs(U-Up))/dt
            if (s+1)*dt >= T_MIN and rate < RATE_TOL:
                break
    wall = time.perf_counter()-t0

    # ---- diagnostics -------------------------------------------------------
    D = diff_matrix(N); w = lgl_weights(N)
    xn, yn, hy = mesh.xnod, mesh.ynod, mesh.hy
    facx = np.array([2.0/(xn[e, -1]-xn[e, 0]) for e in range(mesh.nelem)])[:, None, None]
    facy = (2.0/hy)[:, None, None]
    u, v, p, om = (np.ascontiguousarray(U[..., k]) for k in range(4))
    dx = lambda a: np.einsum('ik,ekj->eij', D, a)*facx
    dy = lambda a: np.einsum('jk,eik->eij', D, a)*facy
    u_x, u_y, v_x, v_y = dx(u), dy(u), dx(v), dy(v)
    p_x, p_y, om_x, om_y = dx(p), dy(p), dx(om), dy(om)
    Rx = u*u_x + v*u_y + p_x + NU*om_y
    Ry = u*v_x + v*v_y + p_y - NU*om_x
    Rc = u_x + v_y
    Rw = om + u_y - v_x
    W = np.einsum('i,j->ij', w, w)[None]*np.array(
        [(xn[e, -1]-xn[e, 0])*hy[e]/4 for e in range(mesh.nelem)])[:, None, None]
    nrm = lambda R: float(np.sqrt(np.sum(R*R*W)/np.sum(W)))

    # profile error at the OUTLET plane (fully developed there)
    xmax = xn.max()
    ys, us = [], []
    for e in range(mesh.nelem):
        if abs(xn[e, -1]-xmax) < 1e-9:
            for j in range(n):
                ys.append(yn[e, j]); us.append(U[e, -1, j, 0])
    ys, us = np.array(ys), np.array(us)
    o = np.argsort(ys); ys, us = ys[o], us[o]
    k = np.concatenate(([True], np.diff(ys) > 1e-12))
    ys, us = ys[k], us[k]
    prof_err = float(np.sqrt(np.mean((us-u_exact(ys))**2))/1.5)   # relative to U_max

    # pressure drop: area-averaged p on the inlet and outlet planes
    def pbar(edge):
        tot = a = 0.0
        for e in range(mesh.nelem):
            xe = xn[e, 0] if edge == 'in' else xn[e, -1]
            ref = xn.min() if edge == 'in' else xmax
            if abs(xe-ref) < 1e-9:
                i = 0 if edge == 'in' else -1
                tot += np.sum(w*U[e, i, :, 2])*(hy[e]/2); a += hy[e]
        return tot/a
    dp = pbar('in')-pbar('out')

    return dict(name=name, dt=dt, ok=True, steps=steps, wall=wall,
                prof_err=prof_err, dp=float(dp),
                dp_err=float(abs(dp-DP_EXACT)/DP_EXACT),
                Rm=float(np.sqrt(nrm(Rx)**2+nrm(Ry)**2)), Rc=nrm(Rc), Rw=nrm(Rw),
                umax=float(np.abs(u).max()))


out = []
print(f"Plane Poiseuille  Re = U_mean*H/nu = {RE:g}   nu = {NU}   L x H = {LX} x {LY}")
print(f"exact:  u = 6y(1-y),  U_max = 1.5,  dp/dx = {DPDX_EXACT},  dp over L = {DP_EXACT}")
print(f"pressure pinned at the INLET lower-left corner;  outflow FREE\n")
for name in VARIANTS:
    N, EX, EY, ik = VARIANTS[name]
    print(f"=== {name}: order {N}, {EX}x{EY} elements, {ik} inlet ===")
    print(f"{'dt':>6}{'steps':>7}{'t_end':>8}{'|u-u_ex|/Umax':>15}{'dp':>10}{'dp err':>10}"
          f"{'|R_mom|':>11}{'|R_div|':>11}{'|R_vort|':>11}{'wall s':>8}")
    for dt in DTS:
        r = run(name, dt)
        out.append(r)
        if not r['ok']:
            print(f"{dt:>6}   DIVERGED"); continue
        print(f"{dt:>6}{r['steps']:>7}{(r['steps']*dt if dt>0 else float('nan')):>8.0f}{r['prof_err']:>15.3e}{r['dp']:>10.5f}"
              f"{r['dp_err']:>10.2e}{r['Rm']:>11.3e}{r['Rc']:>11.3e}{r['Rw']:>11.3e}"
              f"{r['wall']:>8.1f}")
    print()

json.dump(out, open(f'{SC}/poiseuille_dt.json', 'w'), indent=1)
print(f"saved {SC}/poiseuille_dt.json")
