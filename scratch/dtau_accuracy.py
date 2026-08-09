"""Does dtau change the ANSWER?  The falsification test.  (design note sec 6 step 4)

Poiseuille Re=100 is the case that can kill the O(kappa*R) argument.  Its exact
solution is exactly representable, the least-squares residual is driven to
machine zero (J = 5.94e-27, CG_TOLERANCE_FLOOR.md sec 2), so kappa*E^T R = 0 and
dtau MUST leave the answer untouched.  If the profile error moves with dtau
here, the theory in PSEUDO_TIME_DESIGN.md sec 3 is wrong.

Swept at matched KAPPA rather than matched dtau.  kappa = a_flux/dtau = w_mom/dtau
is the damping the operator actually sees; sweeping dtau at fixed w_mom varies
two things at once, which is how the first BFS trial ended up comparing
kappa = 0.1 against kappa = 10 and calling it a dtau comparison.
"""
import os, sys, time
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix, lgl_weights
from lssem2d.lssem import SolverState, ls_pseudo, apply_L
import lssem2d.solver as S

# TIGHT linear solve.  The first run of this script used the default tol=1e-6
# floor and got J = 3.4e-09 with prof err 8.46e-03 -- exactly the
# tolerance-limited number in CG_TOLERANCE_FLOOR.md sec 3.  With R that large
# the R -> 0 limit is never reached and the test cannot falsify anything.
_p = S.pcg_solve


def _tight(state, b, fu, fv, M, mw, pin_p=False, max_iter=5000, tol=1e-6,
           cgsfac=0.0, precond=None, **kw):
    return _p(state, b, fu, fv, M, mw, pin_p=pin_p, max_iter=200000,
              tol=1e-12, cgsfac=1e-10, precond=precond)


S.pcg_solve = _tight

LX, LY, RE = 10., 1., 100.
NU = 1.0*LY/RE
DP = 12.0*NU*1.0/LY**2*LX
N, EX, EY, DT = 8, 10, 2, 0.5
TOL, CAP, TRIP = 1e-11, 600, 50.0
u_ex = lambda y: 6.0*y*(1.0-y)
WMOM = 1.0


def merit(st, U):
    f = np.ascontiguousarray(U[..., 0]/2.0); g = np.ascontiguousarray(U[..., 1]/2.0)
    st.update_linearisation(f, g)
    r = apply_L(st, U, f, g).copy()
    S._drop_pseudo(st, r, U)
    return float(np.sum(r*r/st.mesh.wq[..., None]))


def run(kappa):
    dtau = None if kappa == 0.0 else WMOM/kappa
    m = build_channel(LX, LY, EX, EY, N, bcs=(3, 4, 1, 1)); n = N+1
    pin = next((e, 0, 0) for e in range(m.nelem) if m.bc[e, 0] == 3 and m.bc[e, 2] == 1)
    for e in range(m.nelem):
        if m.bc[e, 1] == 4:
            m.bc[e, 1] = 0
    st = SolverState(m, diff_matrix(N), nu=NU, dt=DT, fac1=1.0,
                     w_mom=WMOM, w_mass=0.0, dtau=dtau)
    inlet = lambda x, y, t: u_ex(y)
    U = np.zeros((m.nelem, n, n, 4)); hist = [U]
    t0 = time.perf_counter(); status = 'cap'; s = 0
    for s in range(CAP):
        Up = hist[0].copy()
        U = S.step_bdf(st, hist, time=0.0, max_newton=1, newton_tol=1e-12,
                       newton_factor=0.0, custom_inlet=inlet, pin_p=pin,
                       cgsfac=1e-3, cg_max_iter=40000, verbose=False)
        if not np.all(np.isfinite(U)):
            status = 'NaN'; break
        if np.abs(U[..., 0]).max() > TRIP:
            status = 'diverged'; break
        if s > 2 and np.max(np.abs(U-Up)) < TOL:
            status = 'converged'; break
    wall = time.perf_counter()-t0
    if status != 'converged':
        return dict(k=kappa, dtau=dtau, status=status, steps=s+1, wall=wall)
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
    return dict(k=kappa, dtau=dtau, status=status, steps=s+1, wall=wall,
                J=merit(st, U),
                prof=float(np.sqrt(np.mean((us-u_ex(ys))**2))/1.5),
                dp=float(pbar('in')-pbar('out')), spread=float(ps.max()-ps.min()))


print("TIGHT linear solve: cgsfac=1e-10, tol=1e-12")
print("Poiseuille Re=100, order 8, 10x2, steady form (w_mass=0, w_mom=1), free outflow, inlet pin")
print(f"exact: dp = {DP},  outlet pressure constant across the channel")
print("The residual here goes to ~0, so kappa*E^T R = 0 and dtau must NOT move the answer.\n")
print(f"{'kappa':>8}{'dtau':>9}{'iters':>7}{'status':>12}{'J':>12}"
      f"{'prof err':>12}{'dp':>11}{'dp err':>11}{'p_out spread':>14}{'wall':>7}")
base = None
for kappa in (0.0, 0.01, 0.1, 1.0, 10.0):
    r = run(kappa)
    if r['status'] != 'converged':
        print(f"{kappa:>8.2f}{str(r['dtau']):>9}{r['steps']:>7}{r['status']:>12}"
              f"{'':>60}{r['wall']:>7.1f}")
    else:
        if base is None:
            base = r
        print(f"{kappa:>8.2f}{(r['dtau'] if r['dtau'] else float('inf')):>9.3g}"
              f"{r['steps']:>7}{r['status']:>12}{r['J']:>12.3e}{r['prof']:>12.3e}"
              f"{r['dp']:>11.6f}{abs(r['dp']-DP):>11.2e}{r['spread']:>14.3e}{r['wall']:>7.1f}")
    sys.stdout.flush()

print(f"\nreference (kappa = 0):  prof err {base['prof']:.3e}   dp {base['dp']:.6f}")
print("If the profile error is flat across kappa, sec 3 of the design note holds.")
