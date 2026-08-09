"""SHORT-domain BFS: restart the converged loose solution with NO pressure pin.

With a free outflow and no pin, nothing fixes the pressure level: p enters the
equations only through p_x and p_y, so adding a constant to p changes no
residual at all.  L^T L therefore has an EXACT null mode and the linear system
is singular (but consistent).  CG on a consistent singular system still
converges to a solution modulo that null space -- the question is whether the
null component drifts, and whether anything physical moves with it.

Everything reported here is split into two kinds:

  shift-INVARIANT (physical)  : outlet p spread, dp across the domain,
                                Qout/Qin, rms div, max|u|, x_r/h, J
  shift-DEPENDENT (the mode)  : mean p, p at the old pin node

If the formulation is sound, the first group should be untouched and only the
second should move.  A pinned control run is included: it must be a 0-CG no-op,
otherwise the comparison means nothing.

Start: bfswms_0.1.npz -- w_mom = 0.1, w_mass = 0, converged at loose tolerance.
"""
import os, sys, time
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
sys.path.insert(0, SC)
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
from fgrid import load
from lssem2d.lgl import diff_matrix, lgl_weights
from lssem2d.lssem import SolverState, apply_L
from lssem2d.operators import dUdx, dUdy
import lssem2d.solver as S
from lssem2d import precond as P

RE, H, WMOM = 389.0, 0.5, 0.1
START = 'bfswms_0.1.npz'
_p = S.pcg_solve
CAP, WALL = 60, 900.0


def build():
    m, _, _ = load('/Users/danielchan/Dropbox/F90_SEM/pmg_clean/cnos_short_grid.dat')
    n = m.N+1
    pin = next((e, n-1, 0) for e in range(m.nelem) if m.bc[e, 1] == 4 and m.bc[e, 2] == 1)
    for e in range(m.nelem):
        if m.bc[e, 1] == 4:
            m.bc[e, 1] = 0                      # free outflow
    return m, n, pin


def merit(st, U):
    f = np.ascontiguousarray(U[..., 0]/2.0); g = np.ascontiguousarray(U[..., 1]/2.0)
    st.update_linearisation(f, g)
    r = apply_L(st, U, f, g)
    return float(np.sum(r*r/st.mesh.wq[..., None]))


def diag(U, m, pin):
    """Shift-invariant physics, then the two shift-dependent numbers."""
    N = m.N; n = N+1
    D = diff_matrix(N); w = lgl_weights(N)
    xn, yn, hy = m.xnod, m.ynod, m.hy
    ux = dUdx(np.ascontiguousarray(U[..., 0]), D, m.facx)
    vy = dUdy(np.ascontiguousarray(U[..., 1]), D, m.facy)
    fl = lambda e, i: np.sum(w*U[e, i, :, 0])*(hy[e]/2)
    xmin, xmax = xn.min(), xn.max()
    INL = [e for e in range(m.nelem) if abs(xn[e, 0]-xmin) < 1e-9 and yn[e, 0] > 0.4]
    OUT = [e for e in range(m.nelem) if abs(xn[e, -1]-xmax) < 1e-9]
    xs, tw = [], []
    for e in range(m.nelem):
        if yn[e, 0] > 0.01 or xn[e, 0] < -1e-9:
            continue
        for i in range(n):
            xs.append(xn[e, i]); tw.append(np.dot(D[0, :], U[e, i, :, 0])*(2.0/hy[e]))
    o = np.argsort(xs); xs, tw = np.array(xs)[o], np.array(tw)[o]; xr = np.nan
    for k in range(len(xs)-1):
        if tw[k] < 0 and tw[k+1] > 0 and xs[k] > 0.05:
            xr = xs[k]-tw[k]*(xs[k+1]-xs[k])/(tw[k+1]-tw[k]); break
    ue = np.array([U[e, -1, j, 0] for e in OUT for j in range(n)])
    pe = np.array([U[e, -1, j, 2] for e in OUT for j in range(n)])
    pi = np.array([U[e, 0, j, 2] for e in INL for j in range(n)])
    return dict(q=float(sum(fl(e, -1) for e in OUT)/sum(fl(e, 0) for e in INL)),
                div=float(np.sqrt(((ux+vy)**2).mean())),
                umax=float(np.abs(U[..., 0]).max()), xr=float(xr/H),
                psp=float(pe.max()-pe.min()),          # shift-invariant
                dp=float(pi.mean()-pe.mean()),         # shift-invariant
                rev=float(100*np.mean(ue < 0)),
                pmean=float((U[..., 2]*m.wq).sum()/m.wq.sum()),   # shift-DEPENDENT
                ppin=float(U[pin[0], pin[1], pin[2], 2]))         # shift-DEPENDENT


def run(use_pin):
    m, n, pin = build(); N = m.N
    st = SolverState(m, diff_matrix(N), nu=1.0/RE, dt=0.5, fac1=1.0,
                     w_mom=WMOM, w_mass=0.0)
    inlet = lambda x, y, t: 6.0*((y-0.5)/0.5)*(1.0-(y-0.5)/0.5)
    nit = [0]
    pp = pin if use_pin else None

    def pcg(state, b, fu, fv, M, mw, pin_p=False, max_iter=5000, tol=None,
            cgsfac=None, precond=None, **kw):
        pre = P.make('pmg2', state, fu, fv, M, pin_p,
                     pc=max(2, N//2), deg=4, coarse_deg=10)
        x, it = _p(state, b, fu, fv, M, mw, pin_p=pin_p, max_iter=300000,
                   tol=1e-6, cgsfac=1e-3, precond=pre)
        nit[0] += it; return x, it
    S.pcg_solve = pcg

    U = np.load(f'{SC}/{START}')['U'].copy()
    hist = [U]; t0 = time.perf_counter(); status = 'cap'; s = 0
    J0 = merit(st, U); d0 = diag(U, m, pin); trace = []
    try:
        for s in range(CAP):
            Up = hist[0].copy()
            U = S.step_bdf(st, hist, time=0.0, max_newton=1, newton_tol=1e-14,
                           newton_factor=0.0, custom_inlet=inlet, pin_p=pp,
                           cgsfac=1e-3, cg_max_iter=300000, verbose=False)
            dU = np.max(np.abs(U-Up)); um = np.abs(U[..., 0]).max()
            pm = float((U[..., 2]*m.wq).sum()/m.wq.sum())
            if s < 8 or s % 10 == 0:
                trace.append((s+1, dU, um, pm))
            if not np.all(np.isfinite(U)):
                status = 'NaN'; break
            if um > 20.0:
                status = f'DIVERGED({um:.1f})'; break
            if s > 1 and dU < 1e-11:
                status = 'conv'; break
            if time.perf_counter()-t0 > WALL:
                status = 'WALL'; break
    finally:
        S.pcg_solve = _p
    ok = np.all(np.isfinite(U)) and np.abs(U[..., 0]).max() < 20.0
    return dict(status=status, it=s+1, cg=nit[0], wall=time.perf_counter()-t0,
                J0=J0, J1=merit(st, U) if ok else np.nan, d0=d0,
                d1=diag(U, m, pin) if ok else None, trace=trace, U=U, m=m)


print("BFS Chan Re=389, SHORT domain, steady form w_mom=0.1 (w_mass=0), p-MG, LOOSE solve")
print(f"restart from {START} (converged WITH a pin) and remove the pressure pin\n")
print("With free outflow and no pin the constant-pressure mode is an EXACT null")
print("mode of L^T L.  Shift-invariant quantities should not move; only the level.\n")

res = {}
for tag, use_pin in (('pinned (control)', True), ('NO PIN', False)):
    r = run(use_pin); res[tag] = r
    d0, d1 = r['d0'], r['d1']
    print(f"=== {tag} ===")
    print(f"  status {r['status']}   iters {r['it']}   CG {r['cg']}   wall {r['wall']:.0f}s")
    print(f"  J   {r['J0']:.4e}  ->  {r['J1']:.4e}")
    if d1 is None:
        print("  field not finite\n"); continue
    print(f"  {'quantity':<26}{'start':>14}{'end':>14}{'change':>14}")
    for k, nm, inv in (('q', 'Qout/Qin', 1), ('div', 'rms div', 1),
                       ('umax', 'max|u|', 1), ('xr', 'x_r/h', 1),
                       ('psp', 'outlet p spread', 1), ('dp', 'dp inlet->outlet', 1),
                       ('rev', 'exit reversed %', 1),
                       ('pmean', 'MEAN p  (level)', 0), ('ppin', 'p at old pin node', 0)):
        mark = '' if inv else '   <- shift-dependent'
        print(f"  {nm:<26}{d0[k]:>14.6g}{d1[k]:>14.6g}{d1[k]-d0[k]:>14.3g}{mark}")
    print("  trace (it, max|dU|, max|u|, mean p): " +
          ", ".join(f"({a},{b:.2e},{c:.2f},{p:+.3f})" for a, b, c, p in r['trace'][:10]))
    np.savez_compressed(f"{SC}/bfsnp_{'pin' if use_pin else 'nopin'}.npz",
                        U=r['U'], xnod=r['m'].xnod, ynod=r['m'].ynod, hy=r['m'].hy)
    print()

# Is the no-pin field the pinned field plus a constant?
a = res['pinned (control)']['U']; b = res['NO PIN']['U']
if np.all(np.isfinite(b)):
    dp = b[..., 2] - a[..., 2]
    print("=== is the no-pin result just a constant shift of the pinned one? ===")
    print(f"  pressure difference : mean {dp.mean():+.6e}   spread {dp.max()-dp.min():.6e}")
    print(f"  velocity difference : max|du| {np.abs(b[...,0]-a[...,0]).max():.3e}"
          f"   max|dv| {np.abs(b[...,1]-a[...,1]).max():.3e}")
    print(f"  vorticity difference: max|dom| {np.abs(b[...,3]-a[...,3]).max():.3e}")
    print("  (a pure null-mode difference would be: pressure spread ~ 0, velocity ~ 0)")
