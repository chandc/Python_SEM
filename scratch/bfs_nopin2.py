"""No pressure pin, done properly.

The first attempt was a null result: restarted from its own converged field, the
loose stopping test finds ||b|| already under threshold and PCG returns after
ZERO iterations, so removing the pin could not have had an effect.  Identical
numbers proved nothing because no linear solve ran.

Two experiments that do run:

  TEST 1 -- is the constant-pressure mode really null?
      Take the converged pinned field, ADD a constant to p everywhere, and
      evaluate.  If the mode is null, J is unchanged to machine precision and
      every shift-invariant quantity is identical.  Then iterate: with no pin
      the shift must SURVIVE (nothing pulls it back); with a pin it must be
      REMOVED (the pin node is forced to 0).

  TEST 2 -- does the pin change the answer when Newton actually works?
      Start from the w_mom = 0.5 field and solve at w_mom = 0.1, so the start
      is genuinely off-minimiser and CG has to do real work.  Run it with and
      without the pin and compare the converged results.

Loose tolerance (cgsfac 1e-3, tol 1e-6) and p-MG throughout, as requested.
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

RE, H = 389.0, 0.5
_p = S.pcg_solve
CAP, WALL = 60, 900.0
SHIFT = 5.0


def build():
    m, _, _ = load('/Users/danielchan/Dropbox/F90_SEM/pmg_clean/cnos_short_grid.dat')
    n = m.N+1
    pin = next((e, n-1, 0) for e in range(m.nelem) if m.bc[e, 1] == 4 and m.bc[e, 2] == 1)
    for e in range(m.nelem):
        if m.bc[e, 1] == 4:
            m.bc[e, 1] = 0
    return m, n, pin


def merit(st, U):
    f = np.ascontiguousarray(U[..., 0]/2.0); g = np.ascontiguousarray(U[..., 1]/2.0)
    st.update_linearisation(f, g)
    r = apply_L(st, U, f, g)
    return float(np.sum(r*r/st.mesh.wq[..., None]))


def diag(U, m, pin):
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
                psp=float(pe.max()-pe.min()), dp=float(pi.mean()-pe.mean()),
                rev=float(100*np.mean(ue < 0)),
                pmean=float((U[..., 2]*m.wq).sum()/m.wq.sum()),
                ppin=float(U[pin[0], pin[1], pin[2], 2]))


def run(wmom, U0, use_pin, cap=CAP):
    m, n, pin = build(); N = m.N
    st = SolverState(m, diff_matrix(N), nu=1.0/RE, dt=0.5, fac1=1.0,
                     w_mom=wmom, w_mass=0.0)
    inlet = lambda x, y, t: 6.0*((y-0.5)/0.5)*(1.0-(y-0.5)/0.5)
    nit = [0]; pp = pin if use_pin else None

    def pcg(state, b, fu, fv, M, mw, pin_p=False, max_iter=5000, tol=None,
            cgsfac=None, precond=None, **kw):
        pre = P.make('pmg2', state, fu, fv, M, pin_p,
                     pc=max(2, N//2), deg=4, coarse_deg=10)
        x, it = _p(state, b, fu, fv, M, mw, pin_p=pin_p, max_iter=300000,
                   tol=1e-6, cgsfac=1e-3, precond=pre)
        nit[0] += it; return x, it
    S.pcg_solve = pcg

    U = U0.copy(); hist = [U]; t0 = time.perf_counter(); status = 'cap'; s = 0
    trace = []
    try:
        for s in range(cap):
            Up = hist[0].copy()
            U = S.step_bdf(st, hist, time=0.0, max_newton=1, newton_tol=1e-14,
                           newton_factor=0.0, custom_inlet=inlet, pin_p=pp,
                           cgsfac=1e-3, cg_max_iter=300000, verbose=False)
            dU = np.max(np.abs(U-Up)); um = np.abs(U[..., 0]).max()
            pm = float((U[..., 2]*m.wq).sum()/m.wq.sum())
            if s < 6 or s % 10 == 0:
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
                J=merit(st, U) if ok else np.nan,
                d=diag(U, m, pin) if ok else None, trace=trace, U=U, m=m, pin=pin)


KEYS = (('q', 'Qout/Qin'), ('div', 'rms div'), ('umax', 'max|u|'), ('xr', 'x_r/h'),
        ('psp', 'outlet p spread'), ('dp', 'dp inlet->outlet'), ('rev', 'exit rev %'),
        ('pmean', 'MEAN p  (level)'), ('ppin', 'p at pin node'))

m0, _, pin0 = build()
U_conv = np.load(f'{SC}/bfswms_0.1.npz')['U']

print("=" * 96)
print("TEST 1 -- is the constant-pressure mode an exact null mode?")
print("=" * 96)
st0 = SolverState(m0, diff_matrix(m0.N), nu=1.0/RE, dt=0.5, fac1=1.0,
                  w_mom=0.1, w_mass=0.0)
U_shift = U_conv.copy(); U_shift[..., 2] += SHIFT
J_a, J_b = merit(st0, U_conv), merit(st0, U_shift)
print(f"  J(converged)            = {J_a:.10e}")
print(f"  J(converged, p += {SHIFT:g})  = {J_b:.10e}")
print(f"  relative change         = {abs(J_b-J_a)/J_a:.3e}"
      f"    -> {'NULL MODE CONFIRMED' if abs(J_b-J_a)/J_a < 1e-10 else 'NOT a null mode'}")

print(f"\n  now iterate from the SHIFTED field (p += {SHIFT:g}), loose solve:\n")
print(f"  {'':<20}{'status':>10}{'it':>5}{'CG':>9}{'J':>13}"
      + "".join(f"{nm:>18}" for _, nm in KEYS[-2:]))
for tag, up in (('WITH pin', True), ('NO pin', False)):
    r = run(0.1, U_shift, up)
    d = r['d']
    print(f"  {tag:<20}{r['status']:>10}{r['it']:>5}{r['cg']:>9}{r['J']:>13.4e}"
          + "".join(f"{d[k]:>18.6g}" for k, _ in KEYS[-2:]))
    print(f"  {'':<20}trace: " + ", ".join(f"({a},{b:.1e},{c:.2f},p={p:+.3f})"
                                           for a, b, c, p in r['trace'][:6]))
    np.savez_compressed(f"{SC}/bfsnp2_shift_{'pin' if up else 'nopin'}.npz",
                        U=r['U'], xnod=r['m'].xnod, ynod=r['m'].ynod, hy=r['m'].hy)
    sys.stdout.flush()
print(f"\n  reference: the pinned converged field has mean p = "
      f"{diag(U_conv, m0, pin0)['pmean']:.6g}, p at pin node = 0")

print("\n" + "=" * 96)
print("TEST 2 -- with the pin removed, does Newton reach a different answer?")
print("        start = w_mom 0.5 field, solve at w_mom 0.1 (genuinely off-minimiser)")
print("=" * 96)
U_off = np.load(f'{SC}/bfswms_0.5.npz')['U']
out = {}
print(f"  {'':<12}{'status':>10}{'it':>5}{'CG':>9}{'wall':>7}{'J':>12}"
      + "".join(f"{nm:>17}" for _, nm in KEYS[:6]))
for tag, up in (('WITH pin', True), ('NO pin', False)):
    r = run(0.1, U_off, up); out[tag] = r; d = r['d']
    print(f"  {tag:<12}{r['status']:>10}{r['it']:>5}{r['cg']:>9}{r['wall']:>7.0f}"
          f"{r['J']:>12.4e}" + "".join(f"{d[k]:>17.6g}" for k, _ in KEYS[:6]))
    print(f"  {'':<12}level: mean p = {d['pmean']:+.6g}   p at pin node = {d['ppin']:+.6g}")
    np.savez_compressed(f"{SC}/bfsnp2_off_{'pin' if up else 'nopin'}.npz",
                        U=r['U'], xnod=r['m'].xnod, ynod=r['m'].ynod, hy=r['m'].hy)
    sys.stdout.flush()

a, b = out['WITH pin']['U'], out['NO pin']['U']
if np.all(np.isfinite(b)):
    dpf = b[..., 2] - a[..., 2]
    print("\n  difference, no-pin minus pinned:")
    print(f"    pressure : mean {dpf.mean():+.6e}   spread {dpf.max()-dpf.min():.6e}")
    print(f"    velocity : max|du| {np.abs(b[...,0]-a[...,0]).max():.4e}"
          f"   max|dv| {np.abs(b[...,1]-a[...,1]).max():.4e}")
    print(f"    vorticity: max|dom| {np.abs(b[...,3]-a[...,3]).max():.4e}")
    print("    a PURE null-mode difference is: pressure spread ~ 0 and velocity ~ 0,")
    print("    i.e. the two solutions differ only by an arbitrary constant in p.")
