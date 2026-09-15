"""SHORT-domain BFS: restart a CONVERGED steady solution and tighten the solve.

On the LONG domain, tightening the linear solve with p-MG DIVERGED
(max|u| 1.51 -> 40.05 in one Newton step) while the loose solve converged in 11
iterations -- solver inexactness had been acting as accidental damping.  This
asks whether the short domain does the same thing, starting from a state that is
already a converged minimiser at that w_mom, so any motion is caused by the
tolerance change alone and nothing else.

Ladder: (cgsfac, tol) = (1e-3,1e-6) control -> (1e-5,1e-8) -> (1e-8,1e-10).
The control should be a no-op; if it moves, the "converged" label was wrong.

Reports the least-squares functional J at start and end, a per-iteration trace
of (max|dU|, max|u|), and the usual exact references.
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
CAP = 30                 # Newton iterations
WALL = 900.0             # seconds per run -- the long domain burned 40 min once


def build():
    m, _, _ = load('/Users/danielchan/Dropbox/F90_SEM/pmg_clean/cnos_short_grid.dat')
    n = m.N+1
    pin = next((e, n-1, 0) for e in range(m.nelem) if m.bc[e, 1] == 4 and m.bc[e, 2] == 1)
    for e in range(m.nelem):
        if m.bc[e, 1] == 4:
            m.bc[e, 1] = 0
    return m, n, pin


def merit(st, U):
    """The least-squares functional J = int R^2 that Newton minimises."""
    f = np.ascontiguousarray(U[..., 0]/2.0)
    g = np.ascontiguousarray(U[..., 1]/2.0)
    st.update_linearisation(f, g)
    r = apply_L(st, U, f, g)
    wq = st.mesh.wq[..., None]
    return float(np.sum(r*r/wq))


def metrics(U, m):
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
    return dict(q=float(sum(fl(e, -1) for e in OUT)/sum(fl(e, 0) for e in INL)),
                div=float(np.sqrt(((ux+vy)**2).mean())),
                umax=float(np.abs(U[..., 0]).max()), xr=float(xr/H),
                psp=float(pe.max()-pe.min()), rev=float(100*np.mean(ue < 0)))


def run(wmom, U0, cgsfac, tol, line_search=False):
    m, n, pin = build(); N = m.N
    st = SolverState(m, diff_matrix(N), nu=1.0/RE, dt=0.5, fac1=1.0,
                     w_mom=wmom, w_mass=0.0)
    inlet = lambda x, y, t: 6.0*((y-0.5)/0.5)*(1.0-(y-0.5)/0.5)
    nit = [0]

    def pcg(state, b, fu, fv, M, mw, pin_p=False, max_iter=5000, tol_=None,
            cgsfac_=None, precond=None, **kw):
        pre = P.make('pmg2', state, fu, fv, M, pin_p,
                     pc=max(2, N//2), deg=4, coarse_deg=10)
        x, it = _p(state, b, fu, fv, M, mw, pin_p=pin_p, max_iter=300000,
                   tol=tol, cgsfac=cgsfac, precond=pre)
        nit[0] += it; return x, it
    S.pcg_solve = pcg

    U = U0.copy(); hist = [U]; t0 = time.perf_counter()
    J0 = merit(st, U); status = 'cap'; trace = []; s = 0
    try:
        for s in range(CAP):
            Up = hist[0].copy()
            U = S.step_bdf(st, hist, time=0.0, max_newton=1, newton_tol=1e-14,
                           newton_factor=0.0, custom_inlet=inlet, pin_p=pin,
                           cgsfac=cgsfac, cg_max_iter=300000, verbose=False,
                           line_search=line_search)
            dU = np.max(np.abs(U-Up)); um = np.abs(U[..., 0]).max()
            if s < 6 or dU > 1.0:
                trace.append((s+1, dU, um))
            if not np.all(np.isfinite(U)):
                status = 'NaN'; break
            if um > 20.0:
                status = f'DIVERGED(max|u|={um:.1f})'; break
            if s > 1 and dU < 1e-11:
                status = 'conv'; break
            if time.perf_counter()-t0 > WALL:
                status = 'WALL'; break
    finally:
        S.pcg_solve = _p
    ok = np.all(np.isfinite(U)) and np.abs(U[..., 0]).max() < 20.0
    return dict(status=status, it=s+1, cg=nit[0], wall=time.perf_counter()-t0,
                J0=J0, J1=merit(st, U) if ok else np.nan, trace=trace,
                U=U, m=m, mm=metrics(U, m) if ok else None)


print("BFS Chan Re=389, SHORT domain (L/h=5), steady form (w_mass=0), p-MG")
print("restart an ALREADY-CONVERGED field and tighten the linear solve")
print(f"caps: {CAP} Newton iterations, {WALL:.0f} s per run")
print("LONG-domain reference: loose converged in 11 it; tight p-MG DIVERGED to max|u|=40.1\n")

LADDER = [(1e-3, 1e-6, 'control (as converged)'),
          (1e-5, 1e-8, 'tighter'),
          (1e-8, 1e-10, 'tight')]

hdr = (f"{'cgsfac':>8}{'tol':>8}{'iters':>7}{'CG':>10}{'status':>22}"
       f"{'J start':>11}{'J end':>11}{'Qout/Qin':>10}{'rms div':>10}"
       f"{'max|u|':>8}{'x_r/h':>8}{'p_sprd':>8}{'rev':>7}{'wall':>7}")

for wmom, f in ((0.1, 'bfswms_0.1.npz'), (0.5, 'bfswms_0.5.npz')):
    U0 = np.load(f'{SC}/{f}')['U']
    print(f"=== w_mom = {wmom}  (converged loose solution, {f}) ===")
    print(hdr)
    for cgsfac, tol, lab in LADDER:
        r = run(wmom, U0, cgsfac, tol)
        if r['mm'] is None:
            print(f"{cgsfac:>8.0e}{tol:>8.0e}{r['it']:>7}{r['cg']:>10}{r['status']:>22}"
                  f"{r['J0']:>11.3e}{'':>11}{'':>53}{r['wall']:>7.0f}")
        else:
            mm = r['mm']
            print(f"{cgsfac:>8.0e}{tol:>8.0e}{r['it']:>7}{r['cg']:>10}{r['status']:>22}"
                  f"{r['J0']:>11.3e}{r['J1']:>11.3e}{mm['q']:>10.4f}{mm['div']:>10.2e}"
                  f"{mm['umax']:>8.3f}{mm['xr']:>8.3f}{mm['psp']:>8.3f}{mm['rev']:>6.1f}%"
                  f"{r['wall']:>7.0f}")
        if r['trace']:
            print("          trace (it, max|dU|, max|u|): " +
                  ", ".join(f"({a},{b:.2e},{c:.2f})" for a, b, c in r['trace'][:8]))
        if r['mm'] is not None and r['status'] != 'DIVERGED':
            np.savez_compressed(f"{SC}/bfsst_w{wmom:g}_t{tol:g}.npz", U=r['U'],
                                xnod=r['m'].xnod, ynod=r['m'].ynod, hy=r['m'].hy)
        sys.stdout.flush()
    print()

# If tightening breaks it, does the line search rescue the short domain?
print("=== does the line search rescue the tight solve? (w_mom = 0.1) ===")
print(hdr)
U0 = np.load(f'{SC}/bfswms_0.1.npz')['U']
r = run(0.1, U0, 1e-8, 1e-10, line_search=True)
mm = r['mm']
if mm is None:
    print(f"{1e-8:>8.0e}{1e-10:>8.0e}{r['it']:>7}{r['cg']:>10}{r['status']:>22}"
          f"{r['J0']:>11.3e}")
else:
    print(f"{1e-8:>8.0e}{1e-10:>8.0e}{r['it']:>7}{r['cg']:>10}{r['status']+' +LS':>22}"
          f"{r['J0']:>11.3e}{r['J1']:>11.3e}{mm['q']:>10.4f}{mm['div']:>10.2e}"
          f"{mm['umax']:>8.3f}{mm['xr']:>8.3f}{mm['psp']:>8.3f}{mm['rev']:>6.1f}%"
          f"{r['wall']:>7.0f}")
if r['trace']:
    print("          trace: " + ", ".join(f"({a},{b:.2e},{c:.2f})" for a, b, c in r['trace'][:8]))
