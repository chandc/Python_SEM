"""BFS Chan Re=389, long domain: separate the LS momentum weight from the
   effective time step, which the original dt sweep confounded.

At nominal dt, setting weight W and effective step T:
    w_mom = W        w_mass = dt*W/T        (dt_eff = dt*w_mom/w_mass)

ROW A  dt_eff = 0.5 fixed, W varied   -> isolates the WEIGHTING effect
ROW B  W = 0.5 fixed, dt_eff varied   -> isolates the TIME-STEP effect
(W=0.5, T=0.5) is legacy dt=0.5 and appears in both rows as the shared anchor.

Exact references that need no knowledge of the true solution:
    Qout/Qin = 1,  rms div = 0,  max|u| <= 1.5 (inlet peak),  exit reversed = 0%

Guards, learned the hard way: convergence on |dU|/dt_eff (NOT /dt -- they differ
now), a hard max|u| divergence trip so a bad configuration stops in seconds
rather than grinding to the step cap, and a step cap sized from dt_eff.
"""
import os, sys, time, json
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
sys.path.insert(0, SC)
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
from fgrid import load
from lssem2d.lgl import diff_matrix, lgl_weights
from lssem2d.lssem import SolverState, ls_coeffs
from lssem2d.operators import dUdx, dUdy
import lssem2d.solver as S
from lssem2d import precond as P

RE, H, DT = 389.0, 0.5, 0.5
GRID = '/Users/danielchan/Dropbox/F90_SEM/pmg_clean/cnos_long_grid.dat'
TEFF_TARGET, STEP_CAP, UMAX_TRIP = 300.0, 1500, 10.0
RATE_TOL, LB = 1.0e-9, 1.0
_p = S.pcg_solve


def build():
    mesh, _, _ = load(GRID)
    n = mesh.N+1
    pin = next((e, n-1, 0) for e in range(mesh.nelem)
               if mesh.bc[e, 1] == 4 and mesh.bc[e, 2] == 1)
    for e in range(mesh.nelem):
        if mesh.bc[e, 1] == 4:
            mesh.bc[e, 1] = 0
    return mesh, n, pin


def devc_ic(mesh, n):
    """continuous, divergence-free developed IC (see bfs_dt.py)"""
    U = np.zeros((mesh.nelem, n, n, 4))
    ud = lambda y: 3.0*y*(1.0-y)
    dud = lambda y: 3.0-6.0*y
    def us(y):
        if y <= 0.5: return 0.0
        e = 2.0*y-1.0; return 6.0*e*(1.0-e)
    def dus(y):
        if y <= 0.5: return 0.0
        e = 2.0*y-1.0; return 12.0*(1.0-2.0*e)
    def G(y):
        I = 1.5*y*y-y**3
        if y > 0.5:
            e = 2.0*y-1.0; I -= 0.5*(3.0*e*e-2.0*e**3)
        return I
    for e in range(mesh.nelem):
        for i in range(n):
            xx = mesh.xnod[e, i]
            if xx <= 0.0:
                t = sp = spp = 0.0
            else:
                t = min(xx/LB, 1.0)
                sp = (6.0*t-6.0*t*t)/LB
                spp = (6.0-12.0*t)/LB**2
                if xx >= LB: sp = spp = 0.0
            sv = 3.0*t*t-2.0*t**3
            for j in range(n):
                yy = mesh.ynod[e, j]
                if xx < 0.0:
                    eta = (yy-0.5)/0.5
                    U[e, i, j, 0] = 6.0*eta*(1.0-eta)
                    U[e, i, j, 3] = -12.0*(1.0-2.0*eta)
                else:
                    U[e, i, j, 0] = (1.0-sv)*us(yy)+sv*ud(yy)
                    U[e, i, j, 1] = -sp*G(yy)
                    U[e, i, j, 3] = -spp*G(yy)-((1.0-sv)*dus(yy)+sv*dud(yy))
    return U


def run(W, T, tag):
    mesh, n, pin = build()
    N = mesh.N
    w_mom, w_mass = W, DT*W/T
    st = SolverState(mesh, diff_matrix(N), nu=1.0/RE, dt=DT, fac1=1.0,
                     w_mom=w_mom, w_mass=w_mass)
    inlet = lambda x, y, t: 6.0*((y-0.5)/0.5)*(1.0-(y-0.5)/0.5)
    D = diff_matrix(N); w = lgl_weights(N)
    INL = [e for e in range(mesh.nelem) if mesh.bc[e, 0] == 3]
    OUT = [e for e in range(mesh.nelem) if abs(mesh.xnod[e, -1]-mesh.xnod.max()) < 1e-9]
    fl = lambda U, e, i: np.sum(w*U[e, i, :, 0])*(mesh.hy[e]/2)
    nit = [0]

    def pcg(state, b, fu, fv, M, mw, pin_p=False, max_iter=5000, tol=1e-6,
            cgsfac=0.0, precond=None):
        pre = P.make('pmg2', state, fu, fv, M, pin_p, pc=max(2, N//2), deg=4,
                     coarse_deg=10)
        x, it = _p(state, b, fu, fv, M, mw, pin_p=pin_p, max_iter=40000,
                   tol=1e-6, cgsfac=1e-3, precond=pre)
        nit[0] += it
        return x, it

    S.pcg_solve = pcg
    U = devc_ic(mesh, n); hist = [U]
    nstep = min(int(np.ceil(TEFF_TARGET/T)), STEP_CAP)
    t0 = time.perf_counter(); status = 'cap'
    try:
        for s in range(nstep):
            Up = hist[0].copy()
            U = S.step_bdf(st, hist, time=s*DT, max_newton=1, newton_tol=1e-10,
                           newton_factor=0.0, custom_inlet=inlet, pin_p=pin,
                           cgsfac=1e-3, cg_max_iter=40000, verbose=False)
            if not np.all(np.isfinite(U)):
                status = 'NaN'; break
            um = np.abs(U[..., 0]).max()
            if um > UMAX_TRIP:
                status = f'diverged (max|u|={um:.1f})'; break
            if s > 5 and np.max(np.abs(U-Up))/T < RATE_TOL:
                status = 'converged'; break
    finally:
        S.pcg_solve = _p
    wall = time.perf_counter()-t0
    if status in ('NaN',) or status.startswith('diverged'):
        return dict(W=W, T=T, status=status, steps=s+1, wall=wall, cg=nit[0])
    ux = dUdx(np.ascontiguousarray(U[..., 0]), D, mesh.facx)
    vy = dUdy(np.ascontiguousarray(U[..., 1]), D, mesh.facy)
    xs, tw = [], []
    for e in range(mesh.nelem):
        if mesh.ynod[e, 0] > 0.01 or mesh.xnod[e, 0] < -1e-9: continue
        for i in range(n):
            xs.append(mesh.xnod[e, i])
            tw.append(np.dot(D[0, :], U[e, i, :, 0])*(2.0/mesh.hy[e]))
    o = np.argsort(xs); xs, tw = np.array(xs)[o], np.array(tw)[o]
    xr = np.nan
    for k in range(len(xs)-1):
        if tw[k] < 0 and tw[k+1] > 0 and xs[k] > 0.05:
            xr = xs[k]-tw[k]*(xs[k+1]-xs[k])/(tw[k+1]-tw[k]); break
    ue = np.array([U[e, -1, j, 0] for e in OUT for j in range(n)])
    pe = np.array([U[e, -1, j, 2] for e in OUT for j in range(n)])
    np.savez_compressed(f'{SC}/bfsw_{tag}.npz', U=U, W=W, T=T,
                        xnod=mesh.xnod, ynod=mesh.ynod, hy=mesh.hy)
    return dict(W=W, T=T, status=status, steps=s+1, wall=wall, cg=nit[0],
                coef=ls_coeffs(st),
                q=float(sum(fl(U, e, -1) for e in OUT)/sum(fl(U, e, 0) for e in INL)),
                div=float(np.sqrt((( ux+vy)**2).mean())),
                umax=float(np.abs(U[..., 0]).max()), xr=float(xr/H),
                pspread=float(pe.max()-pe.min()), rev=float(100*np.mean(ue < 0)))


def show(r):
    if r['status'] != 'converged':
        print(f"{r['W']:>6}{r['T']:>7}{'':>9}{'':>8}{r['steps']:>7}"
              f"{r['status']:>28}{r['wall']:>8.0f}")
        return
    c = r['coef']
    print(f"{r['W']:>6}{r['T']:>7}{c[0]:>9.3f}{c[1]:>8.3f}{r['steps']:>7}"
          f"{r['q']:>10.4f}{r['div']:>11.3e}{r['umax']:>8.3f}{r['xr']:>8.3f}"
          f"{r['pspread']:>9.3f}{r['rev']:>7.1f}%{r['wall']:>8.0f}")


out = []
print(f"BFS Chan Re={RE:g}, LONG domain, devc IC, SE outlet pin, p-MG, nominal dt={DT}")
print(f"exact refs: Qout/Qin = 1, rms div = 0, max|u| <= 1.5, exit reversed = 0%\n")
hdr = (f"{'W':>6}{'dt_eff':>7}{'a_mass':>9}{'a_flux':>8}{'steps':>7}{'Qout/Qin':>10}"
       f"{'rms div':>11}{'max|u|':>8}{'x_r/h':>8}{'p_sprd':>9}{'rev':>8}{'wall':>8}")
print("=== ROW A: dt_eff = 0.5 fixed, weight varied  (isolates WEIGHTING) ===")
print(hdr)
for W in (0.25, 0.5, 1.0, 2.0):
    r = run(W, 0.5, f'A_W{W:g}'); out.append(r); show(r)
    json.dump(out, open(f'{SC}/bfs_wsweep.json', 'w'), indent=1, default=str)

print(f"\n=== ROW B: weight = 0.5 fixed, dt_eff varied  (isolates TIME STEP) ===")
print(hdr)
for T in (0.25, 0.5, 1.0, 2.0):
    r = run(0.5, T, f'B_T{T:g}'); out.append(r); show(r)
    json.dump(out, open(f'{SC}/bfs_wsweep.json', 'w'), indent=1, default=str)
print(f"\nsaved {SC}/bfs_wsweep.json")
