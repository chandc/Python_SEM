"""Chan (1996) Kovasznay replication: accuracy, timings, Mflops.

Chan, Tables 1 and 2:

  Table 1 -- h-refinement, polynomial order fixed at 2
    Nx x Ny     eps_u      eps_v      eps_p   steps  time(s)  Mflops
    15 x 10   5.49e-2    8.34e-3     0.25       18     31.6    52.4
    30 x 20   1.07e-2    1.77e-3    7.29e-2     19      258    59.7
    60 x 40   1.56e-3    2.69e-4    1.66e-2     19     1916    60.5

  Table 2 -- p-refinement, 8 elements (4 streamwise x 2 vertical)
    N         eps_u      eps_v      eps_p    steps  time(s)  Mflops
    4       6.44e-2    1.31e-2     0.211      19      8.7     25.7
    9       1.56e-6    3.58e-7    3.76e-6     19     30.3     59.7
    14      9.22e-13   4.72e-13   1.47e-11    26      353       96

Setup as stated in the paper:
  u = 1 - e^{lam x} cos(2 pi y),  v = lam e^{lam x} sin(2 pi y)/(2 pi),
  p = (1 - e^{2 lam x})/2,        lam = Re/2 - sqrt(Re^2/4 + 4 pi^2),  Re = 40
  domain [-0.5, 1.0] x [-0.5, 0.5]
  dt and dtau both 1e30  ->  pure steady Newton  ->  w_mass = 0, w_mom = 1
  converged once the residual drops below 1e-10 (1e-13 for N = 14).  NOTE: this
  must mean the ITERATIVE residual, not the PDE residual -- LSSEM has J_min > 0,
  so the PDE residual floors at the discretisation error (6.7e-2 at N=4) and can
  never reach 1e-10.  We converge on the Newton update max|dU| instead and
  report the PDE residual separately.
  CG with a JACOBI preconditioner (so: no p-MG here, matching the paper)
  eps is the r.m.s. error.

eps is reported over UNIQUE global nodes -- element-local arrays duplicate the
shared interfaces, and counting them twice weights interior seams double.  The
duplicated-node value is printed alongside so the convention is visible.
"""
import os, sys, time
import numpy as np
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix, lgl_weights
from lssem2d.lssem import SolverState, apply_L
import lssem2d.solver as S

RE = 40.0
NU = 1.0/RE
LAM = RE/2.0 - np.sqrt(RE**2/4.0 + 4.0*np.pi**2)
LX, LY = 1.5, 1.0


def exact(x, y, t=0.0):
    e = np.exp(LAM*x)
    u = 1.0 - e*np.cos(2*np.pi*y)
    v = LAM*e*np.sin(2*np.pi*y)/(2*np.pi)
    p = (1.0 - np.exp(2*LAM*x))/2.0
    om = e*np.sin(2*np.pi*y)*(LAM**2/(2*np.pi) - 2*np.pi)
    return u, v, p, om


def build(nex, ney, N):
    m = build_channel(LX, LY, nex, ney, N, bcs=(1, 1, 1, 1))
    m.x0 -= 0.5
    m.y0 -= 0.5
    m.setup_derived()
    return m


def fields(m, N):
    n = N+1
    X = np.zeros((m.nelem, n, n)); Y = np.zeros_like(X)
    for e in range(m.nelem):
        X[e], Y[e] = np.meshgrid(m.xnod[e, :], m.ynod[e, :], indexing='ij')
    ue, ve, pe, oe = exact(X, Y)
    E = np.zeros((m.nelem, n, n, 4))
    E[..., 0], E[..., 1], E[..., 2], E[..., 3] = ue, ve, pe, oe
    return E, X, Y


def uniq_mask(X, Y):
    """Boolean mask selecting one copy of each duplicated global node."""
    key = np.stack([np.round(X.ravel(), 10), np.round(Y.ravel(), 10)], 1)
    _, idx = np.unique(key, axis=0, return_index=True)
    mk = np.zeros(X.size, bool); mk[idx] = True
    return mk.reshape(X.shape)


def residual(st, U):
    """rms of the governing-equation residual (apply_L emits wq*R)."""
    f = np.ascontiguousarray(U[..., 0]/2.0); g = np.ascontiguousarray(U[..., 1]/2.0)
    st.update_linearisation(f, g)
    r = apply_L(st, U, f, g)
    R = r/st.mesh.wq[..., None]
    return float(np.sqrt(np.mean(R**2))), float(np.sum(r*r/st.mesh.wq[..., None]))


def run(nex, ney, N, tol, cap=200, cg_tol=1e-12, verbose=False):
    m = build(nex, ney, N); n = N+1
    st = SolverState(m, diff_matrix(N), nu=NU, dt=1.0, fac1=1.0,
                     w_mom=1.0, w_mass=0.0)
    E, X, Y = fields(m, N)
    mk = uniq_mask(X, Y)

    U = np.zeros((m.nelem, n, n, 4)); U[..., 0] = 1.0     # uniform free stream
    hist = [U]
    ncg = [0]
    _p = S.pcg_solve

    def pcg(state, b, fu, fv, M, mw, **kw):
        kw['tol'] = cg_tol
        kw['max_iter'] = 200000
        x, it = _p(state, b, fu, fv, M, mw, **kw)
        ncg[0] += it
        return x, it
    S.pcg_solve = pcg

    t0 = time.perf_counter(); steps = 0; res = np.nan
    dU = np.inf; hist_dU = []
    try:
        for s in range(cap):
            Up = hist[0].copy()
            U = S.step_bdf(st, hist, time=0.0, max_newton=1, newton_tol=1e-16,
                           newton_factor=0.0, exact_solution=exact, pin_p=True,
                           cgsfac=0.0, cg_max_iter=200000, cg_tol=cg_tol)
            steps = s+1
            if not np.all(np.isfinite(U)):
                return dict(status='NaN', steps=steps)
            dU = float(np.max(np.abs(U-Up)))
            res, J = residual(st, U)
            hist_dU.append(dU)
            if verbose:
                print(f"      it {steps:3d}  |dU| {dU:.3e}  res {res:.3e}  "
                      f"cg {ncg[0]}", flush=True)
            if dU < tol:
                break
    finally:
        S.pcg_solve = _p
    wall = time.perf_counter()-t0

    du = U[..., 0]-E[..., 0]; dv = U[..., 1]-E[..., 1]
    p = U[..., 2]-U[..., 2][mk].mean(); pe = E[..., 2]-E[..., 2][mk].mean()
    dp = p-pe
    rms = lambda a: float(np.sqrt(np.mean(a[mk]**2)))
    rms_dup = lambda a: float(np.sqrt(np.mean(a**2)))
    return dict(status='conv' if dU < tol else 'cap', dU=dU, steps=steps, cg=ncg[0],
                wall=wall, res=res, nelem=m.nelem, N=N, npts=int(mk.sum()),
                eu=rms(du), ev=rms(dv), ep=rms(dp),
                eu_d=rms_dup(du), ev_d=rms_dup(dv), ep_d=rms_dup(dp))


CHAN_P = {4: (6.44e-2, 1.31e-2, 0.211, 19, 8.7, 25.7),
          9: (1.56e-6, 3.58e-7, 3.76e-6, 19, 30.3, 59.7),
          14: (9.22e-13, 4.72e-13, 1.47e-11, 26, 353, 96)}
CHAN_H = {(15, 10): (5.49e-2, 8.34e-3, 0.25, 18, 31.6, 52.4),
          (30, 20): (1.07e-2, 1.77e-3, 7.29e-2, 19, 258, 59.7),
          (60, 40): (1.56e-3, 2.69e-4, 1.66e-2, 19, 1916, 60.5)}

if __name__ == '__main__':
    mode = os.environ.get('MODE', 'p')
    print(f"Kovasznay Re = {RE}, lambda = {LAM:.10f}, domain "
          f"[-0.5, 1.0] x [-0.5, 0.5], steady (w_mass = 0), Jacobi-CG\n")
    hdr = (f"{'case':>10}{'elem':>7}{'N':>4}{'pts':>8}{'status':>7}{'steps':>7}"
           f"{'CG':>9}{'wall':>9}{'|dU|':>9}{'res':>10}"
           f"{'eps_u':>11}{'eps_v':>11}{'eps_p':>11}"
           f"{'Chan eps_u':>12}{'Chan eps_v':>12}{'Chan eps_p':>12}{'Chan s':>7}")
    print(hdr)
    if mode == 'p':
        cases = [(4, 2, N) for N in (4, 9, 14)]
    else:
        cases = [(a, b, 2) for (a, b) in ((15, 10), (30, 20), (60, 40))]
    for nex, ney, N in cases:
        tol = 1e-12
        r = run(nex, ney, N, tol, cap=60,
                cg_tol=1e-15 if N == 14 else 1e-13)
        ref = CHAN_P.get(N) if mode == 'p' else CHAN_H.get((nex, ney))
        tag = f"{nex}x{ney}" if mode != 'p' else f"N={N}"
        if r['status'] == 'NaN':
            print(f"{tag:>10}{'':>7}{N:>4}{'':>8}{'NaN':>7}{r['steps']:>7}")
            continue
        print(f"{tag:>10}{r['nelem']:>7}{N:>4}{r['npts']:>8}{r['status']:>7}"
              f"{r['steps']:>7}{r['cg']:>9}{r['wall']:>8.1f}s{r['dU']:>9.1e}{r['res']:>10.1e}"
              f"{r['eu']:>11.3e}{r['ev']:>11.3e}{r['ep']:>11.3e}"
              f"{ref[0]:>12.2e}{ref[1]:>12.2e}{ref[2]:>12.2e}{ref[3]:>7d}")
        sys.stdout.flush()
