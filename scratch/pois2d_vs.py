"""2D periodic channel, body-force driven Poiseuille start-up at Re=100, time
marching (BDF1 then BDF2) from rest: Jacobi vs PMG2 ladder vs vertex-patch
Schwarz (+p=2 coarse) as the CG preconditioner inside the PRODUCTION
step_bdf/newton_step path (hook: state.precond_factory).

Exact transient (u_t = nu u_yy + G, u=0 at y=0,1, rest at t=0, G = 12 nu):
    u(y,t) = 6 y (1-y) - sum_{n odd} 48/(n pi)^3 sin(n pi y) exp(-nu n^2 pi^2 t)
so the centreline history and the final profile are checked against it, not
just against the steady parabola.  Records per step: CG iterations, CG wall,
preconditioner build wall.  Saves everything to scratch/pois2d_vs_<case>.npz.
"""
import os, sys, time
for _v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '8')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, 'scratch')); os.chdir(_R)
import numpy as np
import lssem2d
lssem2d.set_backend('numpy')
from lssem2d import precond as P, solver as S
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState, ls_coeffs
from lssem2d.mesh import build_channel
from vertex_schwarz2d import VertexSchwarz2D, make_coarse

LX, LY, RE, N, EX, EY = 2.0, 1.0, 100.0, 8, 4, 4
NU = 1.0*LY/RE; G = 12.0*NU                      # u_mean = 1, u_max = 1.5
CGSFAC, CGTOL, CGMAX = 1e-8, 1e-10, 60000
REFRESH = 1
ROUGH = 0.0


def u_exact(y, t, nterm=200):
    u = 6.0*y*(1.0-y)
    for n in range(1, 2*nterm, 2):
        u = u - 48.0/(n*np.pi)**3*np.sin(n*np.pi*y)*np.exp(-NU*(n*np.pi)**2*t)
    return u


def ladder(N):
    seq, p = [N], N
    while p > 2:
        p = max(2, p//2); seq.append(p)
    return tuple(seq[1:])


def run(kind, dt, T_end, log_every, refresh=1):
    m = build_channel(LX, LY, EX, EY, N, bcs=(0, 0, 1, 1))
    m.periodic_x = LX; m.compute_global_indices()
    st = SolverState(m, diff_matrix(N), nu=NU, dt=dt, fac1=1.0)
    n = N + 1
    pin = (0, 0, 0)                                          # pressure pinned at one wall corner
    # body force in the x-momentum row: residual row 0 is a_mass*u + a_flux*(... ), so f = a_flux*G
    _, a_flux, _ = ls_coeffs(st)
    fk = np.zeros((m.nelem, n, n, 4)); fk[..., 0] = a_flux*G
    # preconditioner factory (called by newton_step at the CG's linearisation)
    build_t = []
    def timed_factory(f):
        # `refresh`: rebuild the preconditioner every `refresh` calls (= steps at
        # max_newton=1) and reuse it in between -- a slightly stale SPD
        # preconditioner is still a valid preconditioner, and in 3D the operator
        # never changes at all.
        cache = {'n': 0, 'pre': None}
        def g(*a, **k):
            if cache['pre'] is None or cache['n'] % refresh == 0:
                t0 = time.time(); cache['pre'] = f(*a, **k); build_t.append(time.time() - t0)
            cache['n'] += 1
            return cache['pre']
        return g
    if kind == 'jacobi':
        st.precond_factory = None
    elif kind == 'pmg2':
        st.precond_factory = timed_factory(lambda s_, fu, fv, Mi, pp: P.make(
            'pmg2', s_, fu, fv, Mi, pp, pc=ladder(N), deg=6, coarse_solver='direct'))
    elif kind == 'vschwarz':
        st.precond_factory = timed_factory(lambda s_, fu, fv, Mi, pp: VertexSchwarz2D(
            s_, fu, fv, pin_p=pp, coarse=make_coarse(s_, fu, fv, Mi, pp)))
    # time the CG itself by wrapping pcg_solve (newton_step looks it up in S)
    cg_it, cg_t = [], []
    orig = S.pcg_solve
    def pcg_timed(*a, **k):
        t0 = time.time(); out = orig(*a, **k); cg_t.append(time.time() - t0); cg_it.append(int(out[1])); return out
    S.pcg_solve = pcg_timed
    U = np.zeros((m.nelem, n, n, 4)); hist = [U]
    if ROUGH > 0:
        # parabola + random C0 velocity noise: a rough state every step, which is
        # what the DNS hands the solver and what the smooth start-up does not.
        from lssem2d.solver import gather_scatter
        rng = np.random.default_rng(7)
        U[..., 0] = 6.0*m.ynod[:, None, :]*(1.0 - m.ynod[:, None, :])
        noise = rng.standard_normal((m.nelem, n, n, 4)); noise[..., 2:] = 0.0
        mult = gather_scatter(m, np.ones_like(noise)); noise = gather_scatter(m, noise)/np.where(mult < 1e-10, 1.0, mult)
        wall = (np.abs(m.ynod) < 1e-12) | (np.abs(m.ynod - 1.0) < 1e-12)
        noise[:, :, :, :2] *= (~wall)[:, None, :, None]
        U += ROUGH*noise
    nstep = int(round(T_end/dt)); ts, uc, err = [], [], []
    # centreline sample: the node nearest y = 0.5
    ycl = np.abs(m.ynod - 0.5); e_cl, j_cl = np.unravel_index(np.argmin(ycl), ycl.shape)
    t_wall0 = time.time()
    for s in range(nstep):
        t = (s+1)*dt
        U = S.step_bdf(st, hist, time=t, max_newton=1, newton_tol=1e-12, newton_factor=0.0,
                       f_known=fk, pin_p=pin, cgsfac=CGSFAC, cg_tol=CGTOL, cg_max_iter=CGMAX)
        # step_bdf updates `hist` in place (insert new state, keep 2); do NOT
        # reassign it -- doing so made hist = [U, U] and BDF2 ran at 2/3 speed.
        if (s+1) % log_every == 0 or s == nstep-1:
            ue = u_exact(m.ynod[:, None, :], t)                # (nelem,1,n) -> broadcast over x
            e = np.sqrt((m.wq*(U[..., 0] - ue)**2).sum()/(m.wq*ue**2).sum())
            ts.append(t); uc.append(float(U[e_cl, 0, j_cl, 0])); err.append(float(e))
            print(f'  {kind:8s} step {s+1:5d} t={t:7.3f} u_c={uc[-1]:.5f} exact {u_exact(0.5, t):.5f} '
                  f'rel L2 err {e:.2e} | CG it {cg_it[-1]:5d} (mean {np.mean(cg_it):.0f}) '
                  f'cg {np.sum(cg_t):.1f}s build {np.sum(build_t):.1f}s wall {time.time()-t_wall0:.1f}s', flush=True)
    S.pcg_solve = orig
    wall = time.time() - t_wall0
    # final profile against the exact transient at T_end and the steady parabola
    ue = u_exact(m.ynod[:, None, :], T_end); us = 6.0*m.ynod[:, None, :]*(1.0 - m.ynod[:, None, :])
    ef = np.sqrt((m.wq*(U[..., 0]-ue)**2).sum()/(m.wq*ue**2).sum())
    es = np.sqrt((m.wq*(U[..., 0]-us)**2).sum()/(m.wq*us**2).sum())
    out = dict(kind=kind, dt=dt, T_end=T_end, nstep=nstep, cg_it=np.array(cg_it), cg_t=np.array(cg_t),
               build_t=np.array(build_t), wall=wall, ts=np.array(ts), uc=np.array(uc), err=np.array(err),
               err_final_transient=ef, err_final_parabola=es, U=U, y=m.ynod, x=m.xnod, wq=m.wq)
    np.savez_compressed(f'scratch/pois2d_vs_{kind}_dt{dt:g}{"_rough" if ROUGH > 0 else ""}.npz', **out)
    return out


def main():
    # CLI: dt T_end log_every [refresh]; default is the H1-regime case
    a = sys.argv[1:]
    cases = [(float(a[0]), float(a[1]), int(a[2]))] if len(a) >= 3 else [(0.1, 60.0, 100)]
    global REFRESH, ROUGH; REFRESH = int(a[3]) if len(a) >= 4 else 1; ROUGH = float(a[4]) if len(a) >= 5 else 0.0
    for dt, T_end, le in cases:
        print(f'\n=== Poiseuille start-up, Re={RE:.0f}, {EX}x{EY} elements N={N}, periodic x, dt={dt}, T={T_end} ({int(round(T_end/dt))} steps), c=1/dt={1/dt:.0f} (c*~nu p^4/h^2={NU*N**4/(LY/EY)**2:.0f}), CG cgsfac={CGSFAC:g} tol={CGTOL:g}, vschwarz refresh={REFRESH}')
        res = {}
        for kind in ('vschwarz', 'pmg2', 'jacobi'):
            res[kind] = run(kind, dt, T_end, le, refresh=(REFRESH if kind == 'vschwarz' else 1))
        print(f'\n{"precond":>9} {"steps":>5} {"CG it/step (mean/max)":>22} {"CG wall":>8} {"build":>7} {"total":>7} {"err vs exact(T)":>16} {"err vs parabola":>16}')
        for k, r in res.items():
            print(f'{k:>9} {r["nstep"]:5d} {np.mean(r["cg_it"]):11.0f} / {np.max(r["cg_it"]):5d}   {np.sum(r["cg_t"]):7.1f}s {np.sum(r["build_t"]):6.1f}s {r["wall"]:6.1f}s {r["err_final_transient"]:16.2e} {r["err_final_parabola"]:16.2e}')

if __name__ == '__main__':
    main()
