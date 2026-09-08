"""Lid-driven cavity, Re = 32000, N = 30, 4x4 elements, TIME MARCHING (BDF2)
from rest, dt = 2e-3 (c = 1/dt = 500 > c* ~ nu p^4/h^2 ~ 400): the H(div)
regime at high order.  Preconditioners inside the production step_bdf path:

    jac     Jacobi (rebuilt every step, free)
    lad     PMG2 ladder 30->15->8->4->2, Chebyshev deg 4, direct p=2
    pmgv8   same ladder, vertex-patch smoother on the p<=8 levels
    pmgv15  same ladder, vertex-patch smoother on the p<=15 levels
    vs30    fine-level vertex patches (14884 dofs each) + p=2 coarse, no ladder

The PMG preconditioners are built on a FROZEN snapshot of the linearisation
and refreshed every REFRESH steps (pmg_ghia_cavity.snapshot).  Recorded per
step: CG iterations, CG wall, build wall.  All methods must agree on U.
No Ghia data exist at Re = 32000 (their tables stop at 10000); the answer
check here is cross-method agreement, and the profiles are saved.
"""
import os, sys, time
for _v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '8')
os.environ.setdefault('CAV_RE', '32000'); os.environ.setdefault('CAV_EX', '4')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, 'scratch')); os.chdir(_R)
import numpy as np
import lssem2d; lssem2d.set_backend('numpy')
from lssem2d import precond as P, solver as S
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState
from lssem2d.mesh import build_channel
from pmg_ghia_cavity import snapshot, centreline_u, centreline_v, ladder
from pmgv2d import PMGV
from vertex_schwarz2d import VertexSchwarz2D, make_coarse

RE = float(os.environ['CAV_RE']); EX = int(os.environ['CAV_EX'])
N = int(os.environ.get('CAV_N', 30)); DT = float(os.environ.get('CAV_DT', 2e-3))
MAXSTEP = int(os.environ.get('CAV_MAXSTEP', 40)); REFRESH = int(os.environ.get('CAV_REFRESH', 10))
CGTOL, CGMAX = 1e-8, 30000


def run(kind):
    m = build_channel(1.0, 1.0, EX, EX, N, bcs=(1, 1, 1, 2)); m.compute_global_indices()
    st = SolverState(m, diff_matrix(N), nu=1.0/RE, dt=DT, fac1=1.0)
    n = N + 1
    cache = {'n': 0, 'pre': None}; build_t = []; cg_it = []; cg_t = []
    def factory(s_, fu, fv, Mi, pp):
        if kind == 'jac': return None
        if cache['pre'] is None or cache['n'] % REFRESH == 0:
            t0 = time.perf_counter(); snap = snapshot(s_, fu, fv)
            fu, fv = np.ascontiguousarray(fu), np.ascontiguousarray(fv)
            if kind == 'lad':
                cache['pre'] = P.make('pmg2', snap, fu, fv, Mi, pp, pc=ladder(N), deg=4, coarse_solver='direct')
            elif kind == 'vs30':
                # FINE-level vertex patches at N=30: (2N+1)^2*4 = 14884 dofs per
                # patch, 1.77 GB each in float64, 25 patches = 44 GB -- fits this
                # machine's 137 GB.  This is the sec 10 configuration at high order.
                cache['pre'] = VertexSchwarz2D(snap, fu, fv, pin_p=pp, coarse=make_coarse(snap, fu, fv, Mi, pp), verbose=True)
            else:
                cache['pre'] = PMGV(snap, fu, fv, Mi, pp, pc=ladder(N), deg=4, p_patch=int(kind[4:]), coarse_solver='direct')
                if cache['n'] == 0: print(cache['pre'].describe(), flush=True)
            build_t.append(time.perf_counter() - t0)
        cache['n'] += 1
        return cache['pre']
    st.precond_factory = factory
    orig = S.pcg_solve
    def pcg_timed(*a, **k):
        t0 = time.perf_counter(); out = orig(*a, **k); cg_t.append(time.perf_counter() - t0); cg_it.append(int(out[1])); return out
    S.pcg_solve = pcg_timed
    U = np.zeros((m.nelem, n, n, 4)); h = [U]
    t0 = time.perf_counter()
    try:
        for s in range(MAXSTEP):
            U = S.step_bdf(st, h, time=(s+1)*DT, max_newton=1, newton_tol=1e-12, newton_factor=0.0,
                           pin_p=True, cgsfac=0.0, cg_tol=CGTOL, cg_max_iter=CGMAX)
            if not np.all(np.isfinite(U)): print('  NaN at step', s); break
            if (s+1) % 5 == 0 or s == 0:
                print(f'  {kind:7s} step {s+1:4d} t={(s+1)*DT:.4f} max|u|={np.abs(U[...,0]).max():.4f} | CG it {cg_it[-1]:5d} (mean {np.mean(cg_it):.0f}) cg {np.sum(cg_t):.1f}s build {np.sum(build_t):.1f}s wall {time.perf_counter()-t0:.1f}s', flush=True)
    finally:
        S.pcg_solve = orig
    wall = time.perf_counter() - t0
    y, u = centreline_u(U, m, N); x, v = centreline_v(U, m, N)
    out = dict(kind=kind, U=U, cg_it=np.array(cg_it), cg_t=np.array(cg_t), build_t=np.array(build_t), wall=wall,
               y=y, u=u, x=x, v=v, N=N, RE=RE, DT=DT, steps=len(cg_it))
    np.savez_compressed(f'scratch/cavity32k_{kind}_N{N}.npz', **out)
    return out


def main():
    kinds = sys.argv[1:] or ['pmgv8', 'pmgv15', 'lad', 'jac']
    print(f'cavity Re={RE:.0f}, {EX}x{EX} elements N={N}, dt={DT} (c={1/DT:.0f}, c*~{N**4/RE/(1/EX)**2:.0f}), {MAXSTEP} steps BDF, CG abs tol {CGTOL:g}, PMG refresh {REFRESH}, ladder {ladder(N)}')
    res = {}
    for k in kinds:
        res[k] = run(k)
    print(f'\n{"precond":>8} {"steps":>5} {"CG it/step mean/max":>20} {"CG wall":>8} {"build":>7} {"total":>7} {"s/step":>7}')
    for k, r in res.items():
        print(f'{k:>8} {r["steps"]:5d} {np.mean(r["cg_it"]):10.0f} / {np.max(r["cg_it"]):5d}   {np.sum(r["cg_t"]):7.1f}s {np.sum(r["build_t"]):6.1f}s {r["wall"]:6.1f}s {r["wall"]/r["steps"]:6.2f}s')
    ks = list(res)
    for k in ks[1:]:
        print(f'max|U_{k} - U_{ks[0]}| = {np.abs(res[k]["U"] - res[ks[0]]["U"]).max():.2e}')

if __name__ == '__main__':
    main()
