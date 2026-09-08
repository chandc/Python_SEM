"""N-sweep on the Ghia Re=1000 cavity, time marching (BDF) from rest, dt=1e-3,
10 steps, production step_bdf: Jacobi vs condensed vertex-patch Schwarz
(+ p=2 coarse), the latter built ONCE at step 1 on a frozen snapshot and
reused for the 10 steps.  Records iterations, CG wall, build wall, stored
factor bytes and process peak RSS.

    python scratch/nsweep_ghia.py <N> <jac|cond>       one job -> npz + summary line
    python scratch/nsweep_ghia.py collect              table
"""
import os, sys, time, resource
for _v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '2')
os.environ.setdefault('CAV_RE', '1000'); os.environ.setdefault('CAV_EX', '4')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, 'scratch')); os.chdir(_R)
import numpy as np
import lssem2d; lssem2d.set_backend('numpy')
from lssem2d import solver as S
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState
from lssem2d.mesh import build_channel

RE, EX, DT, NSTEP = 1000.0, 4, 1e-3, 10
CGTOL, CGMAX = 1e-8, 60000


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6      # macOS: bytes


def run(N, kind):
    from pmg_ghia_cavity import snapshot
    from vertex_schwarz2d import VertexSchwarzCondensed2D, make_coarse
    m = build_channel(1.0, 1.0, EX, EX, N, bcs=(1, 1, 1, 2)); m.compute_global_indices()
    st = SolverState(m, diff_matrix(N), nu=1.0/RE, dt=DT, fac1=1.0)
    n = N + 1; ndof = (int(m.gidx.max())+1)*4
    rss0 = rss_mb()
    cache = {'pre': None}; build_t = []; stored = [0]; cg_it = []; cg_t = []
    def factory(s_, fu, fv, Mi, pp):
        if kind == 'jac': return None
        if cache['pre'] is None:
            t0 = time.perf_counter(); snap = snapshot(s_, fu, fv)
            fu, fv = np.ascontiguousarray(fu), np.ascontiguousarray(fv)
            pre = VertexSchwarzCondensed2D(snap, fu, fv, pin_p=pp, coarse=make_coarse(snap, fu, fv, Mi, pp))
            cache['pre'] = pre; build_t.append(time.perf_counter() - t0); stored[0] = pre.bytes
        return cache['pre']
    st.precond_factory = factory
    orig = S.pcg_solve
    def pcg_timed(*a, **k):
        t0 = time.perf_counter(); out = orig(*a, **k); cg_t.append(time.perf_counter() - t0); cg_it.append(int(out[1])); return out
    S.pcg_solve = pcg_timed
    U = np.zeros((m.nelem, n, n, 4)); h = [U]
    t0 = time.perf_counter()
    try:
        for s in range(NSTEP):
            U = S.step_bdf(st, h, time=(s+1)*DT, max_newton=1, newton_tol=1e-12, newton_factor=0.0,
                           pin_p=True, cgsfac=0.0, cg_tol=CGTOL, cg_max_iter=CGMAX)
    finally:
        S.pcg_solve = orig
    wall = time.perf_counter() - t0
    out = dict(N=N, kind=kind, ndof=ndof, cg_it=np.array(cg_it), cg_t=np.array(cg_t), build_t=np.array(build_t),
               wall=wall, stored_MB=stored[0]/1e6, rss0_MB=rss0, rss_MB=rss_mb(), U=U, maxu=float(np.abs(U[..., 0]).max()))
    np.savez_compressed(f'scratch/nsweep_ghia_{kind}_N{N}.npz', **out)
    print(f'DONE N={N:2d} {kind:4s} ndof={ndof:6d} CG it/step mean {np.mean(cg_it):7.0f} max {np.max(cg_it):6d} | CG {np.sum(cg_t):7.1f}s build {np.sum(build_t):6.1f}s wall {wall:7.1f}s | '
          f'{1e3*np.sum(cg_t)/np.sum(cg_it):6.2f} ms/it | stored {stored[0]/1e6:7.1f} MB  peak RSS {rss_mb():7.0f} MB', flush=True)


def collect():
    import glob
    rows = {}
    for f in sorted(glob.glob('scratch/nsweep_ghia_*_N*.npz')):
        z = np.load(f, allow_pickle=True); rows[(int(z['N']), str(z['kind']))] = z
    print(f'Ghia Re=1000 cavity, 4x4 elements, dt=1e-3, 10 BDF steps from rest, CG abs tol 1e-8, 2 threads/job, numpy\n')
    print(f'{"N":>3} {"ndof":>6} | {"Jacobi it/step":>14} {"ms/it":>6} {"CG s":>7} {"wall s":>7} {"RSS MB":>7} | {"condensed it/step":>17} {"ms/it":>6} {"CG s":>7} {"build s":>7} {"wall s":>7} {"stored MB":>9} {"RSS MB":>7} | {"it ratio":>8} {"wall ratio":>10} {"|dU| jac-cond":>13}')
    for N in sorted({k[0] for k in rows}):
        j, c = rows.get((N, 'jac')), rows.get((N, 'cond'))
        if j is None or c is None: continue
        du = np.abs(j['U'] - c['U']).max()
        print(f'{N:3d} {int(j["ndof"]):6d} | {np.mean(j["cg_it"]):14.0f} {1e3*np.sum(j["cg_t"])/np.sum(j["cg_it"]):6.2f} {np.sum(j["cg_t"]):7.1f} {float(j["wall"]):7.1f} {float(j["rss_MB"]):7.0f} | '
              f'{np.mean(c["cg_it"]):17.0f} {1e3*np.sum(c["cg_t"])/np.sum(c["cg_it"]):6.1f} {np.sum(c["cg_t"]):7.1f} {np.sum(c["build_t"]):7.1f} {float(c["wall"]):7.1f} {float(c["stored_MB"]):9.1f} {float(c["rss_MB"]):7.0f} | '
              f'{np.mean(j["cg_it"])/np.mean(c["cg_it"]):7.1f}x {float(j["wall"])/float(c["wall"]):9.2f}x {du:13.1e}')


if __name__ == '__main__':
    if sys.argv[1:] == ['collect']: collect()
    else: run(int(sys.argv[1]), sys.argv[2])
