"""Ghia Re=1000 lid-driven cavity, 4x4 elements, time marching (BDF2) from
rest, dt=1e-3, restartable, with the condensed vertex-patch Schwarz
preconditioner (+ p=2 coarse) rebuilt every REFRESH steps.

    python scratch/ghia_n15_run.py --N 15 --nstep 5000 --ckpt 500 [--resume scratch/ghia_n15/checkpoint_XXXXX.npz]

Checkpoint = both BDF history levels, step, time, and the cumulative per-step
iteration/timing records, so a resumed run continues the same trajectory and
the same log.  Log lines are appended (log files are append-mode: split on
time resets when analysing).
"""
import os, sys, time, argparse
for _v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '8')
os.environ.setdefault('CAV_RE', '1000'); os.environ.setdefault('CAV_EX', '4')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, 'scratch')); os.chdir(_R)
import numpy as np
import lssem2d; lssem2d.set_backend('numpy')
from lssem2d import solver as S
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState
from lssem2d.mesh import build_channel
from pmg_ghia_cavity import snapshot, centreline_u, centreline_v, GHIA_X, GHIA_V
from vertex_schwarz2d import VertexSchwarzCondensed2D, make_coarse

RE, EX = 1000.0, 4
CGTOL, CGMAX = 1e-8, 60000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--N', type=int, default=15); ap.add_argument('--nstep', type=int, default=5000)
    ap.add_argument('--ckpt', type=int, default=500); ap.add_argument('--refresh', type=int, default=100)
    ap.add_argument('--resume', default=None); ap.add_argument('--out', default=None); ap.add_argument('--precond', default='cond')
    ap.add_argument('--dt', type=float, default=1e-3); ap.add_argument('--steady', type=float, default=0.0)   # stop when max|dU|/dt < steady
    ap.add_argument('--newton', type=int, default=1); ap.add_argument('--filter', type=float, default=0.0); ap.add_argument('--dtau_p', type=float, default=None); ap.add_argument('--cgsfac', type=float, default=0.0); ap.add_argument('--cgtol', type=float, default=1e-8)   # CG: target = max(cgsfac*|b|, cgtol)
    ap.add_argument('--w_mom', type=float, default=None); ap.add_argument('--w_mass', type=float, default=None)   # None,None = legacy a_flux=dt
    ap.add_argument('--restart', choices=('steady', 'bdf1'), default='steady')
    ap.add_argument('--lid', choices=('ghia', 'reg'), default='ghia')   # reg = regularised lid u = 16 x^2 (1-x)^2 (smooth, no corner singularity)   # history on a dt change: steady = two identical levels (field is a fixed point); bdf1 = one level, self-starting (unsteady field)
    a = ap.parse_args()
    global DT; DT = a.dt
    N = a.N; out = a.out or f'scratch/ghia_n{N}'; os.makedirs(out, exist_ok=True)
    m = build_channel(1.0, 1.0, EX, EX, N, bcs=(1, 1, 1, 2)); m.compute_global_indices()
    st = SolverState(m, diff_matrix(N), nu=1.0/RE, dt=DT, fac1=1.0, w_mom=a.w_mom, w_mass=a.w_mass)
    lid = (lambda x, y, t: 16.0*x**2*(1.0 - x)**2) if a.lid == 'reg' else None
    n = N + 1
    if a.dtau_p is not None: st.dtau_p = a.dtau_p                                   # artificial compressibility on the continuity row
    if a.filter > 0:
        from lssem2d.lgl import lgl_nodes
        def lagmat(xs, xq):
            L = np.ones((len(xq), len(xs)))
            for q, x in enumerate(xq):
                for i in range(len(xs)):
                    for j in range(len(xs)):
                        if i != j: L[q, i] *= (x - xs[j])/(xs[i] - xs[j])
            return L
        xN, xM = lgl_nodes(N), lgl_nodes(N-1)
        Pdown = lagmat(xN, xM); Pup = lagmat(xM, xN)                                 # N -> N-1 -> N projection
        F1 = (1.0 - a.filter)*np.eye(n) + a.filter*(Pup @ Pdown)
        def filt(U):
            V = np.einsum('ai,eijf->eajf', F1, U); return np.einsum('bj,eajf->eabf', F1, V)
    # ---- state: fresh or resumed ----
    if a.resume:
        z = np.load(a.resume)
        h = [z['U0']] + ([z['U1']] if z['U1'].size else [])
        step0 = int(z['step']); t = float(z['t']); t_off = t - step0*DT       # t_off != 0 when resuming with a different dt
        if abs(t_off) > 1e-12 or a.restart == 'bdf1':
            h = [h[0], h[0].copy()] if a.restart == 'steady' else h[:1]       # dt changed: 'steady' = two identical levels (stored field is a fixed point; a one-level restart there injects a 4e-3 transient, restart_probe.py); 'bdf1' = one level, self-starting, for an UNSTEADY stored field (local error O(dt^2) once, global order kept)
        cg_it = list(z['cg_it']); cg_t = list(z['cg_t']); build_t = list(z['build_t'])
        print(f'resumed from {a.resume}: step {step0}, t={t:.4f}, {len(cg_it)} steps of records'
              + (f' (dt changed from {float(z["dt"]):g} to {DT:g}: t_off={t_off:.4f}, restart={a.restart})' if abs(t_off) > 1e-12 else ''), flush=True)
    else:
        h = [np.zeros((m.nelem, n, n, 4))]; step0 = 0; t = 0.0; t_off = 0.0; cg_it, cg_t, build_t = [], [], []
    # ---- preconditioner factory: rebuild every `refresh` calls on a frozen snapshot ----
    cache = {'n': 0, 'pre': None}
    def factory(s_, fu, fv, Mi, pp):
        if a.precond == 'jac': return None
        if cache['pre'] is None or cache['n'] % a.refresh == 0:
            t0 = time.perf_counter(); snap = snapshot(s_, fu, fv); snap.dtau_p = getattr(s_, 'dtau_p', None)
            fu, fv = np.ascontiguousarray(fu), np.ascontiguousarray(fv)
            cache['pre'] = VertexSchwarzCondensed2D(snap, fu, fv, pin_p=pp, coarse=make_coarse(snap, fu, fv, Mi, pp))
            build_t.append(time.perf_counter() - t0)
        cache['n'] += 1
        return cache['pre']
    st.precond_factory = factory
    orig = S.pcg_solve
    def pcg_timed(*args, **k):
        t0 = time.perf_counter(); o = orig(*args, **k); cg_t.append(time.perf_counter() - t0); cg_it.append(int(o[1])); return o
    S.pcg_solve = pcg_timed

    def save(step, tag=None):
        f = f'{out}/checkpoint_{step:05d}.npz' if tag is None else f'{out}/{tag}.npz'
        np.savez_compressed(f, U0=h[0], U1=(h[1] if len(h) > 1 else np.zeros(0)), step=step, t=(step)*DT + t_off,
                            cg_it=np.array(cg_it), cg_t=np.array(cg_t), build_t=np.array(build_t),
                            N=N, dt=DT, RE=RE, EX=EX, precond=a.precond, refresh=a.refresh)
        return f

    mask_f = st.get_global_mask(pin_p=True) if a.filter > 0 else None
    wall0 = time.perf_counter()
    try:
        for step in range(step0, a.nstep):
            t = (step+1)*DT + t_off
            U = S.step_bdf(st, h, time=t, max_newton=a.newton, newton_tol=1e-12, newton_factor=0.0,
                           pin_p=True, cgsfac=a.cgsfac, cg_tol=a.cgtol, cg_max_iter=CGMAX, custom_lid=lid)
            if a.filter > 0:
                h[0][...] = filt(h[0])*mask_f + h[0]*(1 - mask_f)                   # filter free dofs only; keep prescribed values
                U = h[0]
            if not np.all(np.isfinite(U)):
                print(f'NaN at step {step+1}', flush=True); save(step+1, 'nan'); return
            if (step+1) % 50 == 0 or step == step0:
                print(f'step {step+1:5d} t={t:.3f} max|u|={np.abs(U[...,0]).max():.4f} max|v|={np.abs(U[...,1]).max():.4f} | CG it {cg_it[-1]:5d} (mean {np.mean(cg_it):.0f}) '
                      f'cg {np.sum(cg_t):.0f}s build {np.sum(build_t):.0f}s wall {time.perf_counter()-wall0:.0f}s', flush=True)
            if (step+1) % a.ckpt == 0:
                print('  checkpoint ->', save(step+1), flush=True)
            if a.steady > 0 and len(h) > 1:
                rate = float(np.abs(h[0] - h[1]).max())/DT; rate_uv = float(np.abs(h[0] - h[1])[..., :2].max())/DT
                if (step+1) % 50 == 0: print(f'    max|dU|/dt = {rate:.3e}  (u,v only: {rate_uv:.3e})', flush=True)
                if rate < a.steady:
                    print(f'STEADY at step {step+1}, t={t:.3f}: max|dU|/dt = {rate:.3e} < {a.steady:g}', flush=True)
                    a.nstep = step+1; break
    finally:
        S.pcg_solve = orig
    f = save(a.nstep, 'final')
    y, u = centreline_u(h[0], m, N); x, v = centreline_v(h[0], m, N)
    gh = np.load('cavity_re1000_data.npz'); gy, gu = gh['ghia_y'], gh['ghia_u']
    rms_u = float(np.sqrt(np.mean((np.interp(gy, y, u) - gu)**2))); o = np.argsort(GHIA_X)
    rms_v = float(np.sqrt(np.mean((np.interp(GHIA_X[o], x, v) - GHIA_V[o])**2)))
    np.savez_compressed(f'{out}/profiles_final.npz', y=y, u=u, x=x, v=v, rms_u=rms_u, rms_v=rms_v, t=a.nstep*DT + t_off)
    print(f'DONE {a.nstep} steps, t={a.nstep*DT + t_off:.2f}: CG it/step mean {np.mean(cg_it):.0f} max {np.max(cg_it)}, CG {np.sum(cg_t):.0f}s, builds {len(build_t)} ({np.sum(build_t):.0f}s), '
          f'wall this session {time.perf_counter()-wall0:.0f}s | rms vs Ghia: u {rms_u:.3e} v {rms_v:.3e} (flow at t={a.nstep*DT + t_off:.1f}, not steady) | {f}', flush=True)

if __name__ == '__main__':
    main()
