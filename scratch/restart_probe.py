"""Probe: from the balanced dt=1e-2 steady field, take 3 steps at various dt with
BDF1 (one history level) or BDF2 (two identical levels) and report max|dU| per step.
Separates 'the fixed point moved with dt' from 'BDF1 restart transient'."""
import os, sys, time
for _v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ.setdefault(_v, '4')
os.environ.setdefault('CAV_RE', '1000'); os.environ.setdefault('CAV_EX', '4')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, 'scratch')); os.chdir(_R)
import numpy as np
import lssem2d; lssem2d.set_backend('numpy')
from lssem2d import solver as S
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState
from lssem2d.mesh import build_channel
from pmg_ghia_cavity import snapshot
from vertex_schwarz2d import VertexSchwarzCondensed2D, make_coarse

if __name__ == '__main__':
    N = 15; m = build_channel(1.0, 1.0, 4, 4, N, bcs=(1, 1, 1, 2)); m.compute_global_indices()
    Uref = np.load('scratch/ghia_n15_w0.1_steady/final.npz')['U0']
    for dt, nh in [(1e-2, 2), (1e-2, 1), (1e-3, 2), (1e-3, 1), (1e-4, 2)]:
        w = np.sqrt(dt)
        st = SolverState(m, diff_matrix(N), nu=1e-3, dt=dt, fac1=1.0, w_mom=w, w_mass=w)
        cache = {}
        def factory(s_, fu, fv, Mi, pp):
            if 'p' not in cache:
                snap = snapshot(s_, fu, fv); fu, fv = np.ascontiguousarray(fu), np.ascontiguousarray(fv)
                cache['p'] = VertexSchwarzCondensed2D(snap, fu, fv, pin_p=pp, coarse=make_coarse(snap, fu, fv, Mi, pp))
            return cache['p']
        st.precond_factory = factory
        h = [Uref.copy()] + ([Uref.copy()] if nh == 2 else [])
        out = []
        for k in range(3):
            Uo = h[0].copy(); t0 = time.perf_counter()
            U = S.step_bdf(st, h, time=(k+1)*dt, max_newton=1, newton_tol=1e-12, newton_factor=0.0, pin_p=True, cgsfac=1e-6, cg_tol=1e-14, cg_max_iter=60000)
            out.append((np.abs(U - Uo)[..., :2].max(), np.abs(U - Uref)[..., :2].max(), time.perf_counter() - t0))
        print(f"dt={dt:g} BDF{nh} start: " + '  '.join(f"step{k+1}: |dU|={a:.2e} |U-Uref|={b:.2e} ({c:.0f}s)" for k, (a, b, c) in enumerate(out)), flush=True)
