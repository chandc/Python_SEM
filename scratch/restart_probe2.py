"""Where is the per-step change located when dt changes from 1e-2 to 1e-3/1e-4 (BDF2, steady history)?"""
import os, sys
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
    for dt in (1e-3, 1e-4):
        w = np.sqrt(dt)
        st = SolverState(m, diff_matrix(N), nu=1e-3, dt=dt, fac1=1.0, w_mom=w, w_mass=w)
        cache = {}
        def factory(s_, fu, fv, Mi, pp):
            if 'p' not in cache:
                snap = snapshot(s_, fu, fv); fu, fv = np.ascontiguousarray(fu), np.ascontiguousarray(fv)
                cache['p'] = VertexSchwarzCondensed2D(snap, fu, fv, pin_p=pp, coarse=make_coarse(snap, fu, fv, Mi, pp))
            return cache['p']
        st.precond_factory = factory
        h = [Uref.copy(), Uref.copy()]
        U = S.step_bdf(st, h, time=dt, max_newton=1, newton_tol=1e-12, newton_factor=0.0, pin_p=True, cgsfac=1e-6, cg_tol=1e-14, cg_max_iter=60000)
        dU = U - Uref
        for f, nm in ((0, 'u'), (1, 'v'), (2, 'p'), (3, 'om')):
            a = np.abs(dU[..., f]); i = np.unravel_index(a.argmax(), a.shape)
            x, y = m.xnod[i[0], i[1]], m.ynod[i[0], i[2]]
            rms = np.sqrt(np.sum(a**2*m.wq)/np.sum(m.wq))
            far = a[(np.abs(m.ynod[:, None, :]*0 + m.ynod[:, None, :]) < 0.9) & (np.abs(m.xnod[:, :, None]*0 + m.xnod[:, :, None]) > -1)] if False else None
            print(f"dt={dt:g} {nm:2s}: max|d{nm}|={a.max():.2e} at (x,y)=({x:.3f},{y:.3f})  rms={rms:.2e}", flush=True)
        # max over the region away from the lid (y<0.9)
        mask = (m.ynod[:, None, :] < 0.9) * np.ones((1, N+1, 1), bool)
        print(f"dt={dt:g} max|du| for y<0.9: {np.abs(dU[..., 0])[mask].max():.2e}, y<0.98: {np.abs(dU[..., 0])[(m.ynod[:, None, :] < 0.98) * np.ones((1, N+1, 1), bool)].max():.2e}", flush=True)
