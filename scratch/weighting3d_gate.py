"""Phase-3 gate for the balanced weighting in 3D (BALANCED_CONDENSED_PLAN.md 5.3):
laminar Poiseuille + decaying z-rolls on the Stage-5 rig, 20 RKW3 steps,
legacy vs balanced (vs mom_exp 1.5): rms div u, mean-profile error,
perturbation energy decay, CG iterations -- Jacobi and vertex-patch."""
import os, sys, time
for _v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ.setdefault(_v, '4')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, 'scratch')); os.chdir(_R)
os.environ.setdefault('LSSEM3D_BACKEND', 'numpy')
import numpy as np
import lssem3d; lssem3d.set_backend('numpy')
from lssem3d import operator as OP
import channel3d as C

if __name__ == '__main__':
    N, ex, ey, nz = 6, 2, 2, 8
    dt, nstep = float(os.environ.get('DT', 5e-3)), int(os.environ.get('NSTEP', 20))
    s = C.setup(N=N, ex=ex, ey=ey, nz=nz, re=180.0)
    U0 = C.initial_state(s, amp=1e-3)
    E0 = C.perturbation_energy(s, U0)
    print(f'Stage-5 rig {ex}x{ey} N={N} nz={nz}, dt={dt}, {nstep} steps; c per stage = {[round(1/(b*dt)) for b in (37/160, 5/24, 1/6)]}')
    print(f'  initial: div {C.divergence(s, U0):.2e}  profile err {C.mean_profile_error(s, U0):.2e}  E_pert {E0:.3e}')
    for precond in os.environ.get('P', 'jacobi,vschwarz').split(','):
        for weighting, mom_exp in [w for w in (('legacy', None), ('balanced', None), (None, 1.5)) if (w[0] or f'q={w[1]}') in os.environ.get('W', 'legacy,balanced,q=1.5').split(',')]:
            tag = weighting or f'q={mom_exp}'
            t0 = time.perf_counter()
            Minv = C.make_precond(s, dt, 0.0, rowweight=True, precond=precond, weighting=weighting, mom_exp=mom_exp)
            tb = time.perf_counter() - t0
            U = U0.copy(); Nprev = np.zeros(OP.to_complex(U).shape[:-2] + (3, s['nk']), dtype=complex)
            its = []; t0 = time.perf_counter()
            for i in range(nstep):
                U, Nprev, it = C.step(s, U, Nprev, dt, 0.0, rowweight=True, Minv=Minv, tol=1e-8, max_iter=20000, weighting=weighting, mom_exp=mom_exp)
                its.append(it)
            tw = time.perf_counter() - t0
            print(f'  {precond:8s} {tag:9s} {nstep} steps: div {C.divergence(s, U):.2e}  profile err {C.mean_profile_error(s, U):.2e}  '
                  f'E/E0 {C.perturbation_energy(s, U)/E0:.5f}  CG it/stage max {max(its)} mean {np.mean(its):.0f}  build {tb:.1f}s  {tw/nstep:.2f} s/step', flush=True)
