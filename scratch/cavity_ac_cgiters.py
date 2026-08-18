"""Measure CG iterations per time step, AC off vs on, on the cavity.

    uv run --quiet python scratch/cavity_ac_cgiters.py

Wraps solver.pcg_solve to accumulate the iteration count it returns, so this is
a direct measurement rather than an inference from wall time.  Every case does
identical work -- same steps, same nsub, same tolerance, 200 CG calls -- so the
only difference between rows is kappa_p.

Produces the table plotted by scratch/cavity_ac_cgplot.py and documented in
ARTIFICIAL_COMPRESSIBILITY.md sec 5.2.  If you re-measure and the numbers move,
update the DATA literal in cavity_ac_cgplot.py to match.
"""
import sys, os, time
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo/scratch')
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np, lssem2d
lssem2d.set_backend('numpy')
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState
import lssem2d.solver as S

RE, EX, N = 1000.0, 6, 10
NSTEP = 40
_orig = S.pcg_solve
COUNT = {'it': 0, 'calls': 0}

def counting_pcg(*a, **k):
    x, it = _orig(*a, **k)
    COUNT['it'] += it; COUNT['calls'] += 1
    return x, it
S.pcg_solve = counting_pcg

print(f'Cavity Re={RE:.0f}, {EX}x{EX} elem N={N}, {NSTEP} steps from rest, nsub=5,')
print('cg_tol=1e-8, cgsfac=1e-3.  CG iterations counted inside pcg_solve.\n')
print(f"{'dt':>7}{'a_mass':>8}{'kappa_p':>9}{'CG its':>10}{'per step':>10}"
      f"{'CG calls':>10}{'its/call':>10}{'wall':>8}")
for dt in (0.25, 0.05):
    a_mass = 1.5/dt
    for tag, kap in (('off', None), ('half', a_mass/2), ('match', a_mass)):
        n = N+1
        mesh = build_channel(1.0, 1.0, EX, EX, N, bcs=(1, 1, 1, 2))
        st = SolverState(mesh, diff_matrix(N), nu=1.0/RE, dt=dt, fac1=1.0,
                         w_mom=1.0, w_mass=1.0)
        st.dtau_p = None if kap is None else 1.0/kap
        U = np.zeros((mesh.nelem, n, n, 4)); hist = [U.copy()]
        COUNT['it'] = 0; COUNT['calls'] = 0
        t0 = time.perf_counter()
        for s in range(NSTEP):
            U = S.step_bdf(st, hist, time=(s+1)*dt, max_newton=5,
                           newton_tol=1e-13, newton_factor=1e-6, pin_p=True,
                           cgsfac=1e-3, cg_tol=1e-8, cg_max_iter=60000,
                           line_search=True)
            if not np.all(np.isfinite(U)):
                break
        w = time.perf_counter()-t0
        print(f'{dt:>7g}{a_mass:>8.4g}{(0.0 if kap is None else kap):>9.4g}'
              f'{COUNT["it"]:>10d}{COUNT["it"]/NSTEP:>10.1f}{COUNT["calls"]:>10d}'
              f'{COUNT["it"]/max(COUNT["calls"],1):>10.1f}{w:>7.1f}s', flush=True)
