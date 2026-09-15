"""Does AC's benefit scale with the GLL polynomial order N?  Cavity Re = 1000.

    uv run --quiet python scratch/cavity_ac_nsweep.py <N> <off|half|match>

Every AC result in ARTIFICIAL_COMPRESSIBILITY.md was measured at a SINGLE order
per flow -- N = 10 on the cavity and channel, N = 6 on the BFS -- and those two
differ in flow, mesh and BCs as well, so nothing about N could be extracted from
them.  This sweep varies N alone: mesh fixed at 6x6 elements, dt fixed at 0.05
(a_mass = 30, where AC bites hardest in sec 5.2), N = 4 .. 14.

WHY IT MATTERS.  The documented mechanism is that Jacobi has no a33 pressure
diagonal and AC supplies one.  Spectral conditioning of L^T L grows steeply with
N, so the hole AC fills is plausibly N-dependent -- the 27.5x at N = 10 could
grow, saturate or shrink.  And the kappa_p ~ a_mass rule balances kappa_p against
a_mass, which carries NO N dependence, while the operator norms it is balancing
against certainly do.  If the optimum drifts with N, sec 6's recommendation is
only valid at N = 10.

WHAT THIS MEASURES: conditioning, i.e. CG iterations per solve over 40 steps from
rest -- the same protocol as scratch/cavity_ac_cgiters.py, so the N = 10 column
should reproduce that table exactly.  It does NOT measure converged accuracy:
40 steps at dt = 0.05 is t = 2, nowhere near steady.  The rms field in the npz is
recorded for completeness but is NOT an accuracy figure; accuracy-versus-N needs
converged runs and is a separate, much more expensive study.

ONE FILE PER RUN, no shared csv.  These are launched in parallel, and a
checkpointing shared csv is exactly the race that silently truncated a plot
earlier in this study.

CG ITERATION COUNTS ARE VALID UNDER PARALLEL LOAD -- they are deterministic and
load-independent (verified: re-measuring a_mass = 6 AC-off in a different pass
returned bit-identical 138098).  The WALL column is not; it is recorded but must
not be compared across runs from a parallel launch.
"""
import os, sys, time
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')          # must precede numpy import
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
sys.path.insert(0, SC)
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
import lssem2d
lssem2d.set_backend('numpy')
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState
import lssem2d.solver as S

RE, EX, DT = 1000.0, 6, 0.05
NSTEP = 40
GH = np.load('cavity_re1000_data.npz')

_orig = S.pcg_solve
COUNT = {'it': 0, 'calls': 0}


def counting_pcg(*a, **k):
    x, it = _orig(*a, **k)
    COUNT['it'] += it; COUNT['calls'] += 1
    return x, it


S.pcg_solve = counting_pcg


def lagrange(xn, xq):
    n = len(xn); w = np.ones(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                w[i] /= (xn[i]-xn[j])
    dd = xq-xn
    if np.any(np.abs(dd) < 1e-13):
        L = np.zeros(n); L[np.argmin(np.abs(dd))] = 1.0; return L
    num = w/dd
    return num/num.sum()


def centreline_u(mesh, U, n):
    ys, us = [], []
    for e in range(mesh.nelem):
        xs = mesh.xnod[e]
        if xs[0]-1e-9 <= 0.5 <= xs[-1]+1e-9:
            L = lagrange(xs, 0.5)
            for j in range(n):
                ys.append(mesh.ynod[e, j]); us.append(np.dot(L, U[e, :, j, 0]))
    o = np.argsort(ys); ys, us = np.array(ys)[o], np.array(us)[o]
    k = np.concatenate(([True], np.diff(ys) > 1e-9))
    return ys[k], us[k]


def run(N, kspec):
    n = N+1
    mesh = build_channel(1.0, 1.0, EX, EX, N, bcs=(1, 1, 1, 2))
    st = SolverState(mesh, diff_matrix(N), nu=1.0/RE, dt=DT, fac1=1.0,
                     w_mom=1.0, w_mass=1.0)
    a_mass = 1.5/DT                                   # BDF2 fac1 = 1.5
    if kspec == 'off':
        kap = None
    elif kspec == 'half':
        kap = a_mass/2.0
    elif kspec == 'match':
        kap = a_mass
    else:
        kap = float(kspec)
    st.dtau_p = None if kap is None else 1.0/kap
    U = np.zeros((mesh.nelem, n, n, 4)); hist = [U.copy()]
    COUNT['it'] = 0; COUNT['calls'] = 0
    t0 = time.perf_counter(); ok = True
    for s in range(NSTEP):
        U = S.step_bdf(st, hist, time=(s+1)*DT, max_newton=5, newton_tol=1e-13,
                       newton_factor=1e-6, pin_p=True, cgsfac=1e-3,
                       cg_tol=1e-8, cg_max_iter=60000, line_search=True)
        if not np.all(np.isfinite(U)):
            ok = False; break
    wall = time.perf_counter()-t0
    rms = np.nan
    if ok:
        ys, us = centreline_u(mesh, U, n)
        rms = float(np.sqrt(np.mean((np.interp(GH['ghia_y'], ys, us)
                                     - GH['ghia_u'])**2)))
    dofs = mesh.nelem*n*n*4
    np.savez(f'{SC}/nsweep_N{N}_{kspec}.npz', N=N, EX=EX, dt=DT, a_mass=a_mass,
             kappa_p=(0.0 if kap is None else kap), kspec=kspec, dofs=dofs,
             cg_its=COUNT['it'], cg_calls=COUNT['calls'], wall_s=wall,
             its_per_call=COUNT['it']/max(COUNT['calls'], 1), ok=ok,
             rms_t2=rms, nstep=NSTEP)
    return dict(N=N, kspec=kspec, kap=(0.0 if kap is None else kap), dofs=dofs,
                its=COUNT['it'], calls=COUNT['calls'],
                per_call=COUNT['it']/max(COUNT['calls'], 1), wall=wall, ok=ok)


if __name__ == '__main__':
    N = int(sys.argv[1]); kspec = sys.argv[2]
    r = run(N, kspec)
    print(f"N={r['N']:<3} {r['kspec']:>6} kappa_p={r['kap']:<5g} "
          f"dofs={r['dofs']:>7} cg={r['its']:>9} calls={r['calls']:>4} "
          f"its/call={r['per_call']:>8.1f} wall={r['wall']:>7.1f}s "
          f"{'' if r['ok'] else 'DIVERGED'}", flush=True)
