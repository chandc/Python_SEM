"""Effect of artificial compressibility on PRESSURE convergence -- cavity Re=1000.

    uv run --quiet python scratch/cavity_ac_pconv.py            # full serial sweep
    uv run --quiet python scratch/cavity_ac_pconv.py 0.05       # one dt

AC adds kappa_p*(p - p_prev) to the continuity row, so pressure is the field it
acts on directly.  This measures what that does to the convergence of p, charged
against the two costs that actually matter:

    * CG iterations  -- machine-independent, deterministic, the honest unit
    * wall seconds   -- what the user waits, but load-dependent

Per step we record max|dp| and max|du| separately.  Reporting only a combined
|dU| would hide the effect: |dU| is dominated by vorticity, whose magnitude here
is ~300x the pressure's, so a large change in p is invisible inside it.

FIXED STEP BUDGET, no convergence test.  Earlier scripts in this study used a
|dU| threshold that turned out to sit below the solver's own floor, so runs that
had converged kept going for hours.  A fixed budget cannot have that failure
mode, and "CG iterations to reach |dp| < tol" is then a DERIVED quantity read
off the recorded history -- no re-running to change the threshold.

RUN SERIALLY.  CG counts are deterministic and unaffected by load; wall times
are not, and half the point here is the wall column.

    scratch/cavity_ac_pconv_dt{dt}_k{kappa}.npz     per-step history
    scratch/cavity_ac_pconv.csv                     summary, merged across runs
"""
import os, sys, time, csv
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

RE, EX, N = 1000.0, 6, 10
NSTEP = 300
KFRAC = [None, 0.25, 0.5, 1.0, 2.0]          # None = AC off; else kappa_p/a_mass
DTS = [0.05, 0.25]
CSV = f'{SC}/cavity_ac_pconv.csv'
COLS = ['dt', 'a_mass', 'kappa_p', 'kfrac', 'steps', 'cg_its', 'wall_s',
        'dp_final', 'du_final', 'cg_to_dp_1e-4', 'wall_to_dp_1e-4',
        'cg_to_dp_1e-6', 'wall_to_dp_1e-6']

_orig = S.pcg_solve
COUNT = {'it': 0}


def counting_pcg(*a, **k):
    x, it = _orig(*a, **k)
    COUNT['it'] += it
    return x, it


S.pcg_solve = counting_pcg


def run(dt, kfrac):
    n = N+1
    mesh = build_channel(1.0, 1.0, EX, EX, N, bcs=(1, 1, 1, 2))
    st = SolverState(mesh, diff_matrix(N), nu=1.0/RE, dt=dt, fac1=1.0,
                     w_mom=1.0, w_mass=1.0)
    a_mass = 1.5/dt                                   # BDF2 fac1 = 1.5
    kap = None if kfrac is None else kfrac*a_mass
    st.dtau_p = None if kap is None else 1.0/kap
    U = np.zeros((mesh.nelem, n, n, 4)); hist = [U.copy()]
    COUNT['it'] = 0
    # columns: step, cumulative CG its, cumulative wall, max|dp|, max|du|
    H = np.zeros((NSTEP, 5))
    t0 = time.perf_counter()
    for s in range(NSTEP):
        Up = hist[0].copy()
        U = S.step_bdf(st, hist, time=(s+1)*dt, max_newton=5, newton_tol=1e-13,
                       newton_factor=1e-6, pin_p=True, cgsfac=1e-3,
                       cg_tol=1e-8, cg_max_iter=60000, line_search=True)
        if not np.all(np.isfinite(U)):
            H = H[:s]; break
        H[s] = (s+1, COUNT['it'], time.perf_counter()-t0,
                np.abs(U[..., 2]-Up[..., 2]).max(),
                np.abs(U[..., 0]-Up[..., 0]).max())
        if (s+1) % 50 == 0:
            print(f'    step {s+1:4d}  |dp| = {H[s,3]:.3e}  |du| = {H[s,4]:.3e}'
                  f'  cg = {int(H[s,1]):>8d}  {H[s,2]:6.0f}s', flush=True)
    np.savez(f'{SC}/cavity_ac_pconv_dt{dt:g}_k{0 if kap is None else kap:g}.npz',
             hist=H, dt=dt, a_mass=a_mass, kappa_p=(0.0 if kap is None else kap),
             kfrac=(0.0 if kfrac is None else kfrac), U=U, xnod=mesh.xnod,
             ynod=mesh.ynod)

    def first(col, tol):
        """Cumulative cost at the first step with |dp| below tol (nan if never).
        Read off the recorded history, so the tolerance can be changed later
        without re-solving anything."""
        i = np.where(H[:, 3] < tol)[0]
        return np.nan if len(i) == 0 else H[i[0], col]

    return dict(dt=f'{dt:g}', a_mass=f'{a_mass:g}',
                kappa_p=f'{0.0 if kap is None else kap:g}',
                kfrac=('off' if kfrac is None else f'{kfrac:g}'),
                steps=len(H), cg_its=int(H[-1, 1]), wall_s=f'{H[-1,2]:.1f}',
                dp_final=f'{H[-1,3]:.3e}', du_final=f'{H[-1,4]:.3e}',
                **{'cg_to_dp_1e-4': f'{first(1,1e-4):.0f}',
                   'wall_to_dp_1e-4': f'{first(2,1e-4):.1f}',
                   'cg_to_dp_1e-6': f'{first(1,1e-6):.0f}',
                   'wall_to_dp_1e-6': f'{first(2,1e-6):.1f}'})


if __name__ == '__main__':
    dts = [float(a) for a in sys.argv[1:]] or DTS
    rows = {}
    if os.path.exists(CSV):
        with open(CSV) as fh:
            rows = {(r['dt'], r['kappa_p']): r for r in csv.DictReader(fh)}
    print(f'Cavity Re={RE:.0f}, {EX}x{EX} elem N={N}, {NSTEP} steps from rest, '
          f'nsub=5, cg_tol=1e-8.\nPressure convergence vs CG iterations and wall '
          f'time.  kappa_p given as a fraction of a_mass.\n')
    for dt in dts:
        for kfrac in KFRAC:
            lab = 'off' if kfrac is None else f'{kfrac:g}*a_mass'
            print(f'  dt = {dt:g}, kappa_p = {lab}', flush=True)
            r = run(dt, kfrac)
            rows[(r['dt'], r['kappa_p'])] = r
            with open(CSV, 'w', newline='') as fh:
                wr = csv.DictWriter(fh, COLS); wr.writeheader()
                for k in sorted(rows, key=lambda k: (-float(k[0]), float(k[1]))):
                    wr.writerow(rows[k])
            print(f'    -> cg {r["cg_its"]}, wall {r["wall_s"]}s, '
                  f'|dp| {r["dp_final"]}\n', flush=True)
    print(f'-> {CSV}')
