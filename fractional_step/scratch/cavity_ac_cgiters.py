"""Measure CG iterations per solve, AC off vs on, on the cavity.

    uv run --quiet python scratch/cavity_ac_cgiters.py [dt ...]

Wraps solver.pcg_solve to accumulate the iteration count it returns, so this is
a direct measurement rather than an inference from wall time.  Every case does
identical work -- same steps, same nsub, same tolerance, 200 CG calls -- so the
only thing that differs between the three rows of a dt is kappa_p.

Writes scratch/cavity_ac_cgiters.csv, which scratch/cavity_ac_cgplot.py reads.
The csv is the record: re-running merges by (dt, kappa_p) rather than starting
over, so a dt added later joins the existing sweep instead of replacing it.

RUN THIS SERIALLY.  The iteration counts are deterministic and load-independent,
but the wall column is not -- parallel jobs on the same machine inflate it and
make the columns non-comparable.  Documented in ARTIFICIAL_COMPRESSIBILITY.md
sec 5.2.
"""
import sys, os, time, csv
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
sys.path.insert(0, SC)
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np, lssem2d
lssem2d.set_backend('numpy')
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState
import lssem2d.solver as S

RE, EX, N = 1000.0, 6, 10
NSTEP = 40
CSV = f'{SC}/cavity_ac_cgiters.csv'
COLS = ['dt', 'a_mass', 'kappa_p', 'tag', 'cg_its', 'cg_calls', 'its_per_call',
        'its_per_step', 'wall_s']

_orig = S.pcg_solve
COUNT = {'it': 0, 'calls': 0}


def counting_pcg(*a, **k):
    x, it = _orig(*a, **k)
    COUNT['it'] += it; COUNT['calls'] += 1
    return x, it


S.pcg_solve = counting_pcg


def measure(dt, kap, w_mass=1.0):
    n = N+1
    mesh = build_channel(1.0, 1.0, EX, EX, N, bcs=(1, 1, 1, 2))
    st = SolverState(mesh, diff_matrix(N), nu=1.0/RE, dt=dt, fac1=1.0,
                     w_mom=1.0, w_mass=w_mass)
    st.dtau_p = None if kap is None else 1.0/kap
    U = np.zeros((mesh.nelem, n, n, 4)); hist = [U.copy()]
    COUNT['it'] = 0; COUNT['calls'] = 0
    t0 = time.perf_counter()
    for s in range(NSTEP):
        U = S.step_bdf(st, hist, time=(s+1)*dt, max_newton=5, newton_tol=1e-13,
                       newton_factor=1e-6, pin_p=True, cgsfac=1e-3,
                       cg_tol=1e-8, cg_max_iter=60000, line_search=True)
        if not np.all(np.isfinite(U)):
            break
    return COUNT['it'], COUNT['calls'], time.perf_counter()-t0


def load_csv():
    if not os.path.exists(CSV):
        return {}
    with open(CSV) as fh:
        return {(float(r['dt']), float(r['kappa_p'])): r
                for r in csv.DictReader(fh)}


def save_csv(rows):
    with open(CSV, 'w', newline='') as fh:
        w = csv.DictWriter(fh, COLS); w.writeheader()
        for k in sorted(rows, key=lambda k: (-k[0], k[1])):
            w.writerow(rows[k])


def measure_steady():
    """The STEADY form (w_mass = 0 => a_mass = 0) is the a_mass -> 0 limit of the
    same curve, measured under identical conditions.  It goes to its OWN csv
    because a_mass = 0 has no position on the log axis cavity_ac_cgplot.py draws
    -- it belongs there as a horizontal reference line, not a plotted point."""
    its, calls, wall = measure(1.0, None, w_mass=0.0)
    with open(f'{SC}/cavity_ac_cgiters_steady.csv', 'w', newline='') as fh:
        wr = csv.DictWriter(fh, COLS); wr.writeheader()
        wr.writerow(dict(dt=1.0, a_mass=0, kappa_p=0, tag='steady', cg_its=its,
                         cg_calls=calls, its_per_call=f'{its/max(calls,1):.1f}',
                         its_per_step=f'{its/NSTEP:.1f}', wall_s=f'{wall:.1f}'))
    print(f'{"steady":>7}{0:>8}{0:>9}{its:>10d}{its/NSTEP:>10.1f}'
          f'{calls:>10d}{its/max(calls,1):>10.1f}{wall:>7.1f}s   (w_mass = 0)',
          flush=True)


HDR = (f"{'dt':>7}{'a_mass':>8}{'kappa_p':>9}{'CG its':>10}{'per step':>10}"
       f"{'CG calls':>10}{'its/call':>10}{'wall':>8}")

if __name__ == '__main__':
    ARGS = sys.argv[1:]
    print(f'Cavity Re={RE:.0f}, {EX}x{EX} elem N={N}, {NSTEP} steps from rest, '
          f'nsub=5,\ncg_tol=1e-8, cgsfac=1e-3.  CG iterations counted inside '
          f'pcg_solve.\n')
    print(HDR); print('-'*len(HDR))
    if ARGS == ['steady']:                   # just the a_mass = 0 reference point
        measure_steady()
        raise SystemExit
    DTS = [float(a) for a in ARGS] or [1.0, 0.5, 0.25, 0.1, 0.05]
    rows = load_csv()
    for dt in DTS:
        a_mass = 1.5/dt                                  # BDF2 fac1 = 1.5
        for tag, kap in (('off', None), ('half', a_mass/2), ('match', a_mass)):
            its, calls, wall = measure(dt, kap)
            kv = 0.0 if kap is None else kap
            rows[(dt, kv)] = dict(dt=f'{dt:g}', a_mass=f'{a_mass:g}',
                                  kappa_p=f'{kv:g}', tag=tag, cg_its=its,
                                  cg_calls=calls,
                                  its_per_call=f'{its/max(calls,1):.1f}',
                                  its_per_step=f'{its/NSTEP:.1f}',
                                  wall_s=f'{wall:.1f}')
            print(f'{dt:>7g}{a_mass:>8.4g}{kv:>9.4g}{its:>10d}{its/NSTEP:>10.1f}'
                  f'{calls:>10d}{its/max(calls,1):>10.1f}{wall:>7.1f}s', flush=True)
            save_csv(rows)                               # checkpoint every case
    print()
    measure_steady()
    print(f'\n-> {CSV}')
