"""w_obc sweep of the Dong outlet on the nx18-refined Gartling Re = 800 BFS
-- Stage 3 of OUTFLOW_DONG_OBC_PLAN.md, on the finer grid.

    uv run --quiet python scratch/dong_wobc_sweep.py <w_obc> [N]
    uv run --quiet python scratch/dong_wobc_sweep.py table    # collect results
    uv run --quiet python scratch/dong_wobc_sweep.py plot     # x_r vs w_obc + profiles

One process per w_obc value so the sweep parallelises across cores (BLAS is
told to stay single-threaded; oversubscription would slow every job).

Grid: gartling_nx18_N7_grid.dat (18x4 uniform, order 7) -- generated for this
sweep; nx18 is Chan's fig-6 refinement level, at which the coarse-grid
limit-cycle artifact disappears.  N = 6 uses his actual fig-6 grid.  Steady
form, identical protocol to scratch/dong_gartling.py.  The plan's sec 4
warning is the motivation: a boundary row integrated over a 1D edge is
dimensionally unlike a volume row, so w_obc = 1 is arbitrary -- the sweep
looks for the flat region, as gartling_wmom_plot.py did for w_mom.

The w_obc = 1 run doubles as the nx18 refinement check for the v(x = 7)
overshoot seen on nx11 (DONG_OBC_RESULTS.md sec 6): if v converges onto
Gartling's curve here, that overshoot was resolution error steered by the
BC, not a defect of the condition.

Each run saves scratch/dong_wobc_nx18_N<N>_w<w>.npz.
"""
import os, sys, time
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
SC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SC)
sys.path.insert(0, ROOT)
sys.path.insert(0, SC)
os.chdir(ROOT)
import numpy as np
import lssem2d
lssem2d.set_backend('numpy')
from fgrid import load
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState
import lssem2d.solver as S
from gartling_run import inlet_profile, features, NU

WVALS = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0]


def run(w_obc, N=7, cap=None, wallcap=3600.0):
    cap = cap or int(os.environ.get('DONG_CAP', '400'))
    m, _, _ = load(f'grids/gartling_nx18_N{N}_grid.dat')
    D = diff_matrix(m.N); n = m.N + 1
    for e in range(m.nelem):
        if m.bc[e, 1] == 4:
            m.bc[e, 1] = 6
    st = SolverState(m, D, nu=NU, dt=1.0, fac1=1.0, w_mom=1.0, w_mass=0.0)
    st.obc_w = w_obc
    st.obc_D0 = 0.0
    st.obc_delta = None
    inl = lambda x, y, t: inlet_profile(y)
    U = np.zeros((m.nelem, n, n, 4)); h = [U.copy()]
    t0 = time.perf_counter(); status = 'CAP'; d = np.nan
    for s in range(cap):
        prev = h[0].copy()
        U = S.step_bdf(st, h, time=0.0, max_newton=1, newton_tol=1e-12,
                       newton_factor=0.0, custom_inlet=inl, pin_p=False,
                       cgsfac=1e-3, cg_tol=1e-6, cg_max_iter=200000,
                       line_search=True)
        if not np.all(np.isfinite(U)):
            status = 'NaN'; break
        d = float(np.abs(U - prev).max())
        if np.abs(U[..., 0]).max() > 20.0:
            status = 'BLEWUP'; break
        if d < 1e-10:
            status = 'conv'; break
        if time.perf_counter() - t0 > wallcap:
            status = 'WALLCAP'; break
    lo, us, ur = features(U, m, D)
    np.savez(f'{SC}/dong_wobc_nx18_N{N}_w{w_obc:g}.npz', U=U, xnod=m.xnod,
             ynod=m.ynod, hy=m.hy, hx=m.hx, N=m.N, nu=NU, status=status,
             iters=s + 1, dU=d, w_obc=w_obc, lo_reatt=lo, up_sep=us,
             up_reatt=ur)
    print(f'w_obc = {w_obc:g}  N = {N}: {status} in {s + 1} it, '
          f'|dU| = {d:.3e}, x_r = {lo:.3f}, up = {us:.3f}/{ur:.3f}, '
          f'{time.perf_counter() - t0:.0f}s')


def table():
    import glob, re
    rows = []
    for f in sorted(glob.glob(f'{SC}/dong_wobc_nx18_N*_w*.npz')):
        d = np.load(f, allow_pickle=True)
        mm = re.search(r'N(\d+)_w([\d.]+)\.npz', f)
        rows.append((float(mm.group(2)), int(mm.group(1)), str(d['status']),
                     int(d['iters']), float(d['dU']), float(d['lo_reatt']),
                     float(d['up_sep']), float(d['up_reatt'])))
    rows.sort(key=lambda r: (r[1], r[0]))
    hdr = (f"{'w_obc':>7}{'N':>4}{'status':>9}{'iters':>7}{'|dU|':>11}"
           f"{'x_r':>8}{'up sep':>8}{'up re':>8}")
    print(hdr); print('-' * len(hdr))
    for r in rows:
        print(f'{r[0]:>7g}{r[1]:>4}{r[2]:>9}{r[3]:>7}{r[4]:>11.3e}'
              f'{r[5]:>8.3f}{r[6]:>8.3f}{r[7]:>8.3f}')
    print('\nChan & Mittal: 6.1, 4.8, 10.5.  nx11 N7 Dong (w_obc = 1): '
          '6.153, 4.915, 10.497.')


def plot():
    import glob, re
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from dong_gartling import profile_at
    ws, xr, us_, ur_ = [], [], [], []
    for f in sorted(glob.glob(f'{SC}/dong_wobc_nx18_N7_w*.npz')):
        d = np.load(f, allow_pickle=True)
        if 'conv' not in str(d['status']):
            continue
        ws.append(float(d['w_obc'])); xr.append(float(d['lo_reatt']))
        us_.append(float(d['up_sep'])); ur_.append(float(d['up_reatt']))
    o = np.argsort(ws)
    ws, xr, us_, ur_ = (np.array(a)[o] for a in (ws, xr, us_, ur_))

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3)
    ax = fig.add_subplot(gs[0, :])
    ax.semilogx(ws, xr, 'o-', label='lower reattach (Chan 6.1)')
    ax.semilogx(ws, us_, 's-', label='upper sep (Chan 4.8)')
    ax.semilogx(ws, ur_, '^-', label='upper reattach (Chan 10.5)')
    for v, c in ((6.1, 'C0'), (4.8, 'C1'), (10.5, 'C2')):
        ax.axhline(v, color=c, ls=':', lw=1)
    ax.set_xlabel('w_obc'); ax.set_ylabel('x')
    ax.set_title('Gartling Re = 800, nx18 N = 7, Dong outlet: wall features '
                 'vs boundary-row weight w_obc')
    ax.grid(alpha=.3); ax.legend(fontsize=9)

    # v(x = 7) refinement check at w_obc = 1: nx18 vs nx11 vs benchmark
    for col, xs in enumerate((7.0, 15.0)):
        axp = fig.add_subplot(gs[1, col])
        ref = np.genfromtxt(f'reference/gartling_re800_x{int(xs)}_profiles.csv',
                            delimiter=',', names=True, skip_header=2)
        axp.plot(ref['v'], ref['y'], 'ko', ms=4, mfc='none',
                 label='Gartling (Chan fig. 3)')
        d18 = np.load(f'{SC}/dong_wobc_nx18_N7_w1.npz', allow_pickle=True)
        y18, r18 = profile_at(d18, xs)
        axp.plot(r18[:, 1], y18, 'C0-', lw=2, label='Dong nx18 N7')
        d11 = np.load(f'{SC}/dong_gartling_steady_nx11_N7.npz',
                      allow_pickle=True)
        y11, r11 = profile_at(d11, xs)
        axp.plot(r11[:, 1], y11, 'C3--', lw=1.5, label='Dong nx11 N7')
        axp.set_title(f'v at x = {xs:g}  (w_obc = 1)', fontsize=10)
        axp.set_xlabel('v'); axp.grid(alpha=.3)
        if col == 0:
            axp.set_ylabel('y'); axp.legend(fontsize=8)
    axw = fig.add_subplot(gs[1, 2])
    for f in sorted(glob.glob(f'{SC}/dong_wobc_nx18_N7_w*.npz')):
        d = np.load(f, allow_pickle=True)
        if 'conv' not in str(d['status']):
            continue
        y7, r7 = profile_at(d, 7.0)
        axw.plot(r7[:, 1], y7, lw=1.2, label=f"w = {float(d['w_obc']):g}")
    axw.plot(np.genfromtxt(f'reference/gartling_re800_x7_profiles.csv',
                           delimiter=',', names=True, skip_header=2)['v'],
             np.genfromtxt(f'reference/gartling_re800_x7_profiles.csv',
                           delimiter=',', names=True, skip_header=2)['y'],
             'ko', ms=4, mfc='none', label='benchmark')
    axw.set_title('v at x = 7 across w_obc', fontsize=10)
    axw.set_xlabel('v'); axw.grid(alpha=.3); axw.legend(fontsize=7)
    fig.tight_layout()
    out = f'{ROOT}/figs/dong_wobc_sweep.png'
    fig.savefig(out, dpi=130)
    print('wrote', out)


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'table'
    if mode == 'table':
        table()
    elif mode == 'plot':
        plot()
    else:
        run(float(mode), int(sys.argv[2]) if len(sys.argv) > 2 else 7)
