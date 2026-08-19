"""Dong OBC on the SHORT Armaly-specification BFS, compared against the
figs/armaly_profiles.png cases (LONG/P+Z reference, SHORT/P+Z, SHORT/free).

    uv run --quiet python scratch/dong_armaly.py run    # solve + save npz
    uv run --quiet python scratch/dong_armaly.py plot   # profile figure

Armaly spec (armaly_run.py): ER 1.94, no-slip top, Re = 389 via D = 2h
(nu = 2/389), parabolic inlet on y in [0.94, 1.94], dt = 1, cold start.
Short grid: L = 5, x_r ~ 7.6 lies beyond it, so the bottom-wall backflow
CROSSES the outlet -- a genuine-backflow exit, like the cnos short domain.

Two Dong variants on the short grid:
  off  D0 = 1.94 (= 1/U_c, U_c = 1/1.94), switch OFF, nsub = 1 -- the exact
       protocol of the P+Z/free runs in the figure.
  on   switch ARMED (delta = 0.05, U0 = 1) in PICARD form with nsub = 5 --
       the lagged-explicit form blew up on the cnos short domain
       (scratch/dong_bfs.py), so the semi-implicit one is used here.

The plot mirrors figs/armaly_profiles.png (u, v, p - p_inlet at x/S = 1..4,
the stations inside the short domain) with the Dong runs overlaid.
"""
import os, sys, time
SC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SC)
sys.path.insert(0, ROOT)
sys.path.insert(0, SC)
os.chdir(ROOT)
import numpy as np
import lssem2d
lssem2d.set_backend('numpy')
from fgrid import load
from lssem2d.lgl import diff_matrix, lgl_weights
from lssem2d.lssem import SolverState
import lssem2d.solver as S
from armaly_run import inlet_profile, reattach, GRIDS, RE, NU, S_STEP

H_TOT = 1.94
D0 = 1.94                          # 1/U_c, U_c = flux 1.0 / height 1.94


def run(variant, cap=1200, wallcap=2400.0):
    m, _, _ = load(GRIDS['short']); N = m.N; n = N + 1
    for e in range(m.nelem):
        if m.bc[e, 1] == 4:
            m.bc[e, 1] = 6
    D = diff_matrix(N)
    st = SolverState(m, D, nu=NU, dt=1.0, fac1=1.0, w_mom=1.0, w_mass=1.0)
    st.obc_D0 = D0
    if variant == 'on':
        st.obc_delta = 0.05
        st.obc_picard = True
        nsub = 5
    else:
        st.obc_delta = None
        nsub = 1
    inl = lambda x, y, t: inlet_profile(y)
    U = np.zeros((m.nelem, n, n, 4)); h = [U.copy()]
    t0 = time.perf_counter(); status = 'CAP'; d = np.nan
    for s in range(cap):
        prev = h[0].copy()
        U = S.step_bdf(st, h, time=s * 1.0, max_newton=nsub, newton_tol=1e-12,
                       newton_factor=(1e-6 if nsub > 1 else 0.0),
                       custom_inlet=inl, pin_p=False,
                       cgsfac=1e-3, cg_tol=1e-6, cg_max_iter=200000,
                       line_search=(nsub > 1))
        if not np.all(np.isfinite(U)):
            status = 'NaN'; break
        d = float(np.abs(U - prev).max())
        if np.abs(U[..., 0]).max() > 20.0:
            status = 'BLEWUP'; break
        if d < 1e-11:
            status = 'conv'; break
        if time.perf_counter() - t0 > wallcap:
            status = 'WALLCAP'; break
    ok = np.all(np.isfinite(U)) and status != 'BLEWUP'
    out = [e for e in range(m.nelem) if m.bc[e, 1] == 6]
    umin = min(U[e, -1, :, 0].min() for e in out) if ok else np.nan
    np.savez(f'{SC}/armaly_short_dong_{variant}.npz', U=U, xnod=m.xnod,
             ynod=m.ynod, hy=m.hy, N=N, nu=NU, dt=1.0, status=status,
             steps=s + 1, dU=d)
    xr = reattach(U, m.xnod, m.ynod, m.hy, N) if ok else np.nan
    print(f'  short / Dong-{variant}: {status:>8} {s + 1:>5} steps  '
          f'|dU| = {d:.3e}  max|u| = {(np.abs(U[..., 0]).max() if ok else np.nan):.4f}  '
          f'min u out = {umin:.3f}  x_r = {xr:.3f}  '
          f'{time.perf_counter() - t0:.0f}s', flush=True)


def plot():
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation, LinearTriInterpolator
    CASES = [('armaly_long_pz.npz', 'LONG / P+Z (reference)', 'tab:green',
              dict(ls='-', lw=2.2)),
             ('armaly_short_pz.npz', 'SHORT / P+Z', 'tab:blue',
              dict(ls='none', marker='o', ms=4.6, mfc='none', mew=1.3)),
             ('armaly_short_free.npz', 'SHORT / free', 'tab:red',
              dict(ls='none', marker='s', ms=4.2, mfc='none', mew=1.1)),
             ('armaly_short_dong_off.npz', 'SHORT / Dong (switch off)',
              'tab:purple', dict(ls='--', lw=1.8)),
             ('armaly_short_dong_on.npz', 'SHORT / Dong (switch ON, Picard)',
              'darkorange', dict(ls='-', lw=1.6))]

    def pack(f, lab, col, sty):
        d = np.load(f'{SC}/{f}')
        U, xn, yn, hy = d['U'].copy(), d['xnod'], d['ynod'], d['hy']
        n = U.shape[1]; wq = lgl_weights(n - 1); xmin = xn.min()
        tot = a = 0.0
        for e in range(U.shape[0]):
            if abs(xn[e, 0] - xmin) < 1e-9:
                tot += np.sum(wq * U[e, 0, :, 2]) * (hy[e] / 2); a += hy[e]
        U[..., 2] -= tot / a
        px, py, q = [], [], [[], [], []]
        for e in range(U.shape[0]):
            for i in range(n):
                for j in range(n):
                    px.append(xn[e, i]); py.append(yn[e, j])
                    for k in range(3):
                        q[k].append(U[e, i, j, k])
        px, py = np.array(px), np.array(py)
        tri = Triangulation(px, py)
        cx = px[tri.triangles].mean(1); cy = py[tri.triangles].mean(1)
        tri.set_mask((cx < 0) & (cy < S_STEP))
        return dict(lab=lab, col=col, sty=sty, xmax=px.max(),
                    f=[LinearTriInterpolator(tri, np.array(q[k]))
                       for k in range(3)])

    C = [pack(*c) for c in CASES]
    ST = [1.0, 2.0, 3.0, 4.0]
    yy = np.linspace(0.002, H_TOT - 0.002, 420)
    NM = ['u  (axial)', 'v  (vertical)', 'p - p_inlet']
    fig, axs = plt.subplots(3, len(ST), figsize=(3.4 * len(ST), 10.6),
                            sharey=True)
    for r in range(3):
        for k, xs_ in enumerate(ST):
            x = xs_ * S_STEP
            ax = axs[r, k]
            for c in C:
                if x > c['xmax'] + 1e-9:
                    continue
                v = np.array(c['f'][r](np.full_like(yy, x), yy).filled(np.nan))
                if c['sty'].get('marker'):
                    ax.plot(v[::18], yy[::18], color=c['col'], label=c['lab'],
                            **c['sty'])
                else:
                    ax.plot(v, yy, color=c['col'], label=c['lab'], **c['sty'])
            ax.axvline(0, color='k', lw=.7, ls=':')
            ax.axhline(S_STEP, color='0.6', lw=.7, ls=':')
            ax.grid(alpha=.3)
            if r == 0:
                ax.set_title(f'x/S = {xs_:g}   (x = {x:.2f})', fontsize=10)
            if r == 2:
                ax.set_xlabel('value')
            if k == 0:
                ax.set_ylabel(f'{NM[r]}\n\ny')
    axs[1, 0].legend(fontsize=7.5, loc='upper left')
    fig.suptitle('Armaly specification, Re = 389, SHORT domain (L = 5, backflow '
                 'crosses the outlet) -- Dong OBC vs the armaly_profiles.png '
                 'cases.\nGreen = long-domain P+Z reference; the short-domain '
                 'runs differ from it only through their outlet condition.',
                 fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = f'{SC}/dong_armaly_profiles.png'
    fig.savefig(out, dpi=130, bbox_inches='tight')
    print('wrote', out)


def streamlines():
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation, LinearTriInterpolator
    CASES = [('armaly_long_pz.npz', 'LONG / P+Z (reference)'),
             ('armaly_short_pz.npz', 'SHORT / P+Z'),
             ('armaly_short_dong_off.npz', 'SHORT / Dong (switch off)'),
             ('armaly_short_dong_on.npz', 'SHORT / Dong (switch ON, Picard)')]
    fig, axs = plt.subplots(len(CASES), 1, figsize=(14.5, 2.9 * len(CASES)))
    for ax, (f, lab) in zip(axs, CASES):
        d = np.load(f'{SC}/{f}')
        U, xn, yn = d['U'], d['xnod'], d['ynod']
        n = U.shape[1]
        px, py, qu, qv = [], [], [], []
        for e in range(U.shape[0]):
            for i in range(n):
                for j in range(n):
                    px.append(xn[e, i]); py.append(yn[e, j])
                    qu.append(U[e, i, j, 0]); qv.append(U[e, i, j, 1])
        px, py = np.array(px), np.array(py)
        tri = Triangulation(px, py)
        cx = px[tri.triangles].mean(1); cy = py[tri.triangles].mean(1)
        tri.set_mask((cx < 0) & (cy < S_STEP))
        gx = np.linspace(px.min(), px.max(), 1200)
        gy = np.linspace(0, H_TOT, 260)
        GX, GY = np.meshgrid(gx, gy)
        ui = np.array(LinearTriInterpolator(tri, np.array(qu))(GX, GY).filled(np.nan))
        vi = np.array(LinearTriInterpolator(tri, np.array(qv))(GX, GY).filled(np.nan))
        ax.contourf(GX, GY, np.ma.masked_invalid(np.hypot(ui, vi)), levels=40,
                    cmap='viridis', vmin=0, vmax=1.6)
        rev = np.ma.masked_where(np.nan_to_num(ui) >= 0, np.ones_like(GX))
        ax.contourf(GX, GY, rev, levels=[.5, 1.5], colors=['red'], alpha=.28)
        ax.streamplot(gx, gy, np.nan_to_num(ui), np.nan_to_num(vi),
                      density=2.4, color='w', linewidth=.6, arrowsize=.65)
        ax.add_patch(plt.Rectangle((px.min(), 0), -px.min(), S_STEP,
                                   fc='0.85', ec='k', lw=1.1, zorder=5))
        ax.axvspan((8.05 - 0.7) * S_STEP, (8.05 + 0.7) * S_STEP, color='gold',
                   alpha=.22, zorder=1)
        ax.axvline(8.05 * S_STEP, color='goldenrod', lw=2.0, ls='--', zorder=6)
        ax.axvline(px.max(), color='yellow', lw=3, zorder=6)
        ax.set_xlim(-2, 17.2); ax.set_ylim(0, H_TOT); ax.set_ylabel('y')
        ax.set_title(f"{lab} ({d['status']})   |   max|u| = "
                     f"{np.abs(U[..., 0]).max():.4f}", fontsize=10)
    axs[-1].set_xlabel('x')
    fig.suptitle('Armaly specification, Re = 389.  Red = reversed flow; gold '
                 'dashed = Armaly measured x_r/S = 8.05 (+/- 0.7 band); '
                 'yellow = outlet.\nThe SHORT domain (L = 5) truncates the '
                 'recirculation: its outlet sits in backflow.', fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = f'{ROOT}/figs/dong_armaly_streamlines.png'
    fig.savefig(out, dpi=120, bbox_inches='tight')
    print('wrote', out)


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'run'
    if mode == 'run':
        run('off')
        run('on')
    elif mode == 'plot':
        plot()
    elif mode == 'streamlines':
        streamlines()
    else:
        raise SystemExit(f'unknown mode {mode!r}')
