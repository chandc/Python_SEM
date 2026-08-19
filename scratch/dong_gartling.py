"""Gartling Re = 800 BFS with the DONG outlet -- Stage 2 of
OUTFLOW_DONG_OBC_PLAN.md, and the profile comparison against Chan & Mittal
(CTR Proc. Summer Program 1996) figs 3-4 / Gartling's benchmark.

    uv run --quiet python scratch/dong_gartling.py run     # solve + save npz
    uv run --quiet python scratch/dong_gartling.py plot    # profile figure

Steady form (w_mass = 0), Chan's own 11x4 grid at N = 7, loose solve with
line search -- identical in every respect to gartling_run.py's run_steady
except the outlet: bc = 6 (Dong, D0 = 0, switch off => traction-free
-p + nu*du/dx = 0 and nu*dv/dx = 0, two scalar conditions) instead of P+Z.
Criterion (plan sec 5): reattachment within the P+Z result's own spread of
Gartling's 6.10 (repo measures 6.100 at N = 7).

The plot overlays, at x = 7 and x = 15: Dong (solid), P+Z (dashed, from
gartling_steady_nx11_N7.npz), and the digitised Gartling benchmark from
Chan's fig. 3 (markers).  No p benchmark exists in the digitisation, so the
p panel is solver-vs-solver only.
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
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState
import lssem2d.solver as S
from gartling_run import inlet_profile, features, NU

NX, N = 11, 7
OUT_NPZ = f'{SC}/dong_gartling_steady_nx{NX}_N{N}.npz'
PZ_NPZ = f'{SC}/gartling_steady_nx{NX}_N{N}.npz'


def run(cap=300):
    m, _, _ = load(f'grids/gartling_nx{NX}_N{N}_grid.dat')
    D = diff_matrix(m.N); n = m.N + 1
    for e in range(m.nelem):
        if m.bc[e, 1] == 4:
            m.bc[e, 1] = 6                       # Dong outlet
    st = SolverState(m, D, nu=NU, dt=1.0, fac1=1.0, w_mom=1.0, w_mass=0.0)
    st.obc_D0 = 0.0                              # steady form: no du/dt term
    st.obc_delta = None                          # no backflow at x = 17
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
    lo, us, ur = features(U, m, D)
    np.savez(OUT_NPZ, U=U, xnod=m.xnod, ynod=m.ynod, hy=m.hy, hx=m.hx, N=m.N,
             nu=NU, status=status, iters=s + 1, dU=d,
             lo_reatt=lo, up_sep=us, up_reatt=ur)
    print(f'DONG steady nx{NX} N{N}: {status} in {s + 1} it, |dU| = {d:.3e}, '
          f'{time.perf_counter() - t0:.0f}s')
    print(f'  lower reattach = {lo:.3f}  (Chan 6.1, repo P+Z 6.100)')
    print(f'  upper sep      = {us:.3f}  (Chan 4.8)')
    print(f'  upper reattach = {ur:.3f}  (Chan 10.5)')


def _lagrange_row(xi_nodes, xi):
    """Lagrange basis values l_i(xi) at the GLL nodes xi_nodes."""
    n = len(xi_nodes)
    L = np.ones(n)
    for i in range(n):
        for k in range(n):
            if k != i:
                L[i] *= (xi - xi_nodes[k]) / (xi_nodes[i] - xi_nodes[k])
    return L


def profile_at(d, xs):
    """(y, u, v, p, om) at station x = xs: exact GLL interpolation along x
    inside each element whose x-range contains xs (the nx11 grid is uniform,
    so x = 7 and 15 are element-interior)."""
    from lssem2d.lgl import lgl_nodes
    U, xn, yn = d['U'], d['xnod'], d['ynod']
    n = U.shape[1]
    xi_nodes = lgl_nodes(n - 1)
    ys, rows = [], []
    for e in range(U.shape[0]):
        x0, x1 = xn[e, 0], xn[e, -1]
        if x0 - 1e-9 <= xs <= x1 + 1e-9:
            xi = 2.0 * (xs - x0) / (x1 - x0) - 1.0
            L = _lagrange_row(xi_nodes, xi)
            vals = np.einsum('i,ijk->jk', L, U[e])       # (n, 4) along y
            for j in range(n):
                ys.append(yn[e, j])
                rows.append(vals[j])
    o = np.argsort(ys)
    ys = np.array(ys)[o]; rows = np.array(rows)[o]
    keep = np.concatenate([[True], np.diff(ys) > 1e-12])
    return ys[keep], rows[keep]


def plot():
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dd = np.load(OUT_NPZ, allow_pickle=True)
    dz = np.load(PZ_NPZ, allow_pickle=True)
    fig, axs = plt.subplots(2, 4, figsize=(16, 8), sharey=True)
    for row, xs in enumerate((7.0, 15.0)):
        ref = np.genfromtxt(f'reference/gartling_re800_x{int(xs)}_profiles.csv',
                            delimiter=',', names=True, skip_header=2)
        yD, rD = profile_at(dd, xs)
        yZ, rZ = profile_at(dz, xs)
        for col, (k, lab) in enumerate(((0, 'u'), (1, 'v'), (2, 'p'),
                                        (3, 'omega'))):
            ax = axs[row, col]
            if lab in ('u', 'v', 'omega'):
                ax.plot(ref[lab], ref['y'], 'ko', ms=4, mfc='none',
                        label='Gartling (Chan fig. 3)')
            ax.plot(rD[:, k], yD, 'C0-', lw=2, label='Dong outlet (bc = 6)')
            ax.plot(rZ[:, k], yZ, 'C3--', lw=1.5, label='P+Z outlet')
            ax.set_title(f'{lab} at x = {xs:g}'
                         + ('' if lab != 'p' else '  (no benchmark)'),
                         fontsize=11)
            ax.grid(alpha=0.3)
            if col == 0:
                ax.set_ylabel('y')
            if row == 1:
                ax.set_xlabel(lab)
    axs[0, 0].legend(fontsize=8, loc='lower right')
    fig.suptitle(f"Gartling Re = 800 BFS, Chan & Mittal 11x4 grid, N = {N}, "
                 f"steady form.  Dong outlet: x_r = {float(dd['lo_reatt']):.3f}, "
                 f"upper sep/reatt = {float(dd['up_sep']):.3f}/"
                 f"{float(dd['up_reatt']):.3f}  "
                 f"(Chan: 6.1, 4.8, 10.5)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = f'{SC}/dong_gartling_profiles.png'
    fig.savefig(out, dpi=150)
    print('wrote', out)


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'run'
    if mode == 'run':
        run()
    elif mode == 'plot':
        plot()
    else:
        raise SystemExit(f'unknown mode {mode!r}')
