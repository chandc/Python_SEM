"""Streamlines + u, v, p, omega profiles for ONE saved Gartling run.

    uv run --quiet python scratch/gartling_one_full.py <run.npz> [tag]

Profiles are taken at x = 7 and x = 15 and overlaid on Gartling's benchmark
(digitised from Chan & Mittal fig. 3).  Pressure has no benchmark -- fig. 3
omits it -- so that row is the solution alone.
"""
import os, sys
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, LinearTriInterpolator

STATIONS = [7.0, 15.0]
QTY = [('u', 0, 'u  (axial velocity)'), ('v', 1, 'v  (vertical velocity)'),
       ('p', 2, 'p  (pressure, p = 0 at outlet)'), ('omega', 3, r'$\omega$  (vorticity)')]


def main():
    f = sys.argv[1]
    tag = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(f).replace('.npz', '')
    d = np.load(f, allow_pickle=True); k = set(d.keys())
    g = lambda n, dv: (d[n] if n in k else dv)
    U, xn, yn = d['U'], d['xnod'], d['ynod']
    h = d['hist'] if 'hist' in k else None
    n = U.shape[1]
    px, py, q = [], [], [[], [], [], []]
    for e in range(U.shape[0]):
        for i in range(n):
            for j in range(n):
                px.append(xn[e, i]); py.append(yn[e, j])
                for c in range(4):
                    q[c].append(U[e, i, j, c])
    tri = Triangulation(np.array(px), np.array(py))
    F = [LinearTriInterpolator(tri, np.array(q[c])) for c in range(4)]

    dt = float(g('dt', 0.1)); wm = float(g('wmom', 1.0)); ws = float(g('wmass', 1.0))
    NX = int(g('NX', 11)); ic = str(g('ic', 'stagnant'))
    tend = float(h[-1, 0]) if h is not None and len(h) else np.nan
    lo = float(h[-1, 3]) if h is not None and len(h) else np.nan
    sub = (f'{NX}x4 N={U.shape[1]-1},  dt = {dt:g}  (dt_eff = {dt*wm/ws:g}),  '
           f'w_mom = {wm:g},  w_mass = {ws:g},  a_mass = {ws*1.5/dt:.3g},  IC = {ic},  '
           f't = {tend:.1f}')

    # ---- streamlines ----
    gx = np.linspace(0, 17, 1400); gy = np.linspace(-0.5, 0.5, 190)
    GX, GY = np.meshgrid(gx, gy)
    ui = np.array(F[0](GX, GY).filled(np.nan)); vi = np.array(F[1](GX, GY).filled(np.nan))
    fig, ax = plt.subplots(figsize=(15.0, 2.6))
    ax.contourf(GX, GY, np.ma.masked_invalid(np.hypot(ui, vi)), levels=40,
                cmap='viridis', vmin=0, vmax=1.5)
    rev = np.ma.masked_where(np.nan_to_num(ui) >= 0, np.ones_like(GX))
    ax.contourf(GX, GY, rev, levels=[.5, 1.5], colors=['red'], alpha=.25)
    ax.streamplot(gx, gy, np.nan_to_num(ui), np.nan_to_num(vi), density=2.7,
                  color='w', linewidth=.6, arrowsize=.6)
    for xv in (6.1, 4.8, 10.5):
        ax.axvline(xv, color='gold', lw=1.3, ls=':', zorder=6)
    for xs in STATIONS:
        ax.axvline(xs, color='k', lw=1.0, ls='--', alpha=.55, zorder=6)
    ax.set_xlim(0, 17); ax.set_ylim(-0.5, 0.5); ax.set_ylabel('y'); ax.set_xlabel('x')
    ax.set_title(f'{sub}   |   lower reattach {lo:.3f} (Gartling 6.1)   |   '
                 f'max|u| = {np.abs(U[..., 0]).max():.4f}', fontsize=10)
    fig.suptitle('Gartling BFS Re = 800.  Red = reversed flow;  gold dotted = Gartling '
                 '6.1 / 4.8 / 10.5;  black dashed = profile stations.', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    o1 = f'figs/gartling_{tag}_streamlines.png'
    fig.savefig(o1, dpi=120, bbox_inches='tight'); print(o1)

    # ---- profiles ----
    yy = np.linspace(-0.4995, 0.4995, 400)
    fig, axs = plt.subplots(4, 2, figsize=(11.0, 15.0), sharey=True)
    for r, (key, c, nm) in enumerate(QTY):
        for ci, xs in enumerate(STATIONS):
            ax = axs[r, ci]
            bf = f'reference/gartling_re800_x{xs:g}_{key}.csv'
            if os.path.exists(bf):
                b = np.loadtxt(bf, delimiter=',', skiprows=9)
                ax.plot(b[:, 1], b[:, 0], '-', color='k', lw=2.8, alpha=.75,
                        label='Gartling benchmark')
            vals = np.array(F[c](np.full_like(yy, xs), yy).filled(np.nan))
            ax.plot(vals, yy, '-', color='tab:blue', lw=2.0, label='this run')
            ax.axvline(0, color='k', lw=.7, ls=':'); ax.axhline(0, color='0.6', lw=.7, ls=':')
            ax.grid(alpha=.3)
            if r == 0:
                ax.set_title(f'x = {xs:g}', fontsize=12)
            if ci == 0:
                ax.set_ylabel(f'{nm}\n\ny')
            if key == 'p':
                ax.text(.03, .04, 'no benchmark:\nChan fig.3 omits p',
                        transform=ax.transAxes, fontsize=8, color='0.35')
    axs[0, 0].legend(fontsize=9, loc='upper left')
    fig.suptitle(f'Gartling BFS Re = 800 -- profiles at x = 7 and x = 15.\n{sub}',
                 fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.945])
    o2 = f'figs/gartling_{tag}_profiles.png'
    fig.savefig(o2, dpi=120, bbox_inches='tight'); print(o2)


if __name__ == '__main__':
    main()
