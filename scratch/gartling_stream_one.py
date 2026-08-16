"""Streamlines from one saved Gartling unsteady run.

    uv run --quiet python scratch/gartling_stream_one.py <run.npz> [out.png]

Plots the final field plus every saved snapshot, with reversed-flow shading and
the steady-solver reattachment for reference.  Reads dt_eff from the file so the
time labels are PHYSICAL time, not nominal step count times dt.
"""
import os, sys
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, LinearTriInterpolator

STEADY_LO = 6.181            # 11x4 N=6 steady-solver reattachment
GX = np.linspace(0, 17, 1400)
GY = np.linspace(-0.5, 0.5, 190)


def interp(U, xn, yn, k):
    n = U.shape[1]
    px, py, q = [], [], []
    for e in range(U.shape[0]):
        for i in range(n):
            for j in range(n):
                px.append(xn[e, i]); py.append(yn[e, j]); q.append(U[e, i, j, k])
    return LinearTriInterpolator(Triangulation(np.array(px), np.array(py)),
                                 np.array(q))


def main():
    f = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else \
        'figs/' + os.path.basename(f).replace('.npz', '_streamlines.png')
    d = np.load(f, allow_pickle=True)
    k = set(d.keys())
    U, xn, yn = d['U'], d['xnod'], d['ynod']
    h = d['hist']
    fields = [(U, float(h[-1, 0]) if len(h) else float('nan'), 'final')]
    if 'snaps' in k and d['snaps'].size:
        for Us, ts in zip(d['snaps'], d['snap_t']):
            fields.insert(-1, (Us, float(ts), 'snapshot'))
    MX, MY = np.meshgrid(GX, GY)
    fig, axs = plt.subplots(len(fields), 1, figsize=(15.0, 2.5*len(fields)),
                            squeeze=False)
    for ax, (Uf, tt, what) in zip(axs[:, 0], fields):
        ui = np.array(interp(Uf, xn, yn, 0)(MX, MY).filled(np.nan))
        vi = np.array(interp(Uf, xn, yn, 1)(MX, MY).filled(np.nan))
        ax.contourf(MX, MY, np.ma.masked_invalid(np.hypot(ui, vi)), levels=40,
                    cmap='viridis', vmin=0, vmax=1.5)
        rev = np.ma.masked_where(np.nan_to_num(ui) >= 0, np.ones_like(MX))
        ax.contourf(MX, MY, rev, levels=[.5, 1.5], colors=['red'], alpha=.26)
        ax.streamplot(GX, GY, np.nan_to_num(ui), np.nan_to_num(vi), density=2.6,
                      color='w', linewidth=.6, arrowsize=.6)
        ax.axvline(STEADY_LO, color='gold', lw=1.5, ls=':', zorder=6)
        ax.set_xlim(0, 17); ax.set_ylim(-0.5, 0.5); ax.set_ylabel('y')
        ax.set_title(f't = {tt:.2f}   ({what})   '
                     f'max|u| = {np.abs(Uf[..., 0]).max():.4f}   '
                     f'max|v| = {np.abs(Uf[..., 1]).max():.4f}', fontsize=10)
    axs[-1, 0].set_xlabel('x')
    dte = float(d['dt_eff']) if 'dt_eff' in k else float(d['dt'])
    wm = float(d['wmom']) if 'wmom' in k else 1.0
    ws = float(d['wmass']) if 'wmass' in k else 1.0
    fig.suptitle(f'Gartling Re = 800, from rest -- dt = {float(d["dt"]):g} '
                 f'(dt_eff = {dte:g}), w_mom = {wm:g}, w_mass = {ws:g}, '
                 f'{str(d["outlet"]) if "outlet" in k else "pz"} outlet, '
                 f'status {str(d["status"])}.\n'
                 f'Red = reversed flow;  gold dotted = steady-solver reattachment '
                 f'{STEADY_LO:.3f}.', fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=118, bbox_inches='tight')
    print(out)
    if len(h):
        print(f'   reached t = {h[-1,0]:.3f}   max|u| = {h[-1,1]:.4f}   '
              f'max|v| = {h[-1,2]:.5f}   lo_reatt = {h[-1,3]:.3f}')


if __name__ == '__main__':
    main()
