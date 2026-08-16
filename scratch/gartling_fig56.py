"""Chan & Mittal figs 5 and 6: the coarse grid invents a limit cycle, the fine one does not.

    uv run --quiet python scratch/gartling_fig56.py

Both runs start from REST (Chan: "initially, the flow is stagnant inside the
domain"), dt = 0.1, BDF2, nsub = 3, P+Z outlet, and run to his endpoint t = 140.
Snapshots at t = 10, 20, 30, 50, 80, 100, 140 are Chan's own fig-5 time list.

THE WEIGHT.  These use w_mom = w_mass = 0.1, not 1.0.  At w_mom = 1 this flow
cannot be time-integrated at all: from a stagnant start it blows up by t ~ 19,
and even STARTING FROM the converged steady field it blows up at t = 62.1, so
the discrete steady state is unstable under the w_mom = 1 time-stepping
operator.  Dropping w_mom to 0.1 lowers the least-squares weight of momentum
against div u = 0 and the vorticity definition by 10x; w_mass is dropped with it
so that dt_eff = dt*w_mom/w_mass = dt and the run stays time-accurate.  With that
change both grids run to t = 140 with max|u| pinned at 1.5000.

    figs/gartling_fig5_nx11_evolution.png   11x4 -- sustained oscillation
    figs/gartling_fig6_nx18_evolution.png   18x4 -- decays to steady
    figs/gartling_fig56_history.png         the quantitative contrast
"""
import os, sys
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, LinearTriInterpolator

RUNS = [(11, 'fig5', 'SUSTAINED oscillation -- Chan fig. 5', 6.181),
        (18, 'fig6', 'DECAYS to steady -- Chan fig. 6', 6.158)]
GX = np.linspace(0, 17, 1300)
GY = np.linspace(-0.5, 0.5, 180)


def interp(U, xn, yn, k):
    n = U.shape[1]
    px, py, q = [], [], []
    for e in range(U.shape[0]):
        for i in range(n):
            for j in range(n):
                px.append(xn[e, i]); py.append(yn[e, j]); q.append(U[e, i, j, k])
    tri = Triangulation(np.array(px), np.array(py))
    return LinearTriInterpolator(tri, np.array(q))


hists = {}
for NX, tag, blurb, steady_lo in RUNS:
    f = f'{SC}/gartling_unsteady_nx{NX}_N6_dt0.1_nsub3_pz_stagnant_wm0.1_ws0.1.npz'
    d = np.load(f, allow_pickle=True)
    snaps, st = d['snaps'], d['snap_t']
    xn, yn = d['xnod'], d['ynod']
    hists[NX] = (d['hist'], steady_lo, blurb)
    MX, MY = np.meshgrid(GX, GY)
    fig, axs = plt.subplots(len(snaps), 1, figsize=(15.0, 1.95*len(snaps)))
    for ax, U, tt in zip(axs, snaps, st):
        ui = np.array(interp(U, xn, yn, 0)(MX, MY).filled(np.nan))
        vi = np.array(interp(U, xn, yn, 1)(MX, MY).filled(np.nan))
        ax.contourf(MX, MY, np.ma.masked_invalid(np.hypot(ui, vi)), levels=36,
                    cmap='viridis', vmin=0, vmax=1.5)
        rev = np.ma.masked_where(np.nan_to_num(ui) >= 0, np.ones_like(MX))
        ax.contourf(MX, MY, rev, levels=[.5, 1.5], colors=['red'], alpha=.25)
        ax.streamplot(GX, GY, np.nan_to_num(ui), np.nan_to_num(vi), density=2.5,
                      color='w', linewidth=.55, arrowsize=.55)
        ax.axvline(steady_lo, color='gold', lw=1.3, ls=':', zorder=6)
        ax.set_xlim(0, 17); ax.set_ylim(-0.5, 0.5)
        ax.set_ylabel(f't = {tt:g}', fontsize=10)
        ax.set_xticks([] if tt != st[-1] else [0, 2, 4, 6, 8, 10, 12, 14, 16])
    axs[-1].set_xlabel('x')
    fig.suptitle(f'Gartling BFS Re = 800 -- {NX}x4 grid, N = 6, from REST, '
                 f'w_mom = w_mass = 0.1, dt = 0.1.   {blurb}\n'
                 f'Red = reversed flow;  gold dotted = the steady solver\'s '
                 f'reattachment ({steady_lo:.3f}).', fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = f'figs/gartling_{tag}_nx{NX}_evolution.png'
    fig.savefig(out, dpi=115, bbox_inches='tight')
    print(out)

# -------- the quantitative contrast --------
fig, axs = plt.subplots(2, 1, figsize=(12.5, 8.0), sharex=True)
col = {11: 'tab:red', 18: 'tab:green'}
for NX in (11, 18):
    h, slo, blurb = hists[NX]
    axs[0].plot(h[:, 0], h[:, 2], color=col[NX], lw=1.3,
                label=f'{NX}x4  ({blurb.split(" --")[0]})')
    axs[1].plot(h[:, 0], h[:, 3], color=col[NX], lw=1.3, label=f'{NX}x4')
    axs[1].axhline(slo, color=col[NX], lw=1.0, ls=':')
axs[0].set_ylabel('max |v|'); axs[0].grid(alpha=.3); axs[0].legend(fontsize=9)
axs[0].set_title('Oscillation amplitude: the 11x4 grid sustains what the 18x4 grid damps',
                 fontsize=11)
axs[1].set_ylabel('lower-wall reattachment'); axs[1].set_xlabel('t')
axs[1].grid(alpha=.3); axs[1].legend(fontsize=9)
axs[1].set_title('Reattachment (dotted = each grid\'s own steady-solver value)', fontsize=11)
fig.suptitle('Chan & Mittal figs 5/6 reproduced: same physics, two grids, from rest.\n'
             'Envelope ratio (last 20 t.u. / first 20 t.u.): 11x4 = 0.430 (sustained), '
             '18x4 = 0.036 (decaying).', fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig('figs/gartling_fig56_history.png', dpi=125, bbox_inches='tight')
print('figs/gartling_fig56_history.png')
