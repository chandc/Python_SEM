"""Streamlines of the two spurious steady-form states vs the correct solution.

    uv run --quiet python scratch/cavity_steady_streamlines.py

Illustrates ARTIFICIAL_COMPRESSIBILITY.md sec 5.3.  Three converged fields on the
SAME 6x6 N=10 mesh, Re = 1000:

  * steady form (w_mass = 0) from rest             -> spurious, RMS u 2.52e-01
  * steady form (w_mass = 0) from the correct field -> spurious, RMS u 1.28e-01
  * time-accurate dt = 0.05, AC on                  -> correct,  RMS u 1.57e-02

The point of the figure is that the first two are NOT small perturbations of the
third -- they are different flows -- yet the least-squares functional cannot tell
them apart (rms momentum 2.07e-01 / 2.00e-01), because the lid corner
singularities dominate the domain integral.

    figs/cavity_steady_spurious_streamlines.png
"""
import os, sys
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, LinearTriInterpolator
import lssem2d
lssem2d.set_backend('numpy')
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix, lgl_weights

CASES = [('cavity_ac_dt1_off_wm0.npz',
          'STEADY  $w_{mass}$ = 0,  from rest', 'spurious'),
         ('cavity_ac_dt1_off_wm0_restart.npz',
          'STEADY  $w_{mass}$ = 0,  from the correct field', 'spurious'),
         ('cavity_ac_dt0.05_match.npz',
          'time-accurate  dt = 0.05,  AC on ($\\kappa_p$ = 30)', 'correct')]
RE, EX, N = 1000.0, 6, 10
nu = 1.0/RE
_m = build_channel(1.0, 1.0, EX, EX, N, bcs=(1, 1, 1, 2))
_D, _w = diff_matrix(N), lgl_weights(N)


def residuals(U):
    """rms of each row of the VVP least-squares functional, over the domain."""
    J = dict(mom=0.0, div=0.0, vort=0.0); area = 0.0
    for e in range(_m.nelem):
        u, v, p, om = (U[e, :, :, k] for k in range(4))
        fx, fy = 2.0/_m.hx[e], 2.0/_m.hy[e]
        ux, uy = (_D @ u)*fx, (u @ _D.T)*fy
        vx, vy = (_D @ v)*fx, (v @ _D.T)*fy
        px, py = (_D @ p)*fx, (p @ _D.T)*fy
        omx, omy = (_D @ om)*fx, (om @ _D.T)*fy
        wq = np.outer(_w, _w)*0.25*_m.hx[e]*_m.hy[e]
        J['mom'] += np.sum(((u*ux+v*uy+px+nu*omy)**2 +
                            (u*vx+v*vy+py-nu*omx)**2)*wq)
        J['div'] += np.sum((ux+vy)**2*wq)
        J['vort'] += np.sum((om+uy-vx)**2*wq)
        area += wq.sum()
    return {k: np.sqrt(val/area) for k, val in J.items()}


def interp(d):
    U, xn, yn = d['U'], d['xnod'], d['ynod']
    n = U.shape[1]
    px, py, q = [], [], [[], []]
    for e in range(U.shape[0]):
        for i in range(n):
            for j in range(n):
                px.append(xn[e, i]); py.append(yn[e, j])
                q[0].append(U[e, i, j, 0]); q[1].append(U[e, i, j, 1])
    tri = Triangulation(np.array(px), np.array(py))
    return [LinearTriInterpolator(tri, np.array(q[k])) for k in (0, 1)]


GHIA_CENTRE = (0.5313, 0.5625)          # Ghia et al. 1982, Re = 1000


def primary_centre(fu, fv):
    """Interior stagnation point = primary vortex centre.  Searched over
    [0.25,0.85]^2: an unrestricted argmin locks onto the bottom-corner eddies,
    which are also stagnation points."""
    g = np.linspace(0.25, 0.85, 700)
    X, Y = np.meshgrid(g, g)
    s = np.hypot(np.array(fu(X, Y).filled(9e9)), np.array(fv(X, Y).filled(9e9)))
    k = np.unravel_index(np.argmin(s), s.shape)
    return X[k], Y[k]


gx = np.linspace(0, 1, 420); gy = np.linspace(0, 1, 420)
GX, GY = np.meshgrid(gx, gy)
fig, axs = plt.subplots(1, 3, figsize=(16.5, 6.0))
for ax, (f, title, kind) in zip(axs, CASES):
    d = np.load(f'{SC}/{f}', allow_pickle=True)
    fu, fv = interp(d)
    ui = np.array(fu(GX, GY).filled(np.nan))
    vi = np.array(fv(GX, GY).filled(np.nan))
    spd = np.hypot(ui, vi)
    ax.contourf(GX, GY, np.ma.masked_invalid(spd), levels=40, cmap='viridis',
                vmin=0, vmax=1.0)
    # Reversed-flow shading makes the spurious layering obvious: the from-rest
    # state has bands of u < 0 stacked under the lid where there should be one
    # co-moving layer.
    rev = np.ma.masked_where(np.nan_to_num(ui) >= 0, np.ones_like(GX))
    ax.contourf(GX, GY, rev, levels=[.5, 1.5], colors=['red'], alpha=.20)
    ax.streamplot(gx, gy, np.nan_to_num(ui), np.nan_to_num(vi), density=2.4,
                  color='w', linewidth=.7, arrowsize=.7)
    cx, cy = primary_centre(fu, fv)
    off = np.hypot(cx-GHIA_CENTRE[0], cy-GHIA_CENTRE[1])
    ax.plot(*GHIA_CENTRE, 'o', ms=13, mfc='none', mec='gold', mew=2.4,
            zorder=10, label='Ghia centre')
    ax.plot(cx, cy, 'x', ms=11, color='red', mew=2.4, zorder=11,
            label='this centre')
    r = residuals(d['U'])
    ax.set_title(f'{title}\nRMS u vs Ghia = {float(d["rms"]):.2e}   '
                 f'({"SPURIOUS" if kind == "spurious" else "correct"})',
                 fontsize=10.5,
                 color='tab:red' if kind == 'spurious' else 'tab:green')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    ax.set_xlabel(f"rms momentum {r['mom']:.3e}   rms div u {r['div']:.3e}\n"
                  f"vortex centre ({cx:.3f}, {cy:.3f}) — {off:.3f} from Ghia",
                  fontsize=9)
    if kind == 'correct':
        ax.legend(fontsize=8, loc='lower left', framealpha=.85)
    ax.set_xticks([0, .5, 1]); ax.set_yticks([0, .5, 1])

fig.suptitle('The steady form converges to spurious states — lid-driven cavity '
             'Re = 1000, 6x6 elements N = 10 (ARTIFICIAL_COMPRESSIBILITY.md '
             '§5.3)\n'
             'Red shading = reversed flow (u < 0).  Gold circle = Ghia\'s primary '
             'vortex centre, red cross = this run\'s.\n'
             'All three are converged fixed points on the same mesh, and the '
             'least-squares functional (below each panel) cannot tell them apart.',
             fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig('figs/cavity_steady_spurious_streamlines.png', dpi=125,
            bbox_inches='tight')
print('figs/cavity_steady_spurious_streamlines.png\n')
for f, title, kind in CASES:
    d = np.load(f'{SC}/{f}', allow_pickle=True)
    r = residuals(d['U'])
    print(f'{f:<38} rms_u {float(d["rms"]):.3e}  mom {r["mom"]:.3e}  '
          f'div {r["div"]:.3e}  vort {r["vort"]:.3e}  [{kind}]')
