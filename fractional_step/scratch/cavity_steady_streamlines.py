"""All four steady-form outcomes, beside the correct solution.  Cavity Re = 1000.

    uv run --quiet python scratch/cavity_steady_streamlines.py

Illustrates ARTIFICIAL_COMPRESSIBILITY.md sec 5.3.  Five converged-or-not fields
on the SAME 6x6 N=10 mesh, all from scratch/cavity_steady_ls.py except the
reference:

  steady (w_mass = 0), from rest,          line search ON  -> STALLED at 2^-25
  steady (w_mass = 0), from rest,          line search OFF -> never converges
  steady (w_mass = 0), from correct field, line search ON  -> STALLED at sweep 1
  steady (w_mass = 0), from correct field, line search OFF -> TRUE fixed point,
                                                              and still spurious
  time-accurate dt = 0.05, AC on                           -> correct

The point of the figure: only ONE of the four steady runs is a genuine fixed
point (|dU| = 0 exactly, no globalisation involved), and that one is still 10x
off Ghia while being indistinguishable from the correct solution by the
least-squares functional, by streamline topology, and by vortex position.  The
other three are solver failures of two different kinds, which look nothing like
each other and neither of which is a solution.

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

# file, title, verdict-key
CASES = [
    ('cavity_steadyls_on_rest.npz',
     'STEADY  from rest,  line search ON', 'stall'),
    ('cavity_steadyls_off_rest.npz',
     'STEADY  from rest,  NO line search', 'diverge'),
    ('cavity_steadyls_on_restart.npz',
     'STEADY  from the correct field,  line search ON', 'stall1'),
    ('cavity_steadyls_off_restart.npz',
     'STEADY  from the correct field,  NO line search', 'spurious'),
    ('cavity_ac_dt0.05_match.npz',
     'time-accurate  dt = 0.05,  AC on ($\\kappa_p$ = 30)', 'correct'),
]
VERDICT = {
    'stall':    ('NOT CONVERGED — line search stalled at $\\alpha=2^{-25}$',
                 'tab:orange'),
    'diverge':  ('NOT CONVERGED — 400 sweeps, $|dU|$ never below 30',
                 'tab:orange'),
    'stall1':   ('NOT CONVERGED — stalled on the very first sweep',
                 'tab:orange'),
    'spurious': ('SPURIOUS — a TRUE fixed point, $|dU|$ = 0 exactly',
                 'tab:red'),
    'correct':  ('correct', 'tab:green'),
}
GHIA_CENTRE = (0.5313, 0.5625)          # Ghia et al. 1982, Re = 1000
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


def primary_centre(fu, fv):
    """Interior stagnation point = primary vortex centre.  Searched over
    [0.25,0.85]^2: an unrestricted argmin locks onto the bottom-corner eddies,
    which are also stagnation points."""
    g = np.linspace(0.25, 0.85, 700)
    X, Y = np.meshgrid(g, g)
    s = np.hypot(np.array(fu(X, Y).filled(9e9)), np.array(fv(X, Y).filled(9e9)))
    k = np.unravel_index(np.argmin(s), s.shape)
    return X[k], Y[k]


gx = np.linspace(0, 1, 400); gy = np.linspace(0, 1, 400)
GX, GY = np.meshgrid(gx, gy)
fig = plt.figure(figsize=(16.5, 11.4))
gs = fig.add_gridspec(2, 6, hspace=0.30, wspace=0.55)
# Row 1: the three failures.  Row 2: the spurious fixed point and the correct
# solution, centred, because those two are the pair the reader must compare.
SLOTS = [gs[0, 0:2], gs[0, 2:4], gs[0, 4:6], gs[1, 1:3], gs[1, 3:5]]
rows = []
for slot, (f, title, key) in zip(SLOTS, CASES):
    ax = fig.add_subplot(slot)
    d = np.load(f'{SC}/{f}', allow_pickle=True)
    fu, fv = interp(d)
    ui = np.array(fu(GX, GY).filled(np.nan))
    vi = np.array(fv(GX, GY).filled(np.nan))
    ax.contourf(GX, GY, np.ma.masked_invalid(np.hypot(ui, vi)), levels=40,
                cmap='viridis', vmin=0, vmax=1.0)
    rev = np.ma.masked_where(np.nan_to_num(ui) >= 0, np.ones_like(GX))
    ax.contourf(GX, GY, rev, levels=[.5, 1.5], colors=['red'], alpha=.20)
    ax.streamplot(gx, gy, np.nan_to_num(ui), np.nan_to_num(vi), density=2.2,
                  color='w', linewidth=.65, arrowsize=.65)
    cx, cy = primary_centre(fu, fv)
    off = np.hypot(cx-GHIA_CENTRE[0], cy-GHIA_CENTRE[1])
    ax.plot(*GHIA_CENTRE, 'o', ms=12, mfc='none', mec='gold', mew=2.2,
            zorder=10, label="Ghia's centre")
    ax.plot(cx, cy, 'x', ms=10, color='red', mew=2.2, zorder=11,
            label="this run's")
    r = residuals(d['U'])
    note, col = VERDICT[key]
    ax.set_title(f'{title}\nRMS u vs Ghia = {float(d["rms"]):.2e}\n{note}',
                 fontsize=9.5, color=col)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    ax.set_xticks([0, .5, 1]); ax.set_yticks([0, .5, 1])
    ax.set_xlabel(f"rms mom {r['mom']:.3e}   rms div u {r['div']:.3e}\n"
                  f"vortex centre ({cx:.3f}, {cy:.3f}) — {off:.3f} from Ghia",
                  fontsize=8.5)
    if key == 'correct':
        ax.legend(fontsize=7.5, loc='lower left', framealpha=.85)
    rows.append((f, float(d['rms']), r, (cx, cy), off, str(d.get('status', ''))))

fig.suptitle('The steady form ($w_{mass}$ = 0) on the lid-driven cavity, '
             'Re = 1000, 6×6 elements N = 10 — ARTIFICIAL_COMPRESSIBILITY.md §5.3\n'
             'Top row: three solver FAILURES, two of them silently reported as '
             'converged.  Bottom row: the one genuine steady fixed point, and the '
             'correct answer.\n'
             'Red shading = reversed flow (u < 0).  Gold circle = Ghia\'s primary '
             'vortex centre, red cross = this run\'s.',
             fontsize=11.5)
fig.subplots_adjust(top=0.86, bottom=0.05, left=0.04, right=0.97)
fig.savefig('figs/cavity_steady_spurious_streamlines.png', dpi=120,
            bbox_inches='tight')
print('figs/cavity_steady_spurious_streamlines.png\n')
hdr = f"{'field':<36}{'RMS u':>11}{'rms mom':>11}{'rms div':>11}{'centre off':>12}"
print(hdr); print('-'*len(hdr))
for f, rms, r, c, off, stat in rows:
    print(f'{f:<36}{rms:>11.3e}{r["mom"]:>11.3e}{r["div"]:>11.3e}{off:>12.3f}')
