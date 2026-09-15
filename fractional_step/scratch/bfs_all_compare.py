"""Everything against everything: Fortran reference and Python, streamlines + u,v,p.

Five cases:
  FORT short / free   the Fortran reference on the truncated domain -- x_r/h = 4.35,
                      max|u| = 1.736.  BOTH WRONG (long reference: 8.154 and 1.500).
  FORT long / free    the reference solution.
  PY short / P+Z      our short domain with the admissible pair.
  PY long / P+Z       our long domain with the pair.
  PY long / free      our long domain, unconstrained outlet.

Pressure has no common datum -- the Fortran pins one node, our P+Z runs set p = 0
on their own outlet (at different x).  All five are re-referenced to the
INLET-PLANE MEAN, which every case has.
"""
import os, sys
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
sys.path.insert(0, SC)
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, LinearTriInterpolator
from lssem2d.lgl import diff_matrix, lgl_weights
from fsol import load_solution

P = '/Users/danielchan/Dropbox/F90_SEM/pmg_clean'
H = 0.5
STATIONS = [0.5, 1.0, 2.0, 2.4, 4.0, 8.0]


def pack(U, xn, yn, hy, lab, col, sty):
    U = U.copy()
    n = U.shape[1]
    wq = lgl_weights(n-1)
    xmin = xn.min()
    tot = a = 0.0
    for e in range(U.shape[0]):
        if abs(xn[e, 0]-xmin) < 1e-9:
            tot += np.sum(wq*U[e, 0, :, 2])*(hy[e]/2); a += hy[e]
    U[..., 2] -= tot/a                       # common datum: inlet-plane mean
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
    tri.set_mask((cx < 0) & (cy < 0.5))
    return dict(lab=lab, col=col, sty=sty, U=U, xn=xn, yn=yn, hy=hy, n=n,
                xmin=px.min(), xmax=px.max(),
                f=[LinearTriInterpolator(tri, np.array(q[k])) for k in range(3)])


C = []
for tag, sol, col, sty in (
        ('FORT short / free', 'run_chan389_short/chan389_short.dat', 'tab:orange',
         dict(ls='-', lw=2.4)),
        ('FORT long / free', 'run_chan389_long/chan389_long.dat', 'k',
         dict(ls='-', lw=2.0))):
    s = load_solution(f'{P}/{sol}')
    hy = np.array([s['ynod'][e, -1]-s['ynod'][e, 0] for e in range(s['nelem'])])
    C.append(pack(s['U'], s['xnod'], s['ynod'], hy, tag, col, sty))
for tag, f, col, sty in (
        ('PY short / P+Z', 'bfs_pz_state.npz', 'tab:blue',
         dict(ls='none', marker='o', ms=4.6, mfc='none', mew=1.3)),
        ('PY long / P+Z', 'bfs_long_pz.npz', 'tab:green', dict(ls='--', lw=1.9)),
        ('PY long / free', 'bfs_long_free.npz', 'tab:red', dict(ls=':', lw=1.9))):
    d = np.load(f'{SC}/{f}')
    C.append(pack(d['U'], d['xnod'], d['ynod'], d['hy'], tag, col, sty))


def reatt(c):
    n = c['n']; D = diff_matrix(n-1); xs, tw = [], []
    for e in range(c['U'].shape[0]):
        if c['yn'][e, 0] > 0.01 or c['xn'][e, 0] < -1e-9:
            continue
        for i in range(n):
            xs.append(c['xn'][e, i])
            tw.append(np.dot(D[0, :], c['U'][e, i, :, 0])*(2.0/c['hy'][e]))
    o = np.argsort(xs); xs, tw = np.array(xs)[o], np.array(tw)[o]
    for k in range(len(xs)-1):
        if tw[k] < 0 and tw[k+1] > 0 and xs[k] > 0.05:
            return xs[k]-tw[k]*(xs[k+1]-xs[k])/(tw[k+1]-tw[k])
    return float('nan')


# ---------------- streamlines ----------------
fig, axs = plt.subplots(len(C), 1, figsize=(14.0, 2.6*len(C)))
for ax, c in zip(axs, C):
    gx = np.linspace(c['xmin'], c['xmax'], 1100); gy = np.linspace(0, 1, 240)
    GX, GY = np.meshgrid(gx, gy)
    ui = np.array(c['f'][0](GX, GY).filled(np.nan))
    vi = np.array(c['f'][1](GX, GY).filled(np.nan))
    ax.contourf(GX, GY, np.ma.masked_invalid(np.hypot(ui, vi)), levels=40,
                cmap='viridis', vmin=0, vmax=1.75)
    rev = np.ma.masked_where(np.nan_to_num(ui) >= 0, np.ones_like(GX))
    ax.contourf(GX, GY, rev, levels=[.5, 1.5], colors=['red'], alpha=.28)
    ax.streamplot(gx, gy, np.nan_to_num(ui), np.nan_to_num(vi), density=2.4,
                  color='w', linewidth=.6, arrowsize=.65)
    ax.add_patch(plt.Rectangle((c['xmin'], 0), -c['xmin'], .5, fc='0.85', ec='k',
                               lw=1.1, zorder=5))
    xr = reatt(c)
    if np.isfinite(xr):
        ax.plot([xr], [0], 'r^', ms=11, zorder=7, clip_on=False)
    ax.axvline(c['xmax'], color='yellow', lw=3, zorder=6)
    ax.set_xlim(-1, 8.6); ax.set_ylim(0, 1); ax.set_ylabel('y')
    s = f"x_r/h = {xr/H:.3f}" if np.isfinite(xr) else "x_r: none in domain"
    ax.set_title(f"{c['lab']}   |   max|u| = {np.abs(c['U'][...,0]).max():.4f}   |   {s}",
                 fontsize=10)
axs[-1].set_xlabel('x')
fig.suptitle('BFS Re = 389 — streamlines.  Red = reversed flow, yellow = outlet, '
             '▲ = reattachment.\nFORT short/free gets x_r/h = 4.35 against the '
             'reference 8.154, and overshoots to max|u| = 1.736.', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(f'{SC}/../figs/bfs_all_streamlines.png', dpi=120, bbox_inches='tight')
print('figs/bfs_all_streamlines.png')

# ---------------- u, v, p profiles ----------------
yy = np.linspace(0.002, 0.998, 400)
NM = ['u  (axial)', 'v  (vertical)', 'p − p_inlet']
fig, axs = plt.subplots(3, len(STATIONS), figsize=(3.0*len(STATIONS), 10.4),
                        sharey=True)
for r in range(3):
    for k, x in enumerate(STATIONS):
        ax = axs[r, k]
        for c in C:
            if x > c['xmax']+1e-9:
                continue
            v = np.array(c['f'][r](np.full_like(yy, x), yy).filled(np.nan))
            if c['sty'].get('marker'):
                ax.plot(v[::16], yy[::16], color=c['col'], label=c['lab'], **c['sty'])
            else:
                ax.plot(v, yy, color=c['col'], label=c['lab'], **c['sty'])
        ax.axvline(0, color='k', lw=.7, ls=':')
        ax.axhline(0.5, color='0.6', lw=.7, ls=':')
        ax.grid(alpha=.3)
        if r == 0:
            ax.set_title(f'x = {x:g}  (x/h = {x/H:g})', fontsize=10)
        if r == 2:
            ax.set_xlabel('value')
        if k == 0:
            ax.set_ylabel(f'{NM[r]}\n\ny')
axs[0, 0].legend(fontsize=7.5, loc='upper left')
fig.suptitle('BFS Re = 389 — u, v and p at streamwise stations.  Short domains end '
             'at x = 2.5.\nPressure re-referenced to the inlet-plane mean.', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(f'{SC}/../figs/bfs_all_profiles.png', dpi=120, bbox_inches='tight')
print('figs/bfs_all_profiles.png')

# ---------------- numbers ----------------
ref = C[1]     # FORT long / free
for r, nm in ((0, 'u'), (1, 'v'), (2, 'p')):
    print(f"\n=== max |{nm} − FORT long| ===")
    print(f"{'x':>6}" + "".join(f"{c['lab']:>20}" for c in C if c is not ref))
    for x in STATIONS:
        a = np.array(ref['f'][r](np.full_like(yy, x), yy).filled(np.nan))
        row = ""
        for c in C:
            if c is ref:
                continue
            if x > c['xmax']+1e-9:
                row += f"{'--':>20}"; continue
            b = np.array(c['f'][r](np.full_like(yy, x), yy).filled(np.nan))
            m = np.isfinite(a) & np.isfinite(b)
            row += f"{np.abs(a[m]-b[m]).max():>20.3e}"
        print(f"{x:>6g}" + row)
