"""Full BFS comparison: u, v, p profiles + pressure contours + streamlines.

From saved states only -- no solving.

PRESSURE DATUM.  The three cases do not share one: short/P+Z and long/P+Z each
impose p = 0 on their own outlet (at x = 2.5 and x = 8.5 respectively), while
long/free is pinned at the inlet corner.  Comparing raw p would compare three
different constants.  Everything below is re-referenced to the INLET-PLANE MEAN,
which exists in all three.

Line style: LONG/P+Z solid (the reference, converged, full domain); the other two
with open symbols, since at several stations the curves coincide to plotting
accuracy and dashes hide underneath each other.
"""
import os, sys
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, LinearTriInterpolator
from lssem2d.lgl import diff_matrix, lgl_weights

H = 0.5
STATIONS = [0.5, 1.0, 2.0, 2.4, 4.0, 8.0]
SPEC = [('bfs_long_pz.npz',  'LONG / P+Z',  'tab:green', dict(ls='-',  lw=2.2, marker=None)),
        ('bfs_pz_state.npz', 'SHORT / P+Z', 'tab:blue',  dict(ls='none', marker='o', ms=4.5,
                                                             mfc='none', mew=1.3)),
        ('bfs_long_free.npz','LONG / free', 'tab:red',   dict(ls='none', marker='s', ms=4.0,
                                                             mfc='none', mew=1.1))]


def load(f, lab, col, sty):
    d = np.load(f'{SC}/{f}')
    U, xn, yn, hy = d['U'].copy(), d['xnod'], d['ynod'], d['hy']
    n = U.shape[1]
    # re-reference pressure to the inlet-plane mean (common to all three)
    wq = lgl_weights(n-1); xmin = xn.min()
    tot = a = 0.0
    for e in range(U.shape[0]):
        if abs(xn[e, 0]-xmin) < 1e-9:
            tot += np.sum(wq*U[e, 0, :, 2])*(hy[e]/2); a += hy[e]
    U[..., 2] -= tot/a
    px, py, q = [], [], [[], [], []]
    for e in range(U.shape[0]):
        for i in range(n):
            for j in range(n):
                px.append(xn[e, i]); py.append(yn[e, j])
                for k in range(3):
                    q[k].append(U[e, i, j, k])
    px, py = np.array(px), np.array(py)
    tri = Triangulation(px, py)
    cx = px[tri.triangles].mean(axis=1); cy = py[tri.triangles].mean(axis=1)
    tri.set_mask((cx < 0) & (cy < 0.5))
    return dict(lab=lab, col=col, sty=sty, U=U, xn=xn, yn=yn, hy=hy, n=n,
                xmin=px.min(), xmax=px.max(), tri=tri,
                f=[LinearTriInterpolator(tri, np.array(q[k])) for k in range(3)],
                status=str(d['status']) if 'status' in d.files else 'conv')


C = [load(*s) for s in SPEC]
for c in C:
    print(f"  {c['lab']:>12}: {c['status']:>8}  x to {c['xmax']:.1f}")

yy = np.linspace(0.002, 0.998, 400)
NAMES = ['u  (axial)', 'v  (vertical)', 'p  (rel. to inlet mean)']

# ---------- figure 1: u, v, p profiles ----------
fig, axs = plt.subplots(3, len(STATIONS), figsize=(3.0*len(STATIONS), 10.2),
                        sharey=True)
for r in range(3):
    for k, x in enumerate(STATIONS):
        ax = axs[r, k]
        for c in C:
            if x > c['xmax']+1e-9:
                continue
            v = np.array(c['f'][r](np.full_like(yy, x), yy).filled(np.nan))
            sty = dict(c['sty'])
            if sty.get('marker'):
                ax.plot(v[::14], yy[::14], color=c['col'], label=c['lab'], **sty)
            else:
                ax.plot(v, yy, color=c['col'], label=c['lab'], **sty)
        ax.axvline(0, color='k', lw=.7, ls=':')
        ax.axhline(0.5, color='0.6', lw=.7, ls=':')
        ax.grid(alpha=.3)
        if r == 0:
            ax.set_title(f'x = {x:g}   (x/h = {x/H:g})', fontsize=10)
        if r == 2:
            ax.set_xlabel('value')
        if k == 0:
            ax.set_ylabel(f'{NAMES[r]}\n\ny')
axs[0, 0].legend(fontsize=8, loc='upper left')
fig.suptitle('BFS Re = 389 — u, v and p profiles.  Short domain ends at x = 2.5;  '
             'y = 0.5 is the step height.\nPressure re-referenced to the inlet-plane mean '
             '(the three cases impose different datums).', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(f'{SC}/../figs/bfs_cmp_profiles.png', dpi=120, bbox_inches='tight')
print('figs/bfs_cmp_profiles.png')

# ---------- figure 2: pressure contours ----------
fig, axs = plt.subplots(3, 1, figsize=(14.0, 9.6))
lv = np.linspace(-0.12, 0.06, 41)
for ax, c in zip(axs, C):
    gx = np.linspace(c['xmin'], c['xmax'], 1000); gy = np.linspace(0, 1, 220)
    GX, GY = np.meshgrid(gx, gy)
    pi = np.array(c['f'][2](GX, GY).filled(np.nan))
    cf = ax.contourf(GX, GY, np.ma.masked_invalid(pi), levels=lv, cmap='coolwarm',
                     extend='both')
    ax.contour(GX, GY, np.ma.masked_invalid(pi), levels=lv[::3], colors='k',
               linewidths=.4, alpha=.5)
    ax.add_patch(plt.Rectangle((c['xmin'], 0), -c['xmin'], .5, fc='0.85', ec='k',
                               lw=1.1, zorder=5))
    ax.axvline(c['xmax'], color='yellow', lw=3, zorder=6)
    ax.set_xlim(-1, 8.6); ax.set_ylim(0, 1); ax.set_ylabel('y')
    ax.set_title(f"{c['lab']}  ({c['status']})", fontsize=10)
    plt.colorbar(cf, ax=ax, pad=.01, fraction=.023, label='p − p_inlet')
axs[-1].set_xlabel('x')
fig.suptitle('Pressure field, common datum and common colour scale.  '
             'Yellow = that case\'s outlet plane.', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(f'{SC}/../figs/bfs_cmp_pressure.png', dpi=120, bbox_inches='tight')
print('figs/bfs_cmp_pressure.png')

# ---------- figure 3: streamlines ----------
def reattach(c):
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


fig, axs = plt.subplots(3, 1, figsize=(14.0, 9.6))
for ax, c in zip(axs, C):
    gx = np.linspace(c['xmin'], c['xmax'], 1100); gy = np.linspace(0, 1, 240)
    GX, GY = np.meshgrid(gx, gy)
    ui = np.array(c['f'][0](GX, GY).filled(np.nan))
    vi = np.array(c['f'][1](GX, GY).filled(np.nan))
    ax.contourf(GX, GY, np.ma.masked_invalid(np.hypot(ui, vi)), levels=40,
                cmap='viridis', vmin=0, vmax=1.55)
    rev = np.ma.masked_where(np.nan_to_num(ui) >= 0, np.ones_like(GX))
    ax.contourf(GX, GY, rev, levels=[.5, 1.5], colors=['red'], alpha=.28)
    ax.streamplot(gx, gy, np.nan_to_num(ui), np.nan_to_num(vi), density=2.6,
                  color='w', linewidth=.65, arrowsize=.7)
    ax.add_patch(plt.Rectangle((c['xmin'], 0), -c['xmin'], .5, fc='0.85', ec='k',
                               lw=1.1, zorder=5))
    xr = reattach(c)
    if np.isfinite(xr):
        ax.plot([xr], [0], 'r^', ms=11, zorder=7, clip_on=False)
    ax.axvline(c['xmax'], color='yellow', lw=3, zorder=6)
    ax.set_xlim(-1, 8.6); ax.set_ylim(0, 1); ax.set_ylabel('y')
    s = f"x_r = {xr:.4f}  (x_r/h = {xr/H:.3f})" if np.isfinite(xr) else "x_r: none in domain"
    ax.set_title(f"{c['lab']}  ({c['status']})   |   {s}", fontsize=10)
axs[-1].set_xlabel('x')
fig.suptitle('Streamlines over |u|.  Red = reversed flow, yellow = outlet, '
             '▲ = reattachment.  Common speed scale.', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(f'{SC}/../figs/bfs_cmp_streamlines.png', dpi=120, bbox_inches='tight')
print('figs/bfs_cmp_streamlines.png')

# ---------- numbers for v and p ----------
for r, nm in ((1, 'v'), (2, 'p')):
    print(f"\n=== max |{nm}| difference vs LONG/P+Z ===")
    print(f"{'x':>6}{'x/h':>7}{'SHORT/P+Z':>14}{'LONG/free':>14}")
    for x in STATIONS:
        ref = np.array(C[0]['f'][r](np.full_like(yy, x), yy).filled(np.nan))
        row = []
        for c in C[1:]:
            if x > c['xmax']+1e-9:
                row.append('     --      '); continue
            v = np.array(c['f'][r](np.full_like(yy, x), yy).filled(np.nan))
            k = np.isfinite(ref) & np.isfinite(v)
            row.append(f"{np.abs(ref[k]-v[k]).max():>14.3e}")
        print(f"{x:>6g}{x/H:>7g}" + "".join(row))
