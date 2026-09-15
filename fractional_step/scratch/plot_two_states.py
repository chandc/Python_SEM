"""Two near-degenerate converged states of the SAME short-domain problem.

Seeded with the long-domain solution and solved WITH the non-monotone line
search, the short domain keeps the physically correct field (max|u| = 1.513,
the inlet peak).  Seeded from its own history it converges instead to a state
with max|u| = 2.494 and an exit pressure spread 17x larger.  The two differ by
1.1% in the least-squares functional.

Rows: the IC, the two line-search results, and the corrupted state.
Right: exit-plane pressure, de-meaned (no pin, so the level is arbitrary),
       on a shared axis and zoomed on the three good ones.
"""
import os, sys
import numpy as np
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, LinearTriInterpolator

H = 0.5
CASES = [
    ('IC: long-domain solution interpolated (the target)',
     'bfsint2_IC.npz',           'tab:green',  '-',  None),
    ('solved from linear IC + line search — CONVERGED, 9 it',
     'bfsint2_linearIC_ls.npz',  'tab:olive',  '--', 3.7327e-05),
    ('solved from spectral IC + line search — WALL at 24 it (still moving)',
     'bfsint2_spectralIC_ls.npz', 'tab:cyan',  '-.', 4.4761e-05),
    ("the short domain's own converged state — CONVERGED, 4 it",
     'bfsnp2_off_nopin.npz',     'tab:red',    ':',  3.6916e-05),
]


def load(f):
    d = np.load(f'{SC}/{f}')
    return d['U'], d['xnod'], d['ynod']


def nodes(U, xn, yn):
    ne, n = U.shape[0], U.shape[1]
    px, py, pu, pv = [], [], [], []
    for e in range(ne):
        for i in range(n):
            for j in range(n):
                px.append(xn[e, i]); py.append(yn[e, j])
                pu.append(U[e, i, j, 0]); pv.append(U[e, i, j, 1])
    return map(np.array, (px, py, pu, pv))


def outlet(U, xn, yn):
    n = U.shape[1]; xmax = xn.max()
    ys, ps, us = [], [], []
    for e in range(U.shape[0]):
        if abs(xn[e, -1]-xmax) < 1e-9:
            for j in range(n):
                ys.append(yn[e, j]); ps.append(U[e, -1, j, 2]); us.append(U[e, -1, j, 0])
    o = np.argsort(ys)
    ys, ps, us = np.array(ys)[o], np.array(ps)[o], np.array(us)[o]
    k = np.concatenate(([True], np.diff(ys) > 1e-12))
    return ys[k], ps[k], us[k]


fig = plt.figure(figsize=(16.2, 9.6))
gs = fig.add_gridspec(4, 3, width_ratios=[2.45, 0.95, 0.95], hspace=.55, wspace=.32)

for row, (lab, f, c, lsty, J) in enumerate(CASES):
    U, xn, yn = load(f)
    n = U.shape[1]
    px, py, pu, pv = nodes(U, xn, yn)
    tri = Triangulation(px, py)
    cx = px[tri.triangles].mean(axis=1); cy = py[tri.triangles].mean(axis=1)
    tri.set_mask((cx < 0) & (cy < 0.5))
    fu = LinearTriInterpolator(tri, pu); fv = LinearTriInterpolator(tri, pv)
    gx = np.linspace(px.min(), px.max(), 780); gy = np.linspace(0, 1, 200)
    GX, GY = np.meshgrid(gx, gy)
    ui = np.array(fu(GX, GY).filled(np.nan)); vi = np.array(fv(GX, GY).filled(np.nan))

    ax = fig.add_subplot(gs[row, 0])
    ax.contourf(GX, GY, np.ma.masked_invalid(np.hypot(ui, vi)), levels=40,
                cmap='viridis', alpha=.82, vmin=0, vmax=2.5)
    rev = np.ma.masked_where(np.nan_to_num(ui) >= 0, np.ones_like(GX))
    ax.contourf(GX, GY, rev, levels=[.5, 1.5], colors=['red'], alpha=.28)
    ax.streamplot(gx, gy, np.nan_to_num(ui), np.nan_to_num(vi), density=2.1,
                  color='w', linewidth=.6, arrowsize=.65)
    ax.add_patch(plt.Rectangle((px.min(), 0), -px.min(), .5, fc='0.85',
                               ec='k', lw=1.1, zorder=5))
    ax.axvline(xn.max(), color='yellow', lw=3, zorder=6)
    y, p, ue = outlet(U, xn, yn)
    t = f'{lab}\nmax|u| = {np.abs(U[...,0]).max():.3f}    exit p spread = {p.max()-p.min():.3f}'
    if J is not None:
        t += f'    J = {J:.4e}'
    ax.set_title(t, fontsize=9)
    ax.set_xlim(px.min(), px.max()); ax.set_ylim(0, 1)
    ax.set_ylabel('y', fontsize=8); ax.tick_params(labelsize=7)
    if row == 3:
        ax.set_xlabel('x', fontsize=9)

axb = fig.add_subplot(gs[:, 1])
axz = fig.add_subplot(gs[:, 2])
for lab, f, c, lsty, J in CASES:
    U, xn, yn = load(f)
    y, p, ue = outlet(U, xn, yn)
    tag = lab.split('—')[0].split(':')[-1].strip()
    axb.plot(p-p.mean(), y, lsty, color=c, lw=2.2,
             label=f'{tag}\n  spread {p.max()-p.min():.3f}')
    if 'own converged' not in lab:
        axz.plot(p-p.mean(), y, lsty, color=c, lw=2.2,
                 label=f'spread {p.max()-p.min():.3f}')

axb.set_xlabel('p - mean(p) at the exit'); axb.set_ylabel('y')
axb.set_title('exit-plane pressure, all four\n(shared axis)', fontsize=9.5)
axb.grid(alpha=.3); axb.set_ylim(0, 1); axb.legend(fontsize=6.5, loc='lower right')

axz.set_xlabel('p - mean(p) at the exit'); axz.set_ylabel('y')
axz.set_title('the three physical ones, zoomed\n(corrupted state omitted)', fontsize=9.5)
axz.grid(alpha=.3); axz.set_ylim(0, 1); axz.legend(fontsize=7.5, loc='lower right')

fig.suptitle('BFS Chan Re=389, SHORT domain, w_mom = 0.1 (w_mass = 0), p-MG, loose solve, no pin — '
             'TWO near-degenerate converged states.\n'
             'Seeded from the long domain WITH a line search the solver keeps max|u| = 1.513 (the physical inlet peak); '
             'seeded from its own history it reaches 2.494.\nThe two differ by 1.1% in J.  Red = reversed flow.',
             fontsize=10.5, y=1.035)
fig.tight_layout(rect=[0, 0, 1, 0.955])
out = f'{SC}/two_states.png'
fig.savefig(out, dpi=145, bbox_inches='tight')
print('saved', out)
for lab, f, c, lsty, J in CASES:
    U, xn, yn = load(f)
    y, p, ue = outlet(U, xn, yn)
    print(f"{lab.split('—')[0][:46]:<48} max|u| {np.abs(U[...,0]).max():.3f}"
          f"  p spread {p.max()-p.min():.4f}  rev {100*np.mean(ue<0):.1f}%")
