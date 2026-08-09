"""Pinned vs unpinned: streamlines and the outflow-plane pressure.

Both fields come from the run where Newton actually worked (start = w_mom 0.5
field, solve at w_mom 0.1, loose solve, p-MG), so the comparison is meaningful
-- restarting from its own converged field does 0 CG iterations and proves
nothing.

The shifted field (converged, p += 5, iterated with NO pin) is drawn on the
pressure panel to show the null mode: identical shape, arbitrary level.
"""
import os, sys
import numpy as np
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, LinearTriInterpolator
from lssem2d.lgl import diff_matrix

H = 0.5
PIN = 'bfsnp2_off_pin.npz'
NOPIN = 'bfsnp2_off_nopin.npz'
SHIFT = 'bfsnp2_shift_nopin.npz'


def load(f):
    d = np.load(f'{SC}/{f}')
    return d['U'], d['xnod'], d['ynod'], d['hy']


def nodes(U, xn, yn):
    ne, n = U.shape[0], U.shape[1]
    px, py, pu, pv = [], [], [], []
    for e in range(ne):
        for i in range(n):
            for j in range(n):
                px.append(xn[e, i]); py.append(yn[e, j])
                pu.append(U[e, i, j, 0]); pv.append(U[e, i, j, 1])
    return map(np.array, (px, py, pu, pv))


def reattach(U, xn, yn, hy):
    n = U.shape[1]; D = diff_matrix(n-1)
    xs, tw = [], []
    for e in range(U.shape[0]):
        if yn[e, 0] > 0.01 or xn[e, 0] < -1e-9:
            continue
        for i in range(n):
            xs.append(xn[e, i]); tw.append(np.dot(D[0, :], U[e, i, :, 0])*(2.0/hy[e]))
    o = np.argsort(xs); xs, tw = np.array(xs)[o], np.array(tw)[o]
    for k in range(len(xs)-1):
        if tw[k] < 0 and tw[k+1] > 0 and xs[k] > 0.05:
            return xs[k]-tw[k]*(xs[k+1]-xs[k])/(tw[k+1]-tw[k])
    return np.nan


def outlet(U, xn, yn):
    n = U.shape[1]; xmax = xn.max()
    ys, ps = [], []
    for e in range(U.shape[0]):
        if abs(xn[e, -1]-xmax) < 1e-9:
            for j in range(n):
                ys.append(yn[e, j]); ps.append(U[e, -1, j, 2])
    o = np.argsort(ys); ys, ps = np.array(ys)[o], np.array(ps)[o]
    k = np.concatenate(([True], np.diff(ys) > 1e-12))
    return ys[k], ps[k]


fig = plt.figure(figsize=(15.4, 7.0))
gs = fig.add_gridspec(2, 3, width_ratios=[2.3, 1.0, 0.85], hspace=.42, wspace=.30)

for row, (lab, f) in enumerate([('WITH pressure pin', PIN), ('NO pressure pin', NOPIN)]):
    U, xn, yn, hy = load(f)
    n = U.shape[1]
    px, py, pu, pv = nodes(U, xn, yn)
    tri = Triangulation(px, py)
    cx = px[tri.triangles].mean(axis=1); cy = py[tri.triangles].mean(axis=1)
    tri.set_mask((cx < 0) & (cy < 0.5))
    fu = LinearTriInterpolator(tri, pu); fv = LinearTriInterpolator(tri, pv)
    gx = np.linspace(px.min(), px.max(), 760); gy = np.linspace(0, 1, 200)
    GX, GY = np.meshgrid(gx, gy)
    ui = np.array(fu(GX, GY).filled(np.nan)); vi = np.array(fv(GX, GY).filled(np.nan))

    ax = fig.add_subplot(gs[row, 0])
    ax.contourf(GX, GY, np.ma.masked_invalid(np.hypot(ui, vi)), levels=40,
                cmap='viridis', alpha=.82, vmin=0, vmax=2.6)
    rev = np.ma.masked_where(np.nan_to_num(ui) >= 0, np.ones_like(GX))
    ax.contourf(GX, GY, rev, levels=[.5, 1.5], colors=['red'], alpha=.28)
    ax.streamplot(gx, gy, np.nan_to_num(ui), np.nan_to_num(vi), density=2.1,
                  color='w', linewidth=.6, arrowsize=.65)
    ax.add_patch(plt.Rectangle((px.min(), 0), -px.min(), .5, fc='0.85',
                               ec='k', lw=1.1, zorder=5))
    xr = reattach(U, xn, yn, hy)
    if np.isfinite(xr):
        ax.plot([xr], [0], 'r^', ms=9, zorder=7, clip_on=False)
    ax.axvline(xn.max(), color='yellow', lw=3, zorder=6)
    if row == 0:            # mark where the pin node sits
        ax.plot([xn.max()], [0.0], marker='o', ms=9, mfc='none', mec='yellow',
                mew=2.4, zorder=8, clip_on=False)
    OUT = [e for e in range(U.shape[0]) if abs(xn[e, -1]-xn.max()) < 1e-9]
    ue = np.array([U[e, -1, j, 0] for e in OUT for j in range(n)])
    ax.set_title(f"{lab}    x_r/h = {xr/H:.3f}    max|u| = {np.abs(U[...,0]).max():.3f}"
                 f"    exit rev = {100*np.mean(ue<0):.0f}%", fontsize=9.5)
    ax.set_xlim(px.min(), px.max()); ax.set_ylim(0, 1)
    ax.set_ylabel('y', fontsize=8); ax.tick_params(labelsize=7)
    if row == 1:
        ax.set_xlabel('x', fontsize=9)

# ---- exit-plane pressure ----------------------------------------------------
axp = fig.add_subplot(gs[:, 1])
Ua, xa, ya, _ = load(PIN); Ub, xb, yb, _ = load(NOPIN); Us, xs_, ys_, _ = load(SHIFT)
y1, p1 = outlet(Ua, xa, ya)
y2, p2 = outlet(Ub, xb, yb)
y3, p3 = outlet(Us, xs_, ys_)
axp.plot(p1, y1, '-', color='tab:blue', lw=2.6, label=f'with pin   spread {p1.max()-p1.min():.4f}')
axp.plot(p2, y2, '--', color='tab:red', lw=1.8, label=f'no pin      spread {p2.max()-p2.min():.4f}')
axp.plot(p3, y3, ':', color='0.45', lw=1.8,
         label=f'converged, p += 5\nno pin   spread {p3.max()-p3.min():.4f}')
axp.set_xlabel('pressure on the outflow plane'); axp.set_ylabel('y')
axp.set_title('exit-plane pressure\npinned and unpinned lie on top of each other', fontsize=9.5)
axp.grid(alpha=.3); axp.set_ylim(0, 1); axp.legend(fontsize=7, loc='lower right')

# ---- the difference ---------------------------------------------------------
axd = fig.add_subplot(gs[:, 2])
axd.plot(p2-p1, y1, '-', color='tab:purple', lw=2.0)
axd.axvline(0, color='k', lw=.8)
axd.axvline((p2-p1).mean(), color='tab:orange', ls='--', lw=1.4,
            label=f'mean {np.mean(p2-p1):+.2e}')
axd.set_xlabel('p(no pin) - p(pin)')
axd.set_title(f'difference, spread {np.ptp(p2-p1):.2e}\n'
              f'(a pure null-mode shift would\nbe a vertical line)', fontsize=9)
axd.grid(alpha=.3); axd.set_ylim(0, 1); axd.legend(fontsize=7, loc='lower right')
axd.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

fig.suptitle('BFS Chan Re=389, SHORT domain, steady form w_mom=0.1 (w_mass=0), p-MG, loose solve — removing the pressure pin.\n'
             'Start = the w_mom 0.5 field so CG does real work (901 vs 902 iterations).   '
             'Yellow line = free outflow, circle = pin node.', fontsize=10.5, y=1.035)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = f'{SC}/nopin_streamlines.png'
fig.savefig(out, dpi=145, bbox_inches='tight')
print('saved', out)
print(f"exit-plane p:  pin spread {p1.max()-p1.min():.6f}   nopin spread {p2.max()-p2.min():.6f}")
print(f"difference  :  mean {np.mean(p2-p1):+.3e}   spread {np.ptp(p2-p1):.3e}")
print(f"shifted     :  mean p on exit plane {p3.mean():+.4f}  vs pinned {p1.mean():+.4f}")
