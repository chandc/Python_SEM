"""Full BFS comparison from saved states -- no solving.

  bfs_pz_state.npz   SHORT domain, P+Z    converged (|dU| = 0)
  bfs_long_pz.npz    LONG  domain, P+Z    converged (|dU| = 0)
  bfs_long_free.npz  LONG  domain, free   WALL-CAPPED at |dU| = 6.1e-05

Three questions:

  A  TRUNCATION.  short/P+Z vs long/P+Z over the overlap x <= 2.5.  Both are
     bit-exact fixed points, so any difference is the artificial boundary, not
     convergence error.  This is the clean measurement.

  B  BC EFFECT on a domain where BOTH work.  long/P+Z vs long/free.  On
     Poiseuille, free outflow converged AND matched dp to 7e-09 while carrying
     730x the whole-field error, so agreement in x_r says little.  Limited here
     by free being wall-capped: differences below ~1e-03 are not resolvable.

  C  REATTACHMENT for each, against the repo gate x_r/h = 8.0 +/- 0.3 (Armaly)
     and the Fortran band 8.135-8.250.
"""
import os, sys
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, LinearTriInterpolator
from lssem2d.lgl import diff_matrix

H = 0.5
CASES = [('bfs_pz_state.npz', 'SHORT / P+Z', 'tab:blue', '-'),
         ('bfs_long_pz.npz', 'LONG / P+Z', 'tab:green', '--'),
         ('bfs_long_free.npz', 'LONG / free', 'tab:red', ':')]
STATIONS = [0.25, 0.5, 1.0, 1.5, 2.0, 2.4, 4.0, 8.0]


def load(f):
    d = np.load(f'{SC}/{f}')
    U, xn, yn, hy = d['U'], d['xnod'], d['ynod'], d['hy']
    n = U.shape[1]
    px, py, pu, pv, pp = [], [], [], [], []
    for e in range(U.shape[0]):
        for i in range(n):
            for j in range(n):
                px.append(xn[e, i]); py.append(yn[e, j])
                pu.append(U[e, i, j, 0]); pv.append(U[e, i, j, 1]); pp.append(U[e, i, j, 2])
    px, py, pu, pv, pp = map(np.array, (px, py, pu, pv, pp))
    tri = Triangulation(px, py)
    cx = px[tri.triangles].mean(axis=1); cy = py[tri.triangles].mean(axis=1)
    tri.set_mask((cx < 0) & (cy < 0.5))
    return dict(U=U, xn=xn, yn=yn, hy=hy, n=n, xmax=px.max(),
                fu=LinearTriInterpolator(tri, pu),
                fv=LinearTriInterpolator(tri, pv),
                fp=LinearTriInterpolator(tri, pp),
                status=str(d['status']) if 'status' in d.files else 'conv')


def reattach(U, xn, yn, hy):
    n = U.shape[1]; D = diff_matrix(n-1)
    xs, tw = [], []
    for e in range(U.shape[0]):
        if yn[e, 0] > 0.01 or xn[e, 0] < -1e-9:
            continue
        for i in range(n):
            xs.append(xn[e, i])
            tw.append(np.dot(D[0, :], U[e, i, :, 0])*(2.0/hy[e]))
    o = np.argsort(xs); xs, tw = np.array(xs)[o], np.array(tw)[o]
    for k in range(len(xs)-1):
        if tw[k] < 0 and tw[k+1] > 0 and xs[k] > 0.05:
            return xs[k]-tw[k]*(xs[k+1]-xs[k])/(tw[k+1]-tw[k])
    return float('nan')


C = {}
for f, lab, col, ls in CASES:
    C[lab] = load(f); C[lab].update(col=col, ls=ls)
    print(f"  {lab:>12}: {C[lab]['status']:>8}, x to {C[lab]['xmax']:.1f}")

print("\n=== C. REATTACHMENT  (gate: x_r/h = 8.0 +/- 0.3;  Fortran 8.135-8.250) ===")
for lab in C:
    c = C[lab]
    xr = reattach(c['U'], c['xn'], c['yn'], c['hy'])
    s = f"{xr:.4f}   x_r/h = {xr/H:.3f}" if np.isfinite(xr) else "none in domain"
    print(f"  {lab:>12}: {s}")

yy = np.linspace(0.002, 0.998, 500)


def prof(c, x):
    return np.array(c['fu'](np.full_like(yy, x), yy).filled(np.nan))


print("\n=== A. TRUNCATION: short/P+Z vs long/P+Z, both converged ===")
print(f"{'x':>6}{'x/h':>7}{'max|du|':>12}{'rms|du|':>12}{'rel to u_max':>14}")
for x in [s for s in STATIONS if s <= 2.45]:
    a, b = prof(C['SHORT / P+Z'], x), prof(C['LONG / P+Z'], x)
    k = np.isfinite(a) & np.isfinite(b)
    du = np.abs(a[k]-b[k])
    print(f"{x:>6g}{x/H:>7g}{du.max():>12.3e}{np.sqrt((du**2).mean()):>12.3e}"
          f"{du.max()/np.abs(b[k]).max()*100:>13.3f}%")

print("\n=== B. BC EFFECT on the LONG domain: P+Z vs free ===")
print(f"{'x':>6}{'x/h':>7}{'max|du|':>12}{'rms|du|':>12}{'rel to u_max':>14}")
for x in STATIONS:
    a, b = prof(C['LONG / P+Z'], x), prof(C['LONG / free'], x)
    k = np.isfinite(a) & np.isfinite(b)
    if k.sum() == 0:
        continue
    du = np.abs(a[k]-b[k])
    print(f"{x:>6g}{x/H:>7g}{du.max():>12.3e}{np.sqrt((du**2).mean()):>12.3e}"
          f"{du.max()/np.abs(a[k]).max()*100:>13.3f}%")
print("  (long/free is WALL-CAPPED at |dU| = 6.1e-05 -- differences below ~1e-03")
print("   are convergence error, not boundary error)")

# ---- figure
fig, axs = plt.subplots(2, 4, figsize=(15.5, 8.0), sharey=True)
axs = axs.ravel()
for k, x in enumerate(STATIONS):
    ax = axs[k]
    for lab in C:
        if x > C[lab]['xmax']+1e-9:
            continue
        ax.plot(prof(C[lab], x), yy, C[lab]['ls'], color=C[lab]['col'], lw=1.9,
                label=lab)
    ax.axvline(0, color='k', lw=.8, ls=':')
    ax.axhline(0.5, color='0.6', lw=.8, ls=':')
    ax.set_title(f'x = {x:g}  (x/h = {x/H:g})', fontsize=10)
    ax.grid(alpha=.3); ax.set_xlabel('u')
    if k % 4 == 0:
        ax.set_ylabel('y')
axs[0].legend(fontsize=8, loc='upper left')
fig.suptitle('BFS Re = 389 — axial velocity, short vs long domain, P+Z vs free outflow.\n'
             'Short domain ends at x = 2.5;  y = 0.5 is the step height;  u < 0 is reversed flow.',
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.91])
fig.savefig(f'{SC}/../figs/bfs_compare.png', dpi=125, bbox_inches='tight')
print('\nfigs/bfs_compare.png')
