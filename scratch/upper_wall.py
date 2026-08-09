"""Upper-wall separation: apply ONE detector to Fortran, p-MG and Jacobi fields.

If the Fortran field yields the same bubble extent as Python, the disagreement
with Chan's 7.84..9.66 is in the detector or in the quoted benchmark numbers,
not in the Python solver.
"""
import sys, numpy as np
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lssem2d.lgl import diff_matrix

SC = '/private/tmp/claude-501/-Users-danielchan-Dropbox-F90-SEM/6eb12f11-0cab-40ba-b8d4-95d1b2eccac6/scratchpad'
H = 0.5


def read_fortran(path):
    t = open(path).read().split(); k = 0
    def take(m):
        nonlocal k
        v = t[k:k+m]; k += m; return v
    time, re = float(take(1)[0]), float(take(1)[0])
    nelem, neig, nterm, ndep, nee = [int(x) for x in take(5)]
    XP = np.zeros((nelem, nterm)); YP = np.zeros((nelem, nterm))
    U = np.zeros((nelem, nterm, nterm, 4))
    for e in range(nelem):
        XP[e] = [float(x) for x in take(nterm)]
        YP[e] = [float(x) for x in take(nterm)]
        f = np.array([float(x) for x in take(nee)]).reshape(neig, ndep)
        for i in range(nterm):
            for j in range(nterm):
                U[e, i, j, :] = f[i*nterm + j]
        take(nee); take(nee); take(nee); take(4); take(4)
    return time, XP, YP, U


def top_shear(U, XP, YP):
    """du/dy at the TOP wall (y=1), returned sorted in x, duplicates averaged.

    tau_wall on the upper wall is -mu du/dy; the flow is attached where
    du/dy < 0 (u decreasing to zero at the wall from below).  Separation is
    du/dy > 0.  We return du/dy itself and test its sign explicitly, rather
    than folding a sign into the caller.
    """
    n = U.shape[1]; D = diff_matrix(n-1)
    xs, g = [], []
    for e in range(U.shape[0]):
        ytop = YP[e, -1]
        if ytop < 0.999:                     # not a top-wall element
            continue
        hy = YP[e, -1] - YP[e, 0]
        for i in range(n):
            xs.append(XP[e, i])
            g.append(np.dot(D[-1, :], U[e, i, :, 0]) * (2.0 / hy))
    xs, g = np.array(xs), np.array(g)
    o = np.argsort(xs, kind='stable'); xs, g = xs[o], g[o]
    # average the duplicated element-interface nodes
    ux, ug = [], []
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j+1] - xs[i] < 1e-9:
            j += 1
        ux.append(xs[i]); ug.append(np.mean(g[i:j+1])); i = j + 1
    return np.array(ux), np.array(ug)


def crossings(x, g, xmin=0.2):
    """All sign changes of g, linearly interpolated. Returns (x, direction)."""
    out = []
    for k in range(len(x)-1):
        if x[k] < xmin:
            continue
        if g[k] == 0.0:
            continue
        if g[k]*g[k+1] < 0:
            xc = x[k] - g[k]*(x[k+1]-x[k])/(g[k+1]-g[k])
            out.append((xc, '+' if g[k+1] > 0 else '-'))
    return out


tF, XPf, YPf, Uf = read_fortran(
    '/Users/danielchan/Dropbox/F90_SEM/pmg_clean/run_chan389_long/chan389_long.dat')
dM = np.load(f'{SC}/dt_dt0p1_dev_pmg_state.npz')
dJ = np.load(f'{SC}/dt_dt0p1_dev_state.npz')

CASES = [('Fortran (validated)', Uf, XPf, YPf, 'k', '-'),
         ('Python p-MG',   dM['U'], dM['xnod'], dM['ynod'], 'r', '--'),
         ('Python Jacobi', dJ['U'], dJ['xnod'], dJ['ynod'], 'tab:blue', ':')]

fig, axs = plt.subplots(2, 1, figsize=(13, 7.4))
print(f"{'case':<22}{'separation x/h':>16}{'reattach x/h':>15}{'length/h':>11}"
      f"{'max du/dy':>12}")
res = {}
for tag, U, XP, YP, c, ls in CASES:
    x, g = top_shear(U, XP, YP)
    res[tag] = (x, g)
    cr = crossings(x, g)
    # bubble = between the first + crossing and the following - crossing
    sep = rea = None
    for k, (xc, d) in enumerate(cr):
        if d == '+':
            sep = xc
            rea = next((xx for xx, dd in cr[k+1:] if dd == '-'), None)
            break
    gm = g[x > 0.2].max()
    if sep is not None and rea is not None:
        print(f"{tag:<22}{sep/H:>16.3f}{rea/H:>15.3f}{(rea-sep)/H:>11.3f}{gm:>12.4f}")
    else:
        print(f"{tag:<22}{'none detected':>16}{'':>15}{'':>11}{gm:>12.4f}")
    res[tag] = (x, g, sep, rea, cr)
    for ax in axs:
        ax.plot(x/H, g, color=c, ls=ls, lw=2.0 if c == 'k' else 1.5, label=tag)

for ax, (lo, hi, ttl) in zip(axs, [
        (0, 17, 'upper-wall du/dy over the whole channel'),
        (6.5, 11.5, 'zoom on the separated region (du/dy > 0)')]):
    ax.axhline(0, color='0.4', lw=1.0)
    ax.axvspan(7.84, 9.66, color='green', alpha=.12,
               label='Chan 1996 quoted 7.84 .. 9.66')
    ax.set_xlim(lo, hi); ax.grid(alpha=.3)
    ax.set_xlabel('x / h'); ax.set_ylabel('du/dy at y=1')
    ax.set_title(ttl, fontsize=10)
axs[0].legend(fontsize=8)
fig.suptitle('Upper-wall shear — Chan 1996 Re=389: is the bubble extent a solver '
             'difference or a detector difference?', fontsize=11)
fig.tight_layout()
fig.savefig(f'{SC}/upper_wall.png', dpi=150, bbox_inches='tight')

print("\nall sign changes of du/dy (x/h, direction):")
for tag in res:
    print(f"  {tag:<22}", [(round(xc/H, 3), d) for xc, d in res[tag][4]])

# how flat is du/dy near the crossings?  a shallow crossing => ill-conditioned
print("\nsensitivity: |du/dy| gradient at each crossing (shallow = poorly located)")
for tag in res:
    x, g, sep, rea, cr = res[tag]
    for xc, d in cr:
        k = np.searchsorted(x, xc) - 1
        if 0 <= k < len(x)-1:
            slope = (g[k+1]-g[k])/(x[k+1]-x[k])
            print(f"  {tag:<22} x/h={xc/H:7.3f} ({d})  d2u/dxdy = {slope:+.4f}")

print(f"\nsaved {SC}/upper_wall.png")
