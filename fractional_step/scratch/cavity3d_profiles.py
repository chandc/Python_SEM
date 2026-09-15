"""M2 gate, plotted: 3D at k_z = 0 against Ghia AND against the 2D solution.

    uv run --quiet python scratch/cavity3d_profiles.py

BOTH COMPONENTS, deliberately.  ARTIFICIAL_COMPRESSIBILITY.md sec 5.1 is the
precedent: RMS u improved with AC at every dt while RMS v did not move at all,
and it was the v column that established AC is accuracy-NEUTRAL rather than
better.  A gate on u alone can be passed by a solution that is wrong in v.

TWO REFERENCES, also deliberately:
  * Ghia et al. (1982) -- the physical benchmark.
  * the converged 2D field (scratch/cavity_ac_dt0.05_match.npz, RMS u 1.568e-02)
    -- the thing the 3D code must actually reproduce at k_z = 0.  If the 3D
    result sits on the 2D curve but both differ from Ghia, that is discretisation
    error and the gate PASSES; if 3D departs from 2D, it is a 3D bug.  Comparing
    only against Ghia cannot separate those two.

    figs/cavity3d_kz0_profiles.png
"""
import os, sys, glob
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lssem3d import operator as OP

GH = np.load('cavity_re1000_data.npz')
GHIA_U, GHIA_Y = GH['ghia_u'], GH['ghia_y']
GHIA_XV = np.array([1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594,
                    0.8047, 0.5000, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781,
                    0.0703, 0.0625, 0.0000])
GHIA_V = np.array([0.0000, -0.21388, -0.27669, -0.33714, -0.39188, -0.51550,
                   -0.42665, -0.31966, 0.02526, 0.32235, 0.33075, 0.37095,
                   0.32627, 0.30353, 0.29012, 0.27485, 0.0000])


def lagrange(xn, xq):
    n = len(xn); w = np.ones(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                w[i] /= (xn[i]-xn[j])
    dd = xq-xn
    if np.any(np.abs(dd) < 1e-13):
        L = np.zeros(n); L[np.argmin(np.abs(dd))] = 1.0; return L
    num = w/dd
    return num/num.sum()


def lines(U, xn, yn, iu, iv, mode=None):
    """u(y) at x=0.5 and v(x) at y=0.5.  mode=None for the 2D layout (var last),
    mode=0 for the 3D layout (var second-to-last, z last)."""
    n = U.shape[1]
    sel = (lambda e, i, j, f: U[e, i, j, f] if mode is None
           else U[e, i, j, f, mode])
    ys, us = [], []
    for e in range(U.shape[0]):
        xs = xn[e]
        if xs[0]-1e-9 <= 0.5 <= xs[-1]+1e-9:
            L = lagrange(xs, 0.5)
            for j in range(n):
                ys.append(yn[e, j])
                us.append(sum(L[i]*sel(e, i, j, iu) for i in range(n)))
    o = np.argsort(ys); ys, us = np.array(ys)[o], np.array(us)[o]
    k = np.concatenate(([True], np.diff(ys) > 1e-9)); ys, us = ys[k], us[k]
    xs_, vs = [], []
    for e in range(U.shape[0]):
        yr = yn[e]
        if yr[0]-1e-9 <= 0.5 <= yr[-1]+1e-9:
            L = lagrange(yr, 0.5)
            for i in range(n):
                xs_.append(xn[e, i])
                vs.append(sum(L[j]*sel(e, i, j, iv) for j in range(n)))
    o = np.argsort(xs_); xs_, vs = np.array(xs_)[o], np.array(vs)[o]
    k = np.concatenate(([True], np.diff(xs_) > 1e-9))
    return ys, us, xs_[k], vs[k]


def rms(ys, us, xs, vs):
    ru = float(np.sqrt(np.mean((np.interp(GHIA_Y, ys, us)-GHIA_U)**2)))
    rv = float(np.sqrt(np.mean((np.interp(GHIA_XV[::-1], xs, vs)[::-1]-GHIA_V)**2)))
    return ru, rv


CASES = []
f3 = f'{SC}/cavity3d_kz0_rkw3.npz'
if os.path.exists(f3):
    d = np.load(f3, allow_pickle=True)
    if np.all(np.isfinite(d['U'])):
        CASES.append(('3D at $k_z$ = 0 (RKW3/CN, AC)', 'tab:red',
                      dict(ls='-', lw=2.4),
                      lines(d['U'], d['xnod'], d['ynod'], OP.U_, OP.V_, mode=0),
                      f" [{str(d['status'])}, {int(d['steps'])} steps]"))
f2 = f'{SC}/cavity_ac_dt0.05_match.npz'
if os.path.exists(f2):
    d = np.load(f2, allow_pickle=True)
    CASES.append(('2D reference (converged)', 'tab:green',
                  dict(ls='--', lw=2.0),
                  lines(d['U'], d['xnod'], d['ynod'], 0, 1), ''))
if not CASES:
    raise SystemExit('no fields yet -- run scratch/cavity3d_kz0.py first')

fig, axs = plt.subplots(1, 2, figsize=(14.5, 6.4))
rows = []
for lab, col, sty, (ys, us, xs, vs), note in CASES:
    ru, rv = rms(ys, us, xs, vs)
    rows.append((lab, ru, rv, note))
    axs[0].plot(us, ys, color=col, **sty, label=f'{lab}\n   RMS u = {ru:.3e}')
    axs[1].plot(xs, vs, color=col, **sty, label=f'{lab}\n   RMS v = {rv:.3e}')
axs[0].plot(GHIA_U, GHIA_Y, 'ko', ms=8, mfc='none', mew=1.9, zorder=9,
            label='Ghia 1982 (Table I)')
axs[1].plot(GHIA_XV, GHIA_V, 'ko', ms=8, mfc='none', mew=1.9, zorder=9,
            label='Ghia 1982 (Table II)')
axs[0].set_xlabel('u'); axs[0].set_ylabel('y')
axs[0].set_title('u(y) on the vertical centreline  x = 0.5', fontsize=12)
axs[1].set_xlabel('x'); axs[1].set_ylabel('v')
axs[1].set_title('v(x) on the horizontal centreline  y = 0.5', fontsize=12)
for a in axs:
    a.grid(alpha=.3); a.axhline(0, color='k', lw=.6); a.axvline(0, color='k', lw=.6)
    a.legend(fontsize=8.5, loc='best')
fig.suptitle('M2 gate: 3D solver at $k_z$ = 0 vs the 2D solution and Ghia — '
             'cavity Re = 1000, 6×6 elements N = 10.\n'
             'The gate is agreement with the 2D CURVE; the Ghia offset both '
             'share is discretisation error, not a 3D bug.', fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig('figs/cavity3d_kz0_profiles.png', dpi=125, bbox_inches='tight')
print('figs/cavity3d_kz0_profiles.png\n')
print(f"{'case':<38}{'RMS u':>12}{'RMS v':>12}")
for lab, ru, rv, note in rows:
    print(f'{lab:<38}{ru:>12.4e}{rv:>12.4e}{note}')
