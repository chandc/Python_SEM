"""Centreline profiles from the cavity AC sweep, against Ghia et al. (1982).

    uv run --quiet python scratch/cavity_ac_plot.py

Ghia Re=1000 references:
  u(y) on the vertical centreline x=0.5   -- cavity_re1000_data.npz (Table I)
  v(x) on the horizontal centreline y=0.5 -- Table II, transcribed below

> NOTE: lssem2d/tests/plot_verification.py carries a DIFFERENT ghia_v -- that
> one is Re=100.  Using it here would silently compare against the wrong Re.

    figs/cavity_ac_centrelines.png
"""
import os, sys, glob, re
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

GH = np.load('cavity_re1000_data.npz')
GHIA_U, GHIA_Y = GH['ghia_u'], GH['ghia_y']
GHIA_XV = np.array([1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047,
                    0.5000, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625, 0.0000])
GHIA_V = np.array([0.0000, -0.21388, -0.27669, -0.33714, -0.39188, -0.51550, -0.42665,
                   -0.31966, 0.02526, 0.32235, 0.33075, 0.37095, 0.32627, 0.30353,
                   0.29012, 0.27485, 0.0000])


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


def centrelines(d):
    U, xn, yn = d['U'], d['xnod'], d['ynod']
    n = U.shape[1]
    ys, us = [], []
    for e in range(U.shape[0]):
        xs = xn[e]
        if xs[0]-1e-9 <= 0.5 <= xs[-1]+1e-9:
            L = lagrange(xs, 0.5)
            for j in range(n):
                ys.append(yn[e, j]); us.append(np.dot(L, U[e, :, j, 0]))
    o = np.argsort(ys); ys, us = np.array(ys)[o], np.array(us)[o]
    k = np.concatenate(([True], np.diff(ys) > 1e-9)); ys, us = ys[k], us[k]
    xs_, vs = [], []
    for e in range(U.shape[0]):
        yr = yn[e]
        if yr[0]-1e-9 <= 0.5 <= yr[-1]+1e-9:
            L = lagrange(yr, 0.5)
            for i in range(n):
                xs_.append(xn[e, i]); vs.append(np.dot(L, U[e, i, :, 1]))
    o = np.argsort(xs_); xs_, vs = np.array(xs_)[o], np.array(vs)[o]
    k = np.concatenate(([True], np.diff(xs_) > 1e-9))
    return ys, us, xs_[k], vs[k]


FILES = sorted(glob.glob(f'{SC}/cavity_ac_dt*.npz'),
               key=lambda s: (-float(re.search(r'dt([\d.]+)_', s).group(1)),
                              s.split('_')[-1]))
STY = {'off': dict(ls='-', lw=2.0), 'half': dict(ls='--', lw=1.8),
       'match': dict(ls=':', lw=2.2),
       # The two w_mass = 0 runs are NOT part of the AC comparison -- they are
       # the steady form, and both converge to spurious states (sec 5.3).  Drawn
       # heavy and black so they cannot be mistaken for one of the AC curves.
       'wm0': dict(ls='-', lw=2.6, color='k', alpha=.85),
       'restart': dict(ls='--', lw=2.6, color='k', alpha=.55)}
COL = {1.0: 'tab:blue', 0.25: 'tab:green', 0.1: 'tab:orange', 0.05: 'tab:red',
       2.0: 'tab:gray'}
LAB = {'wm0': 'STEADY $w_{mass}$=0, from rest (spurious)',
       'restart': 'STEADY $w_{mass}$=0, from converged (spurious)'}

fig, axs = plt.subplots(1, 2, figsize=(14.5, 6.4))
rows = []
for f in FILES:
    d = np.load(f, allow_pickle=True)
    dt = float(d['dt']); kind = f.split('_')[-1].replace('.npz', '')
    if not np.all(np.isfinite(d['U'])):
        continue
    ys, us, xs_, vs = centrelines(d)
    ru = np.sqrt(np.mean((np.interp(GHIA_Y, ys, us)-GHIA_U)**2))
    rv = np.sqrt(np.mean((np.interp(GHIA_XV[::-1], xs_, vs)[::-1]-GHIA_V)**2))
    rows.append((dt, kind, float(d['kappa_p']), str(d['status']), ru, rv))
    sty = dict(STY.get(kind, {}))
    if kind in LAB:
        lab = LAB[kind]
    else:
        lab = f"dt={dt:g}, {kind}" + ("" if kind == 'off'
                                      else f" (k={float(d['kappa_p']):g})")
        sty['color'] = COL.get(dt, 'k')
    axs[0].plot(us, ys, **sty, label=lab, zorder=8 if kind in LAB else 3)
    axs[1].plot(xs_, vs, **sty, label=lab, zorder=8 if kind in LAB else 3)

axs[0].plot(GHIA_U, GHIA_Y, 'ko', ms=7, mfc='none', mew=1.8, zorder=9,
            label='Ghia 1982 (Table I)')
axs[0].set_xlabel('u'); axs[0].set_ylabel('y')
axs[0].set_title('u(y) on the vertical centreline  x = 0.5', fontsize=12)
axs[1].plot(GHIA_XV, GHIA_V, 'ko', ms=7, mfc='none', mew=1.8, zorder=9,
            label='Ghia 1982 (Table II)')
axs[1].set_xlabel('x'); axs[1].set_ylabel('v')
axs[1].set_title('v(x) on the horizontal centreline  y = 0.5', fontsize=12)
for a in axs:
    a.grid(alpha=.3); a.axhline(0, color='k', lw=.6); a.axvline(0, color='k', lw=.6)
axs[0].legend(fontsize=8, loc='lower right')
fig.suptitle('Lid-driven cavity Re = 1000, 6x6 elements N = 10, pressure-pinned.  '
             'Coloured: time-accurate (w_mom = w_mass = 1), AC off/on.\n'
             'Black: the STEADY form (w_mass = 0) -- converges, but to spurious '
             'states, from rest AND from the converged field (sec 5.3).',
             fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig('figs/cavity_ac_centrelines.png', dpi=125, bbox_inches='tight')
print('figs/cavity_ac_centrelines.png\n')
print(f"{'dt':>7}{'AC':>7}{'kappa_p':>9}{'status':>13}{'RMS u':>11}{'RMS v':>11}")
for r in sorted(rows):
    print(f'{r[0]:>7g}{r[1]:>7}{r[2]:>9.4g}{r[3]:>13}{r[4]:>11.4e}{r[5]:>11.4e}')
