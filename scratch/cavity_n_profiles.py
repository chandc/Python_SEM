"""Centreline u and v vs Ghia for every polynomial order N.  Cavity Re = 1000.

    uv run --quiet python scratch/cavity_n_profiles.py

Reads the CONVERGED runs cavity_ac_dt0.05_match[_N*].npz (6x6 elements,
dt = 0.05, kappa_p = a_mass = 30, run to the stall exit).

> NOT the nsweep_N*_*.npz fields -- those are 40 steps, t = 2, nowhere near
> steady.  They measure conditioning (sec 5.2b) and their profiles would show
> early-transient shape, not discretisation error.

Ghia Re=1000 references:
  u(y) on the vertical centreline x=0.5   -- cavity_re1000_data.npz (Table I)
  v(x) on the horizontal centreline y=0.5 -- Table II, transcribed below

> NOTE: lssem2d/tests/plot_verification.py carries a DIFFERENT ghia_v -- that
> one is Re=100.  Using it here would silently compare against the wrong Re.

    figs/cavity_n_profiles.png
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


RUNS = []
for f in glob.glob(f'{SC}/cavity_ac_dt0.05_match*.npz'):
    m = re.search(r'_N(\d+)\.npz$', f)
    N = int(m.group(1)) if m else 10          # unsuffixed file is the N=10 run
    d = np.load(f, allow_pickle=True)
    if not np.all(np.isfinite(d['U'])):
        print(f'skip {f}: non-finite', file=sys.stderr); continue
    ys, us, xs_, vs = centrelines(d)
    RUNS.append(dict(N=N, ys=ys, us=us, xs=xs_, vs=vs, f=f,
                     status=str(d['status']), steps=int(d['steps']),
                     rms_u=float(np.sqrt(np.mean(
                         (np.interp(GHIA_Y, ys, us)-GHIA_U)**2))),
                     rms_v=float(np.sqrt(np.mean(
                         (np.interp(GHIA_XV[::-1], xs_, vs)[::-1]-GHIA_V)**2)))))
RUNS.sort(key=lambda r: r['N'])
if not RUNS:
    raise SystemExit('no converged cavity_ac_dt0.05_match*.npz found')
CM = plt.get_cmap('viridis')
COL = {r['N']: CM(i/max(len(RUNS)-1, 1)) for i, r in enumerate(RUNS)}

fig, axs = plt.subplots(1, 2, figsize=(14.5, 6.4))
for r in RUNS:
    lab = f"N = {r['N']}  (RMS u {r['rms_u']:.2e})"
    axs[0].plot(r['us'], r['ys'], lw=2.0, color=COL[r['N']], label=lab)
    axs[1].plot(r['xs'], r['vs'], lw=2.0, color=COL[r['N']],
                label=f"N = {r['N']}  (RMS v {r['rms_v']:.2e})")
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
fig.suptitle('Convergence in polynomial order — lid-driven cavity Re = 1000, '
             '6x6 elements, dt = 0.05, AC on ($\\kappa_p = a_{mass} = 30$).\n'
             'Converged runs (stall exit), not the 40-step conditioning fields '
             'of sec 5.2b.', fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig('figs/cavity_n_profiles.png', dpi=125, bbox_inches='tight')
print('figs/cavity_n_profiles.png\n')
print(f"{'N':>4}{'dof':>8}{'steps':>7}{'status':>10}{'RMS u':>11}{'RMS v':>11}")
for r in RUNS:
    n = len(r['ys'])
    print(f"{r['N']:>4}{6*6*(r['N']+1)**2*4:>8}{r['steps']:>7}"
          f"{r['status'][:9]:>10}{r['rms_u']:>11.4e}{r['rms_v']:>11.4e}")
