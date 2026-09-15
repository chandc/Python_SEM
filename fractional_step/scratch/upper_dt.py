"""Does the upper-wall bubble depend on dt (the LS momentum-row weight)?

The momentum rows carry  fac1*u + dt*N(u), so dt sets the relative weight of the
momentum residual against continuity/vorticity.  At a steady state the residual
is not identically zero, so the LS minimiser is dt-dependent.  Fortran ran the
Chan case at dt=0.5; the Python p-MG run used dt=0.1.  If that is the cause, the
bubble should march toward the Fortran answer as dt -> 0.5.

Also checks time-convergence from the snapshots, so a still-evolving bubble is
not mistaken for a dt effect.
"""
import os
import sys, glob, re, numpy as np
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lssem2d.lgl import diff_matrix

SC = os.path.dirname(os.path.abspath(__file__))
H = 0.5
sys.path.insert(0, SC)
from upper_wall import read_fortran, top_shear, crossings


def bubble(U, XP, YP):
    x, g = top_shear(U, XP, YP)
    cr = crossings(x, g)
    sep = rea = None
    for k, (xc, d) in enumerate(cr):
        if d == '+':
            sep = xc
            rea = next((xx for xx, dd in cr[k+1:] if dd == '-'), None)
            break
    return sep, rea, g[x > 0.2].max()


def reattach(U, XP, YP):
    """primary (lower-wall) reattachment"""
    n = U.shape[1]; D = diff_matrix(n-1)
    xs, g = [], []
    for e in range(U.shape[0]):
        if YP[e, 0] > 0.01 or XP[e, 0] < -1e-9:
            continue
        hy = YP[e, -1] - YP[e, 0]
        for i in range(n):
            xs.append(XP[e, i]); g.append(np.dot(D[0, :], U[e, i, :, 0])*(2.0/hy))
    o = np.argsort(xs); xs, g = np.array(xs)[o], np.array(g)[o]
    for k in range(len(xs)-1):
        if g[k] < 0 and g[k+1] > 0 and xs[k] > 0.05:
            return xs[k] - g[k]*(xs[k+1]-xs[k])/(g[k+1]-g[k])
    return np.nan


tF, XPf, YPf, Uf = read_fortran(
    '/Users/danielchan/Dropbox/F90_SEM/pmg_clean/run_chan389_long/chan389_long.dat')
sF, rF, mF = bubble(Uf, XPf, YPf)
xrF = reattach(Uf, XPf, YPf)

print("=== 1. steady states across the dt sweep (all Jacobi, zero IC unless _dev) ===")
print(f"{'file':<28}{'dt':>6}{'t_end':>8}{'x_r/h':>9}{'sep x/h':>9}{'rea x/h':>9}"
      f"{'len/h':>8}{'max du/dy':>11}")
print(f"{'FORTRAN dt=0.5':<28}{0.5:>6}{tF:>8.0f}{xrF/H:>9.3f}{sF/H:>9.3f}{rF/H:>9.3f}"
      f"{(rF-sF)/H:>8.3f}{mF:>11.4f}")
rows = []
for f in sorted(glob.glob(f'{SC}/dt_dt*_state.npz')):
    d = np.load(f)
    U, XP, YP = d['U'], d['xnod'], d['ynod']
    dt = float(d['dt']); te = int(d['step'])*dt
    s, r, mx = bubble(U, XP, YP)
    xr = reattach(U, XP, YP)
    tag = f.split('/')[-1].replace('_state.npz', '')
    if s is None or r is None:
        print(f"{tag:<28}{dt:>6}{te:>8.1f}{xr/H:>9.3f}{'no bubble':>9}{'':>9}{'':>8}{mx:>11.4f}")
        continue
    rows.append((dt, s/H, r/H, mx, tag))
    print(f"{tag:<28}{dt:>6}{te:>8.1f}{xr/H:>9.3f}{s/H:>9.3f}{r/H:>9.3f}"
          f"{(r-s)/H:>8.3f}{mx:>11.4f}")

print("\n=== 2. time convergence: is the bubble still moving at the end of a run? ===")
for pat, dt in ((f'{SC}/dt_dt0p1_dev_snap*.npz', 0.1), (f'{SC}/dt_dt0p5_snap*.npz', 0.5)):
    fs = sorted(glob.glob(pat))
    if not fs:
        continue
    print(f"  --- dt={dt} ---")
    print(f"    {'t':>8}{'sep x/h':>10}{'rea x/h':>10}{'len/h':>9}{'max du/dy':>11}")
    for f in fs[-6:]:
        d = np.load(f)
        s, r, mx = bubble(d['U'], d['xnod'], d['ynod'])
        t = int(d['step'])*float(d['dt'])
        if s is None or r is None:
            print(f"    {t:>8.1f}{'no bubble':>10}{'':>10}{'':>9}{mx:>11.4f}")
        else:
            print(f"    {t:>8.1f}{s/H:>10.3f}{r/H:>10.3f}{(r-s)/H:>9.3f}{mx:>11.4f}")

# ------------------------------------------------------------------ plot
rows.sort()
dts = [r[0] for r in rows]
fig, axs = plt.subplots(1, 3, figsize=(14.5, 4.3))
for ax, k, ttl, ref in ((axs[0], 1, 'separation  x/h', sF/H),
                        (axs[1], 2, 'reattachment  x/h', rF/H),
                        (axs[2], 3, 'peak du/dy in the bubble', mF)):
    ax.semilogx(dts, [r[k] for r in rows], 'o-', color='tab:blue', label='Python')
    ax.axhline(ref, color='k', ls='-', lw=2, label='Fortran (dt=0.5)')
    if k == 1: ax.axhline(7.84, color='g', ls='--', label='Chan quoted')
    if k == 2: ax.axhline(9.66, color='g', ls='--', label='Chan quoted')
    ax.axvline(0.5, color='r', ls=':', lw=1.5, label='Fortran dt')
    ax.set_xlabel('dt'); ax.set_title(ttl, fontsize=10); ax.grid(alpha=.3)
    ax.legend(fontsize=7)
fig.suptitle('Upper-wall bubble vs dt — the LS momentum-row weight is dt-dependent',
             fontsize=11)
fig.tight_layout()
fig.savefig(f'{SC}/upper_dt.png', dpi=150, bbox_inches='tight')
print(f"\nsaved {SC}/upper_dt.png")
