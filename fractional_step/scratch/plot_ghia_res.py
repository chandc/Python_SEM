"""Ghia Re=1000 cavity: u and v centreline profiles vs resolution.

Benchmark data:
  u(y) at x=0.5 -- Ghia Table I,  already in cavity_re1000_data.npz
  v(x) at y=0.5 -- Ghia Table II, taken from scratch/plot_cavity.py (Re=1000).
  (Note plot_verification.py carries a DIFFERENT ghia_v -- that one is Re=100,
   used by a Re=100 script.  Do not mix them.)
"""
import os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

SC = os.path.dirname(os.path.abspath(__file__))
d = np.load(f'{SC}/cavity_ghia_res.npz', allow_pickle=True)
keys = [str(k) for k in d['keys']]
ghia_u, ghia_y = d['ghia_u'], d['ghia_y']
ghia_x_v = np.array([1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047,
                     0.5000, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625, 0.0000])
ghia_v = np.array([0.0000, -0.21388, -0.27669, -0.33714, -0.39188, -0.51550, -0.42665,
                   -0.31966, 0.02526, 0.32235, 0.33075, 0.37095, 0.32627, 0.30353,
                   0.29012, 0.27485, 0.0000])

cols = plt.cm.viridis(np.linspace(0.05, 0.8, len(keys)))
fig, axs = plt.subplots(1, 3, figsize=(16, 5.2))
su = max(abs(ghia_u.max()), abs(ghia_u.min()))
sv = max(abs(ghia_v.max()), abs(ghia_v.min()))
print(f"{'mesh':<12}{'DOF':>8}{'RMS u':>10}{'% umax':>9}{'RMS v':>10}{'% vmax':>9}")
dofs, ru_l, rv_l = [], [], []
for c, k in zip(cols, keys):
    y, u = d[f'{k}__y'], d[f'{k}__u']
    x, v = d[f'{k}__x'], d[f'{k}__v']
    ndof = int(d[f'{k}__ndof'])
    ru = np.sqrt(np.mean((np.interp(ghia_y, y, u)-ghia_u)**2))
    rv = np.sqrt(np.mean((np.interp(ghia_x_v[::-1], x, v)[::-1]-ghia_v)**2))
    dofs.append(ndof); ru_l.append(ru); rv_l.append(rv)
    lab = k.replace('_p', ' p=')
    print(f"{lab:<12}{ndof:>8}{ru:>10.4f}{100*ru/su:>8.2f}%{rv:>10.4f}{100*rv/sv:>8.2f}%")
    axs[0].plot(u, y, '-', color=c, lw=1.9, label=f'{lab}  ({ndof//1000}k dof)')
    axs[1].plot(x, v, '-', color=c, lw=1.9, label=lab)

from matplotlib.ticker import NullFormatter, NullLocator, FixedLocator

# u panel: curves sweep bottom-left -> top-right, so the lower-right is clear.
axs[0].plot(ghia_u, ghia_y, 'ko', ms=6, mfc='none', mew=1.7, label='Ghia 1982 (Table I)')
axs[0].set_xlabel('u'); axs[0].set_ylabel('y'); axs[0].set_ylim(0, 1)
axs[0].set_title('u along the vertical centreline  x=0.5', fontsize=10)
axs[0].legend(fontsize=8, loc='lower right', framealpha=.92); axs[0].grid(alpha=.3)
axs[0].axvline(0, color='0.7', lw=.8)

# v panel: the trough at x~0.9 was hidden behind a lower-right legend; the
# upper-right quadrant is empty, so put it there instead.
axs[1].plot(ghia_x_v, ghia_v, 'ko', ms=6, mfc='none', mew=1.7, label='Ghia 1982 (Table II)')
axs[1].set_xlabel('x'); axs[1].set_ylabel('v'); axs[1].set_xlim(0, 1)
axs[1].set_title('v along the horizontal centreline  y=0.5', fontsize=10)
axs[1].legend(fontsize=8, loc='upper right', framealpha=.92); axs[1].grid(alpha=.3)
axs[1].axhline(0, color='0.7', lw=.8)

# convergence panel: default log minor-tick labels collide into an unreadable
# smear, so label only the four meshes actually run, small font, no minor labels.
axs[2].loglog(dofs, 100*np.array(ru_l)/su, 'o-', lw=2, ms=7, label='u profile')
axs[2].loglog(dofs, 100*np.array(rv_l)/sv, 's-', lw=2, ms=7, label='v profile')
axs[2].set_xlabel('degrees of freedom'); axs[2].set_ylabel('RMS error vs Ghia  (% of peak)')
axs[2].set_title('spectral convergence toward Ghia', fontsize=10)
axs[2].grid(alpha=.3, which='both')
axs[2].legend(fontsize=9, loc='upper right', framealpha=.92)
axs[2].xaxis.set_major_locator(FixedLocator(dofs))
axs[2].xaxis.set_minor_locator(NullLocator())
axs[2].xaxis.set_minor_formatter(NullFormatter())
axs[2].set_xticklabels([f"{k.replace('_p',chr(10)+'p=')}\n{n/1000:.1f}k dof"
                        for k, n in zip(keys, dofs)], fontsize=7)
axs[2].set_xlim(dofs[0]*0.75, dofs[-1]*1.35)

fig.suptitle('Lid-driven cavity Re=1000 — LSSEM vs Ghia et al. (1982), resolution study (dt=1.0, steady)',
             fontsize=12)
fig.tight_layout()
out = f'{SC}/cavity_ghia_res.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f"\nsaved {out}")
