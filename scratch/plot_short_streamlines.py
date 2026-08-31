"""Streamlines of the short-domain Armaly BFS with the Dong outlet.

    uv run --quiet python scratch/plot_short_streamlines.py

Plots the SAVED field -- scratch/armaly_short_dong_on.npz, 72 elements at N=10,
Re = 389 (nu = 2/389), ER 1.94, step at x = 0.  Never re-solves.

The SEM field lives on LGL nodes, which cluster at element edges, so it is
interpolated to a uniform grid before streamplot.  Points inside the step
(x < 0, y < 0.94) are masked -- they are outside the flow domain, and letting
griddata extrapolate into them draws streamlines through solid wall.
"""
import os
import sys

SC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SC)
sys.path.insert(0, ROOT); sys.path.insert(0, SC)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

STEP_X, STEP_Y = 0.0, 0.94          # step corner; S = 0.94 is the step height
S_STEP = 0.94
# ARMALY_VALIDATION.md: the LONG domain with P+Z gives x_r/S = 8.145, within
# 1.2% of Armaly's experiment.  That is the truth this short domain is truncating
# -- x_r ~ 7.66, well past its outlet at x = 5.
XR_S_REF = 8.145
CASES = [('armaly_short_free',    'FREE outflow'),
         ('armaly_short_pz',      'P+Z (pressure + zero-gradient)'),
         ('armaly_short_dong_off', 'Dong OBC, backflow switch DISARMED'),
         ('armaly_short_dong_on',  'Dong OBC, backflow switch ARMED')]


def load(tag):
    z = np.load(f'{SC}/{tag}.npz', allow_pickle=True)
    U, xn, yn = z['U'], z['xnod'], z['ynod']
    ne, n = U.shape[0], U.shape[1]
    X = np.repeat(xn[:, :, None], n, axis=2)          # (ne, n, n)
    Y = np.repeat(yn[:, None, :], n, axis=1)
    return (X.ravel(), Y.ravel(),
            U[..., 0].ravel(), U[..., 1].ravel(),
            float(z['nu']), str(z['status']), int(z['steps']))


def outlet_backflow(tag):
    """How much of the outlet plane is in INFLOW?  This is the whole point of
    the short domain: Armaly Re = 389 reattaches near x ~ 7 (ARMALY_VALIDATION),
    well past this domain's x = 5, so the recirculation CROSSES the outlet and
    the boundary condition is asked to handle reversed flow."""
    z = np.load(f'{SC}/{tag}.npz', allow_pickle=True)
    U, xn, yn = z['U'], z['xnod'], z['ynod']
    xmax = xn.max()
    out = [e for e in range(U.shape[0]) if abs(xn[e, -1] - xmax) < 1e-9]
    u = np.concatenate([U[e, -1, :, 0] for e in out])
    return float(u.min()), float((u < 0).sum())/u.size


def reattachment(xg, yg, ug):
    """First x where wall shear changes sign: u at the first row above y=0."""
    j = 1                                            # first interior row
    u = ug[j]
    s = np.sign(u)
    for i in range(1, len(s)):
        if s[i-1] < 0 <= s[i] and xg[i] > 0.2:
            x0, x1 = xg[i-1], xg[i]
            return x0 + (x1-x0)*(-u[i-1])/(u[i]-u[i-1])
    return np.nan


def main():
    fig, axes = plt.subplots(len(CASES), 1, figsize=(13, 12.4), sharex=True)
    for ax, (tag, title) in zip(np.atleast_1d(axes), CASES):
        x, y, u, v, nu, status, steps = load(tag)
        xg = np.linspace(x.min(), x.max(), 900)
        yg = np.linspace(y.min(), y.max(), 190)
        XG, YG = np.meshgrid(xg, yg)
        ug = griddata((x, y), u, (XG, YG), method='linear')
        vg = griddata((x, y), v, (XG, YG), method='linear')
        solid = (XG < STEP_X) & (YG < STEP_Y)         # inside the step
        ug[solid] = np.nan; vg[solid] = np.nan

        spd = np.sqrt(ug**2 + vg**2)
        ax.streamplot(xg, yg, ug, vg, density=(3.2, 1.1), color=spd,
                      cmap='viridis', linewidth=0.8, arrowsize=0.7)
        ax.contour(XG, YG, ug, levels=[0.0], colors='r', linewidths=1.4)
        ax.add_patch(plt.Rectangle((x.min(), 0), -x.min(), STEP_Y,
                                   fc='0.85', ec='0.4', zorder=5))
        xr = reattachment(xg, yg, ug)
        umin, frac = outlet_backflow(tag)
        if np.isfinite(xr):
            ax.plot([xr], [0], 'r*', ms=13, zorder=6)
            err = (xr/S_STEP - XR_S_REF)/XR_S_REF*100
            ax.annotate(f'$x_r/S$ = {xr/S_STEP:.2f}   ({err:+.0f}% vs 8.145)',
                        (xr, 0), (xr-2.4, 0.30), color='r', fontsize=9,
                        arrowprops=dict(arrowstyle='->', color='r', lw=0.8))
        else:
            ax.text(0.5*x.max(), 0.30, 'NO reattachment in domain: '
                    f'$x_r/S$ > {x.max()/S_STEP:.2f}  —  closest to the '
                    'reference 8.145', color='r', fontsize=9, ha='center',
                    zorder=7)
        ax.text(0.995, 0.04, f'outlet backflow: {frac*100:.1f}% of plane,'
                f'  min $u$ = {umin:+.3f}', transform=ax.transAxes,
                ha='right', fontsize=9, color='darkred', zorder=7,
                bbox=dict(fc='white', ec='0.7', alpha=0.85, pad=1.8))
        ax.set_ylim(0, y.max()); ax.set_xlim(x.min(), x.max())
        ax.set_ylabel('y'); ax.set_aspect('equal')
        ax.set_title(f'{title}   —   Re = {2/nu:.0f}, ER 1.94, 72 elem N=10, '
                     f'{steps} steps ({status})', fontsize=10)
    np.atleast_1d(axes)[-1].set_xlabel('x')
    fig.suptitle('Short-domain Armaly BFS, Re = 389, ER 1.94 — predicted '
                 'streamlines by outflow condition\n'
                 'red = $u=0$ dividing line.  Validated reference (long domain, '
                 'within 1.2% of Armaly): $x_r/S$ = 8.145, i.e. $x_r\\approx7.7$ — '
                 'BEYOND this outlet at $x=5$, so the boundary sits in reversed flow.',
                 fontsize=11)
    fig.tight_layout()
    out = f'{SC}/short_bfs_streamlines.png'
    fig.savefig(out, dpi=145, bbox_inches='tight')
    print(f'saved -> {out}')
    for tag, _ in CASES:
        x, y, u, v, nu, status, steps = load(tag)
        print(f'  {tag:24s} min u = {u.min():+.4f}  max|u| = {np.abs(u).max():.4f}'
              f'  status={status} steps={steps}')


if __name__ == '__main__':
    main()
