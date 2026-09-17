"""Near- and far-field vorticity and pressure for the Re = 100 cylinder.

    uv run python scratch/curvi_cyl_fields.py [chk.npz]

Both fields are SOLVED VARIABLES of the velocity-vorticity-pressure system, so
neither panel is a post-processed derivative of a velocity field -- what is drawn
is the discrete unknown.  That is the same property the force integration uses.

Four views, because near and far field ask different questions:

  NEAR  does the body-fitted ring resolve the wall layer and the separation, and
        do the four 45-degree corner elements where the O-ring meets the square
        leave any visible scar?
  FAR   do the vortices convect out through the Dong outflow without reflecting?
        A reflecting outflow shows as a standing pattern piling up near x = Ld
        and as a spurious modulation in C_L; a working one lets the street leave.
"""
import os
import sys

_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R)
os.chdir(_R)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from lssem2d import curvi

OUT = os.path.join(_R, 'figs', 'curvi_cyl_re100_fields.png')
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    _R, 'scratch', '_cyl_re100', 'chk_latest.npz')


def main():
    z = np.load(SRC, allow_pickle=True)
    U = z['U0']
    t = float(z['t'])
    m = curvi.build_cylinder_box()
    X, Y = m.X, m.Y
    tri = mtri.Triangulation(X.ravel(), Y.ravel())
    cx = X.ravel()[tri.triangles].mean(axis=1)
    cy = Y.ravel()[tri.triangles].mean(axis=1)
    tri.set_mask(np.hypot(cx, cy) < m.r_cyl*1.02)

    om = U[..., 3].ravel()
    p = U[..., 2].ravel()
    p = p - p.mean()                     # p is defined up to a constant

    x0, x1, y0, y1 = m.box
    views = (('near', (-1.5, 4.0), (-2.0, 2.0)),
             ('far',  (x0, x1), (y0, y1)))
    fig, ax = plt.subplots(2, 2, figsize=(16.5, 8.2),
                           gridspec_kw=dict(hspace=0.28, wspace=0.16))

    xr_ = X.ravel()
    for r, (fld, lab, cmap, pct) in enumerate(
            ((om, r'vorticity $\omega$', 'seismic', 99.0),
             (p,  r'pressure $p$', 'coolwarm', 99.0))):
        for c, (tag, xl, yl) in enumerate(views):
            a = ax[r, c]
            sel = ((xr_ >= xl[0]) & (xr_ <= xl[1])
                   & (Y.ravel() >= yl[0]) & (Y.ravel() <= yl[1]))
            # CLIP FROM THE REGION BEING SHOWN, EXCLUDING THE WALL LAYER on the
            # far view.  |omega| is 22 at the surface and 0.8 in the wake -- a
            # factor of 27 -- so a single clip level taken over the whole domain
            # saturates the body and leaves the street invisible, which is what
            # the first version of this figure did.  Beyond x = 2 the wall layer
            # is gone and the scale belongs to the vortices.
            if tag == 'far':
                sel = sel & (xr_ > 2.0)
                pct = 99.5
            lim = np.percentile(np.abs(fld[sel]), pct)
            sc = a.tripcolor(tri, fld, shading='gouraud', cmap=cmap,
                             vmin=-lim, vmax=lim)
            fig.colorbar(sc, ax=a, shrink=0.88, extend='both')
            # contour lines for definition: the filled map gives the sign and
            # the magnitude, the lines give the shape of each vortex
            lv = np.array([-0.75, -0.45, -0.2, 0.2, 0.45, 0.75])*lim
            a.tricontour(tri, fld, levels=lv, colors='k', linewidths=0.45,
                         alpha=0.55)
            if tag == 'near':
                n = m.nterm
                for e in range(m.nelem):
                    for idx in (0, n-1):
                        a.plot(X[e, idx, :], Y[e, idx, :], 'k-', lw=0.4, alpha=0.45)
                        a.plot(X[e, :, idx], Y[e, :, idx], 'k-', lw=0.4, alpha=0.45)
            else:
                a.axvline(x1, color='k', lw=2.0)
                a.text(x1 - 0.4, y1 - 1.6, 'Dong\noutflow', ha='right',
                       fontsize=8, rotation=90, va='top')
            a.add_patch(plt.Circle((0, 0), m.r_cyl, color='0.55', zorder=6))
            a.set_xlim(*xl); a.set_ylim(*yl)
            a.set_aspect('equal'); a.tick_params(labelsize=8)
            a.set_title(f'{lab} — {tag} field'
                        + ('   (mesh overlaid)' if tag == 'near' else
                           f'   full domain, {x1-x0:g}D x {y1-y0:g}D'),
                        fontsize=10)

    fig.suptitle(f'Cylinder Re = 100, curvilinear box mesh, $t = {t:.1f}$ — '
                 r'$\omega$ and $p$ are SOLVED VARIABLES, not derivatives',
                 fontsize=12)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches='tight')
    print(f't = {t:.2f}')
    print(f'  |omega| max {np.abs(om).max():8.2f}  (at the wall)')
    print(f'  p range     {p.min():8.4f} to {p.max():.4f}  (mean removed)')
    nearsel = (np.hypot(X, Y).ravel() < 4.0)
    print(f'  |omega| max beyond x = 15: {np.abs(om[X.ravel() > 15]).max():.4f}'
          f'   -- a reflecting outflow would keep this large')
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
