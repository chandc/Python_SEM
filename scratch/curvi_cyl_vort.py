"""Vorticity of the widened cylinder box, near field and far field.

INTERPOLATION.  Plotting the GLL nodes directly is what makes spectral fields
look pixelated: N = 6 gives 7 points per element, clustered at the ends, and any
plotter joining them linearly shows the clustering rather than the field.  Every
element here is evaluated on a uniform sub-grid through the SAME Lagrange basis
the solution is expressed in, so the picture is the actual polynomial.  Element
by element with gouraud shading, so nothing is smeared across a boundary that
the solution does not actually smooth.

WHY AMPLITUDE AND NOT THE VOLUME INTEGRAL.  The outer cells are enormous -- the
last is 5.9 units tall -- so a volume-weighted share of |omega| beyond |y| = 10
reads 2.2 % while the field out there is four orders below the wake.  The
integral measures the cells, not the flow; peak and rms are quoted instead.

PANEL (c) IS THE POINT.  omega is a solved variable in this formulation, not a
post-processed derivative, so what it says about the far field can be taken at
face value.  Saturating the scale 250x shows what actually reaches the region
the widening added, which is the question the domain study was asking.
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
from matplotlib.colors import TwoSlopeNorm, SymLogNorm

import lssem2d
from lssem2d import curvi
from lssem2d.lgl import lgl_nodes

# Two extended-domain runs exist and this plots either; pick with argv[1].
# `old` is where the REFERENCE box's boundary used to sit, drawn as a dotted
# line so the question "does anything live in the region the extension added"
# can be answered by looking rather than by taking my word for it.
CASES = {
    'h40': dict(run='scratch/_cyl_H40_N6dt0.1ac/final.npz',
                kw=lambda c: dict(ys_side=c.nested_lateral_edges(20.0, 2)),
                old=('y', 10.0), xlim=(-10.0, 25.0), ylim=(-20.0, 20.0),
                label='lateral H_full = 40'),
    'xu20': dict(run='scratch/_cyl_Xu20_N6dt0.1ac/final.npz',
                 kw=lambda c: dict(xs_upstream=c.nested_upstream_edges(20.0, 2)),
                 old=('x', -10.0), xlim=(-20.0, 25.0), ylim=(-10.0, 10.0),
                 label='upstream Xu = 20'),
}
CASE = CASES[sys.argv[1] if len(sys.argv) > 1 else 'xu20']
RUN = CASE['run']
N = 6
M = 13                                   # sub-points per element per direction


def lagrange_at(nodes, pts):
    """(len(pts), len(nodes)) matrix evaluating the nodal basis at `pts`."""
    n = len(nodes)
    L = np.ones((len(pts), n))
    for j in range(n):
        for k in range(n):
            if k != j:
                L[:, j] *= (pts - nodes[k])/(nodes[j] - nodes[k])
    return L


def refine(m, F):
    """Element-wise spectral interpolation of coords and field onto M x M."""
    L = lagrange_at(lgl_nodes(m.N), np.linspace(-1.0, 1.0, M))
    ein = lambda A: np.einsum('ai,eij,bj->eab', L, A, L)
    return ein(m.X), ein(m.Y), ein(F)


def panel(ax, Xf, Yf, Ff, vmax, norm=None):
    # CLIPPING IS WHY A SATURATED LINEAR PANEL LOOKS LIKE NOISE.  In the wake
    # |omega| reaches 22 against a 0.02 scale, so every point saturates to solid
    # red or blue and each zero crossing becomes a hard edge -- structure that
    # is entirely an artefact of the colour limits.  Pass a SymLogNorm instead
    # to show four decades at once without clipping anything.
    if norm is None:
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        Ff = np.clip(Ff, -vmax, vmax)
    for e in range(Xf.shape[0]):
        ax.pcolormesh(Xf[e], Yf[e], Ff[e], cmap='RdBu_r', norm=norm,
                      shading='gouraud', rasterized=True)
    return norm


def main():
    lssem2d.set_backend('numpy')
    m = curvi.build_cylinder_box(N=N, **CASE['kw'](curvi))
    m.compute_global_indices()
    z = np.load(RUN, allow_pickle=True)
    om, t = z['U0'][..., 3], float(z['t'])
    Xf, Yf, Of = refine(m, om)
    peak = float(np.abs(om).max())
    print(f'{RUN}: t = {t:.1f}, max|omega| = {peak:.3f}')

    # LAYOUT.  Every panel is aspect-equal; the near field is 2:1 and the full
    # domain nearly square, so one column of three makes two of them postage
    # stamps.  Near field spans the top row, full-domain views sit beneath.
    fig = plt.figure(figsize=(12.6, 12.0))
    gs = fig.add_gridspec(3, 1, hspace=0.30)
    axes = [fig.add_subplot(gs[i]) for i in range(3)]
    specs = ((5.0, (-3.0, 12.0), (-3.5, 3.5), '(a) near field   |omega| <= 5'),
             (1.0, CASE['xlim'], CASE['ylim'], '(b) full domain   |omega| <= 1'),
             (None, CASE['xlim'], CASE['ylim'],
              '(c) same, symmetric log scale -- four decades, nothing clipped'))
    slog = SymLogNorm(linthresh=2e-3, linscale=0.6, vmin=-peak, vmax=peak, base=10)

    for ax, (vmax, xl, yl, ttl) in zip(axes, specs):
        norm = panel(ax, Xf, Yf, Of, vmax, norm=(slog if vmax is None else None))
        ax.add_patch(plt.Circle((0, 0), 0.5, fc='0.75', ec='k', lw=0.9, zorder=5))
        ax_, val = CASE['old']
        if ax_ == 'y':
            for sgn in (+1, -1):
                ax.axhline(sgn*val, color='0.2', lw=1.0, ls=':', zorder=4)
        else:
            ax.axvline(val, color='0.2', lw=1.0, ls=':', zorder=4)
        ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_aspect('equal')
        ax.set_title(ttl, fontsize=10.5, loc='left')
        ax.set_xlabel('x'); ax.set_ylabel('y')
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='RdBu_r'), ax=ax,
                     fraction=0.030, pad=0.02, label='omega')
    ax_, val = CASE['old']
    for ax in axes[1:]:
        if ax_ == 'y':
            ax.text(CASE['xlim'][1] - 1.0, val + 0.9, 'old boundary  y = +-10',
                    fontsize=7.5, color='0.2', ha='right')
        else:
            ax.text(val - 0.4, CASE['ylim'][1] - 1.6,
                    'old inlet  x = -10', fontsize=7.5, color='0.2', ha='right')

    # "far" = the region the extension ADDED, which is the whole question
    far = (np.abs(m.Y) > val) if ax_ == 'y' else (m.X < val)
    up = m.X < -2.0
    pk, rms = float(np.abs(om[far]).max()), float(np.sqrt((om[far]**2).mean()))
    # Element-edge against element-interior rms: if the far-field signal were
    # inter-element noise it would pile up on the shared edges.  Equal rms means
    # a smooth resolved field, and the pattern can be read as physics.
    ei = np.zeros((m.nterm, m.nterm), bool)
    ei[0, :] = ei[-1, :] = ei[:, 0] = ei[:, -1] = True
    sel = ((np.abs(m.Y).max(axis=(1, 2)) > 12.0) if ax_ == 'y'
           else (m.X.max(axis=(1, 2)) < val))
    er = float(np.sqrt((np.abs(om)[:, ei][sel]**2).mean()))
    ir = float(np.sqrt((np.abs(om)[:, ~ei][sel]**2).mean()))
    # omega is a SHARED NODAL UNKNOWN, so co-located values must be identical;
    # a non-zero spread would be an assembly bug rather than a resolution issue.
    # It is zero here, which rules out inter-element noise -- but continuity is
    # not resolution, and the two must not be confused: the outer elements span
    # about one shedding wavelength, so the far-field pattern is a polynomial
    # straining to represent a wave it cannot resolve.  Its AMPLITUDE is
    # meaningful, its shape is not.
    gg = m.gidx.ravel(); vv = om.ravel()
    o = np.argsort(gg); gsv, vsv = gg[o], vv[o]
    bnd = np.flatnonzero(np.diff(gsv)) + 1
    jump = max((vsv[i:j].max() - vsv[i:j].min())
               for i, j in zip(np.r_[0, bnd], np.r_[bnd, len(gsv)]) if j - i > 1)
    hxe = (m.X.max(axis=(1, 2)) - m.X.min(axis=(1, 2)))[sel]
    hxmin, hxmax = float(hxe.min()), float(hxe.max())
    lam = 1.0/0.1682

    fig.suptitle(
        f'Cylinder Re = 100, {CASE["label"]}, N = 6, dt = 0.1, '
        f't = {t:.0f}\n'
        f'in the region the extension added: peak |omega| {pk:.4f}, '
        f'rms {rms:.4f}  --  '
        f'{pk/peak:.1e} of the wake peak {peak:.1f}\n'
        f'far field is exactly continuous (co-located nodes agree to '
        f'{jump:.0e}) but UNDER-RESOLVED: outer elements {hxmin:.1f}-{hxmax:.1f} '
        f'wide against a {lam:.1f} shedding wavelength, so read its amplitude, '
        f'not its shape', fontsize=11.5)
    out_png = f'figs/curvi_cyl_{sys.argv[1] if len(sys.argv) > 1 else "xu20"}_vorticity.png'
    fig.savefig(out_png, dpi=140, bbox_inches='tight')
    print('wrote', out_png)
    print(f'added region: peak {pk:.5f}, rms {rms:.5f}; '
          f'upstream x<-2: peak {float(np.abs(om[up]).max()):.5f}')
    print(f'far elements: edge rms {er:.5f}, interior rms {ir:.5f} -> '
          f'{"smooth" if abs(er - ir)/ir < 0.25 else "EDGE-CONCENTRATED"}')


if __name__ == '__main__':
    main()
