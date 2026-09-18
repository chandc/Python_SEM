"""Draw the cylinder-in-box mesh, current vs the proposed lateral extension.

The figure exists to check one claim by eye: that widening the box changes the
domain and NOTHING else.  That claim is false for the obvious way of widening
it -- `_geom_edges` normalises its widths to span a0..a1, so raising H alone
rescales every lateral division and the wide mesh is not a refinement of the
narrow one.  Behr et al. (1995) ran their domain study on NESTED meshes for
exactly this reason, and `curvi.nested_lateral_edges` reproduces that here: the
reference mesh's edges survive untouched and new elements are appended outside
them.  Panel (c) is the proof -- the two ladders coincide up to y = 10.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import lssem2d
from lssem2d import curvi

N = 6                       # the order the N sweep says resolves St and C_D
H_REF = 10.0                # current half-height
H_NEW, N_EXTRA = 20.0, 2    # proposed: H_full 20 -> 40, two elements appended
GREY, ORANGE, BLUE = '0.42', '#c2410c', '#1d4ed8'


def edges_of(m, e):
    X, Y = m.X[e], m.Y[e]
    return ((X[0, :], Y[0, :]), (X[-1, :], Y[-1, :]),
            (X[:, 0], Y[:, 0]), (X[:, -1], Y[:, -1]))


def draw(ax, m, keep=None, colour=GREY, lw=0.45, z=1):
    for e in range(m.nelem):
        if keep is not None and not keep(m.Y[e].min(), m.Y[e].max()):
            continue
        for xs, ys in edges_of(m, e):
            ax.plot(xs, ys, '-', color=colour, lw=lw, zorder=z)


def dof(m):
    return (int(m.gidx.max()) + 1)*4


def main():
    lssem2d.set_backend('numpy')
    ys = curvi.nested_lateral_edges(H_NEW, N_EXTRA, H=H_REF)
    mn = curvi.build_cylinder_box(N=N)
    mw = curvi.build_cylinder_box(N=N, ys_side=ys)
    for m in (mn, mw):
        m.compute_global_indices()

    pn = set(zip(*[np.round(v.ravel(), 9) for v in (mn.X, mn.Y)]))
    pw = set(zip(*[np.round(v.ravel(), 9) for v in (mw.X, mw.Y)]))
    shared = len(pn & pw)/len(pn)

    fig = plt.figure(figsize=(14.5, 7.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.15, 1.0],
                          hspace=0.30, wspace=0.20)
    ax0, ax1 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
    axz, axl = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])

    inner = lambda lo, hi: lo >= -H_REF - 1e-9 and hi <= H_REF + 1e-9
    outer = lambda lo, hi: not inner(lo, hi)

    for ax, m, H, tag in (
            (ax0, mn, H_REF, f'(a) current   H_full = {2*H_REF:.0f}   '
                             f'{mn.nelem} elements   {dof(mn):,} dof'),
            (ax1, mw, H_NEW, f'(b) proposed  H_full = {2*H_NEW:.0f}   '
                             f'{mw.nelem} elements   {dof(mw):,} dof   '
                             f'(+{dof(mw)/dof(mn) - 1:.0%})')):
        draw(ax, m, keep=inner, colour=GREY)
        draw(ax, m, keep=outer, colour=ORANGE, lw=0.6, z=2)
        ax.add_patch(plt.Circle((0, 0), 0.5, fc='0.80', ec='k', lw=0.8, zorder=3))
        for s in (+1, -1):
            ax.axhline(s*H, color=BLUE, lw=1.3, ls='--', zorder=4)
        ax.set_xlim(-11.5, 26.5); ax.set_ylim(-H_NEW - 2.5, H_NEW + 2.5)
        ax.set_aspect('equal'); ax.set_ylabel('y')
        ax.set_title(tag, fontsize=10.5, loc='left')
        ax.text(0, H + 0.9, 'symmetry  (v = 0, omega = 0)', fontsize=7.5,
                color=BLUE, ha='center', va='bottom')
        ax.text(-11.2, -H_NEW - 1.2, 'free stream  x = -10', fontsize=7.5, color=BLUE)
        ax.text(26.2, -H_NEW - 1.2, 'Dong outflow  x = 25', fontsize=7.5,
                color=BLUE, ha='right')
    ax1.set_xlabel('x')

    # ---- (c) near field: unchanged by construction ----
    draw(axz, mw, colour=ORANGE, lw=2.4)
    draw(axz, mn, colour=GREY, lw=0.9)
    axz.add_patch(plt.Circle((0, 0), 0.5, fc='0.80', ec='k', lw=1.0, zorder=3))
    axz.set_xlim(-3.4, 3.4); axz.set_ylim(-3.4, 3.4); axz.set_aspect('equal')
    axz.set_title('(c) near field, wide (thick orange) under narrow (thin grey)\n'
                  'no orange shows = the meshes coincide', fontsize=9.5, loc='left')
    axz.set_xlabel('x'); axz.set_ylabel('y')

    # ---- (d) the lateral edge ladders, which is where nesting is won or lost ----
    naive = curvi._geom_edges(1.5, H_NEW, 4 + N_EXTRA, 1.45)
    ref = curvi._geom_edges(1.5, H_REF, 4, 1.45)
    for k, (lab, e, c, mk) in enumerate((
            (f'current, H = {H_REF:.0f}', ref, GREY, 'o'),
            ('proposed: nested', ys, ORANGE, 's'),
            ('naive: raise H, ny_side', naive, '#6b7280', 'x'))):
        axl.plot(e, np.full_like(e, -k), mk + '-', color=c, ms=5, lw=1.2,
                 mfc='none' if k else c)
        axl.text(-0.6, -k, lab, ha='right', va='center', fontsize=8.5, color=c)
    axl.axvspan(1.5, H_REF, color=GREY, alpha=0.10)
    axl.text((1.5 + H_REF)/2, 0.42, 'shared with the current mesh',
             ha='center', fontsize=8, color=GREY)
    axl.set_xlim(-11.5, 21.5); axl.set_ylim(-2.7, 0.8)
    axl.set_yticks([]); axl.set_xlabel('y of the lateral element edges')
    axl.set_title('(d) why the naive widening is not a domain test:\n'
                  'it moves every interior edge as well', fontsize=9.5, loc='left')
    axl.spines[['left', 'right', 'top']].set_visible(False)

    fig.suptitle('Cylinder in a box: first lateral extension, H_full 20 -> 40   '
                 f'(N = {N}, ratio_out = 1.45 fixed, Lu/Ld untouched)   '
                 f'narrow-mesh nodes retained: {shared:.1%}', fontsize=12)
    out = 'figs/curvi_cyl_box_grid_H40.png'
    fig.savefig(out, dpi=140, bbox_inches='tight')
    print('wrote', out)
    print('nested edges  :', np.round(ys, 4))
    print('naive edges   :', np.round(naive, 4))
    print(f'shared nodes  : {shared:.4%}   dof {dof(mn):,} -> {dof(mw):,}')


if __name__ == '__main__':
    main()
