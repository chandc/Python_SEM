"""Gate G3, shown rather than tabulated: the annular grid and the computed fields.

    uv run python scratch/curvi_g3_fields.py [N]

The convergence table in `curvi_g3.py` says the error falls exponentially; this
says what the solution actually looks like on the mesh that produced it.  Every
panel carries its analytic answer, because all four are known in closed form:

    u_th = A r + B/r,   u_r = 0,   omega = 2A,   p = A^2 r^2/2 + 2AB ln r - B^2/2r^2

The two that are CONSTANTS -- u_r = 0 and omega = 2A -- are the informative ones
on a curvilinear mesh.  A constant is what a metric error cannot fake: it has no
shape to hide in, so any departure is the discretisation's own, and both are
plotted against every node rather than a mean so the spread is visible.
"""
import os
import sys

_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R)
sys.path.insert(0, os.path.join(_R, 'scratch'))
os.chdir(_R)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import lssem2d
from curvi_g3 import solve, _A, _B, A_IN, B_OUT, NU, OM_IN

OUT = os.path.join(_R, 'figs', 'curvi_g3_fields.png')


def main(N=8, E_r=3, E_th=4):
    lssem2d.set_backend('numpy')
    e = solve(N, E_r, E_th)
    m, U = e['m'], e['U']
    n = m.nterm
    X, Y = m.X, m.Y
    r = np.hypot(X, Y)
    ur = (U[..., 0]*X + U[..., 1]*Y)/r
    ut = (U[..., 1]*X - U[..., 0]*Y)/r
    rr = np.linspace(A_IN, B_OUT, 400)

    fig, ax = plt.subplots(1, 4, figsize=(18, 4.4))

    # ---- 1. the grid itself
    for el in range(m.nelem):
        ax[0].plot(X[el].ravel(), Y[el].ravel(), '.', ms=2.2, color='C0', alpha=0.75)
        for idx in (0, n-1):
            ax[0].plot(X[el, idx, :], Y[el, idx, :], 'k-', lw=1.3)
            ax[0].plot(X[el, :, idx], Y[el, :, idx], 'k-', lw=1.3)
    ng = int(m.gidx.max()) + 1
    ax[0].set_aspect('equal'); ax[0].tick_params(labelsize=8)
    ax[0].set_xlabel('$x$'); ax[0].set_ylabel('$y$')
    ax[0].set_title(f'the mesh: {E_r}x{E_th} elements, $N = {N}$\n'
                    f'{ng} global nodes, no straight edge', fontsize=10)

    # ---- 2. speed, with the flow direction
    # Delaunay spans the CONVEX HULL, so the bore of the annulus gets painted as
    # if it were fluid.  Mask any triangle whose centroid falls outside the
    # annulus: the domain has a hole and the picture must show it.
    import matplotlib.tri as mtri
    tri = mtri.Triangulation(X.ravel(), Y.ravel())
    cx = X.ravel()[tri.triangles].mean(axis=1)
    cy = Y.ravel()[tri.triangles].mean(axis=1)
    cr = np.hypot(cx, cy)
    tri.set_mask((cr < A_IN*1.001) | (cr > B_OUT*0.999))
    sp = ax[1].tripcolor(tri, np.hypot(U[..., 0], U[..., 1]).ravel(),
                         shading='gouraud', cmap='viridis')
    for el in range(m.nelem):
        for idx in (0, n-1):
            ax[1].plot(X[el, idx, :], Y[el, idx, :], 'w-', lw=0.7, alpha=0.8)
            ax[1].plot(X[el, :, idx], Y[el, :, idx], 'w-', lw=0.7, alpha=0.8)
    k = max(1, n//5)
    ax[1].quiver(X[:, ::k, ::k].ravel(), Y[:, ::k, ::k].ravel(),
                 U[:, ::k, ::k, 0].ravel(), U[:, ::k, ::k, 1].ravel(),
                 color='w', scale=6, width=0.004, alpha=0.9)
    fig.colorbar(sp, ax=ax[1], shrink=0.85)
    ax[1].set_aspect('equal'); ax[1].tick_params(labelsize=8)
    ax[1].set_title(r'computed $|\mathbf{u}|$' + '\n'
                    f'inner cylinder turning, outer at rest', fontsize=10)

    # ---- 3. the azimuthal profile, every node against the analytic curve
    ax[2].plot(rr, _A*rr + _B/rr, 'k-', lw=2.0, zorder=1,
               label=r'exact  $u_\theta = Ar + B/r$')
    ax[2].plot(r.ravel(), ut.ravel(), 'o', ms=3.0, color='C3', alpha=0.5,
               zorder=2, label='computed, every node')
    ax[2].plot(r.ravel(), ur.ravel(), 's', ms=2.5, color='C0', alpha=0.6,
               label=r'computed $u_r$  (exact value 0)')
    ax[2].axhline(0.0, color='C7', ls=':', lw=1)
    ax[2].set_xlabel('$r$'); ax[2].set_ylabel('velocity component')
    ax[2].set_title(f'velocity profile\n'
                    r'$\max|u_r| = $' + f'{np.abs(ur).max():.1e}', fontsize=10)
    ax[2].legend(fontsize=8, loc='upper right'); ax[2].grid(alpha=0.3)

    # ---- 4. pressure and vorticity, both against their analytic forms
    pe = 0.5*_A**2*rr**2 + 2*_A*_B*np.log(rr) - 0.5*_B**2/rr**2
    pc = U[..., 2].ravel()
    # p is determined up to a constant (a node is pinned); align the means
    pex_nodes = 0.5*_A**2*r.ravel()**2 + 2*_A*_B*np.log(r.ravel()) - 0.5*_B**2/r.ravel()**2
    pc = pc - pc.mean() + pex_nodes.mean()
    # TWO QUANTITIES ON TWO AXES, so every series is named in ONE legend and the
    # right-hand axis is coloured to match its own curves.  Without that the
    # vorticity diamonds read as pressure values -- they are not; they sit on the
    # right axis, which spans only 2A +/- 0.02 so that the scatter is visible at
    # all.
    h1, = ax[3].plot(rr, pe, '-', lw=2.0, color='k', label='exact $p$  (left)')
    h2, = ax[3].plot(r.ravel(), pc, 'o', ms=3.0, color='C1', alpha=0.55,
                     label='computed $p$, every node  (left)')
    axb = ax[3].twinx()
    h3, = axb.plot(rr, np.full_like(rr, 2*_A), '--', lw=2.0, color='C4',
                   label=r'exact $\omega \equiv 2A$  (right)')
    h4, = axb.plot(r.ravel(), U[..., 3].ravel(), 'D', ms=2.6, color='C4', alpha=0.5,
                   label=r'computed $\omega$, every node  (right)')
    axb.set_ylabel(r'$\omega$', color='C4')
    axb.tick_params(labelcolor='C4', labelsize=8)
    axb.spines['right'].set_color('C4')
    axb.set_ylim(2*_A - 0.02, 2*_A + 0.02)
    ax[3].set_xlabel('$r$'); ax[3].set_ylabel('$p$')
    ax[3].set_title('pressure (left axis) and vorticity (right axis)\n'
                    r'$2A = $' + f'{2*_A:.4f},  '
                    r'$\max|\omega - 2A| = $' + f'{e["omcore"]:.1e}', fontsize=10)
    ax[3].legend(handles=[h1, h2, h3, h4], fontsize=7, loc='upper left',
                 framealpha=0.9)
    ax[3].grid(alpha=0.3)

    fig.suptitle(f'Taylor-Couette between $r = {A_IN}$ and $r = {B_OUT}$, '
                 r'$\Omega_{\rm in} = $' + f'{OM_IN}, ' + r'$\nu = $' + f'{NU}, '
                 f'Re = {OM_IN*A_IN*(B_OUT-A_IN)/NU:.0f}   —   '
                 r'relative $L^2$ error in $\mathbf{u}$: ' + f'{e["uv"]:.2e}, '
                 r'$\max|\nabla\cdot\mathbf{u}| = $' + f'{e["divmax"]:.1e}',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=140)
    print(f'N = {N}: |u| err {e["uv"]:.3e}   u_r {e["ur"]:.3e}   p {e["p"]:.3e}   '
          f'max|div u| {e["divmax"]:.3e}   max|om-2A| {e["omcore"]:.3e}')
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
