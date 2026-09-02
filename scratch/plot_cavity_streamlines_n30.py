"""Predicted streamlines, Ghia Re=1000 cavity at N=30.

    uv run --quiet python scratch/plot_cavity_streamlines_n30.py

Plots the SAVED field scratch/pmg_ghia_cavity_lad_N30_r1.npz -- never re-solves.
16 elements at N=30 (58,564 DOF), p-multigrid ladder (15,7,3,2) + direct coarse.

The streamfunction is obtained by integrating u = dpsi/dy upward from the floor,
which is path-independent only for a divergence-free field.  LSSEM only PENALISES
div u (project lesson L5), so the residual divergence is reported alongside --
if it were large the psi contours would not be meaningful.
"""
import os
import sys

_SC = os.path.dirname(os.path.abspath(__file__))
_R = os.path.dirname(_SC)
sys.path.insert(0, _R); sys.path.insert(0, _SC)
os.chdir(_R)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

import lssem2d
lssem2d.set_backend('numpy')
from lssem2d.mesh import build_channel
from lssem2d.operators import dUdx, dUdy

N, EX = 30, 4
# Ghia et al. (1982) reported vortex centres at Re = 1000.
GHIA_CENTRES = {'primary': (0.5313, 0.5625),
                'BL1': (0.0859, 0.0781),
                'BR1': (0.8594, 0.1094)}
LEVELS = [-0.1175, -0.115, -0.11, -0.1, -0.09, -0.07, -0.05, -0.03, -0.01,
          -1e-4, -1e-5, -1e-7, 0.0, 1e-8, 1e-6, 1e-5, 5e-5, 1e-4, 2.5e-4,
          5e-4, 1e-3, 1.5e-3, 3e-3]


def main():
    z = np.load(f'{_SC}/pmg_ghia_cavity_lad_N30_r1.npz', allow_pickle=True)
    U = z['U']
    m = build_channel(1.0, 1.0, EX, EX, N, bcs=(1, 1, 1, 2))
    n = N + 1

    D = _D(m, N)
    ux = dUdx(np.ascontiguousarray(U[..., 0]), D, m.facx)
    uy = dUdy(np.ascontiguousarray(U[..., 0]), D, m.facy)
    vx = dUdx(np.ascontiguousarray(U[..., 1]), D, m.facx)
    vy = dUdy(np.ascontiguousarray(U[..., 1]), D, m.facy)
    wq = m.wq
    L2 = lambda a: float(np.sqrt((wq*a**2).sum()))
    dv = ux + vy
    # ||div u||/||u|| carries units of 1/L and is not interpretable; the
    # dimensionless ratio is against ||grad u||.
    gradn = float(np.sqrt((wq*(ux**2 + uy**2 + vx**2 + vy**2)).sum()))
    div = L2(dv)/gradn

    X = np.stack([np.repeat(m.xnod[e][:, None], n, axis=1) for e in range(m.nelem)])
    Y = np.stack([np.repeat(m.ynod[e][None, :], n, axis=0) for e in range(m.nelem)])
    pts = np.column_stack([X.ravel(), Y.ravel()])
    ug, vg = U[..., 0].ravel(), U[..., 1].ravel()

    g = 401
    xi = np.linspace(0, 1, g); yi = np.linspace(0, 1, g)
    XI, YI = np.meshgrid(xi, yi)
    ui = griddata(pts, ug, (XI, YI), method='cubic')
    vi = griddata(pts, vg, (XI, YI), method='cubic')
    ui = np.nan_to_num(ui); vi = np.nan_to_num(vi)

    # psi from u = dpsi/dy, integrated up from the floor (psi = 0 on the walls)
    psi = np.zeros_like(ui)
    dy = yi[1] - yi[0]
    psi[1:] = np.cumsum(0.5*(ui[1:] + ui[:-1])*dy, axis=0)

    Xn = np.stack([np.repeat(m.xnod[e][:, None], n, axis=1) for e in range(m.nelem)])
    Yn = np.stack([np.repeat(m.ynod[e][None, :], n, axis=0) for e in range(m.nelem)])
    corner = ((Xn < 0.02) | (Xn > 0.98)) & (Yn > 0.98)
    frac = float((wq*dv**2)[corner].sum()/(wq*dv**2).sum())
    print(f'N={N}, {EX*EX} elements, {int(z["gdof"])} DOF, {int(z["steps"])} steps')
    print(f'||div u||/||grad u|| = {div:.3e}   '
          f'(max|div| = {np.abs(dv).max():.1f} overall, '
          f'{np.abs(dv[~corner]).max():.1f} off the lid corners)')
    print(f'{frac*100:.1f}% of ||div||^2 sits in the two LID CORNERS, where u jumps')
    print(f'1 -> 0 discontinuously -- the classic cavity singularity, not a solver defect.')
    print(f'psi range: [{psi.min():.6f}, {psi.max():.6f}]  (Ghia psi_min = -0.117929)\n')

    def centre(mask, sign):
        p = np.where(mask, psi*sign, -np.inf)
        k = np.unravel_index(np.argmax(p), p.shape)
        return xi[k[1]], yi[k[0]], psi[k]

    IN = (XI > 0.02) & (XI < 0.98) & (YI > 0.02) & (YI < 0.98)
    print(f'{"vortex":>9}{"x (ours)":>10}{"y (ours)":>10}{"psi":>12}   '
          f'{"x (Ghia)":>9}{"y (Ghia)":>9}')
    px, py, pp = centre(IN, -1)
    gxp, gyp = GHIA_CENTRES['primary']
    print(f'{"primary":>9}{px:10.4f}{py:10.4f}{pp:12.6f}   {gxp:9.4f}{gyp:9.4f}')
    for tag, box in (('BL1', (XI < 0.3) & (YI < 0.3)),
                     ('BR1', (XI > 0.7) & (YI < 0.3))):
        cx, cy, cp = centre(box & IN, +1)
        gx_, gy_ = GHIA_CENTRES[tag]
        print(f'{tag:>9}{cx:10.4f}{cy:10.4f}{cp:12.3e}   {gx_:9.4f}{gy_:9.4f}')

    fig = plt.figure(figsize=(14.5, 5.6))
    a0 = fig.add_subplot(1, 3, 1)
    spd = np.sqrt(ui**2 + vi**2)
    a0.streamplot(xi, yi, ui, vi, density=2.6, color=spd, cmap='viridis',
                  linewidth=0.8, arrowsize=0.7)
    a0.set_aspect('equal'); a0.set_xlim(0, 1); a0.set_ylim(0, 1)
    a0.set_title('streamlines (lid moves →)'); a0.set_xlabel('x'); a0.set_ylabel('y')

    a1 = fig.add_subplot(1, 3, 2)
    a1.contour(XI, YI, psi, levels=LEVELS, colors='k', linewidths=0.7)
    for tag, (gx_, gy_) in GHIA_CENTRES.items():
        a1.plot(gx_, gy_, 'r+', ms=11, mew=1.8)
    a1.plot(px, py, 'bo', ms=6, mfc='none', mew=1.6)
    a1.set_aspect('equal'); a1.set_xlim(0, 1); a1.set_ylim(0, 1)
    a1.set_title('ψ contours (Ghia levels)\nred + = Ghia centres, blue ○ = ours')
    a1.set_xlabel('x')

    a2 = fig.add_subplot(1, 3, 3)
    a2.contour(XI, YI, psi, levels=np.linspace(0, 2.0e-3, 22), colors='k',
               linewidths=0.7)
    a2.plot(*GHIA_CENTRES['BR1'], 'r+', ms=11, mew=1.8)
    a2.set_xlim(0.6, 1.0); a2.set_ylim(0.0, 0.4); a2.set_aspect('equal')
    a2.set_title('bottom-right corner vortex (BR1)'); a2.set_xlabel('x')

    fig.suptitle(f'Lid-driven cavity Re = 1000 — LSSEM {EX}×{EX} elements at '
                 f'N = {N} ({int(z["gdof"])} DOF), p-multigrid ladder + direct '
                 f'coarse   |   ||div u||/||grad u|| = {div:.2e}', fontsize=11)
    fig.tight_layout()
    out = f'{_SC}/cavity_n30_streamlines.png'
    fig.savefig(out, dpi=145, bbox_inches='tight')
    print(f'\nsaved -> {out}')


def _D(m, N):
    from lssem2d.lgl import diff_matrix
    return diff_matrix(N)


if __name__ == '__main__':
    main()
