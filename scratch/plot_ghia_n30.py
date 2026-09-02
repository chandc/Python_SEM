"""Ghia Re=1000 cavity at N=30: centreline profiles against Ghia Tables I & II.

    uv run --quiet python scratch/plot_ghia_n30.py

Plots the SAVED field scratch/pmg_ghia_cavity_lad_N30_r1.npz -- never re-solves.
16 elements at N=30 (58,564 DOF), p-multigrid ladder 30->15->7->3->2 with a
direct coarse solve.
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

import lssem2d
lssem2d.set_backend('numpy')
from lssem2d.mesh import build_channel
import pmg_ghia_cavity as G

N, EX = 30, 4


def main():
    z = np.load(f'{_SC}/pmg_ghia_cavity_lad_N30_r1.npz', allow_pickle=True)
    U = z['U']
    m = build_channel(1.0, 1.0, EX, EX, N, bcs=(1, 1, 1, 2))
    y, u = G.centreline_u(U, m, N)
    x, v = G.centreline_v(U, m, N)

    gh = np.load(f'{_R}/cavity_re1000_data.npz')
    gy, gu = gh['ghia_y'], gh['ghia_u']
    o = np.argsort(G.GHIA_X)
    gx, gv = G.GHIA_X[o], G.GHIA_V[o]

    ui = np.interp(gy, y, u)
    vi = np.interp(gx, x, v)

    print(f'N={N}, {EX*EX} elements, {int(z["gdof"])} DOF, '
          f'{int(z["steps"])} steps, ladder r1\n')
    print(f'{"y":>8}{"u (ours)":>11}{"u (Ghia)":>11}{"diff":>11}   |'
          f'{"x":>8}{"v (ours)":>11}{"v (Ghia)":>11}{"diff":>11}')
    for i in range(len(gy)):
        j = len(gy)-1-i
        print(f'{gy[i]:8.4f}{ui[i]:11.5f}{gu[i]:11.5f}{ui[i]-gu[i]:+11.5f}   |'
              f'{gx[j]:8.4f}{vi[j]:11.5f}{gv[j]:11.5f}{vi[j]-gv[j]:+11.5f}')
    print(f'\n  RMS  u = {np.sqrt(np.mean((ui-gu)**2)):.4e}   '
          f'v = {np.sqrt(np.mean((vi-gv)**2)):.4e}')
    print(f'  MAX |diff|  u = {np.abs(ui-gu).max():.4e} at y={gy[np.argmax(np.abs(ui-gu))]:.4f}'
          f'   v = {np.abs(vi-gv).max():.4e} at x={gx[np.argmax(np.abs(vi-gv))]:.4f}')

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.9))
    ax[0].plot(u, y, '-', lw=1.8, color='#1f77b4', label=f'LSSEM N={N}')
    ax[0].plot(gu, gy, 'o', ms=6, mfc='none', mew=1.5, color='crimson',
               label='Ghia et al. (1982)')
    ax[0].set_xlabel('u'); ax[0].set_ylabel('y')
    ax[0].set_title('u on the vertical centreline  x = 0.5')
    ax[0].axvline(0, color='0.8', lw=0.8); ax[0].legend(); ax[0].grid(alpha=.3)

    ax[1].plot(x, v, '-', lw=1.8, color='#1f77b4', label=f'LSSEM N={N}')
    ax[1].plot(gx, gv, 's', ms=6, mfc='none', mew=1.5, color='crimson',
               label='Ghia et al. (1982)')
    ax[1].set_xlabel('x'); ax[1].set_ylabel('v')
    ax[1].set_title('v on the horizontal centreline  y = 0.5')
    ax[1].axhline(0, color='0.8', lw=0.8); ax[1].legend(); ax[1].grid(alpha=.3)

    ax[2].plot(ui-gu, gy, 'o-', ms=4, color='#1f77b4', label='u error (vs y)')
    ax[2].plot(vi-gv, gx, 's-', ms=4, color='seagreen', label='v error (vs x)')
    ax[2].axvline(0, color='k', lw=0.8)
    ax[2].set_xlabel('ours − Ghia'); ax[2].set_ylabel('y  /  x')
    ax[2].set_title(f'pointwise difference\nRMS u={np.sqrt(np.mean((ui-gu)**2)):.2e}, '
                    f'v={np.sqrt(np.mean((vi-gv)**2)):.2e}')
    ax[2].legend(); ax[2].grid(alpha=.3)

    fig.suptitle(f'Lid-driven cavity Re = 1000 — LSSEM {EX}x{EX} elements at '
                 f'N = {N} ({int(z["gdof"])} DOF), p-multigrid ladder '
                 f'{G.ladder(N)} + direct coarse solve', fontsize=11)
    fig.tight_layout()
    out = f'{_SC}/ghia_n30_profiles.png'
    fig.savefig(out, dpi=145, bbox_inches='tight')
    print(f'\nsaved -> {out}')


if __name__ == '__main__':
    main()
