"""Streamwise vorticity on successive CROSS-STREAM (z, y) planes.

The classic view of quasi-streamwise vortices: in a (z,y) section they appear as
compact counter-rotating patches hugging the wall, and stepping along x shows
whether the same structures PERSIST -- which is what "streamwise" means.  A
vortex of length ~200-300 wall units should survive several stations; turbulence
that is merely noisy would decorrelate between them.

omega_x is a PRIMARY UNKNOWN in FOSLS, so these are the solver's own values.
Planes are evaluated from the spectral interpolant (semplot.cross), not
triangulated -- see semplot.py.
"""
import os, sys, glob
for _v in ('OMP_NUM_THREADS',): os.environ.setdefault(_v, '4')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, 'scratch')); os.chdir(_R)
import numpy as np, semplot
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.path.join(_R, 'scratch', 'run01_omegax_planes.png')
RT, YMAX, NPL = 180.0, 110.0, 6

def main():
    import lssem3d; lssem3d.set_backend('numpy')
    from lssem3d import operator as OP, fourier as FR
    import minchan as MC
    s = MC.setup(); m = s['m']; N = s['N']; nz = s['nz']; LZ = s['lz']
    f = sorted(glob.glob(os.path.join(_R, 'scratch/run01_ck/checkpoint_*.npz')))[-1]
    d = np.load(f); t = float(d['t'])
    ox = FR.to_physical(np.ascontiguousarray(
        OP.to_complex(d['U'])[..., OP.OX_:OP.OX_+1, :]), nz)[..., 0, :]

    LX = np.pi
    stations = np.linspace(0, LX, NPL, endpoint=False)
    imgs = [semplot.cross(ox, m, N, xs, RT=RT, ymax_plus=YMAX, nz=nz)
            for xs in stations]
    lim = np.percentile(np.abs(np.array([i[2] for i in imgs])), 99)

    fig, axes = plt.subplots(NPL, 1, figsize=(9.0, 2.05*NPL), sharex=True)
    for ax, xs, (z, y, img) in zip(axes, stations, imgs):
        sc = ax.pcolormesh(z*LZ*RT, y, img, cmap='RdBu_r', vmin=-lim, vmax=lim,
                           shading='gouraud', rasterized=True)
        ax.contour(z*LZ*RT, y, img, levels=[-0.45*lim, 0.45*lim], colors='k',
                   linewidths=0.4, alpha=.5)
        ax.set(ylabel='$y^+$', ylim=(0, YMAX))
        ax.text(0.985, 0.88, f'$x^+$={xs*RT:.0f}', transform=ax.transAxes,
                ha='right', va='top', fontsize=10,
                bbox=dict(fc='w', ec='0.6', alpha=.85, pad=2))
        plt.colorbar(sc, ax=ax, fraction=0.020, pad=0.01)
    axes[-1].set_xlabel('$z^+$')
    fig.suptitle(f'run01 FOSLS-3D minimal channel $Re_\\tau$=180, t={t:.2f} — '
                 f'$\\omega_x$ on cross-stream planes  (colour scale shared)', y=0.997)
    fig.tight_layout(); fig.savefig(OUT, dpi=125, bbox_inches='tight')

    # does a structure PERSIST from one station to the next?
    print(f't={t:.2f}, {NPL} stations, dx+ = {stations[1]*RT:.0f} wall units\n')
    print(f'{"x+":>6}  {"rms":>7}  {"corr with x+=0":>15}  {"corr with previous":>19}')
    ref = imgs[0][2]
    for k, (xs, (z, y, img)) in enumerate(zip(stations, imgs)):
        c0 = np.corrcoef(ref.ravel(), img.ravel())[0, 1]
        cp = np.corrcoef(imgs[k-1][2].ravel(), img.ravel())[0, 1] if k else 1.0
        print(f'{xs*RT:6.0f}  {img.std():7.2f}  {c0:15.3f}  {cp:19.3f}')
    print('\n  high station-to-station correlation => the vortices are genuinely')
    print('  streamwise-coherent, not independent noise in each plane.')
    print(f'saved -> {OUT}')

if __name__ == '__main__':
    main()
