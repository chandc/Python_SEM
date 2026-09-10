"""omega_x cross-planes and streamwise coherence: FOSLS vs FRACTIONAL STEP.

The fractional-step field carries no vorticity, so its omega_x is computed the
way its users would -- by spectral differentiation,

    omega_x = dw/dy - i k_z v

against FOSLS's PRIMARY unknown.  That difference is the point of the
comparison, not a caveat to it.

The two are independent realisations at different instants (FS at its own
t=15.95, ~16 turnovers and long converged; FOSLS at t=4.32), so nothing
point-by-point is meaningful -- only the STATISTICS and the STRUCTURE.  The
streamwise correlation is the quantity of interest, because the FOSLS run
measured a coherence far shorter than the canonical 200-300 wall units and the
question is whether the fractional-step field on the SAME MESH does better.
"""
import os, sys, glob
for _v in ('OMP_NUM_THREADS',): os.environ.setdefault(_v, '4')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, 'scratch')); os.chdir(_R)
import numpy as np, semplot
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.path.join(_R, 'scratch', 'omegax_fosls_vs_fs.png')
RT, YMAX = 180.0, 110.0
FS = os.path.join(_R, 'results/minchan_re180_E/state_t15.95.npz')

def main():
    import lssem3d; lssem3d.set_backend('numpy')
    from lssem3d import operator as OP, fourier as FR, deriv as DV
    import minchan as MC
    s = MC.setup(); m = s['m']; N = s['N']; nz = s['nz']; kz = s['kz']; LZ = s['lz']

    # ---- FOSLS: omega_x is a primary unknown ----
    f = sorted(glob.glob(os.path.join(_R, 'scratch/run01_ck/checkpoint_*.npz')))[-1]
    dF = np.load(f); tF = float(dF['t'])
    oxF = FR.to_physical(np.ascontiguousarray(
        OP.to_complex(dF['U'])[..., OP.OX_:OP.OX_+1, :]), nz)[..., 0, :]
    uF = FR.to_physical(np.ascontiguousarray(
        OP.to_complex(dF['U'])[..., OP.U_:OP.U_+1, :]), nz)[..., 0, :]

    # ---- fractional step: omega_x = dw/dy - i k_z v, by differentiation ----
    dS = np.load(FS); tS = float(dS['t']); Uc = dS['U']
    oxc = (DV.ddy(np.ascontiguousarray(Uc[..., 2:3, :]), s['D'], m.facy)
           - 1j*kz*Uc[..., 1:2, :])
    oxS = FR.to_physical(np.ascontiguousarray(oxc), nz)[..., 0, :]
    uS = FR.to_physical(np.ascontiguousarray(Uc[..., 0:1, :]), nz)[..., 0, :]

    LX = np.pi
    stations = np.array([0.0, LX/3, 2*LX/3])
    fig, axes = plt.subplots(len(stations), 2, figsize=(15.0, 2.3*len(stations)),
                             sharex=True, sharey=True)
    allimg = []
    for col, (fld, lab) in enumerate(((oxF, f'FOSLS  (primary $\\omega_x$)  t={tF:.2f}'),
                                      (oxS, f'fractional step (differentiated)  t={tS:.2f}'))):
        for row, xs in enumerate(stations):
            z, y, img = semplot.cross(fld, m, N, xs, RT=RT, ymax_plus=YMAX, nz=nz)
            allimg.append((col, row, z, y, img))
    lim = np.percentile(np.abs(np.concatenate([i[4].ravel() for i in allimg])), 99)
    for col, row, z, y, img in allimg:
        ax = axes[row, col]
        sc = ax.pcolormesh(z*LZ*RT, y, img, cmap='RdBu_r', vmin=-lim, vmax=lim,
                           shading='gouraud', rasterized=True)
        ax.set(ylim=(0, YMAX))
        if col == 0: ax.set_ylabel('$y^+$')
        if row == 0:
            ax.set_title(('FOSLS  —  $\\omega_x$ a PRIMARY unknown' if col == 0
                          else 'fractional step  —  $\\omega_x$ DIFFERENTIATED'),
                         fontsize=11)
        ax.text(0.985, 0.87, f'$x^+$={stations[row]*RT:.0f}', transform=ax.transAxes,
                ha='right', va='top', fontsize=9,
                bbox=dict(fc='w', ec='0.6', alpha=.85, pad=2))
    for ax in axes[-1]: ax.set_xlabel('$z^+$')
    fig.colorbar(sc, ax=axes, fraction=0.018, pad=0.01)
    fig.suptitle(f'$\\omega_x$ cross-stream planes, SAME MESH — shared colour scale', y=0.995)
    fig.savefig(OUT, dpi=125, bbox_inches='tight')

    # ---- streamwise coherence, the quantitative comparison ----
    def coh(fld, yv):
        xf, zf, img = semplot.plane(fld, m, N, yv, RT=RT, nx_per_elem=16, nz=nz)
        a = img - img.mean(axis=1, keepdims=True)
        F = np.fft.rfft(a, axis=1)
        R = np.fft.irfft((F*np.conj(F)).real, a.shape[1], axis=1).mean(axis=0)
        R /= R[0]
        dx = np.arange(len(R))*(LX/len(R))*RT
        h = np.flatnonzero(R[:len(R)//2] < 0.5)
        return (dx[h[0]] if len(h) else np.nan), R.std(), np.abs(fld).std()
    print(f'FOSLS t={tF:.2f}   fractional step t={tS:.2f}   (same 6x18 N=8 Nz=32 mesh)\n')
    print(f'{"quantity":22s} {"FOSLS":>18} {"fractional step":>18}   [canonical]')
    for nm, fF, fS, yv, ref in (('omega_x  y+=15', oxF, oxS, 15., '~200-300'),
                                ('omega_x  y+=30', oxF, oxS, 30., '~200-300'),
                                ("u'       y+=15", uF,  uS,  15., 'O(1000)')):
        LF, _, rF = coh(fF, yv); LS, _, rS = coh(fS, yv)
        print(f'{nm:22s}  L(R=0.5) dx+={LF:6.0f}     L(R=0.5) dx+={LS:6.0f}   [{ref}]')
    for nm, fF, fS, yv in (('omega_x rms  y+=15', oxF, oxS, 15.),
                           ('omega_x rms  y+=30', oxF, oxS, 30.)):
        _, _, a = coh(fF, yv); _, _, b = coh(fS, yv)
        print(f'{nm:22s} {a:18.2f} {b:18.2f}')
    print(f'\nsaved -> {OUT}')

if __name__ == '__main__':
    main()
