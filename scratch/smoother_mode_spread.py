"""Does one shared rho across all Fourier modes cripple the 3D smoother?

estimate_lambda_max sums over the WHOLE array, mode axis included, so Chebyshev4
gets a single rho = max over modes.  Each k_z is an independent operator, so a
mode whose true lam_max is 100x smaller than the bound receives a polynomial
designed for an interval 100x too wide -- it damps almost nothing there.  2D has
no mode axis, which is a candidate for why the 2D ladder is p-independent and
the 3D one is not.

Measures lam_max(M^-1 A) PER MODE and reports the spread.
"""
import os, sys
for _v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '8')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R,'scratch')); os.chdir(_R)
os.environ.setdefault('LSSEM3D_BACKEND', 'numba')
import numpy as np

def main():
    import lssem3d; lssem3d.set_backend(os.environ['LSSEM3D_BACKEND'])
    from lssem3d import operator as OP, precond as P3, solver3d as S3, timestep as T
    import minchan as MC
    s = MC.setup(); dt = 8e-4
    cc = T.implicit_coeff(dt, 0); rw = OP.momentum_row_weights(cc)
    lv = P3._Level(s['m'], s['nk'], s['nz'], s['nu'], cc, s['kz'], 0.0, rw,
                   False, mask=s['mask'])
    shape = lv.shape
    rng = np.random.default_rng(0)

    # power iteration run INDEPENDENTLY per mode: the operator never couples
    # modes, so a per-mode normalisation isolates each spectrum.
    v = rng.standard_normal(shape)*lv.mask
    nrm = lambda x: np.sqrt((x*x).sum(axis=(0,1,2,3)))     # per mode
    v /= np.maximum(nrm(v), 1e-300)
    lam = np.zeros(shape[-1])
    for _ in range(40):
        w = lv.M_inv*lv.A(v)
        nw = nrm(w)
        lam = nw/np.maximum(nrm(v), 1e-300)
        v = w/np.maximum(nw, 1e-300)
    print(f'channel p=8, c={cc:.0f}, nk={shape[-1]}\n')
    print('  k  |    k_z   |  lam_max(M^-1 A)')
    for k in range(shape[-1]):
        print(f'{k:3d}  | {float(s["kz"][k]):8.2f} | {lam[k]:.4e}')
    print(f'\nshared rho would be 1.3 * {lam.max():.4e} = {1.3*lam.max():.4e}')
    print(f'spread lam_max(worst)/lam_max(best) = {lam.max()/lam.min():.2f}x')
    print(f'the mode with the SMALLEST lam_max is over-damped by a factor '
          f'{lam.max()/lam.min():.1f} in interval width')

if __name__ == '__main__':
    main()
