"""Does w7 = 1e-3 buy anything over 1e-4?  Measured on the CHANNEL, the only
flow here where R_7 is actually active.

WHY NOT TAYLOR-GREEN.  Both TG configurations in tgv3d.py have |R_7| = 0 on
their initial state (rotxz 1.7e-14, exactly zero to roundoff; tgv3d a 0.000%
share), because an exact TG field has divergence-free vorticity.  The sec 7E.1
order gate that reports "2.00, 2.00, 2.00, row weights on" is therefore BLIND to
w7 by construction and cannot be used to price it.  The channel seed carries
genuine transverse vorticity, which is what sec 7J used too.

sec 7J states 1e-2/1e-3/1e-4 are "bit-identical (4.8691e-08, 48315 CG)".  If that
holds here, raising 1e-4 -> 1e-3 buys no accuracy and costs ~4% conditioning, and
the sec 7S.5 recommendation must be withdrawn.

NOTE the default-argument trap: momentum_row_weights(c, w7=ROW7_WEIGHT) binds
ROW7_WEIGHT at DEF time, so monkeypatching operator.ROW7_WEIGHT does nothing at
all -- silently.  The function itself has to be wrapped.
"""
import os, sys, time
for _v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '8')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R,'scratch')); os.chdir(_R)
os.environ.setdefault('LSSEM3D_BACKEND', 'numba')
import numpy as np

SEED = os.path.join(_R, 'scratch', 'fs_seed', 'seed_ckpt.npz')
OUT  = os.path.join(_R, 'scratch', 'fosls3d_row7_channel.npz')

def main():
    import lssem3d; lssem3d.set_backend(os.environ['LSSEM3D_BACKEND'])
    from lssem3d import operator as OP, deriv as DV
    import minchan as MC

    _orig = OP.momentum_row_weights
    def force(w7):
        OP.momentum_row_weights = (lambda c, w7=None, w_mom=None, w_vort=None, _w=w7:
                                   _orig(c, w7=_w, w_mom=w_mom, w_vort=w_vort))
    d = np.load(SEED)
    s = MC.setup()
    U0 = d['U']; t0v = float(d['t']) if 't' in d else 0.0
    dt = 8.0e-4
    print(f'channel seed t={t0v:.4f}, dt={dt:g}, backend={os.environ["LSSEM3D_BACKEND"]}')
    C0 = OP.to_complex(U0)
    print(f'max|om_x|={np.abs(C0[...,OP.OX_,:]).max():.2f}  '
          f'max|om_y|={np.abs(C0[...,OP.OY_,:]).max():.2f}  '
          f'max|om_z|={np.abs(C0[...,OP.OZ_,:]).max():.2f}\n')

    def divom(U):
        C = OP.to_complex(U)
        dx = DV.ddx(C[..., OP.OX_:OP.OX_+1, :], s['D'], s['m'].facx)
        dy = DV.ddy(C[..., OP.OY_:OP.OY_+1, :], s['D'], s['m'].facy)
        dz = 1j*s['kz']*C[..., OP.OZ_:OP.OZ_+1, :]
        r = dx + dy + dz
        return float(np.sqrt((np.abs(r)**2).mean())), float(np.abs(C[..., 3:6, :]).max())

    print(f'{"w7":>8} {"CG it":>7} {"wall s":>8} {"rms div om":>12} '
          f'{"vs w7=1":>11} {"vs 1e-4":>11}')
    print('-'*64)
    res = {}
    for w7 in (1.0, 1e-2, 1e-3, 1e-4):
        force(w7)
        Minv = MC._precond(s, dt)
        Nprev = np.zeros(OP.to_complex(U0).shape[:-2] + (3, s['nk']), dtype=complex)
        tw = time.perf_counter()
        U1, _, it = MC.advance(s, U0.copy(), Nprev, dt, Minv, tol=1e-8)
        tw = time.perf_counter() - tw
        rms, om = divom(U1)
        res[w7] = (U1, it, tw, rms)
        d1 = np.abs(U1-res[1.0][0]).max()/max(np.abs(res[1.0][0]).max(),1e-300)
        d4 = '' if 1e-4 not in res else f'{np.abs(U1-res[1e-4][0]).max()/np.abs(U1).max():11.3e}'
        print(f'{w7:8.0e} {it:7d} {tw:8.2f} {rms:12.4e} {d1:11.3e} {d4:>11}')
    U4 = res[1e-4][0]
    print(f'\n  1e-3 vs 1e-4 max|dU| / max|U| = '
          f'{np.abs(res[1e-3][0]-U4).max()/np.abs(U4).max():.3e}')
    print(f'  1e-3 vs 1e-4 bitwise identical: {np.array_equal(res[1e-3][0], U4)}')
    print(f'  CG iterations 1e-3 vs 1e-4: {res[1e-3][1]} vs {res[1e-4][1]}')
    np.savez_compressed(OUT, w7=np.array([1.0,1e-2,1e-3,1e-4]),
                        it=np.array([res[w][1] for w in (1.0,1e-2,1e-3,1e-4)]),
                        wall=np.array([res[w][2] for w in (1.0,1e-2,1e-3,1e-4)]),
                        rmsdiv=np.array([res[w][3] for w in (1.0,1e-2,1e-3,1e-4)]))
    print(f'\nsaved -> {OUT}')

if __name__ == '__main__':
    main()
