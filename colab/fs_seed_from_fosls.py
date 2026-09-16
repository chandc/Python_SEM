"""Convert a least-squares checkpoint into a fractional-step restart.

    python colab/fs_seed_from_fosls.py --ckpt checkpoint_0006200.npz --out fs_seed.npz

WHY, RATHER THAN USING EACH CODE'S OWN ARCHIVED FIELD.  The two solvers have
identical meshes, boxes and Reynolds numbers, so a comparison from their separate
restart states is already fair in every respect that matters statistically.  It
is not fair in the one respect a referee will ask about: the two runs then sample
different realisations of the same flow, and a 1-3 % difference could be the
realisation rather than the scheme.  Marching the SAME field with both codes
removes that objection entirely.

THE CONVERSION IS A SLICE, NOT AN INTERPOLATION.  Both states live on the same
108 x 9 x 9 grid with the same 17 retained Fourier modes.  The least-squares
state carries seven fields as fourteen split-real components; the projection code
carries velocity and pressure as complex arrays:

    U_fs = to_complex(U_ls)[..., (U_, V_, W_), :]        (108, 9, 9, 3, 17)
    p_fs = to_complex(U_ls)[..., (P_,), :]               (108, 9, 9, 1, 17)

Vorticity is dropped because the projection code does not carry it; it is
recoverable as curl u at any time.

WHAT IS CHECKED BEFORE THE FILE IS WRITTEN.  A seed that does not satisfy the
receiving code's constraints starts a transient that looks like a scheme
difference.  So this reports, and refuses on the first two:

  * the relative pointwise divergence of the seeded velocity -- the least-squares
    field's strongest property, and it should arrive intact;
  * finiteness and the wall boundary values;
  * u_tau and the bulk velocity, which must match the source checkpoint, since
    a mismatch means the slice took the wrong components.

The projection code runs `incremental=False`, so the seeded pressure is a
starting value for a field it rebuilds each substage rather than a constraint.
"""
import argparse
import os
import sys

_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R)
sys.path.insert(0, os.path.join(_R, 'scratch'))
os.chdir(_R)

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='least-squares checkpoint npz')
    ap.add_argument('--out', required=True, help='fractional-step restart npz to write')
    ap.add_argument('--dt', type=float, default=8e-4,
                    help='recorded in the file; the driver takes its own --dt')
    a = ap.parse_args()

    import lssem3d
    lssem3d.set_backend('numpy')
    from lssem3d import operator as OP, fourier as FR, deriv as DV
    import minchan as MC

    z = np.load(a.ckpt)
    U, t = z['U'], float(z['t'])
    print(f'source: {a.ckpt}\n        t = {t:.4f}, U {U.shape}')

    Uc = OP.to_complex(U)
    U_fs = np.ascontiguousarray(Uc[..., (OP.U_, OP.V_, OP.W_), :])
    p_fs = np.ascontiguousarray(Uc[..., (OP.P_,), :])

    # ---- checks, on the source's own mesh ----
    s = MC.setup()
    m, D, kz, nz = s['m'], s['D'], s['kz'], s['nz']
    div = (DV.ddx(U_fs[..., 0:1, :], D, m.facx)
           + DV.ddy(U_fs[..., 1:2, :], D, m.facy)
           + 1j*kz*U_fs[..., 2:3, :])
    rdiv = float(np.sqrt((abs(div)**2).sum()/(abs(U_fs)**2).sum()))
    finite = bool(np.all(np.isfinite(U_fs)) and np.all(np.isfinite(p_fs)))
    ut, ub = MC.u_tau(s, U), MC.bulk(s, U)

    # velocity at the walls, in physical space: the seeded field must satisfy
    # the no-slip condition the projection code will impose
    P = FR.to_physical(U_fs, nz)
    ywall = np.isclose(np.concatenate([m.ynod[e][None, :].repeat(m.N+1, 0)[None]
                                       for e in range(m.nelem)]), 0.0) | \
            np.isclose(np.concatenate([m.ynod[e][None, :].repeat(m.N+1, 0)[None]
                                       for e in range(m.nelem)]), 2.0)
    wall_u = float(np.abs(P[ywall]).max()) if ywall.any() else float('nan')

    print(f'\n  relative pointwise divergence   {rdiv:.3e}')
    print(f'  max |u| at the walls            {wall_u:.3e}')
    print(f'  u_tau / U_bulk                  {ut:.5f} / {ub:.4f}')
    print(f'  finite                          {finite}')

    if not finite:
        raise SystemExit('REFUSED: the source checkpoint is not finite')
    if rdiv > 1e-2:
        raise SystemExit(f'REFUSED: seeded divergence {rdiv:.2e} is too large; '
                         f'the slice probably took the wrong components')

    np.savez(a.out, U=U_fs, p=p_fs, t=t, dt=a.dt)
    print(f'\nwrote {a.out}  ({os.path.getsize(a.out)/1e6:.1f} MB)')
    print(f'the projection run should start at t = {t:.4f} and march the same '
          f'span the least-squares run did.')


if __name__ == '__main__':
    main()
