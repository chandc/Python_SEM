"""Seed the FOSLS-3D minimal channel from a CONVERGED fractional-step field.

    uv run --quiet python scratch/minchan_seed_fs.py check    # gates only
    uv run --quiet python scratch/minchan_seed_fs.py write    # gates + write IC

WHY THIS EXISTS.  Both FOSLS minimal-channel runs were retired without reaching
turbulence: minchan_001 carried relative divergence 1.1e-01 from a non-solenoidal
trip, and minchan_002 relaminarised because the solenoidal fix made the trip 1.8x
weaker (3D_STATUS sec 7P).  Reaching turbulence from a cold start costs ~12 days.

The fractional-step code already HAS a statistically stationary Re_tau = 180
field: `~/lssem_fs/scratch/_minchan_stat_E`, u_tau = 1.0014 +/- 0.0205 over the
last 2000 of 7401 samples, t = 3.0 -> 16.0, 79 h of accumulated runtime.  Mean
profile U+ = 1.048 at y+ = 1.0 and 14.12/15.35/17.18 at y+ = 30/50/100;
u_rms peak 2.86 against KMM's 2.7.  Seeding from it skips the transient entirely.

THE GRIDS ARE IDENTICAL, which is what makes this a copy rather than a project:
both codes use RE_TAU=180, DELTA=1, LX=pi, LZ=0.34*pi, FX=1 and N=8, ex=6, ey=18,
nz=32 -> 108 elements, 9 nodes, 17 modes.  The fractional-step minchan.py is a
direct descendant of the FOSLS one.

WHAT HAS TO BE BUILT.  Fractional step stores 4 complex fields (u,v,w and p);
FOSLS carries 7 (u,v,w,omega_x,omega_y,omega_z,p) as 14 split-real.  So:

    u, v, w  -> U_, V_, W_     direct copy
    p        -> P_             direct copy
    omega    -> OX_, OY_, OZ_  DERIVED, by the DISCRETE curl

The curl must be discrete, not analytic: FOSLS carries vorticity as INDEPENDENT
unknowns whose definition rows R_1..R_3 are least-squares residuals, so an
inconsistent omega shows up as a large J at t = 0.  channel3d._set_vorticity uses
"the same discrete derivatives as the operator", which is exactly right.

ON THE DIVERGENCE.  The source run logs div ~ 2e-01, the same order that retired
minchan_001.  That is NOT the same defect: fractional step enforces
div u = 0 WEAKLY (a projection against the pressure test space), so a STRONG-form
pointwise measure reads large by construction.  The gate below reports both the
strong-form residual and its weak/assembled counterpart so the distinction is on
the record rather than assumed.
"""
import os
import sys

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
_SC = os.path.dirname(os.path.abspath(__file__))
_R = os.path.dirname(_SC)
sys.path.insert(0, _R); sys.path.insert(0, _SC)
os.chdir(_R)

import numpy as np

from lssem3d import operator as OP, fourier as FR, deriv as DV
import channel3d as C
import minchan as MC

SRC = f'{_SC}/fs_seed/chk_latest.npz'
DST = f'{_SC}/fs_seed/minchan_ic_from_fs.npz'


def load_and_convert(src=SRC):
    """fractional-step (u,v,w)+p  ->  FOSLS 14-field split-real state."""
    z = np.load(src, allow_pickle=True)
    Ufs, pfs, t0 = z['U'], z['p'], float(z['t'])
    s = MC.setup()
    m, nk = s['m'], s['kz'].size
    ne, n = m.nelem, m.N + 1
    assert Ufs.shape == (ne, n, n, 3, nk), \
        f'grid mismatch: source {Ufs.shape} vs FOSLS ({ne},{n},{n},3,{nk})'

    Uc = np.zeros((ne, n, n, OP.NVAR, nk), dtype=complex)
    Uc[..., OP.U_, :] = Ufs[..., 0, :]
    Uc[..., OP.V_, :] = Ufs[..., 1, :]
    Uc[..., OP.W_, :] = Ufs[..., 2, :]
    Uc[..., OP.P_, :] = pfs[..., 0, :]
    C._set_vorticity(s, Uc)                    # discrete curl -- see module docstring
    U = np.concatenate([Uc.real, Uc.imag], axis=-2)
    return s, U*s['mask'], t0, Uc


def divergence(s, Uc):
    """Strong-form div u, and the assembled (weak) residual for contrast."""
    m, D, kz = s['m'], s['D'], s['kz']
    u, v, w = (Uc[..., OP.U_, :], Uc[..., OP.V_, :], Uc[..., OP.W_, :])
    d = DV.ddx(u, D, m.facx) + DV.ddy(v, D, m.facy) + 1j*kz*w
    nz = 2*(kz.size - 1)
    dphys = FR.to_physical(d, nz)
    uph = FR.to_physical(u, nz)
    wq = m.wq[..., None]
    strong = float(np.sqrt((wq*dphys**2).sum()))
    unorm = float(np.sqrt((wq*uph**2).sum()))
    return strong, unorm, float(np.abs(dphys).max())


def functional_by_row(s, U):
    """J = sum_r rho_r |R_r|^2 W, broken down by row.

    THE DECISIVE DIAGNOSTIC.  FOSLS PENALISES div u; the source field satisfies it
    only WEAKLY.  So the question is not whether the strong-form divergence is
    large -- it is -- but how much of the least-squares functional it accounts
    for, i.e. how hard the FOSLS solve will fight the imported field.  A
    continuity row that dominates J means a violent t=0 adjustment; one that sits
    alongside the others means the field is acceptable as an IC.

    F4' validated J as an error estimator for exactly this kind of question, and
    it rose 8.6e9 on the minchan_001 defect -- so it is the right instrument.
    """
    m, D, kz, nu = s['m'], s['D'], s['kz'], s['nu']
    rw = OP.momentum_row_weights(s.get('c', 525.0))
    R = OP.apply_L(U*s['mask'], D, m.facx, m.facy, kz, nu,
                   s.get('c', 525.0), m.wq, 0.0, rw)
    nrow = R.shape[-2]//2
    names = ['continuity', 'om_x def', 'om_y def', 'om_z def',
             'x-momentum', 'y-momentum', 'z-momentum', 'div omega']
    tot = float((R**2).sum())
    out = []
    for r in range(nrow):
        e = float((R[..., r, :]**2).sum() + (R[..., r+nrow, :]**2).sum())
        out.append((names[r] if r < len(names) else f'row {r}', e, e/max(tot, 1e-300)))
    return tot, out


def gates(s, U, t0, Uc):
    print(f'source t = {t0:.4f}\n')
    print(f'  state shape {U.shape}   finite: {np.all(np.isfinite(U))}')
    ut = MC.u_tau(s, U)
    ub = MC.bulk(s, U)
    rw = MC.rms_w(s, U)
    print(f'  u_tau  = {ut:.4f}      (target 1.0; source run 1.0014 +/- 0.0205)')
    print(f'  U_bulk = {ub:.4f}      (KMM ~15.6)')
    print(f'  rms_w  = {rw:.4f}      (minchan_002 relaminarised below ~0.135)')
    st, un, mx = divergence(s, Uc)
    print(f'\n  STRONG-form divergence  ||div u||/||u|| = {st/un:.3e}   max|div| = {mx:.3e}')
    print(f'     minchan_001 (retired) carried 1.1e-01; the validated Stokes case 5e-09.')
    print(f'     Fractional step enforces div u = 0 WEAKLY, so a strong-form measure')
    print(f'     is expected to be large -- see the module docstring.')
    E = MC.energy(s, U)
    print(f'\n  energy diagnostics: {E}')

    tot, rows = functional_by_row(s, U)
    print(f'\n  FOSLS functional J = {tot:.6e}, by row:')
    for nm, e, f in sorted(rows, key=lambda x: -x[1]):
        bar = '#'*int(round(40*f))
        print(f'     {nm:12s} {e:12.4e}  {f*100:6.2f}%  {bar}')
    cont = [f for nm, e, f in rows if nm == 'continuity'][0]
    print(f'\n  -> continuity accounts for {cont*100:.1f}% of J.')
    print('     Dominant  => FOSLS will fight the field hard at t=0.')
    print('     Comparable => the field is acceptable as an IC.')
    return dict(u_tau=ut, bulk=ub, rms_w=rw, div_rel=st/un, div_max=mx, t=t0,
                J=tot, J_continuity_frac=cont)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'check'
    s, U, t0, Uc = load_and_convert()
    g = gates(s, U, t0, Uc)
    if mode == 'write':
        np.savez_compressed(DST, U=U, t=t0, **{k: v for k, v in g.items()
                                               if k != 't'})
        print(f'\nwrote -> {DST}')
    else:
        print('\n(check only; pass "write" to save the IC)')


if __name__ == '__main__':
    main()
