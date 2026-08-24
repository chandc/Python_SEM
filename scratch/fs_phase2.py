"""PHASE 2: TGV with convection -- Gates 2 and 3 on the fractional-step path.

    python scratch/fs_phase2.py

Gate 3 is the parameter-free one and the more useful: for decaying turbulence
the energy and enstrophy must satisfy

    -dE/dt = 2 nu Omega

exactly, with no fitted constant.  The least-squares path holds it to 6.65e-06.
It is a joint statement about the convective term, the viscous term and the
time integrator, so it fails if any of the three is wrong -- and it does not
care that the pressure is now computed by projection rather than solved for
simultaneously.

convect.convective is reused UNCHANGED: it reads only u, v, w at indices 0, 1,
2, which is exactly the fractional-step velocity layout, and returns
(nelem, n, n, 3, nmode) -- the same shape.  It is UNSIGNED, so N = -convective.
"""
import sys
sys.path.insert(0, '.'); sys.path.insert(0, 'scratch')
import numpy as np
import lssem3d; lssem3d.set_backend('numpy')
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem3d import (project as PJ, helmholtz as HH, convect as CV,
                     fourier as FR, solver3d as S3, timestep as T, deriv as DV)

L = 2*np.pi


def build(N=8, ne=4, nz=16, nu=0.01, tol=1e-10, backend='numpy'):
    m = build_channel(L, L, ne, ne, N, bcs=(0, 0, 0, 0))
    m.periodic_x = L; m.periodic_y = L; m.compute_global_indices()
    nk, n = nz//2 + 1, N + 1
    X = np.empty((m.nelem, n, n)); Y = np.empty_like(X)
    for e in range(m.nelem):
        X[e] = m.xnod[e][:, None]; Y[e] = m.ynod[e][None, :]
    mask_u = PJ.build_masks(m, nk, nz, 3, wall=False)
    mask_p = PJ.build_masks(m, nk, nz, 1, wall=False)
    ind = np.zeros(mask_p.shape); ind[0, 0, 0, 0, 0] = 1.0
    mask_p[..., 0, 0] *= (S3.gs(m, ind)[..., 0, 0] < 0.5)
    kz = FR.wavenumbers(nz, L)
    D = diff_matrix(N)
    if backend == 'cupy':
        import cupy as cp
        g = lambda a: cp.asarray(np.ascontiguousarray(a))
    else:
        g = lambda a: a
    v = np.ones(mask_p[..., 0:1, 0:1].shape)*mask_p[..., 0:1, 0:1]
    mw1 = S3.multiplicity_weight(m, mask_p.shape)[..., 0:1, 0:1]
    s = dict(m=m, D=D, N=N, nz=nz, nk=nk, nu=nu, kz=kz, lz=L, X=X, Y=Y,
             tol=tol, incremental=False, wall_u=None, ubc=None,
             # host copies kept for setup; the hot path uses the *g versions
             mask_u=g(mask_u), mask_p=g(mask_p),
             Dg=g(D), fxg=g(m.facx), fyg=g(m.facy), wqg=g(m.wq), kzg=g(kz),
             wq3=g(m.wq[..., None, None]), wq1=g(m.wq[..., None, None]),
             mw1=g(mw1), null_kz0=g(v),
             null_norm=float((v*v*mw1).sum()), backend=backend)
    like = s['mask_p']
    s['Mp'] = HH.fdm_preconditioner(m, N, kz**2, 1.0, s['mask_p'], 2, nk,
                                    like=like)
    return s


def ic_tgv(s, to_device=True):
    x, y = s['X'][..., None], s['Y'][..., None]
    z = (L/s['nz'])*np.arange(s['nz']).reshape(1, 1, 1, -1)
    up = np.zeros((s['m'].nelem, s['N']+1, s['N']+1, 3, s['nz']))
    up[..., 0, :] = np.sin(x)*np.cos(y)*np.cos(z)
    up[..., 1, :] = -np.cos(x)*np.sin(y)*np.cos(z)
    Uc = FR.to_modes(up)
    if to_device and s.get('backend') == 'cupy':
        import cupy as cp
        return cp.asarray(np.ascontiguousarray(Uc))
    return Uc


def diagnostics(s, Uc):
    """E and Omega, both from the velocity (vorticity computed here).

    DEVICE arrays.  This reached into s['m'].facx, which is host whatever the
    backend, and --price never calls it -- so pricing passed while the run died
    on the first diagnostic line.  A price path that does not exercise the run
    path is not a smoke test.
    """
    D = s.get('Dg', s['D'])
    fx = s.get('fxg', s['m'].facx)
    fy = s.get('fyg', s['m'].facy)
    kz = s.get('kzg', s['kz'])
    u, v, w = (Uc[..., i:i+1, :] for i in range(3))
    ox = DV.ddy(w, D, fy) - 1j*kz*v
    oy = 1j*kz*u - DV.ddx(w, D, fx)
    oz = DV.ddx(v, D, fx) - DV.ddy(u, D, fy)
    wz = L/s['nz']
    # wq is (nelem, n, n); the physical field is (nelem, n, n, F, nz), so it
    # needs TWO trailing axes, not one.  With one it broadcasts against the
    # field axis and silently weights the wrong thing -- here it raised, which
    # was luck.
    w2 = s.get('wqg', s['m'].wq)[..., None, None]
    xp = np if isinstance(Uc, np.ndarray) else __import__('cupy')
    E = 0.5*wz*float(xp.sum(xp.abs(FR.to_physical(Uc, s['nz']))**2 * w2))
    Om = 0.5*wz*float(xp.sum(xp.abs(FR.to_physical(
        xp.concatenate([ox, oy, oz], axis=-2), s['nz']))**2 * w2))
    return E, Om


def main():
    print('GATE 3  TGV: the parameter-free balance  -dE/dt = 2 nu Omega')
    s = build()
    Uc = ic_tgv(s)
    pc = np.zeros((s['m'].nelem, s['N']+1, s['N']+1, 1, s['nk']), dtype=complex)
    Nprev = np.zeros((s['m'].nelem, s['N']+1, s['N']+1, 3, s['nk']), dtype=complex)
    dt = 0.01
    E, Om = diagnostics(s, Uc)
    print(f'   t=0.000  E={E:.8f}  Omega={Om:.6f}')
    worst = 0.0
    for i in range(10):
        Ep, Op = E, Om
        for k in range(T.NSTAGE):
            lam = T.implicit_coeff(dt, k) + s['nu']*(s['kz']**2)
            s['Mu'] = HH.fdm_preconditioner(s['m'], s['N'], lam, s['nu'],
                                            s['mask_u'], 6, s['nk'])
            Nk = -CV.convective(Uc, s['D'], s['m'].facx, s['m'].facy, s['kz'],
                                s['nz'])
            Uc, pc, _ = PJ.substage(s, Uc, pc, Nk, Nprev, k, dt)
            Nprev = Nk
        E, Om = diagnostics(s, Uc)
        bal = (-(E - Ep)/dt)/(2*s['nu']*0.5*(Om + Op))
        worst = max(worst, abs(bal - 1.0))
        if i % 3 == 0 or i == 9:
            print(f'   t={(i+1)*dt:.3f}  E={E:.8f}  Omega={Om:.6f}  '
                  f'balance={bal:.6f}')
    print(f'\n   worst deviation from 1: {worst:.2e}   '
          f'{"PASS" if worst < 1e-4 else "FAIL"}')
    print('   (least-squares path: 6.65e-06)')


if __name__ == '__main__':
    main()
