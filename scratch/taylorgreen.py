"""Taylor-Green decay: the ONLY test where convection is active AND the answer is known.

    uv run --quiet python scratch/taylorgreen.py [rw|norw] [ac|noac]

THE GAP THIS CLOSES.  The temporal order of the explicit RK3 half has never been
measured through the PDE:

  * the order-2.00 Stokes capstone runs with convection switched OFF by
    construction (`stokes3d.stage` has no convective term at all);
  * the 3.025 explicit-only result runs on a SCALAR model ODE, which validates
    the gamma/zeta table but bypasses rhs_explicit -> dealiased convective() ->
    stage-RHS assembly entirely;
  * `test_convect.py` checks that plumbing SPATIALLY (each component contributes,
    dealiasing exact, CFL falls with N) but never in time.

So a dt-factor slip, a sign, or the wrong stage state in the convective assembly
would leave the Stokes capstone at a pristine 2.00 and the scalar test at 3.025
while silently degrading the real mixed scheme.  Given nine silent bugs in this
project, three of them in exactly this kind of assembly, that is not hypothetical.

THE SOLUTION, exact for incompressible Navier-Stokes on a doubly-periodic box:

    u = -cos(x) sin(y) F(t)        F(t) = exp(-2 nu t)
    v =  sin(x) cos(y) F(t)
    om_z = 2 cos(x) cos(y) F(t)
    p = -(1/4)(cos 2x + cos 2y) F(t)^2

Verified by hand: u_t = 2 nu cos x sin y F, u.grad u = -(1/2) sin(2x) F^2,
-p_x = -(1/2) sin(2x) F^2, nu lap u = 2 nu cos x sin y F -- the convective term
is balanced POINTWISE by the pressure gradient, which is exactly the coupling
Stokes decay cannot exercise.

Also the first user of `mesh.periodic_y`: the 2D code supports double
periodicity but nothing in the repo had ever exercised it.
"""
import os, sys, time
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
sys.path.insert(0, SC)
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem3d import (operator as OP, solver3d as S3, bc as BC, convect as CV,
                     fourier as FR, timestep as T, parallel as PAR)

L = 2.0*np.pi
NU = 0.1
NZ = 1                      # k_z = 0 only: convection lives in (x, y)


def setup(N=8, ex=3, ey=3):
    m = build_channel(L, L, ex, ey, N, bcs=(0, 0, 0, 0))
    m.periodic_x = L
    m.periodic_y = L                       # first user of double periodicity
    m.compute_global_indices()
    mask = BC.build_mask(m, NZ, pin_p=False, nz=1)
    BC.pin_dof(m, mask, OP.P_, 0)          # ALL copies: the seam node has 4
    mask[..., OP.NVAR:, :] = 0.0           # k_z = 0: imaginary half unphysical
    n = N+1
    X = np.empty((m.nelem, n, n)); Y = np.empty_like(X)
    for e in range(m.nelem):
        X[e] = m.xnod[e][:, None]
        Y[e] = m.ynod[e][None, :]
    return dict(m=m, D=diff_matrix(N), N=N, nz=NZ, nk=NZ, lz=L, nu=NU,
                kz=np.zeros(NZ), mask=mask, X=X, Y=Y)


def exact(s, t):
    """The analytic state at time t, in the split-real 3D layout."""
    m, N, X, Y = s['m'], s['N'], s['X'], s['Y']
    F = np.exp(-2.0*NU*t)
    Uc = np.zeros((m.nelem, N+1, N+1, OP.NVAR, NZ), dtype=complex)
    Uc[..., OP.U_, 0] = -np.cos(X)*np.sin(Y)*F
    Uc[..., OP.V_, 0] = np.sin(X)*np.cos(Y)*F
    Uc[..., OP.OZ_, 0] = 2.0*np.cos(X)*np.cos(Y)*F
    Uc[..., OP.P_, 0] = -0.25*(np.cos(2*X) + np.cos(2*Y))*F*F
    return np.concatenate([Uc.real, Uc.imag], axis=-2)


def _fw(rw, n2):
    return (1.0 if rw is None
            else np.concatenate([rw, rw]).reshape((1, 1, 1, n2, 1)))


def stage(s, U, Nprev, k, dt, kap, Minv=None, rw=None, tol=1e-12,
          max_iter=40000):
    """One RKW3/CN stage WITH the convective term -- the path under test."""
    m, D, kz, mask, nu = s['m'], s['D'], s['kz'], s['mask'], s['nu']
    c = T.implicit_coeff(dt, k)
    Uc = OP.to_complex(U)
    Nk = -CV.convective(Uc, D, m.facx, m.facy, kz, s['nz'])
    R0 = OP.apply_L0_complex(Uc, D, m.facx, m.facy, kz, nu, 0.0, kap)
    Lk = -R0[..., 4:7, :]

    fc = np.zeros(Uc.shape[:-2] + (OP.NROW, Uc.shape[-1]), dtype=complex)
    for row, fld in ((4, OP.U_), (5, OP.V_), (6, OP.W_)):
        i = row - 4
        fc[..., row, :] = c*(Uc[..., fld, :] + dt*(
            T.GAMMA[k]*Nk[..., i, :] + T.ZETA[k]*Nprev[..., i, :]
            + T.ALPHA[k]*Lk[..., i, :]))
    fc[..., 0, :] = kap*Uc[..., OP.P_, :]
    f = np.concatenate([fc.real, fc.imag], axis=-2)

    wqR = m.wq[..., None, None]
    r = OP.apply_LT(
        OP.apply_L(U, D, m.facx, m.facy, kz, nu, c, m.wq, kap, rw)
        - f*wqR*_fw(rw, f.shape[-2]),
        D, m.facx, m.facy, kz, nu, c, kap)
    b = -S3.gs(m, r)*mask
    dU, it, _ = PAR.pcg(b, D, m.facx, m.facy, kz, nu, c, mesh=m, mask=mask,
                        M_inv=None if Minv is None else Minv[k], tol=tol,
                        max_iter=max_iter, wq=m.wq, kap=kap, rw=rw)
    return U + dU, Nk, it


def make_precond(s, dt, kap, rowweight):
    shape = (s['m'].nelem, s['N']+1, s['N']+1, OP.NVAR_R, NZ)
    out = []
    for k in range(T.NSTAGE):
        cc = T.implicit_coeff(dt, k)
        rw = OP.momentum_row_weights(cc) if rowweight else None
        out.append(S3.jacobi_inverse(S3.jacobi_diagonal(
            shape, s['D'], s['m'].facx, s['m'].facy, s['kz'], s['nu'], cc,
            s['m'], s['mask'], s['m'].wq, kap, rw=rw), s['mask']))
    return out


def err_vs_exact(s, U, t):
    """L2 velocity error, quadrature-weighted."""
    Ue = exact(s, t)
    a, b = OP.to_complex(U), OP.to_complex(Ue)
    d = sum(np.abs(a[..., f, 0] - b[..., f, 0])**2 for f in (OP.U_, OP.V_))
    return float(np.sqrt(np.sum(d*s['m'].wq)))


def run(dt, rowweight, ac, tend=0.4, grid=None):
    s = setup(**(grid or {}))
    U = exact(s, 0.0)
    kap = T.a_mass_worst(dt) if ac else 0.0
    Minv = make_precond(s, dt, kap, rowweight)
    Nprev = np.zeros(OP.to_complex(U).shape[:-2] + (3, NZ), dtype=complex)
    tot, capped = 0, False
    for _ in range(int(round(tend/dt))):
        for k in range(T.NSTAGE):
            rw = (OP.momentum_row_weights(T.implicit_coeff(dt, k))
                  if rowweight else None)
            U, Nprev, it = stage(s, U, Nprev, k, dt, kap, Minv=Minv, rw=rw)
            tot += it
            capped |= (it >= 40000)
        if not np.all(np.isfinite(U)):
            return dict(status='BLEWUP', dt=dt)
    return dict(status='ok', dt=dt, err=err_vs_exact(s, U, tend), cg=tot,
                capped=capped)


if __name__ == '__main__':
    rowweight = (sys.argv[1] if len(sys.argv) > 1 else 'norw') == 'rw'
    ac = (sys.argv[2] if len(sys.argv) > 2 else 'ac') == 'ac'
    print(f'Taylor-Green, nu={NU}, t=0.4, doubly periodic, CONVECTION ACTIVE'
          f'  [N={int(os.environ.get("TG_N", 8))}]')
    print(f'row weights={"on" if rowweight else "off"}  '
          f'AC={"on" if ac else "off"}\n')
    print(f"{'dt':>10}{'L2 err':>13}{'CG':>10}{'capped':>8}")
    N = int(os.environ.get('TG_N', 8))
    dts = [float(x) for x in os.environ.get('TG_DTS', '0.02,0.01,0.005,0.0025').split(',')]
    errs = []
    for dt in dts:
        t0 = time.perf_counter()
        r = run(dt, rowweight, ac, grid=dict(N=N))
        if r['status'] != 'ok':
            print(f"{dt:>10g}   {r['status']}"); break
        errs.append(r['err'])
        print(f"{dt:>10g}{r['err']:>13.4e}{r['cg']:>10}"
              f"{('YES' if r['capped'] else 'no'):>8}", flush=True)
    if len(errs) > 1:
        o = [np.log2(a/b) for a, b in zip(errs[:-1], errs[1:])]
        print(f"\n  order: {[f'{x:.2f}' for x in o]}")
        print('  Expect 2.00 -- RK3 is third order on the convective half alone;')
        print('  Crank-Nicolson caps the mixed scheme at 2.')
