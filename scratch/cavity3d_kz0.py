"""M2 GATE: the 3D solver at k_z = 0 must reproduce the 2D Ghia result.

    uv run --quiet python scratch/cavity3d_kz0.py [dt] [nstep]

3D_DEVELOPMENT_PLAN.md Stage 1 calls this the anchor: RMS u = 1.568e-02 against
Ghia is already measured by the 2D code on this mesh (ARTIFICIAL_COMPRESSIBILITY.md
sec 5.1), so it is a number, not a judgement call.  test_stage1_vs_2d.py already
showed the OPERATOR matches to 1e-13; this exercises the whole stack --
operator + quadrature weights + gather-scatter + multiplicity + BC masking +
batched CG -- on an actual flow.

TIME INTEGRATION HERE IS DELIBERATELY FIRST ORDER.  The target is the STEADY
answer, which is a property of the fixed point and not of the integrator, so a
simple implicit step with explicit convection is sufficient and much less code
than RKW3/CN.  Temporal ORDER is Stage 4's business and is already tested
separately (test_solver3d.py measures RKW3 slope 3.0 on a linear ODE).

Momentum row of the operator is   c*u + p_x + nu*(oz_y - i k oy) = f
so with backward Euler, c = 1/dt and

    f = c*u^n - (u.grad u)^n          <- explicit convection, per the plan

and the four constraint rows (continuity, three vorticity definitions, vorticity
divergence) have zero right-hand side.
"""
import os, sys, time
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem3d import operator as OP, solver3d as S3, bc as BC, convect as CV

RE, EX, N = 1000.0, 6, 10
NU = 1.0/RE
NZ, LZ = 1, 2.0*np.pi          # a single k_z = 0 mode
GH = np.load('cavity_re1000_data.npz')


def lagrange(xn, xq):
    n = len(xn); w = np.ones(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                w[i] /= (xn[i]-xn[j])
    dd = xq-xn
    if np.any(np.abs(dd) < 1e-13):
        L = np.zeros(n); L[np.argmin(np.abs(dd))] = 1.0; return L
    num = w/dd
    return num/num.sum()


def centreline_u(mesh, U, n):
    """u(y) on x = 0.5, from the real part of the k_z = 0 mode."""
    ys, us = [], []
    for e in range(mesh.nelem):
        xs = mesh.xnod[e]
        if xs[0]-1e-9 <= 0.5 <= xs[-1]+1e-9:
            L = lagrange(xs, 0.5)
            for j in range(n):
                ys.append(mesh.ynod[e, j])
                us.append(np.dot(L, U[e, :, j, OP.U_, 0]))
    o = np.argsort(ys); ys, us = np.array(ys)[o], np.array(us)[o]
    k = np.concatenate(([True], np.diff(ys) > 1e-9))
    return ys[k], us[k]


def run(dt=0.05, nstep=600, tol=1e-9):
    n = N+1
    mesh = build_channel(1.0, 1.0, EX, EX, N, bcs=(1, 1, 1, 2))
    D = diff_matrix(N)
    kz = np.zeros(NZ)                       # k_z = 0 only
    c = 1.0/dt                              # backward Euler
    shape = (mesh.nelem, n, n, OP.NVAR_R, NZ)

    mask = BC.build_mask(mesh, NZ, pin_p=True)
    U = np.zeros(shape)
    BC.apply_values(mesh, U, NZ, lid_speed=1.0, pin_p=True)

    # Jacobi diagonal: probed once (c is constant, so the operator is too)
    t0 = time.perf_counter()
    diag = S3.jacobi_diagonal(shape, D, mesh.facx, mesh.facy, kz, NU, c,
                              mesh, mask, mesh.wq)
    M_inv = np.where(np.abs(diag) > 1e-300, 1.0/np.where(diag == 0, 1, diag), 0.0)
    print(f'  jacobi probed in {time.perf_counter()-t0:.0f}s', flush=True)

    Uc = np.zeros((mesh.nelem, n, n, OP.NVAR, NZ), dtype=complex)
    t0 = time.perf_counter(); status = 'CAP'
    for s in range(nstep):
        Up = U.copy()
        # --- explicit convection, on the complex view of the state ---
        Uc[:] = OP.to_complex(U)
        Nc = CV.convective(Uc, D, mesh.facx, mesh.facy, kz, NZ)   # (...,3,nk)

        # --- right-hand side: momentum rows only ---
        fc = np.zeros((mesh.nelem, n, n, OP.NROW, NZ), dtype=complex)
        for row, fld in ((4, OP.U_), (5, OP.V_), (6, OP.W_)):
            fc[..., row, :] = c*Uc[..., fld, :] - Nc[..., row-4, :]
        f = np.concatenate([fc.real, fc.imag], axis=-2)

        # DEFECT CORRECTION, not a direct solve for U.  Solving A U = L^T W f
        # with a masked A leaves the PRESCRIBED values out of the equations
        # entirely: the lid is written into U after the solve and the interior
        # never sees it, giving a converged but motionless flow (measured RMS
        # 3.27e-01 against a target of 1.57e-02).  lssem2d avoids this by
        # forming the residual at the CURRENT state -- which already carries the
        # boundary values -- and solving for a masked increment.
        wqR = mesh.wq[..., None, None]
        r = OP.apply_LT(OP.apply_L(U, D, mesh.facx, mesh.facy, kz, NU, c,
                                   mesh.wq) - f*wqR,
                        D, mesh.facx, mesh.facy, kz, NU, c)
        b = -S3.gs(mesh, r)*mask

        dU, it, _ = S3.pcg(b, D, mesh.facx, mesh.facy, kz, NU, c, mesh=mesh,
                           mask=mask, M_inv=M_inv, tol=1e-8, max_iter=20000,
                           wq=mesh.wq)
        U = U + dU                      # dU is masked, so the BCs survive

        if not np.all(np.isfinite(U)):
            status = f'NaN@{s+1}'; break
        dU = float(np.abs(U-Up).max())
        if (s+1) % 50 == 0 or s < 3:
            ys, us = centreline_u(mesh, U, n)
            rms = float(np.sqrt(np.mean((np.interp(GH['ghia_y'], ys, us)
                                         - GH['ghia_u'])**2)))
            print(f'  step {s+1:4d}  |dU| {dU:.3e}  cg {it:5d}  '
                  f'RMS u {rms:.4e}  {time.perf_counter()-t0:6.0f}s', flush=True)
        if s > 3 and dU < tol:
            status = 'conv'; break

    ys, us = centreline_u(mesh, U, n)
    rms = float(np.sqrt(np.mean((np.interp(GH['ghia_y'], ys, us)-GH['ghia_u'])**2)))
    np.savez(f'{SC}/cavity3d_kz0_dt{dt:g}.npz', U=U, xnod=mesh.xnod,
             ynod=mesh.ynod, dt=dt, rms=rms, status=status, steps=s+1)
    return rms, status, s+1, time.perf_counter()-t0


if __name__ == '__main__':
    dt = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
    nstep = int(sys.argv[2]) if len(sys.argv) > 2 else 600
    print(f'3D solver at k_z = 0, cavity Re={RE:g}, {EX}x{EX} N={N}, dt={dt:g}')
    print(f'GATE: RMS u vs Ghia should approach 1.568e-02 '
          f'(the measured 2D value on this mesh)\n')
    rms, status, steps, wall = run(dt, nstep)
    print(f'\nRMS u = {rms:.4e}   target 1.568e-02   ratio {rms/1.5682e-02:.3f}'
          f'   [{status}, {steps} steps, {wall:.0f}s]')
