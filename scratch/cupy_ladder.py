"""CuPy backend: whole-solve parity, then a device-resident physics step.

    docker run --rm --gpus all -v "$PWD":/work -w /work lssem-cupy:latest \
           python scratch/cupy_ladder.py

A. WHOLE-SOLVE PARITY.  `normal_op` parity (scratch/cupy_parity.py) only proves
   one matvec.  This runs the entire preconditioned CG -- hundreds of
   iterations, every dot product, the gather-scatter, the convergence test --
   device-resident, and compares the SOLUTION to the NumPy reference.  The
   iteration counts are NOT expected to match bitwise: `gs_cupy` accumulates
   with atomics, so reduction order varies (the same caveat the torch port
   records for `index_add_`).  What must match is the answer.

B. PHYSICS STEP.  A real TGV stage -- convection, the dealiased FFT, the
   defect-corrected RHS, the solve -- run on the device and compared against
   the host path in the quantities that are actually validated: kinetic energy
   and enstrophy.
"""
import sys, time
sys.path.insert(0, '.')
import numpy as np, cupy as cp
import lssem3d
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem3d import (operator as OP, solver3d as S3, bc as BC, fourier as FR,
                     convect as CV, timestep as T)

L = 2*np.pi
g = cp.asarray


def case(N=8, ex=4, nz=16, nu=1/180., c=525.0):
    m = build_channel(L, L, ex, ex, N, bcs=(0, 0, 0, 0))
    m.periodic_x = L; m.periodic_y = L; m.compute_global_indices()
    nk = nz//2 + 1
    mask = BC.build_mask(m, nk, pin_p=False, nz=nz)
    BC.pin_dof(m, mask, OP.P_, 0)
    D = diff_matrix(N); kz = FR.wavenumbers(nz, L)
    rw = OP.momentum_row_weights(c)
    shape = (m.nelem, N+1, N+1, OP.NVAR_R, nk)
    return dict(m=m, D=D, kz=kz, mask=mask, rw=rw, shape=shape, nu=nu, c=c,
                N=N, nz=nz, nk=nk)


def part_A(s):
    m, D, kz, mask, rw = s['m'], s['D'], s['kz'], s['mask'], s['rw']
    nu, c = s['nu'], s['c']
    x_true = S3.make_continuous(m, np.random.default_rng(1).standard_normal(s['shape']))*mask
    kw = dict(mesh=m, mask=mask, wq=m.wq, kap=0.0, rw=rw)
    lssem3d.set_backend('numpy')
    b = S3.normal_op(x_true, D, m.facx, m.facy, kz, nu, c, **kw)
    Mi = S3.jacobi_inverse(S3.jacobi_diagonal_analytic(
        s['shape'], D, m.facx, m.facy, kz, nu, c, m, mask, m.wq, 0.0, rw=rw), mask)
    t0 = time.perf_counter()
    x_np, it_np, _ = S3.pcg(b, D, m.facx, m.facy, kz, nu, c, mesh=m, mask=mask,
                            M_inv=Mi, tol=1e-10, max_iter=5000, wq=m.wq, rw=rw)
    t_np = time.perf_counter() - t0

    lssem3d.set_backend('cupy')
    kwg = dict(mesh=m, mask=g(mask), wq=g(m.wq), kap=0.0, rw=g(rw))
    t0 = time.perf_counter()
    x_cp, it_cp, _ = S3.pcg(g(b), g(D), g(m.facx), g(m.facy), g(kz), nu, c,
                            mesh=m, mask=g(mask), M_inv=g(Mi), tol=1e-10,
                            max_iter=5000, wq=g(m.wq), rw=g(rw))
    cp.cuda.Stream.null.synchronize()
    t_cp = time.perf_counter() - t0
    assert isinstance(x_cp, cp.ndarray), 'solution left the device'
    scale = float(np.abs(x_np).max())
    err_ref = float(np.abs(x_np - x_true).max())/scale
    err_gpu = float(cp.abs(x_cp - g(x_np)).max())/scale
    print('A. WHOLE-SOLVE PARITY (device-resident CG)')
    print(f'   numpy : {it_np:5d} iters  {t_np:7.2f} s   err vs planted solution {err_ref:.2e}')
    print(f'   cupy  : {it_cp:5d} iters  {t_cp:7.2f} s   err vs planted solution '
          f'{float(cp.abs(x_cp - g(x_true)).max())/scale:.2e}')
    # ACCEPTANCE, self-calibrated -- and getting this yardstick right took
    # two wrong tries.  An ABSOLUTE threshold on |x_cupy - x_numpy| is wrong:
    # at this conditioning both solves sit ~1e-6 from the planted solution at
    # tol = 1e-10, so demanding 1e-8 asks the backends to agree far more
    # closely than either agrees with the truth.  A fraction-of-solver-error
    # threshold is wrong too, because `gs_cupy` accumulates with atomics, so
    # CuPy does not even reproduce ITSELF run to run.
    #
    # The floor has to be measured, not assumed (3D_STATUS.md L5).  Solve
    # twice on the device: that spread IS the reduction noise.  The port is
    # correct if the cupy-vs-numpy difference is the same size as
    # cupy-vs-cupy, and if both solutions independently meet the residual
    # tolerance they were asked for.
    x_cp2, _, _ = S3.pcg(g(b), g(D), g(m.facx), g(m.facy), g(kz), nu, c,
                         mesh=m, mask=g(mask), M_inv=g(Mi), tol=1e-10,
                         max_iter=5000, wq=g(m.wq), rw=g(rw))
    spread = float(cp.abs(x_cp - x_cp2).max())/scale
    lssem3d.set_backend('numpy')
    res_np = np.abs(b - S3.normal_op(x_np, D, m.facx, m.facy, kz, nu, c,
                                     **kw)).max()/np.abs(b).max()
    lssem3d.set_backend('cupy')
    res_cp = float(cp.abs(g(b) - S3.normal_op(x_cp, g(D), g(m.facx), g(m.facy),
                                              g(kz), nu, c, **kwg)).max()
                   )/float(cp.abs(g(b)).max())
    ok = (err_gpu < 5*max(spread, 1e-300)) and (res_cp < 1e-8) and (res_np < 1e-8)
    print(f'   cupy vs numpy      : {err_gpu:.2e}')
    print(f'   cupy vs cupy (rerun): {spread:.2e}   <- the atomics floor, measured')
    print(f'   relative residual   : numpy {res_np:.2e}   cupy {res_cp:.2e}')
    print(f'   verdict: {"PASS" if ok else "FAIL"} -- backend difference is '
          f'{err_gpu/max(spread,1e-300):.1f}x the self-spread   ({t_np/t_cp:.1f}x wall)')
    return ok


def part_B(s):
    """One RKW3 stage with convection, host vs device, compared in E and Omega."""
    m, D, kz, mask, nu = s['m'], s['D'], s['kz'], s['mask'], s['nu']
    n, nz, nk = s['N']+1, s['nz'], s['nk']
    X = np.empty((m.nelem, n, n)); Y = np.empty_like(X)
    for e in range(m.nelem):
        X[e] = m.xnod[e][:, None]; Y[e] = m.ynod[e][None, :]
    Z = (L/nz)*np.arange(nz)
    P = np.zeros((m.nelem, n, n, OP.NVAR, nz))
    x, y, z = X[..., None], Y[..., None], Z.reshape(1, 1, 1, -1)
    P[..., OP.U_, :] = np.sin(x)*np.cos(y)*np.cos(z)
    P[..., OP.V_, :] = -np.cos(x)*np.sin(y)*np.cos(z)
    P[..., OP.OX_, :] = -np.cos(x)*np.sin(y)*np.sin(z)
    P[..., OP.OY_, :] = -np.sin(x)*np.cos(y)*np.sin(z)
    P[..., OP.OZ_, :] = 2*np.sin(x)*np.sin(y)*np.cos(z)
    U0 = OP.to_real(FR.to_modes(P))
    dt, k = 0.02, 2
    c = T.implicit_coeff(dt, k)
    rw = OP.momentum_row_weights(c)

    def stage(U, dev):
        cv = (lambda a: g(a)) if dev else (lambda a: a)
        Dl, fx, fy, kzl = cv(D), cv(m.facx), cv(m.facy), cv(kz)
        wq, mk = cv(m.wq), cv(mask)
        Uc = OP.to_complex(U)
        Nk = -CV.convective(Uc, Dl, fx, fy, kzl, nz)
        R0 = OP.apply_L0_complex(Uc, Dl, fx, fy, kzl, nu, 0.0, 0.0) if not dev \
            else __import__('lssem3d.kernels_cupy', fromlist=['_L0'])._L0(
                Uc, Dl, fx, fy, kzl, nu, 0.0, 0.0)
        Lk = -R0[..., 4:7, :]
        xp = cp if dev else np
        fc = xp.zeros(Uc.shape[:-2] + (OP.NROW, Uc.shape[-1]), dtype=complex)
        for row, fld in ((4, OP.U_), (5, OP.V_), (6, OP.W_)):
            i = row - 4
            fc[..., row, :] = c*(Uc[..., fld, :] + dt*(
                T.GAMMA[k]*Nk[..., i, :] + T.ALPHA[k]*Lk[..., i, :]))
        f = xp.concatenate([fc.real, fc.imag], axis=-2)
        fw = xp.concatenate([cv(rw), cv(rw)]).reshape((1, 1, 1, f.shape[-2], 1))
        r = OP.apply_LT(OP.apply_L(U, Dl, fx, fy, kzl, nu, c, wq, 0.0, cv(rw))
                        - f*wq[..., None, None]*fw, Dl, fx, fy, kzl, nu, c, 0.0)
        b = -S3.gs(m, r)*mk
        Mi = S3.jacobi_inverse(S3.jacobi_diagonal_analytic(
            U0.shape, D, m.facx, m.facy, kz, nu, c, m, mask, m.wq, 0.0, rw=rw), mask)
        dU, it, _ = S3.pcg(b, Dl, fx, fy, kzl, nu, c, mesh=m, mask=mk,
                           M_inv=cv(Mi), tol=1e-9, max_iter=5000, wq=wq, rw=cv(rw))
        return U + dU, it

    def diag(U, dev):
        xp = cp if dev else np
        Pp = FR.to_physical(OP.to_complex(U), nz)
        wz = L/nz; wq = (g(m.wq) if dev else m.wq)[..., None]
        E = 0.5*wz*sum(float(xp.sum(xp.abs(Pp[..., f, :])**2*wq))
                       for f in (OP.U_, OP.V_, OP.W_))
        Om = 0.5*wz*sum(float(xp.sum(xp.abs(Pp[..., f, :])**2*wq))
                        for f in (OP.OX_, OP.OY_, OP.OZ_))
        return E, Om

    lssem3d.set_backend('numpy')
    Un, itn = stage(U0, False)
    En, On = diag(Un, False)
    lssem3d.set_backend('cupy')
    Ug, itg = stage(g(U0), True)
    Eg, Og = diag(Ug, True)
    print('\nB. PHYSICS STAGE (convection + dealiased FFT + solve), host vs device')
    print(f'   numpy: E = {En:.10f}  Omega = {On:.8f}  ({itn} CG)')
    print(f'   cupy : E = {Eg:.10f}  Omega = {Og:.8f}  ({itg} CG)')
    dE, dO = abs(Eg-En)/En, abs(Og-On)/On
    print(f'   rel diff: E {dE:.2e}   Omega {dO:.2e}   '
          f'{"PASS" if max(dE, dO) < 1e-10 else "FAIL"}')
    return max(dE, dO) < 1e-10


if __name__ == '__main__':
    s = case()
    ok = part_A(s) & part_B(s)
    print('\nLADDER:', 'PASS' if ok else 'FAIL')
