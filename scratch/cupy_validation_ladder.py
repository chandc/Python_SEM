"""The validation ladder, re-run on the CuPy path.

    docker run --rm --gpus all -v "$PWD":/work -w /work lssem-cupy:latest \
           python scratch/cupy_validation_ladder.py [gate]

A backend is not trusted until it re-passes the ladder -- symmetry and
self-parity tests cannot find a consistently wrong operator (3D_STATUS.md L1),
and Phase 2 proved only that ONE stage agrees with NumPy.  These three gates
compare against things NumPy cannot influence:

  1  STOKES DECAY   sigma against the ANALYTIC 9.3137399 (an exact unsteady
                    solution; convection off, so the reference rate is exact)
  2  ROTATED (x,z) TG   temporal order against the DESIGN order 2.00, and the
                    errors against the recorded NumPy run (3D_STATUS sec 7E.1:
                    5.724e-07 / 1.431e-07 / 3.577e-08 at dt = 0.02/0.01/0.005)
  3  TGV BALANCE    the parameter-free -dE/dt = 2 nu Omega, which needs no
                    reference data at all

Every gate runs device-resident: the state never returns to the host inside a
step.  Gate 2's configuration matches the recorded one exactly (N = 12, 3x3,
Nz = 8, tol 1e-12) so the numbers are directly comparable.
"""
import sys, time
sys.path.insert(0, '.'); sys.path.insert(0, 'scratch')
import numpy as np, cupy as cp
import lssem3d
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem3d import (operator as OP, solver3d as S3, bc as BC, fourier as FR,
                     convect as CV, timestep as T)
from lssem3d.kernels_cupy import _L0 as L0_cupy

L = 2*np.pi
g = cp.asarray


def stage(s, U, Nprev, k, dt, Minv, rw, convect, tol, max_iter=40000):
    """One RKW3/CN stage, device-resident.  `convect=False` gives the Stokes
    operator (the reference rate is then exact, which is the point of gate 1)."""
    m, D, kz, mask, nu, nz = (s['m'], s['Dg'], s['kzg'], s['maskg'], s['nu'],
                              s['nz'])
    c = T.implicit_coeff(dt, k)
    Uc = OP.to_complex(U)
    if convect:
        Nk = -CV.convective(Uc, D, s['fxg'], s['fyg'], kz, nz)
    else:
        Nk = cp.zeros(Uc.shape[:-2] + (3, Uc.shape[-1]), dtype=cp.complex128)
    R0 = L0_cupy(Uc, D, s['fxg'], s['fyg'], kz, nu, 0.0, 0.0)
    Lk = -R0[..., 4:7, :]
    fc = cp.zeros(Uc.shape[:-2] + (OP.NROW, Uc.shape[-1]), dtype=cp.complex128)
    for row, fld in ((4, OP.U_), (5, OP.V_), (6, OP.W_)):
        i = row - 4
        fc[..., row, :] = c*(Uc[..., fld, :] + dt*(
            T.GAMMA[k]*Nk[..., i, :] + T.ZETA[k]*Nprev[..., i, :]
            + T.ALPHA[k]*Lk[..., i, :]))
    f = cp.concatenate([fc.real, fc.imag], axis=-2)
    wq = s['wqg']
    fw = cp.concatenate([rw, rw]).reshape((1, 1, 1, f.shape[-2], 1))
    r = OP.apply_LT(
        OP.apply_L(U, D, s['fxg'], s['fyg'], kz, nu, c, wq, 0.0, rw)
        - f*wq[..., None, None]*fw, D, s['fxg'], s['fyg'], kz, nu, c, 0.0)
    b = -S3.gs(m, r)*mask
    dU, it, _ = S3.pcg(b, D, s['fxg'], s['fyg'], kz, nu, c, mesh=m, mask=mask,
                       M_inv=Minv[k], tol=tol, max_iter=max_iter, wq=wq, rw=rw)
    return U + dU, Nk, it


def to_device(s):
    m = s['m']
    s['Dg'], s['kzg'] = g(s['D']), g(s['kz'])
    s['fxg'], s['fyg'] = g(m.facx), g(m.facy)
    s['wqg'], s['maskg'] = g(m.wq), g(s['mask'])
    return s


def precond(s, dt):
    m = s['m']
    shape = (m.nelem, s['N']+1, s['N']+1, OP.NVAR_R, s['nk'])
    out, rws = [], []
    for k in range(T.NSTAGE):
        cc = T.implicit_coeff(dt, k)
        rw = OP.momentum_row_weights(cc)
        rws.append(g(rw))
        out.append(g(S3.jacobi_inverse(S3.jacobi_diagonal_analytic(
            shape, s['D'], m.facx, m.facy, s['kz'], s['nu'], cc, m, s['mask'],
            m.wq, 0.0, rw=rw), s['mask'])))
    return out, rws


# ------------------------------------------------------------------ gate 1
def gate_stokes():
    import stokes3d as SD
    print('GATE 1  Stokes decay: sigma against the ANALYTIC 9.3137399')
    s = to_device(SD.setup(N=8))
    U0, meta = SD.initial_state(s, mode='kz0')
    rows = []
    for dt in (0.01, 0.005, 0.0025):
        Minv, rws = precond(s, dt)
        U = g(U0)
        nstep = int(round(0.05/dt))
        ts, Es = [0.0], [float(cp.asarray(SD.energy(s, cp.asnumpy(U))))]
        Np_ = cp.zeros(OP.to_complex(U).shape[:-2] + (3, s['nk']),
                       dtype=cp.complex128)
        t0 = time.perf_counter()
        for i in range(nstep):
            for k in range(T.NSTAGE):
                U, _, _ = stage(s, U, Np_, k, dt, Minv, rws[k], False, 1e-12)
            ts.append((i+1)*dt)
            Es.append(float(cp.asarray(SD.energy(s, cp.asnumpy(U)))))
        ts, Es = np.array(ts), np.array(Es)
        k0 = len(ts)//2
        sig = -0.5*np.polyfit(ts[k0:], np.log(Es[k0:]/Es[0]), 1)[0]
        rel = abs(sig - SD.SIGMA_2D)/SD.SIGMA_2D
        rows.append((dt, sig, rel))
        print(f'   dt = {dt:<8g} sigma = {sig:.7f}   rel err {rel:.3e}   '
              f'({time.perf_counter()-t0:.0f}s)', flush=True)
    order = np.log2(rows[0][2]/rows[-1][2])/2
    ok = rows[-1][2] < 2e-5 and order > 1.7
    print(f'   convergence order in dt = {order:.2f} (expect ~2)   '
          f'{"PASS" if ok else "FAIL"}')
    return ok


# ------------------------------------------------------------------ gate 2
def gate_order():
    print('\nGATE 2  Rotated (x,z) TG: temporal order, z-convection active')
    N, ex, nz, nu = 12, 3, 8, 0.1
    m = build_channel(L, L, ex, ex, N, bcs=(0, 0, 0, 0))
    m.periodic_x = L; m.periodic_y = L; m.compute_global_indices()
    nk = nz//2 + 1
    mask = BC.build_mask(m, nk, pin_p=False, nz=nz)
    BC.pin_dof(m, mask, OP.P_, 0)
    s = to_device(dict(m=m, D=diff_matrix(N), N=N, nz=nz, nk=nk, nu=nu,
                       kz=FR.wavenumbers(nz, L), mask=mask))
    n = N+1
    X = np.empty((m.nelem, n, n))
    for e in range(m.nelem):
        X[e] = m.xnod[e][:, None]
    Z = (L/nz)*np.arange(nz)

    def exact(t):
        F = np.exp(-2.0*nu*t)
        x, z = X[..., None], Z.reshape(1, 1, 1, -1)
        P = np.zeros((m.nelem, n, n, OP.NVAR, nz))
        P[..., OP.U_, :] = -np.cos(x)*np.sin(z)*F
        P[..., OP.W_, :] = np.sin(x)*np.cos(z)*F
        P[..., OP.OY_, :] = -2.0*np.cos(x)*np.cos(z)*F
        P[..., OP.P_, :] = -0.25*(np.cos(2*x) + np.cos(2*z))*F*F
        return OP.to_real(FR.to_modes(P))

    ref = {0.02: 5.724e-07, 0.01: 1.431e-07, 0.005: 3.577e-08}
    errs = []
    for dt in (0.02, 0.01, 0.005):
        Minv, rws = precond(s, dt)
        U = g(exact(0.0))
        Np_ = cp.zeros(OP.to_complex(U).shape[:-2] + (3, nk),
                       dtype=cp.complex128)
        t0 = time.perf_counter()
        for _ in range(int(round(0.4/dt))):
            for k in range(T.NSTAGE):
                U, Np_, _ = stage(s, U, Np_, k, dt, Minv, rws[k], True, 1e-12)
        d = OP.to_complex(U) - g(OP.to_complex(exact(0.4)))
        wq = s['wqg'][..., None]
        e = float(cp.sqrt(sum(cp.sum(cp.abs(d[..., f, :])**2*wq)
                              for f in (OP.U_, OP.W_))))
        errs.append(e)
        print(f'   dt = {dt:<8g} L2 err = {e:.4e}   NumPy record {ref[dt]:.3e}'
              f'   ratio {e/ref[dt]:.3f}   ({time.perf_counter()-t0:.0f}s)',
              flush=True)
    o = [np.log2(a/b) for a, b in zip(errs[:-1], errs[1:])]
    ok = all(abs(x - 2.0) < 0.1 for x in o)
    print(f'   order: {[f"{x:.2f}" for x in o]}   (expect 2.00)   '
          f'{"PASS" if ok else "FAIL"}')
    return ok


# ------------------------------------------------------------------ gate 3
def gate_balance():
    print('\nGATE 3  TGV: the parameter-free balance -dE/dt = 2 nu Omega')
    N, ex, nz, nu = 8, 3, 24, 0.01
    m = build_channel(L, L, ex, ex, N, bcs=(0, 0, 0, 0))
    m.periodic_x = L; m.periodic_y = L; m.compute_global_indices()
    nk = nz//2 + 1
    mask = BC.build_mask(m, nk, pin_p=False, nz=nz)
    BC.pin_dof(m, mask, OP.P_, 0)
    s = to_device(dict(m=m, D=diff_matrix(N), N=N, nz=nz, nk=nk, nu=nu,
                       kz=FR.wavenumbers(nz, L), mask=mask))
    n = N+1
    X = np.empty((m.nelem, n, n)); Y = np.empty_like(X)
    for e in range(m.nelem):
        X[e] = m.xnod[e][:, None]; Y[e] = m.ynod[e][None, :]
    Z = (L/nz)*np.arange(nz)
    x, y, z = X[..., None], Y[..., None], Z.reshape(1, 1, 1, -1)
    P = np.zeros((m.nelem, n, n, OP.NVAR, nz))
    P[..., OP.U_, :] = np.sin(x)*np.cos(y)*np.cos(z)
    P[..., OP.V_, :] = -np.cos(x)*np.sin(y)*np.cos(z)
    P[..., OP.OX_, :] = -np.cos(x)*np.sin(y)*np.sin(z)
    P[..., OP.OY_, :] = -np.sin(x)*np.cos(y)*np.sin(z)
    P[..., OP.OZ_, :] = 2*np.sin(x)*np.sin(y)*np.cos(z)
    U = g(OP.to_real(FR.to_modes(P)))

    def EO(U):
        Pp = FR.to_physical(OP.to_complex(U), nz)
        wz = L/nz; wq = s['wqg'][..., None]
        E = 0.5*wz*sum(float(cp.sum(cp.abs(Pp[..., f, :])**2*wq))
                       for f in (OP.U_, OP.V_, OP.W_))
        Om = 0.5*wz*sum(float(cp.sum(cp.abs(Pp[..., f, :])**2*wq))
                        for f in (OP.OX_, OP.OY_, OP.OZ_))
        return E, Om

    dt = 0.02
    Minv, rws = precond(s, dt)
    Np_ = cp.zeros(OP.to_complex(U).shape[:-2] + (3, nk), dtype=cp.complex128)
    E0, Om0 = EO(U)
    V = L**3
    print(f'   E(0) = {E0:.6f} (exact {V/8:.6f});  '
          f'Omega(0) = {Om0:.6f} (exact {3*V/8:.6f})')
    Es, Oms = [E0], [Om0]
    for i in range(10):
        for k in range(T.NSTAGE):
            U, Np_, _ = stage(s, U, Np_, k, dt, Minv, rws[k], True, 1e-9)
        e, o = EO(U); Es.append(e); Oms.append(o)
    bal = [-(Es[i+1]-Es[i])/dt/(2*nu*0.5*(Oms[i+1]+Oms[i]))
           for i in range(len(Es)-1)]
    worst = max(abs(b - 1.0) for b in bal[1:])
    ok = abs(E0 - V/8)/(V/8) < 1e-12 and worst < 1e-3
    print(f'   balance ratio over 10 steps: min {min(bal[1:]):.6f}  '
          f'max {max(bal[1:]):.6f}   worst deviation {worst:.2e}   '
          f'{"PASS" if ok else "FAIL"}')
    return ok


if __name__ == '__main__':
    lssem3d.set_backend('cupy')
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    res = {}
    if which in ('all', '1'):
        res['stokes'] = gate_stokes()
    if which in ('all', '2'):
        res['order'] = gate_order()
    if which in ('all', '3'):
        res['balance'] = gate_balance()
    print('\nLADDER on the CuPy path:',
          ' '.join(f'{k}={"PASS" if v else "FAIL"}' for k, v in res.items()))
    sys.exit(0 if all(res.values()) else 1)
