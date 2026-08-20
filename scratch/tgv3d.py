"""3D Taylor-Green ladder: rotated (x,z) order gate, then the interacting TGV.

    uv run --quiet python scratch/tgv3d.py order            # rotated-TG dt sweep
    uv run --quiet python scratch/tgv3d.py run re100        # TGV shakedown
    uv run --quiet python scratch/tgv3d.py run re400        # TGV validation

THE LADDER (review recommendation, in order):

1. `order` -- Taylor-Green rotated into the (x, z) plane:

       u = -cos x sin z F(t),  v = 0,  w = sin x cos z F(t),  F = e^{-2 nu t}
       omega_y = -2 cos x cos z F,  p = -(1/4)(cos 2x + cos 2z) F^2

   Same exact non-interacting structure as the (x,y) test in taylorgreen.py,
   but the convection now lives in THE Z DIRECTION: w d/dz products, the i*k_z
   terms, and the 3/2-rule dealiased mode convolution inside the stage loop --
   the one path of the time splitting never order-tested through the PDE.
   Expect 2.00 (CN caps the mixed scheme; RK3 is the convective half alone).

2/3. `run` -- the classical interacting TGV on the (2 pi)^3 triply periodic box:

       u =  sin x cos y cos z          omega_x = -cos x sin y sin z
       v = -cos x sin y cos z          omega_y = -sin x cos y sin z
       w = 0                           omega_z = 2 sin x cos y ... (analytic)
       p = (1/16)(cos 2x + cos 2y)(cos 2z + 2)

   Vortex stretching (absent in 2D) couples modes from t = 0+: this is a
   PHYSICS benchmark, not an order test -- there is no exact solution past
   t = 0.  Judged on (a) the energy-dissipation history vs Brachet et al.
   (1983) and (b) the parameter-free internal check  -dE/dt = 2 nu Omega,
   which any incompressible solve must satisfy and which meters divergence
   error and numerical dissipation with no reference data at all.

   re100: nu = 0.01,  4x4 elems N = 8, Nz = 32, t -> 12   (resolved, cheap)
   re400: nu = 0.0025, 8x8 elems N = 8, Nz = 64, t -> 15  (Brachet curve)

DATA SAVED (movie + diagnostics, per the request):
  scratch/tgv_frames_<tag>/frame_####.npz   complex64 full mode-space state +
                                            t, for movies and any later field
                                            analysis (every snap_dt)
  scratch/tgv_frames_<tag>/chk_####.npz     float64 checkpoints (restart)
  scratch/tgv_diag_<tag>.npz                per-step t, E, enstrophy, max|u|,
                                            CG its, capped flag, dt, balance

Configuration: row weights ON, operator-AC OFF (3D_STATUS sec 8.2), every
solve guarded, all-copies pressure pin (sec 7C -- the seam corner has
multiplicity 4).
"""
import os, sys, time, json
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


def setup(N=8, ex=3, ey=3, nz=8, nu=0.1):
    m = build_channel(L, L, ex, ey, N, bcs=(0, 0, 0, 0))
    m.periodic_x = L
    m.periodic_y = L
    m.compute_global_indices()
    nk = nz//2 + 1
    mask = BC.build_mask(m, nk, pin_p=False, nz=nz)   # freezes imag @ k=0, Nyq
    BC.pin_dof(m, mask, OP.P_, 0)                     # ALL copies (sec 7C)
    n = N + 1
    X = np.empty((m.nelem, n, n)); Y = np.empty_like(X)
    for e in range(m.nelem):
        X[e] = m.xnod[e][:, None]
        Y[e] = m.ynod[e][None, :]
    return dict(m=m, D=diff_matrix(N), N=N, nz=nz, nk=nk, lz=L, nu=nu,
                kz=FR.wavenumbers(nz, L), mask=mask, X=X, Y=Y,
                zpl=(L/nz)*np.arange(nz))


def _to_state(s, phys):
    """physical (nelem,n,n,7,nz) real -> split-real mode state."""
    return OP.to_real(FR.to_modes(phys))


def ic_rotxz(s, t):
    """Exact rotated TG at time t (order gate)."""
    X, Z = s['X'][..., None, None], s['zpl'].reshape(1, 1, 1, 1, -1)
    F = np.exp(-2.0*s['nu']*t)
    P = np.zeros(X.shape[:3] + (OP.NVAR, s['nz']))
    P[..., OP.U_, :] = -np.cos(X[..., 0, :])*np.sin(Z[..., 0, :])*F
    P[..., OP.W_, :] = np.sin(X[..., 0, :])*np.cos(Z[..., 0, :])*F
    P[..., OP.OY_, :] = -2.0*np.cos(X[..., 0, :])*np.cos(Z[..., 0, :])*F
    P[..., OP.P_, :] = -0.25*(np.cos(2*X[..., 0, :]) + np.cos(2*Z[..., 0, :]))*F*F
    return _to_state(s, P)


def ic_tgv(s):
    """Classical TGV initial condition, vorticity and pressure analytic."""
    X = s['X'][..., None]; Y = s['Y'][..., None]
    Z = s['zpl'].reshape(1, 1, 1, -1)
    P = np.zeros(X.shape[:3] + (OP.NVAR, s['nz']))
    sx, cx = np.sin(X), np.cos(X)
    sy, cy = np.sin(Y), np.cos(Y)
    sz, cz = np.sin(Z), np.cos(Z)
    P[..., OP.U_, :] = sx*cy*cz
    P[..., OP.V_, :] = -cx*sy*cz
    P[..., OP.OX_, :] = -cx*sy*sz
    P[..., OP.OY_, :] = -sx*cy*sz
    P[..., OP.OZ_, :] = 2.0*sx*sy*cz
    P[..., OP.P_, :] = (1.0/16.0)*(np.cos(2*X) + np.cos(2*Y))*(np.cos(2*Z) + 2.0)
    return _to_state(s, P)


def stage(s, U, Nprev, k, dt, Minv, rw, tol=1e-10, max_iter=40000):
    """One RKW3/CN stage with convection, AC off.  From taylorgreen.py."""
    m, D, kz, mask, nu = s['m'], s['D'], s['kz'], s['mask'], s['nu']
    c = T.implicit_coeff(dt, k)
    Uc = OP.to_complex(U)
    Nk = -CV.convective(Uc, D, m.facx, m.facy, kz, s['nz'])
    R0 = OP.apply_L0_complex(Uc, D, m.facx, m.facy, kz, nu, 0.0, 0.0)
    Lk = -R0[..., 4:7, :]
    fc = np.zeros(Uc.shape[:-2] + (OP.NROW, Uc.shape[-1]), dtype=complex)
    for row, fld in ((4, OP.U_), (5, OP.V_), (6, OP.W_)):
        i = row - 4
        fc[..., row, :] = c*(Uc[..., fld, :] + dt*(
            T.GAMMA[k]*Nk[..., i, :] + T.ZETA[k]*Nprev[..., i, :]
            + T.ALPHA[k]*Lk[..., i, :]))
    f = np.concatenate([fc.real, fc.imag], axis=-2)
    wqR = m.wq[..., None, None]
    fw = (1.0 if rw is None
          else np.concatenate([rw, rw]).reshape((1, 1, 1, f.shape[-2], 1)))
    r = OP.apply_LT(
        OP.apply_L(U, D, m.facx, m.facy, kz, nu, c, m.wq, 0.0, rw)
        - f*wqR*fw,
        D, m.facx, m.facy, kz, nu, c, 0.0)
    b = -S3.gs(m, r)*mask
    dU, it, _ = PAR.pcg(b, D, m.facx, m.facy, kz, nu, c, mesh=m, mask=mask,
                        M_inv=Minv[k], tol=tol, max_iter=max_iter,
                        wq=m.wq, kap=0.0, rw=rw)
    return U + dU, Nk, it


def make_precond(s, dt):
    shape = (s['m'].nelem, s['N']+1, s['N']+1, OP.NVAR_R, s['nk'])
    out, rws = [], []
    for k in range(T.NSTAGE):
        cc = T.implicit_coeff(dt, k)
        rw = OP.momentum_row_weights(cc)
        rws.append(rw)
        out.append(S3.jacobi_inverse(S3.jacobi_diagonal(
            shape, s['D'], s['m'].facx, s['m'].facy, s['kz'], s['nu'], cc,
            s['m'], s['mask'], s['m'].wq, 0.0, rw=rw), s['mask']))
    return out, rws


def fields_physical(s, U):
    return FR.to_physical(OP.to_complex(U), s['nz'])


def energy_enstrophy(s, U):
    """E = 1/2 int |u|^2, Omega = 1/2 int |omega|^2 (solver's own omega)."""
    P = fields_physical(s, U)
    wz = s['lz']/s['nz']
    wq = s['m'].wq[..., None]
    E = 0.5*wz*sum(np.sum(np.abs(P[..., f, :])**2*wq)
                   for f in (OP.U_, OP.V_, OP.W_))
    Om = 0.5*wz*sum(np.sum(np.abs(P[..., f, :])**2*wq)
                    for f in (OP.OX_, OP.OY_, OP.OZ_))
    return float(E.real), float(Om.real)


def err_rotxz(s, U, t):
    d = OP.to_complex(U) - OP.to_complex(ic_rotxz(s, t))
    wq = s['m'].wq[..., None]
    e = sum(np.sum(np.abs(d[..., f, :])**2*wq) for f in (OP.U_, OP.W_))
    return float(np.sqrt(e))


# ---------------------------------------------------------------------------
def order_gate():
    """Rotated (x,z) TG: the z-convection order test."""
    print('Rotated (x,z) Taylor-Green: z-convection through the full stage '
          'assembly.\nN=12, 3x3, Nz=8, nu=0.1, t=0.4, row weights on, AC off.\n')
    s = setup(N=12, ex=3, ey=3, nz=8, nu=0.1)
    print(f"{'dt':>10}{'L2 err (u,w)':>15}{'CG':>10}{'capped':>8}")
    errs = []
    for dt in (0.02, 0.01, 0.005, 0.0025):
        Minv, rws = make_precond(s, dt)
        U = ic_rotxz(s, 0.0)
        Np_ = np.zeros(OP.to_complex(U).shape[:-2] + (3, s['nk']), dtype=complex)
        tot, capped = 0, False
        for _ in range(int(round(0.4/dt))):
            for k in range(T.NSTAGE):
                U, Np_, it = stage(s, U, Np_, k, dt, Minv, rws[k], tol=1e-12)
                tot += it
                capped |= (it >= 40000)
        e = err_rotxz(s, U, 0.4)
        errs.append(e)
        print(f"{dt:>10g}{e:>15.4e}{tot:>10}{('YES' if capped else 'no'):>8}",
              flush=True)
    o = [np.log2(a/b) for a, b in zip(errs[:-1], errs[1:])]
    print(f"\n  order: {[f'{x:.2f}' for x in o]}   (expect 2.00)")
    np.savez(f'{SC}/tgv3d_order.npz', errs=np.array(errs),
             dts=np.array([0.02, 0.01, 0.005, 0.0025]))


# ---------------------------------------------------------------------------
CASES = dict(
    # re100 sizing note: the first attempt (4x4, Nz=32, cfl=0.6, tol 1e-9) ran
    # at 147 s/step -- 52 h to t=12 -- with the balance check already 1.0000 at
    # step 10.  Re = 100 does not need that resolution; this sizing is ~4x
    # cheaper and still comfortably resolved (smallest TGV scales at Re=100 are
    # O(1); 25x25 GLL points per (2 pi)^2 plane and 13 z-modes cover them).
    re100=dict(nu=0.01, N=8, ex=3, ey=3, nz=24, tend=12.0, snap=0.25,
               cfl=1.0, chk_every=2.0, tol=1e-8),
    # re400 sizing: the 8x8/Nz=64 (64^3) version prices at ~19 min/step in
    # numpy -- weeks, not days.  48^3 with tol 1e-7 runs in ~3-4 days and the
    # BALANCE RATIO is the honest referee of what the coarser grid costs: at
    # Re = 400 the cascade reaches finer scales than Re = 100, so expect the
    # ratio to dip further than re100's 0.993 floor near peak enstrophy.  The
    # 64^3 rerun is an M6 (numba) deliverable, not a numpy one.
    re400=dict(nu=0.0025, N=8, ex=6, ey=6, nz=48, tend=15.0, snap=0.5,
               cfl=1.1, chk_every=2.5, tol=1e-7),
)


def run_tgv(tag):
    cfg = CASES[tag]
    s = setup(N=cfg['N'], ex=cfg['ex'], ey=cfg['ey'], nz=cfg['nz'],
              nu=cfg['nu'])
    U = ic_tgv(s)
    Pph = fields_physical(s, U)
    dt = float(CV.max_dt_for_cfl(Pph[..., :OP.NVAR, :], s['D'], s['m'].facx,
                                 s['m'].facy, s['lz'], s['nz'], cfg['cfl']))
    dt = min(dt, 0.02)
    nstep = int(np.ceil(cfg['tend']/dt))
    fdir = f'{SC}/tgv_frames_{tag}'
    os.makedirs(fdir, exist_ok=True)
    print(f'TGV {tag}: nu={cfg["nu"]}, {cfg["ex"]}x{cfg["ey"]} N={cfg["N"]}, '
          f'Nz={cfg["nz"]}, dt={dt:.4g} ({nstep} steps to t={cfg["tend"]}), '
          f'a_mass_worst={T.a_mass_worst(dt):.0f}', flush=True)
    Minv, rws = make_precond(s, dt)
    Np_ = np.zeros(OP.to_complex(U).shape[:-2] + (3, s['nk']), dtype=complex)
    E0, Om0 = energy_enstrophy(s, U)
    diag = dict(t=[0.0], E=[E0], Om=[Om0], maxu=[float(np.abs(Pph[..., :3, :]).max())],
                cg=[0], capped=[0])
    np.savez_compressed(f'{fdir}/frame_0000.npz',
                        U=OP.to_complex(U).astype(np.complex64), t=0.0)
    nfr, next_snap, next_chk = 1, cfg['snap'], cfg['chk_every']
    t0 = time.perf_counter()
    for step_i in range(1, nstep + 1):
        tot, capped = 0, False
        for k in range(T.NSTAGE):
            U, Np_, it = stage(s, U, Np_, k, dt, Minv, rws[k],
                               tol=cfg.get('tol', 1e-9))
            tot += it
            capped |= (it >= 40000)
        t = step_i*dt
        if not np.all(np.isfinite(U)):
            print(f'BLEWUP at t={t:.3f}', flush=True)
            break
        E, Om = energy_enstrophy(s, U)
        Pph = fields_physical(s, U)
        diag['t'].append(t); diag['E'].append(E); diag['Om'].append(Om)
        diag['maxu'].append(float(np.abs(Pph[..., :3, :]).max()))
        diag['cg'].append(tot); diag['capped'].append(int(capped))
        if t + 1e-12 >= next_snap:
            np.savez_compressed(f'{fdir}/frame_{nfr:04d}.npz',
                                U=OP.to_complex(U).astype(np.complex64), t=t)
            nfr += 1; next_snap += cfg['snap']
        if t + 1e-12 >= next_chk:
            np.savez(f'{fdir}/chk_{step_i:05d}.npz', U=U, t=t, dt=dt)
            next_chk += cfg['chk_every']
        if step_i % 10 == 0 or step_i == nstep:
            eps_num = -(diag['E'][-1] - diag['E'][-2])/dt
            bal = eps_num/(2*cfg['nu']*0.5*(diag['Om'][-1] + diag['Om'][-2]))
            print(f't={t:7.3f}  E={E:.6f}  Om={Om:.5f}  '
                  f'-dE/dt / 2nuOm={bal:.4f}  max|u|={diag["maxu"][-1]:.3f}  '
                  f'CG={tot}  {"CAP!" if capped else ""}  '
                  f'[{time.perf_counter()-t0:.0f}s]', flush=True)
        np.savez(f'{SC}/tgv_diag_{tag}.npz',
                 **{k2: np.array(v) for k2, v in diag.items()},
                 nu=cfg['nu'], dt=dt)
    print(f'done: {nfr} movie frames in {fdir}, diagnostics in '
          f'tgv_diag_{tag}.npz, wall {time.perf_counter()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'order'
    if mode == 'order':
        order_gate()
    elif mode == 'run':
        run_tgv(sys.argv[2])
    else:
        raise SystemExit(f'unknown mode {mode!r}')
