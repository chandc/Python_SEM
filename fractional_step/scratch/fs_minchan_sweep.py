"""Trip-amplitude sweep for the minimal channel on the RK3-CN projection path.

    python scratch/fs_minchan_sweep.py [--gate-only]

Question (3D_STATUS.md sec 7P): minchan_002's solenoidal trip at amp 1.0/0.3
was sub-threshold and the flow relaminarised.  The candidates 2.0/1.0 and
3.0/1.5 were characterised statically but never run.  Verdict criterion:
rms_w inflecting UPWARD and u_tau holding near 1 by t ~ 3, vs the monotonic
decay minchan_002 showed by t ~ 1.

Scheme: RKW3/CN `substage` (3 projections per step) with SKEW-SYMMETRIC
convection -- the TGV runs showed the advective form blowing up mid-transition
and skew surviving; a tripping channel lives in exactly that regime.

Threads: M3 Max, 12 performance cores -- do NOT let minchan.py's import pin
them to 1 (it setdefault()s the BLAS env vars, so set ours FIRST).
"""
import os, sys, time
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '12'
sys.path.insert(0, '.'); sys.path.insert(0, 'scratch')
import numpy as np

import lssem3d; lssem3d.set_backend('numpy')
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem3d import (project as PJ, helmholtz as HH, convect as CV,
                     fourier as FR, solver3d as S3, timestep as T,
                     deriv as DV, operator as OP, hpmg)
import minchan as MC          # rig constants + the tested trip construction

RE_TAU, DELTA = 180.0, 1.0
LX, LZ, FX = np.pi, 0.34*np.pi, 1.0
N, EX, EY, NZ = 8, 6, 18, 32
NU, TOL = 1.0/RE_TAU, 1e-6
DT = 5.0e-4                   # CFL 0.55 / 0.65 for the two trips, room to 1.73
TEND = 3.0
OUT = 'scratch/_minchan_fs'
CONSISTENT = '--consistent' in sys.argv


def build():
    m = build_channel(LX, 2.0*DELTA, EX, EY, N, bcs=(0, 0, 1, 1))
    m.periodic_x = LX
    m.compute_global_indices()
    nk, n = NZ//2 + 1, N + 1
    kz = FR.wavenumbers(NZ, LZ)
    mask_u = PJ.build_masks(m, nk, NZ, 3, wall=True)
    mask_p = PJ.build_masks(m, nk, NZ, 1, wall=False)
    if CONSISTENT:
        # E's null vector is the pure constant: do NOT pin (a pinned dof
        # rotates the null space and CG amplifies it); purged per-iteration.
        pass
    else:
        ind = np.zeros(mask_p.shape); ind[0, 0, 0, 0, 0] = 1.0
        mask_p[..., 0, 0] *= (S3.gs(m, ind)[..., 0, 0] < 0.5)
    v = np.ones(mask_p[..., 0:1, 0:1].shape)*mask_p[..., 0:1, 0:1]
    mw1 = S3.multiplicity_weight(m, mask_p.shape)[..., 0:1, 0:1]
    D = diff_matrix(N)
    s = dict(m=m, D=D, N=N, nz=NZ, nk=nk, nu=NU, kz=kz, lz=LZ, tol=TOL,
             incremental=False, mask_u=mask_u, mask_p=mask_p,
             Dg=D, fxg=m.facx, fyg=m.facy, wqg=m.wq, kzg=kz,
             wq3=m.wq[..., None, None], wq1=m.wq[..., None, None],
             mw1=mw1, null_kz0=v, null_norm=float((v*v*mw1).sum()),
             wall_u=PJ.wall_indicator(m, nk, NZ, 3), ubc=None,
             backend='numpy', check_every=None)
    t0 = time.perf_counter()
    s['Mp'] = hpmg.HelmholtzPMG(m, N, kz**2, 1.0, 1, nk, NZ, wall=False,
                                pin_kz0=not CONSISTENT, deg=6, like=mask_p)
    s['consistent_p'] = CONSISTENT
    print(f'pressure: p-multigrid (setup {time.perf_counter()-t0:.1f}s)',
          flush=True)
    s['Mu_stages'] = [HH.fdm_preconditioner(
        m, N, T.implicit_coeff(DT, k) + NU*(kz**2), NU, mask_u, 6, nk,
        like=mask_u) for k in range(T.NSTAGE)]
    # constant body force, in mode space once
    fp = np.zeros((m.nelem, n, n, 3, NZ)); fp[..., 0, :] = FX
    s['Fm'] = FR.to_modes(fp)[..., :nk]
    return s


def trip_ic(s, amp_roll, amp_noise, seed=0):
    """minchan.initial_state (tested, solenoidal), sliced to u,v,w complex."""
    smc = MC.setup(N=N, ex=EX, ey=EY, nz=NZ)
    U = MC.initial_state(smc, amp_roll=amp_roll, amp_noise=amp_noise, seed=seed)
    Uc = OP.to_complex(U)[..., (OP.U_, OP.V_, OP.W_), :]
    return PJ._join(PJ._split(Uc)*s['mask_u'])   # exact no-slip at t=0


def trip_ic_shaped(s, amp_roll, amp_noise, seed=0):
    """Rolls as before; noise potential built from LOW-k harmonics only.

    Measured on a20/a30: the white-spectrum curl noise carries its energy at
    near-cutoff wavenumbers and decays ~e^{-22 t} (E3d 6677 -> 1.4 by t=0.38
    at amp 3.0/1.5) -- gone before lift-up can feed anything.  Streak-instability
    scales: lambda_x+ ~ 300 -> k_x = {1,2}*(2 pi/L_x); k_z at the streak
    spacing.  nu k^2 ~ 0.2/t there: e-fold 5 time units instead of 0.05.
    """
    smc = MC.setup(N=N, ex=EX, ey=EY, nz=NZ)
    U = MC.initial_state(smc, amp_roll=amp_roll, amp_noise=0.0, seed=seed)
    Uc = OP.to_complex(U)[..., (OP.U_, OP.V_, OP.W_), :]
    m = s['m']
    X, Y = np.empty((m.nelem, N+1, N+1)), np.empty((m.nelem, N+1, N+1))
    for e in range(m.nelem):
        X[e] = m.xnod[e][:, None]; Y[e] = m.ynod[e][None, :]
    X, Y = X[..., None], Y[..., None]
    Z = (LZ/NZ)*np.arange(NZ).reshape(1, 1, 1, -1)
    rng = np.random.default_rng(seed)
    wall = np.sin(np.pi*Y/(2.0*DELTA))**2
    kzr = 2.0*np.pi/LZ
    A = []
    for _ in range(3):
        a = np.zeros(X.shape[:3] + (NZ,))
        for jx in (1, 2):
            for jz in (1, 2):
                c = rng.standard_normal(); ph = rng.uniform(0, 2*np.pi)
                ps = rng.uniform(0, 2*np.pi)
                a += c*(np.cos(jx*(2*np.pi/LX)*X + ph)
                        * np.cos(jz*kzr*Z + ps))
        A.append(wall*a)
    Ah = [FR.to_modes(a)[..., :s['nk']] for a in A]
    ikz = 1j*s['kz']
    dx = lambda q: DV.ddx(q, s['D'], m.facx)
    dy = lambda q: DV.ddy(q, s['D'], m.facy)
    cx = FR.to_physical(dy(Ah[2]) - ikz*Ah[1], NZ)
    cy = FR.to_physical(ikz*Ah[0] - dx(Ah[2]), NZ)
    cz = FR.to_physical(dx(Ah[1]) - dy(Ah[0]), NZ)
    sc = amp_noise/max(np.abs(cx).max(), np.abs(cy).max(),
                       np.abs(cz).max(), 1e-30)
    P = FR.to_physical(Uc, NZ)
    P[..., 0, :] += sc*cx; P[..., 1, :] += sc*cy; P[..., 2, :] += sc*cz
    Un = FR.to_modes(P)[..., :s['nk']]
    return PJ._join(PJ._split(Un)*s['mask_u'])


def diag(s, Uc):
    m, D = s['m'], s['D']
    du = DV.ddy(Uc[..., 0:1, :], D, m.facy)[..., 0, 0].real/NZ
    Y = np.empty((m.nelem, N+1, N+1))
    for e in range(m.nelem):
        Y[e] = m.ynod[e][None, :]
    lo = np.abs(Y - Y.min()) < 1e-12
    hi = np.abs(Y - Y.max()) < 1e-12
    tw = 0.5*(np.abs(du[lo]).mean() + np.abs(du[hi]).mean())*NU
    utau = float(np.sqrt(max(tw, 0.0)))
    ub = float(Uc[..., 0, 0].real.mean()/NZ)
    P = FR.to_physical(Uc, NZ)
    rmsw = float(np.sqrt(np.mean(P[..., 2, :]**2)))
    d = PJ.divergence(Uc, D, m.facx, m.facy, s['kz'])
    rdiv = float(np.sqrt((np.abs(d)**2).sum()/(np.abs(Uc)**2).sum()))
    full = np.concatenate([P, np.zeros(P.shape[:-2] + (4, NZ))], axis=-2)
    dtmax = CV.max_dt_for_cfl(full, D, m.facx, m.facy, LZ, NZ, np.sqrt(3.0))
    return utau, ub, rmsw, rdiv, DT/dtmax*np.sqrt(3.0)


def run(s, tag, amp_roll, amp_noise, tend, log_every=20, chk_min=15.0,
        resume=False):
    t0 = 0.0
    if resume:
        d = np.load(f'{OUT}/chk_{tag}.npz')
        Uc, pc, t0 = d['U'], d['p'], float(d['t'])
        print(f'resuming {tag} from t={t0:.4f} at dt={DT:g}', flush=True)
    else:
        Uc = (trip_ic_shaped if tag.startswith('s') else trip_ic)(
            s, amp_roll, amp_noise)
        pc = np.zeros(Uc[..., 0:1, :].shape, dtype=complex)
    Nprev = np.zeros_like(Uc)
    s['ubc'] = None
    log = open(f'{OUT}/{tag}.log', 'a' if resume else 'w')
    hdr = (f'# minchan fs sweep {tag}: amp_roll={amp_roll} amp_noise={amp_noise} '
           f'dt={DT} RKW3/CN substage skew=True {EX}x{EY} N={N} Nz={NZ}')
    print(hdr, flush=True); log.write(hdr + '\n')
    nstep = int(round((tend - t0)/DT))
    w0 = last = time.perf_counter()
    for i in range(nstep):
        tot = 0
        for k in range(T.NSTAGE):
            s['Mu'] = s['Mu_stages'][k]
            Nk = -CV.convective(Uc, s['Dg'], s['fxg'], s['fyg'], s['kzg'], NZ,
                                skew=True) + s['Fm']
            Uc, pc, inf = PJ.substage(s, Uc, pc, Nk, Nprev, k, DT)
            Nprev = Nk
            tot += inf[0] + inf[2]
        t = t0 + (i + 1)*DT
        if not np.isfinite(np.abs(Uc).max()):
            line = f'BLEW UP at t={t:.4f}'
            print(line, flush=True); log.write(line + '\n'); log.flush()
            return 'blowup'
        if i % log_every == log_every - 1 or i == nstep - 1:
            utau, ub, rmsw, rdiv, cfl = diag(s, Uc)
            line = (f't={t:7.4f}  u_tau={utau:.4f}  U_b={ub:.3f}  '
                    f'rms_w={rmsw:.4f}  CFL={cfl:.2f}  div={rdiv:.1e}  '
                    f'CG={tot}  [{time.perf_counter()-w0:.0f}s]')
            print(line, flush=True); log.write(line + '\n'); log.flush()
        if time.perf_counter() - last > chk_min*60:
            np.savez(f'{OUT}/chk_{tag}.npz', U=Uc, p=pc, t=t, dt=DT)
            last = time.perf_counter()
    np.savez(f'{OUT}/final_{tag}.npz', U=Uc, p=pc, t=t, dt=DT)
    return 'done'


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    for a in sys.argv[1:]:
        if a.startswith('--dt='):
            DT = float(a.split('=')[1])
        if a.startswith('--tol='):
            TOL = float(a.split('=')[1])
    s = build()
    if '--with-gate' in sys.argv or '--gate-only' in sys.argv:
        print('\n=== gate: laminar control, 200 steps ===', flush=True)
        st = run(s, 'gate_a00', 0.0, 0.0, 200*DT)
        if st != 'done':
            sys.exit(f'gate failed: {st}')
        if '--gate-only' in sys.argv:
            sys.exit(0)
    cand = {'a20': (2.0, 1.0), 'a30': (3.0, 1.5),
            's20': (2.0, 1.0), 's30': (3.0, 1.5)}
    want = [a for a in sys.argv[1:] if a in cand] or list(cand)
    resume = '--resume' in sys.argv
    for tag in want:
        ar, an = cand[tag]
        print(f'\n=== sweep {tag}: amp_roll={ar} amp_noise={an} '
              f'to t={TEND} ===', flush=True)
        run(s, tag, ar, an, TEND, resume=resume)
    print('\nsweep complete', flush=True)
