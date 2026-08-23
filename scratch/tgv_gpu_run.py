"""Checkpointed, restartable TGV driver for the CuPy/GPU path.

    python scratch/tgv_gpu_run.py <case> --outdir DIR [--budget SEC] [--price]

Built for Colab, whose VMs expire: a run survives as a chain of sessions, each
resuming the last checkpoint, working until its wall-clock budget is nearly
spent, checkpointing, and exiting with a message saying whether more sessions
are needed.

WHY RESTARTS ARE EXACT HERE.  RKW3's first stage has ZETA[0] = 0, so the
convective history N_prev is multiplied by zero at the top of every step and
carries no information ACROSS steps.  A checkpoint taken between steps
therefore needs only the state and the time -- no history, no stage index --
and a restarted run is bit-identical to an uninterrupted one, not merely
close.  `--selftest` proves that rather than asserting it.

WHAT IS WRITTEN (all to --outdir, which on Colab should be under Drive):
    chk_latest.npz   float64 state + t + step + config fingerprint
    chk_prev.npz     the previous one, kept so a crash mid-write cannot
                     destroy the only copy
    diag.csv         appended every step: t, E, Omega, balance, max|u|, CG
    frame_####.npz   complex64 snapshots for movies, every `snap` time units

Checkpoints are written to a temporary name and renamed, so a checkpoint file
is either complete or absent -- never half-written.
"""
import argparse, os, sys, time
sys.path.insert(0, '.'); sys.path.insert(0, 'scratch')
import numpy as np

import lssem3d
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem3d import (operator as OP, solver3d as S3, bc as BC, fourier as FR,
                     convect as CV, timestep as T)

L = 2*np.pi

CASES = {
    # The CORIA-CFD benchmark case (TGV_VALIDATION.md sec 9): Re = 1600,
    # integrated to t = 20, referenced to a 512^3 pseudo-spectral solution.
    # 16x16 elements at N = 8 gives 128 unique points per periodic direction.
    're1600_128': dict(nu=6.25e-4, N=8, ex=16, ey=16, nz=128, tend=20.0,
                       snap=1.0, cfl=1.0, tol=1e-6),
    # Reproduces the Mac's 88^3 Re = 800 run -- a cross-check of the GPU path
    # against a trajectory already recorded on CPU.
    're800_88':   dict(nu=1/800., N=8, ex=11, ey=11, nz=88, tend=4*np.pi,
                       snap=1.0, cfl=1.0, tol=1e-6),
    # Tiny, for exercising the restart machinery itself in seconds.
    'smoke':      dict(nu=0.01, N=8, ex=3, ey=3, nz=16, tend=0.4,
                       snap=0.1, cfl=1.0, tol=1e-8),
}


def setup(cfg, xp):
    m = build_channel(L, L, cfg['ex'], cfg['ey'], cfg['N'], bcs=(0, 0, 0, 0))
    m.periodic_x = L; m.periodic_y = L; m.compute_global_indices()
    nz = cfg['nz']; nk = nz//2 + 1
    mask = BC.build_mask(m, nk, pin_p=False, nz=nz)
    BC.pin_dof(m, mask, OP.P_, 0)
    n = cfg['N'] + 1
    X = np.empty((m.nelem, n, n)); Y = np.empty_like(X)
    for e in range(m.nelem):
        X[e] = m.xnod[e][:, None]; Y[e] = m.ynod[e][None, :]
    g = (lambda a: xp.asarray(a)) if xp is not np else (lambda a: a)
    return dict(m=m, D=diff_matrix(cfg['N']), N=cfg['N'], nz=nz, nk=nk,
                nu=cfg['nu'], kz=FR.wavenumbers(nz, L), mask=mask, X=X, Y=Y,
                zpl=(L/nz)*np.arange(nz), xp=xp, g=g,
                Dg=g(diff_matrix(cfg['N'])), kzg=g(FR.wavenumbers(nz, L)),
                fxg=g(m.facx), fyg=g(m.facy), wqg=g(m.wq), maskg=g(mask))


def ic_tgv(s):
    m, n, nz = s['m'], s['N']+1, s['nz']
    x = s['X'][..., None]; y = s['Y'][..., None]
    z = s['zpl'].reshape(1, 1, 1, -1)
    P = np.zeros((m.nelem, n, n, OP.NVAR, nz))
    P[..., OP.U_, :] = np.sin(x)*np.cos(y)*np.cos(z)
    P[..., OP.V_, :] = -np.cos(x)*np.sin(y)*np.cos(z)
    P[..., OP.OX_, :] = -np.cos(x)*np.sin(y)*np.sin(z)
    P[..., OP.OY_, :] = -np.sin(x)*np.cos(y)*np.sin(z)
    P[..., OP.OZ_, :] = 2*np.sin(x)*np.sin(y)*np.cos(z)
    P[..., OP.P_, :] = (1/16.)*(np.cos(2*x) + np.cos(2*y))*(np.cos(2*z) + 2.)
    return OP.to_real(FR.to_modes(P))


def precond(s, dt):
    m = s['m']
    shape = (m.nelem, s['N']+1, s['N']+1, OP.NVAR_R, s['nk'])
    out, rws = [], []
    for k in range(T.NSTAGE):
        cc = T.implicit_coeff(dt, k)
        rw = OP.momentum_row_weights(cc)
        rws.append(s['g'](rw))
        out.append(s['g'](S3.jacobi_inverse(S3.jacobi_diagonal_analytic(
            shape, s['D'], m.facx, m.facy, s['kz'], s['nu'], cc, m, s['mask'],
            m.wq, 0.0, rw=rw), s['mask'])))
    return out, rws


def stage(s, U, Nprev, k, dt, Minv, rw, tol, max_iter=60000):
    xp = s['xp']
    m, D, kz, mask, nu, nz = (s['m'], s['Dg'], s['kzg'], s['maskg'], s['nu'],
                              s['nz'])
    c = T.implicit_coeff(dt, k)
    Uc = OP.to_complex(U)
    Nk = -CV.convective(Uc, D, s['fxg'], s['fyg'], kz, nz)
    if xp is np:
        R0 = OP.apply_L0_complex(Uc, D, s['fxg'], s['fyg'], kz, nu, 0.0, 0.0)
    else:
        from lssem3d.kernels_cupy import _L0
        R0 = _L0(Uc, D, s['fxg'], s['fyg'], kz, nu, 0.0, 0.0)
    Lk = -R0[..., 4:7, :]
    fc = xp.zeros(Uc.shape[:-2] + (OP.NROW, Uc.shape[-1]), dtype=complex)
    for row, fld in ((4, OP.U_), (5, OP.V_), (6, OP.W_)):
        i = row - 4
        fc[..., row, :] = c*(Uc[..., fld, :] + dt*(
            T.GAMMA[k]*Nk[..., i, :] + T.ZETA[k]*Nprev[..., i, :]
            + T.ALPHA[k]*Lk[..., i, :]))
    f = xp.concatenate([fc.real, fc.imag], axis=-2)
    wq = s['wqg']
    fw = xp.concatenate([rw, rw]).reshape((1, 1, 1, f.shape[-2], 1))
    r = OP.apply_LT(
        OP.apply_L(U, D, s['fxg'], s['fyg'], kz, nu, c, wq, 0.0, rw)
        - f*wq[..., None, None]*fw, D, s['fxg'], s['fyg'], kz, nu, c, 0.0)
    b = -S3.gs(m, r)*mask
    dU, it, _ = S3.pcg(b, D, s['fxg'], s['fyg'], kz, nu, c, mesh=m, mask=mask,
                       M_inv=Minv[k], tol=tol, max_iter=max_iter, wq=wq, rw=rw)
    return U + dU, Nk, it


def diagnostics(s, U):
    xp = s['xp']
    P = FR.to_physical(OP.to_complex(U), s['nz'])
    wz = L/s['nz']; wq = s['wqg'][..., None]
    E = 0.5*wz*sum(float(xp.sum(xp.abs(P[..., f, :])**2*wq))
                   for f in (OP.U_, OP.V_, OP.W_))
    Om = 0.5*wz*sum(float(xp.sum(xp.abs(P[..., f, :])**2*wq))
                    for f in (OP.OX_, OP.OY_, OP.OZ_))
    mx = float(xp.abs(P[..., :3, :]).max())
    return E, Om, mx


def fingerprint(cfg):
    return '|'.join(f'{k}={cfg[k]}' for k in sorted(cfg))


def save_checkpoint(outdir, U, t, step, dt, cfg, xp):
    """Atomic, with one generation of history kept."""
    Uh = U if xp is np else xp.asnumpy(U)
    tmp = os.path.join(outdir, 'chk_tmp.npz')
    latest = os.path.join(outdir, 'chk_latest.npz')
    prev = os.path.join(outdir, 'chk_prev.npz')
    np.savez(tmp, U=Uh, t=t, step=step, dt=dt, fp=fingerprint(cfg))
    if os.path.exists(latest):
        os.replace(latest, prev)
    os.replace(tmp, latest)          # rename is atomic: never half-written


def load_checkpoint(outdir, cfg):
    for name in ('chk_latest.npz', 'chk_prev.npz'):
        p = os.path.join(outdir, name)
        if not os.path.exists(p):
            continue
        try:
            d = np.load(p, allow_pickle=False)
        except Exception as e:
            print(f'  {name} unreadable ({e}); trying the previous one')
            continue
        if str(d['fp']) != fingerprint(cfg):
            sys.exit(f'ERROR: {name} was written for a DIFFERENT configuration.\n'
                     f'  file: {d["fp"]}\n  this: {fingerprint(cfg)}\n'
                     f'Use a different --outdir rather than mixing runs.')
        return d['U'], float(d['t']), int(d['step']), float(d['dt'])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('case', choices=sorted(CASES))
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--budget', type=float, default=1e9,
                    help='wall-clock seconds for THIS session; the run '
                         'checkpoints and exits cleanly before spending it')
    ap.add_argument('--backend', default='cupy')
    ap.add_argument('--chk-minutes', type=float, default=10.0)
    ap.add_argument('--price', action='store_true',
                    help='time a few steps, print the projected total, exit')
    ap.add_argument('--selftest', action='store_true',
                    help='prove a restart reproduces an uninterrupted run')
    a = ap.parse_args()
    cfg = CASES[a.case]
    os.makedirs(a.outdir, exist_ok=True)
    lssem3d.set_backend(a.backend)
    xp = np
    if a.backend in ('cupy',):
        import cupy as cp; xp = cp
    s = setup(cfg, xp)

    U0 = ic_tgv(s)
    Pph = FR.to_physical(OP.to_complex(U0), s['nz'])
    dt = float(CV.max_dt_for_cfl(Pph[..., :OP.NVAR, :], s['D'], s['m'].facx,
                                 s['m'].facy, L, s['nz'], cfg['cfl']))
    dt = min(dt, 0.02)
    nstep = int(np.ceil(cfg['tend']/dt))
    dof = U0.size

    if a.selftest:
        return selftest(s, cfg, dt, a)

    resumed = load_checkpoint(a.outdir, cfg)
    if resumed is None:
        U, t, step = s['g'](U0), 0.0, 0
        print(f'START  {a.case}: {cfg["ex"]}x{cfg["ey"]} N={cfg["N"]} '
              f'Nz={cfg["nz"]} ({dof/1e6:.2f} M dof), dt={dt:.5g}, '
              f'{nstep} steps to t={cfg["tend"]:g}')
        mode = 'w'
    else:
        Uh, t, step, dt_ck = resumed
        U = s['g'](Uh); dt = dt_ck
        print(f'RESUME {a.case} at t={t:.4f} (step {step}/{nstep})')
        mode = 'a'
    Minv, rws = precond(s, dt)
    Np_ = xp.zeros(OP.to_complex(U).shape[:-2] + (3, s['nk']), dtype=complex)

    dpath = os.path.join(a.outdir, 'diag.csv')
    if mode == 'w' or not os.path.exists(dpath):
        with open(dpath, 'w') as fh:
            fh.write('t,E,Omega,balance,maxu,cg,capped\n')

    if a.price:
        t0 = time.perf_counter()
        for _ in range(3):
            for k in range(T.NSTAGE):
                U, Np_, it = stage(s, U, Np_, k, dt, Minv, rws[k], cfg['tol'])
        if xp is not np:
            xp.cuda.Stream.null.synchronize()
        per = (time.perf_counter()-t0)/3
        print(f'PRICE  {per:.1f} s/step  ->  {nstep*per/3600:.1f} h total '
              f'({nstep} steps)  |  sessions at {a.budget/3600:.1f} h: '
              f'{int(np.ceil(nstep*per/a.budget))}')
        return

    E, Om, mx = diagnostics(s, U)
    t_start, wall0, last_chk = t, time.perf_counter(), time.perf_counter()
    next_snap = (int(t/cfg['snap']) + 1)*cfg['snap']
    nfr = int(round(t/cfg['snap']))
    prevE, prevOm = E, Om
    while step < nstep:
        tot, capped = 0, False
        for k in range(T.NSTAGE):
            U, Np_, it = stage(s, U, Np_, k, dt, Minv, rws[k], cfg['tol'])
            tot += it; capped |= (it >= 60000)
        step += 1; t += dt
        E, Om, mx = diagnostics(s, U)
        bal = (-(E - prevE)/dt)/(2*cfg['nu']*0.5*(Om + prevOm)) if step > 1 else 1.0
        prevE, prevOm = E, Om
        with open(dpath, 'a') as fh:
            fh.write(f'{t:.6f},{E:.10f},{Om:.8f},{bal:.6f},{mx:.6f},'
                     f'{tot},{int(capped)}\n')
        if t + 1e-12 >= next_snap:
            Uc = OP.to_complex(U)
            arr = Uc if xp is np else xp.asnumpy(Uc)
            np.savez_compressed(os.path.join(a.outdir, f'frame_{nfr:04d}.npz'),
                                U=arr.astype(np.complex64), t=t)
            nfr += 1; next_snap += cfg['snap']
        now = time.perf_counter()
        if now - last_chk > a.chk_minutes*60:
            save_checkpoint(a.outdir, U, t, step, dt, cfg, xp)
            last_chk = now
        if step % 10 == 0:
            rate = (now - wall0)/max(step - int(round(t_start/dt)), 1)
            print(f't={t:8.4f} E={E:.6f} Om={Om:.4f} bal={bal:.4f} '
                  f'CG={tot} [{rate:.1f} s/step]', flush=True)
        # leave room to checkpoint before the session is cut off
        if now - wall0 > a.budget - 120:
            save_checkpoint(a.outdir, U, t, step, dt, cfg, xp)
            left = (nstep - step)*(now - wall0)/max(step - int(round(t_start/dt)), 1)
            print(f'\nBUDGET REACHED at t={t:.4f} (step {step}/{nstep}). '
                  f'Checkpoint written.\nRun this cell again to continue; '
                  f'~{left/3600:.1f} h of compute remain.')
            return
    save_checkpoint(a.outdir, U, t, step, dt, cfg, xp)
    print(f'\nDONE at t={t:.4f} ({step} steps). Checkpoint and {nfr} frames '
          f'in {a.outdir}')


def selftest(s, cfg, dt, a):
    """A restart must reproduce an uninterrupted run -- prove it."""
    xp = s['xp']
    Minv, rws = precond(s, dt)
    U0 = s['g'](ic_tgv(s))
    Np0 = xp.zeros(OP.to_complex(U0).shape[:-2] + (3, s['nk']), dtype=complex)

    def advance(U, n):
        Np_ = xp.zeros_like(Np0)
        for _ in range(n):
            for k in range(T.NSTAGE):
                U, Np_, _ = stage(s, U, Np_, k, dt, Minv, rws[k], cfg['tol'])
        return U
    straight = advance(U0, 6)
    half = advance(U0, 3)
    save_checkpoint(a.outdir, half, 3*dt, 3, dt, cfg, xp)
    Uh, t, step, dt2 = load_checkpoint(a.outdir, cfg)
    restarted = advance(s['g'](Uh), 3)
    d = float(xp.abs(straight - restarted).max())
    scale = float(xp.abs(straight).max())
    print(f'SELFTEST  6 steps straight vs 3+checkpoint+3:')
    print(f'  max|difference| = {d:.3e}  (relative {d/scale:.3e})')
    print(f'  {"PASS -- restart is exact" if d == 0.0 else "PASS (within solver tolerance)" if d/scale < 1e-10 else "FAIL"}')
    print(f'  RKW3 ZETA[0] = {T.ZETA[0]} -- this is WHY no convective history '
          f'has to be carried across the checkpoint')


if __name__ == '__main__':
    main()
