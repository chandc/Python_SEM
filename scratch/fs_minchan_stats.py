"""Minimal-channel Re_tau=180 STATISTICS run -- RK3-CN, skew, consistent
P_N-P_N projection, resuming from the tripped turbulent state.

    python scratch/fs_minchan_stats.py --restart FILE [--backend cupy]
        [--tend 28] [--dt 3.5e-4] [--consistent] [--outdir DIR]

Statistics: plane-averaged (x,z, quadrature-weighted in x) running sums of
U, u^2, v^2, w^2, uv at every distinct y, accumulated every ACC_EVERY steps,
plus the u_tau time series.  Saved with every checkpoint -- a killed run
keeps its statistics.
"""
import os, sys, time
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '12')
sys.path.insert(0, '.'); sys.path.insert(0, 'scratch')
import numpy as np

def main():
    arg = lambda f, d: (sys.argv[sys.argv.index(f) + 1] if f in sys.argv else d)
    backend = arg('--backend', 'cupy')
    restart = arg('--restart', 'scratch/_minchan_fs/final_s20.npz')
    outdir = arg('--outdir', 'scratch/_minchan_stat')
    TEND = float(arg('--tend', 28.0))
    DT = float(arg('--dt', 3.5e-4))
    CONSISTENT = '--consistent' in sys.argv
    ACC_EVERY, LOG_EVERY, CHK_MIN = 5, 100, 20.0

    import lssem3d; lssem3d.set_backend(backend)
    from lssem2d.mesh import build_channel
    from lssem2d.lgl import diff_matrix
    from lssem3d import (project as PJ, helmholtz as HH, convect as CV,
                         fourier as FR, solver3d as S3, timestep as T,
                         deriv as DV, hpmg)

    RE_TAU, DELTA = 180.0, 1.0
    LX, LZ, FX = np.pi, 0.34*np.pi, 1.0
    N, EX, EY, NZ = 8, 6, 18, 32
    NU, TOL = 1.0/RE_TAU, 1e-7

    m = build_channel(LX, 2.0*DELTA, EX, EY, N, bcs=(0, 0, 1, 1))
    m.periodic_x = LX
    m.compute_global_indices()
    nk, n = NZ//2 + 1, N + 1
    kz = FR.wavenumbers(NZ, LZ)
    mask_u = PJ.build_masks(m, nk, NZ, 3, wall=True)
    mask_p = PJ.build_masks(m, nk, NZ, 1, wall=False)
    if not CONSISTENT:
        ind = np.zeros(mask_p.shape); ind[0, 0, 0, 0, 0] = 1.0
        mask_p[..., 0, 0] *= (S3.gs(m, ind)[..., 0, 0] < 0.5)
    if backend == 'cupy':
        import cupy as xp
        g = lambda a: xp.asarray(np.ascontiguousarray(a))
        print(f'GPU  {xp.cuda.runtime.getDeviceProperties(0)["name"].decode()}',
              flush=True)
    else:
        xp = np; g = lambda a: a
    v = np.ones(mask_p[..., 0:1, 0:1].shape)*mask_p[..., 0:1, 0:1]
    mw1 = S3.multiplicity_weight(m, mask_p.shape)[..., 0:1, 0:1]
    D = diff_matrix(N)
    s = dict(m=m, D=D, N=N, nz=NZ, nk=nk, nu=NU, kz=kz, lz=LZ, tol=TOL,
             incremental=False, mask_u=g(mask_u), mask_p=g(mask_p),
             Dg=g(D), fxg=g(m.facx), fyg=g(m.facy), wqg=g(m.wq), kzg=g(kz),
             wq3=g(m.wq[..., None, None]), wq1=g(m.wq[..., None, None]),
             mw1=g(mw1), null_kz0=g(v), null_norm=float((v*v*mw1).sum()),
             wall_u=g(PJ.wall_indicator(m, nk, NZ, 3)), ubc=None,
             backend=backend, check_every=None, consistent_p=CONSISTENT)
    t0 = time.perf_counter()
    if CONSISTENT:
        from lssem3d import epmg
        s['Mp'] = epmg.ConsistentPMG(m, N, kz, nk, NZ, deg=6, like=s['mask_p'])
        s['tol_p'] = float(arg('--tolp', '1e-4'))
        print(f'pressure: E-multigrid deg=6, tol_p={s["tol_p"]:g} '
              f'(setup {time.perf_counter()-t0:.1f}s)', flush=True)
    else:
        s['Mp'] = hpmg.HelmholtzPMG(m, N, kz**2, 1.0, 1, nk, NZ, wall=False,
                                    pin_kz0=True, deg=6, like=s['mask_p'])
        print(f'pressure: pmg pin_kz0=True '
              f'(setup {time.perf_counter()-t0:.1f}s)', flush=True)
    pre = [HH.fdm_preconditioner(m, N, T.implicit_coeff(DT, k) + NU*(kz**2), NU,
                                 s['mask_u'], 6, nk, like=s['mask_u'])
           for k in range(T.NSTAGE)]
    fp = np.zeros((m.nelem, n, n, 3, NZ)); fp[..., 0, :] = FX
    Fm = g(FR.to_modes(fp)[..., :nk])

    NW = int(arg('--modepar', '0'))
    if NW > 0 and backend == 'numpy':
        from lssem3d.modepar import ModePool
        _cfg = dict(N=N, ex=EX, ey=EY, nz=NZ, nu=NU, dt=DT, tol=TOL,
                    lx=float(LX), lz=float(LZ))
        s['modepool'] = ModePool(_cfg, nworkers=NW, blas_threads=3)
        print(f'mode-space pool: {NW} workers x 3 BLAS threads', flush=True)

    d = np.load(restart)
    Uc, pc, t = g(d['U']), g(d['p']), float(d['t'])
    Nprev = xp.zeros_like(Uc)
    print(f'restart {restart} at t={t:.4f}; dt={DT:g}, to t={TEND}', flush=True)

    # ---- statistics machinery (host side) ----
    Y = np.empty((m.nelem, n, n)); W = m.wq
    for e in range(m.nelem):
        Y[e] = m.ynod[e][None, :]
    yk = np.round(Y, 10).ravel()
    order = np.argsort(yk, kind='stable')
    splits = np.flatnonzero(np.diff(yk[order]) > 1e-9) + 1
    groups = np.split(np.arange(len(yk)), splits)
    yvals = np.array([yk[order][gr[0]] for gr in groups])
    wcol = (W.ravel())[order]
    gw = [wcol[gr] for gr in groups]
    gwsum = np.array([w.sum() for w in gw])
    NY = len(groups)
    sums = np.zeros((5, NY))          # U, uu, vv, ww, uv (plane means)
    nsamp = 0
    utau_series = []

    def accumulate(Uc_host):
        nonlocal nsamp
        P = FR.to_physical(Uc_host, NZ)
        flat = P.reshape(-1, 3, NZ)[order]
        for iq, quant in enumerate((lambda f: f[:, 0, :],
                                    lambda f: f[:, 0, :]**2,
                                    lambda f: f[:, 1, :]**2,
                                    lambda f: f[:, 2, :]**2,
                                    lambda f: f[:, 0, :]*f[:, 1, :])):
            for j, gr in enumerate(groups):
                q = quant(flat[gr])
                sums[iq, j] += float((q.mean(axis=1)*gw[j]).sum()/gwsum[j])
        nsamp += 1

    def utau_of(Uc_host):
        du = DV.ddy(Uc_host[..., 0:1, :], s['D'], m.facy)[..., 0, 0].real/NZ
        lo = np.abs(Y - 0.0) < 1e-12; hi = np.abs(Y - 2.0) < 1e-12
        return float(np.sqrt(0.5*(np.abs(du[lo]).mean()
                                  + np.abs(du[hi]).mean())*NU))

    def save_stats(path):
        np.savez(path, y=yvals, sums=sums, nsamp=nsamp, t=t,
                 utau_series=np.array(utau_series), nu=NU, dt=DT,
                 note='sums = plane means of U,uu,vv,ww,uv; divide by nsamp')

    def step():
        nonlocal Uc, pc, Nprev
        tot = 0
        for k in range(T.NSTAGE):
            s['Mu'] = pre[k]
            Nk = -CV.convective(Uc, s['Dg'], s['fxg'], s['fyg'], s['kzg'], NZ,
                                skew=True) + Fm
            Uc, pc, inf = PJ.substage(s, Uc, pc, Nk, Nprev, k, DT)
            Nprev = Nk
            tot += inf[0] + inf[2]
        return tot

    os.makedirs(outdir, exist_ok=True)
    log = open(f'{outdir}/stats_run.log', 'a')
    cvt = (lambda a: a) if backend == 'numpy' else xp.asnumpy
    nstep = int(round((TEND - t)/DT))
    w0 = last = time.perf_counter()
    for i in range(nstep):
        tot = step()
        t += DT
        if i % ACC_EVERY == ACC_EVERY - 1:
            Uh = cvt(Uc)
            accumulate(Uh)
            ut = utau_of(Uh)
            utau_series.append((t, ut))
        if i % LOG_EVERY == LOG_EVERY - 1:
            Uh = cvt(Uc)
            if not np.isfinite(np.abs(Uh).max()):
                line = f'BLEW UP at t={t:.4f}'
                print(line, flush=True); log.write(line + '\n'); log.flush()
                break
            dd = PJ.divergence(g(Uh), s['Dg'], s['fxg'], s['fyg'], s['kzg'])
            rdiv = float(xp.sqrt((abs(dd)**2).sum()/(abs(Uc)**2).sum()))
            ut = utau_of(Uh)
            line = (f't={t:8.4f}  u_tau={ut:.4f}  div={rdiv:.1e}  CG={tot}  '
                    f'nsamp={nsamp}  [{time.perf_counter()-w0:.0f}s]')
            print(line, flush=True); log.write(line + '\n'); log.flush()
        if time.perf_counter() - last > CHK_MIN*60:
            np.savez(f'{outdir}/chk_tmp.npz', U=cvt(Uc), p=cvt(pc), t=t, dt=DT)
            os.replace(f'{outdir}/chk_tmp.npz', f'{outdir}/chk_latest.npz')
            save_stats(f'{outdir}/stats_latest.npz')
            last = time.perf_counter()
    np.savez(f'{outdir}/final_state.npz', U=cvt(Uc), p=cvt(pc), t=t, dt=DT)
    save_stats(f'{outdir}/stats_final.npz')
    print(f'DONE t={t:.4f} in {(time.perf_counter()-w0)/3600:.2f} h, '
          f'{nsamp} samples', flush=True)


if __name__ == '__main__':
    main()
