"""GATE 3: Armaly backward-facing step, Re = 389, quasi-2D (thin span).

Geometry (Armaly 1983 / in-house 2D validation ARMALY_VALIDATION.md):
inlet height h = 1.0 above a step S = 0.94 (ER 1.94); inlet channel length 2;
outlet length 18 (~19 S).  Re = U_m * 2h / nu = 389 with U_m = 1.
PASS: reattachment x_r/S -> 8.1 +/- 0.4 (in-house 2D: 8.145; Armaly: 8.05).
"""
import os, sys, time
for v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS',
          'VECLIB_MAXIMUM_THREADS'):
    os.environ[v] = os.environ.get('BFS_THREADS', '12')
sys.path.insert(0,'.'); sys.path.insert(0,'scratch')
import numpy as np


def main():
    import lssem3d; lssem3d.set_backend('numpy')
    from lssem2d.mesh import build_bfs
    from lssem2d.lgl import diff_matrix
    from lssem3d import (project as PJ, helmholtz as HH, convect as CV,
                         fourier as FR, solver3d as S3, timestep as T,
                         deriv as DV, hpmg)
    # geometry: --er sets the expansion ratio (H_in + S)/H_in.
    # 1.94 = Armaly's rig (S = 0.94); 2.0 = the Erturk/literature standard.
    _er = float(sys.argv[sys.argv.index('--er')+1]) if '--er' in sys.argv else 1.94
    H_in = 1.0
    S_h = H_in*(_er - 1.0)
    UM = 1.0
    arg = lambda f, dflt: (sys.argv[sys.argv.index(f)+1] if f in sys.argv
                           else dflt)
    RE = float(arg('--re', 389.0))
    NU = UM*2*H_in/RE
    NZ = int(arg('--nz', 4))
    N = int(arg('--order', 7))
    XPOW = float(arg('--xpow', 1.0))
    EIX, EOX, EY = 3, 16, 3
    LIN = 2.0
    # Re=600 references put the upper-wall bubble's reattachment at ~17 step
    # heights: an 18-unit outlet ends AT the bubble and the open boundary
    # interacts with it (measured: x_r oscillating 12.6 <-> 17.6).  Scale the
    # domain with Re.
    LOUT = float(sys.argv[sys.argv.index('--lout')+1]) if '--lout' in sys.argv         else 18.0
    m = build_bfs(N, E_in_x=EIX, E_out_x=EOX, E_y=EY, L_in=LIN, L_out=LOUT,
                  H_in=H_in, H_step=S_h, xpow=XPOW)
    nk, n = NZ//2 + 1, N + 1
    LZ = float(arg('--lz', 1.0))
    kz = FR.wavenumbers(NZ, LZ)
    mask_u = PJ.build_masks(m, nk, NZ, 3, wall=True)
    mask_p = PJ.build_masks(m, nk, NZ, 1, wall=False, outflow_p=True)
    D = diff_matrix(N)
    X = np.empty((m.nelem, n, n)); Y = np.empty_like(X)
    for e in range(m.nelem):
        X[e] = m.xnod[e][:, None]; Y[e] = m.ynod[e][None, :]
    mw1 = S3.multiplicity_weight(m, mask_p.shape)[..., 0:1, 0:1]
    s = dict(m=m, D=D, N=N, nz=NZ, nk=nk, nu=NU, kz=kz, lz=LZ,
             tol=1e-7, incremental=False, mask_u=mask_u, mask_p=mask_p,
             Dg=D, fxg=m.facx, fyg=m.facy, wqg=m.wq, kzg=kz,
             wq3=m.wq[..., None, None], wq1=m.wq[..., None, None],
             mw1=mw1, null_kz0=None, null_norm=1.0,
             wall_u=None, ubc=None, backend='numpy', check_every=1)
    # dt from the ACTUAL mesh via the CFL rule -- hand-picked dt failed twice
    # when grading shrank the smallest column 5.3x.  The estimate uses the
    # steady inflow profile as the velocity scale; 0.35 safety covers the
    # corner acceleration (local speeds ~1.5x inflow there).
    up0 = np.zeros((m.nelem, n, n, 7, NZ))
    yy0 = np.clip((Y - S_h)/H_in, 0.0, 1.0)
    up0[..., 0, :] = (6.0*UM*yy0*(1.0 - yy0)
                      * ((Y > S_h) & (Y < S_h + H_in)))[..., None]
    dt_cfl = CV.max_dt_for_cfl(up0, D, m.facx, m.facy, LZ, NZ, 1.0)
    dt = min(float(arg('--dt', 1e9)), 0.35*dt_cfl)
    print(f'dt = {dt:.2e}  (0.35 x unit-CFL {dt_cfl:.2e})', flush=True)
    t0 = time.time()
    s['Mp'] = hpmg.HelmholtzPMG(m, N, kz**2, 1.0, 1, nk, NZ, wall=False,
                                pin_kz0=False, outflow_p=True, deg=6,
                                like=mask_p)
    print(f'pressure PMG (outflow) setup {time.time()-t0:.1f}s', flush=True)
    pre = [HH.fdm_preconditioner(m, N, T.implicit_coeff(dt, k) + NU*kz**2,
                                 NU, mask_u, 6, nk, like=mask_u)
           for k in range(T.NSTAGE)]
    # inflow parabola in the inlet span y in [S, S+h]; zero elsewhere
    yy = np.clip((Y - S_h)/H_in, 0.0, 1.0)
    prof = 6.0*UM*yy*(1.0 - yy)          # mean UM, max 1.5 UM
    prof = np.where((Y > S_h) & (Y < S_h + H_in), prof, 0.0)
    up = np.zeros((m.nelem, n, n, 3, NZ))
    up[..., 0, :] = prof[..., None]
    Uprof = FR.to_modes(up)[..., :nk]
    lift = PJ._split(Uprof)*(1.0 - mask_u)
    s['ubc_in'] = lift; s['ubc'] = lift
    TRAMP = float(arg('--ramp', 0.0))   # smooth inflow spin-up over TRAMP
    # IC: the profile field masked to interior (top stream flows, bottom still)
    Uc = PJ._join(PJ._split(Uprof)*mask_u + lift)
    pc = np.zeros((m.nelem, n, n, 1, nk), dtype=complex)
    Nprev = np.zeros_like(Uc)
    # reattachment: du/dy at the BOTTOM wall (y=0), zero crossing, x>0
    bot = []
    for e in range(m.nelem):
        if m.bc[e, 2] == 1 and abs(m.y0[e]) < 1e-12 and m.x0[e] >= -1e-12:
            bot.append(e)
    bot = np.array(sorted(bot, key=lambda e: m.x0[e]))
    def reattach(Uc):
        du = DV.ddy(Uc[..., 0:1, :], D, m.facy)[..., 0, 0].real/NZ
        xs, tw = [], []
        for e in bot:
            for i in range(n):
                xs.append(X[e][i, 0]); tw.append(du[e, i, 0])
        xs, tw = np.array(xs), np.array(tw)
        o = np.argsort(xs); xs, tw = xs[o], tw[o]
        # A SHED EDDY between the primary bubble and the exit adds crossing
        # pairs; reporting only the LAST crossing conflated eddy transits
        # with reattachment swings (measured: 'x_r' 12.6 <-> 17.6 while the
        # primary reattachment sat near 6).  Report FIRST (primary), LAST,
        # and the crossing count; skip the corner micro-features (x < 0.5 S).
        sgn = np.sign(tw)
        idx = np.flatnonzero((sgn[:-1] < 0) & (sgn[1:] >= 0))
        idx = [i for i in idx if xs[i] > 0.5*S_h]
        if len(idx) == 0:
            return 0.0, 0.0, 0
        def xz(i):
            x0, x1, f0, f1 = xs[i], xs[i+1], tw[i], tw[i+1]
            return (x0 - f0*(x1 - x0)/(f1 - f0))/S_h
        return xz(idx[0]), xz(idx[-1]), len(idx)
    nstep = int(round(float(sys.argv[sys.argv.index('--tend')+1])
                      if '--tend' in sys.argv else 150.0)/dt) if True else 0
    nstep = int(round((float(sys.argv[sys.argv.index('--tend')+1])
                       if '--tend' in sys.argv else 150.0)/dt))
    w0 = last = time.time()
    OUT = arg('--outdir', 'scratch/_bfs')
    os.makedirs(OUT, exist_ok=True)
    for i in range(nstep):
        if TRAMP > 0:
            r = min(1.0, (i*dt)/TRAMP)
            r = 0.5 - 0.5*np.cos(np.pi*r)      # C1 ramp
            s['ubc_in'] = lift*r
            if i*dt <= TRAMP + 2*dt:
                s['ubc'] = lift*r
        tot = 0
        for k in range(T.NSTAGE):
            s['Mu'] = pre[k]
            Nk = -CV.convective(Uc, D, m.facx, m.facy, kz, NZ, skew=True)
            Uc, pc, inf = PJ.substage(s, Uc, pc, Nk, Nprev, k, dt)
            Nprev = Nk
            tot += inf[0] + inf[2]
        if '--forensic' in sys.argv and i % 50 == 49:
            P0 = np.abs(Uc).max()
            am = np.unravel_index(np.argmax(np.abs(Uc[..., 0])), Uc[..., 0].shape)
            print(f'  t={(i+1)*dt:.3f} max|U|={P0:.3e} at elem {am[0]} '
                  f'(x0={m.x0[am[0]]:.2f}, y0={m.y0[am[0]]:.2f}) '
                  f'field {am[3]} mode {am[4] if len(am)>4 else 0}', flush=True)
            np.savez(f'{OUT}/forensic_{i+1}.npz', U=Uc, t=(i+1)*dt)
        if not np.isfinite(np.abs(Uc).max()):
            print(f'BLEW UP at t={(i+1)*dt:.3f}', flush=True); return
        if i % 500 == 499:
            xr1, xrN, nc = reattach(Uc)
            print(f't={(i+1)*dt:7.2f}  x_r/S={xr1:6.3f}  xlast/S={xrN:6.3f}  '
                  f'ncross={nc}  CG={tot}  [{time.time()-w0:.0f}s]',
                  flush=True)
        if time.time() - last > 20*60:
            np.savez(f'{OUT}/chk.npz', U=Uc, p=pc, t=(i+1)*dt)
            last = time.time()
    np.savez(f'{OUT}/final.npz', U=Uc, p=pc, t=nstep*dt)
    xr1, xrN, nc = reattach(Uc)
    print(f'DONE x_r/S={xr1:.3f} (last {xrN:.3f}, ncross {nc})', flush=True)


if __name__ == '__main__':
    main()
