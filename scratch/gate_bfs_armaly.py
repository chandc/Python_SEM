"""GATE 3: Armaly backward-facing step, Re = 389, quasi-2D (thin span).

Geometry (Armaly 1983 / in-house 2D validation ARMALY_VALIDATION.md):
inlet height h = 1.0 above a step S = 0.94 (ER 1.94); inlet channel length 2;
outlet length 18 (~19 S).  Re = U_m * 2h / nu = 389 with U_m = 1.
PASS: reattachment x_r/S -> 8.1 +/- 0.4 (in-house 2D: 8.145; Armaly: 8.05).
"""
import os, sys, time
for v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS',
          'VECLIB_MAXIMUM_THREADS'):
    os.environ[v] = '12'
sys.path.insert(0,'.'); sys.path.insert(0,'scratch')
import numpy as np


def main():
    import lssem3d; lssem3d.set_backend('numpy')
    from lssem2d.mesh import build_bfs
    from lssem2d.lgl import diff_matrix
    from lssem3d import (project as PJ, helmholtz as HH, convect as CV,
                         fourier as FR, solver3d as S3, timestep as T,
                         deriv as DV, hpmg)
    S_h, H_in = 0.94, 1.0
    UM = 1.0
    NU = UM*2*H_in/389.0
    NZ = 4
    N, EIX, EOX, EY = 7, 3, 16, 3
    LIN, LOUT = 2.0, 18.0
    m = build_bfs(N, E_in_x=EIX, E_out_x=EOX, E_y=EY, L_in=LIN, L_out=LOUT,
                  H_in=H_in, H_step=S_h)
    nk, n = NZ//2 + 1, N + 1
    LZ = 1.0
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
    dt = 4e-3
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
        # last sign change from negative (recirc) to positive
        sgn = np.sign(tw)
        idx = np.flatnonzero((sgn[:-1] < 0) & (sgn[1:] >= 0))
        if len(idx) == 0:
            return 0.0
        i = idx[-1]
        x0, x1, f0, f1 = xs[i], xs[i+1], tw[i], tw[i+1]
        return (x0 - f0*(x1 - x0)/(f1 - f0))/S_h
    nstep = int(round(float(sys.argv[sys.argv.index('--tend')+1])
                      if '--tend' in sys.argv else 150.0)/dt) if True else 0
    nstep = int(round((float(sys.argv[sys.argv.index('--tend')+1])
                       if '--tend' in sys.argv else 150.0)/dt))
    w0 = last = time.time()
    os.makedirs('scratch/_bfs', exist_ok=True)
    for i in range(nstep):
        tot = 0
        for k in range(T.NSTAGE):
            s['Mu'] = pre[k]
            Nk = -CV.convective(Uc, D, m.facx, m.facy, kz, NZ, skew=True)
            Uc, pc, inf = PJ.substage(s, Uc, pc, Nk, Nprev, k, dt)
            Nprev = Nk
            tot += inf[0] + inf[2]
        if not np.isfinite(np.abs(Uc).max()):
            print(f'BLEW UP at t={(i+1)*dt:.3f}', flush=True); return
        if i % 500 == 499:
            xr = reattach(Uc)
            print(f't={(i+1)*dt:7.2f}  x_r/S={xr:6.3f}  CG={tot}  '
                  f'[{time.time()-w0:.0f}s]', flush=True)
        if time.time() - last > 20*60:
            np.savez('scratch/_bfs/chk.npz', U=Uc, p=pc, t=(i+1)*dt)
            last = time.time()
    np.savez('scratch/_bfs/final.npz', U=Uc, p=pc, t=nstep*dt)
    print(f'DONE x_r/S={reattach(Uc):.3f}', flush=True)


if __name__ == '__main__':
    main()
