"""GATES for the projection-path outflow OBC.  --vortex = Gate 2.

Channel x in [0, 2pi] (inflow W, outflow E), walls y, Fourier z.  Inflow is
the exact parabola; the correct steady state IS the parabola everywhere with
p linear in x.  Pass: velocity relative error vs exact < 1e-3 at t = 5, no
drift, exit plane clean.
"""
import os, sys, time
for v in ('OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ[v]='12'
sys.path.insert(0,'.'); sys.path.insert(0,'scratch')
import numpy as np


def main():
    import lssem3d; lssem3d.set_backend('numpy')
    from lssem2d.mesh import build_channel
    from lssem2d.lgl import diff_matrix
    from lssem3d import (project as PJ, helmholtz as HH, convect as CV,
                         fourier as FR, solver3d as S3, timestep as T, hpmg)
    RE = 100.0
    NU = 1.0/RE
    LX, LY, NZ = 2*np.pi, 2.0, 8
    EX, EY, N = 6, 6, 6
    # W=wall-code(=inflow Dirichlet via lifting), E=OUTFLOW, S/N walls
    m = build_channel(LX, LY, EX, EY, N, bcs=(1, PJ.OUTFLOW, 1, 1))
    m.compute_global_indices()
    nk, n = NZ//2 + 1, N + 1
    kz = FR.wavenumbers(NZ, 0.34*np.pi)
    mask_u = PJ.build_masks(m, nk, NZ, 3, wall=True)
    mask_p = PJ.build_masks(m, nk, NZ, 1, wall=False, outflow_p=True)
    D = diff_matrix(N)
    Y = np.empty((m.nelem, n, n)); X = np.empty_like(Y)
    for e in range(m.nelem):
        X[e] = m.xnod[e][:, None]; Y[e] = m.ynod[e][None, :]
    mw1 = S3.multiplicity_weight(m, mask_p.shape)[..., 0:1, 0:1]
    s = dict(m=m, D=D, N=N, nz=NZ, nk=nk, nu=NU, kz=kz, lz=0.34*np.pi,
             tol=1e-9, incremental=False, mask_u=mask_u, mask_p=mask_p,
             Dg=D, fxg=m.facx, fyg=m.facy, wqg=m.wq, kzg=kz,
             wq3=m.wq[..., None, None], wq1=m.wq[..., None, None],
             mw1=mw1, null_kz0=None, null_norm=1.0,   # NONSINGULAR: no null
             wall_u=None, ubc=None, backend='numpy', check_every=1)
    dt = 2e-3
    s['Mp'] = HH.fdm_preconditioner(m, N, kz**2, 1.0, mask_p, 2, nk,
                                    like=mask_p)
    pre = [HH.fdm_preconditioner(m, N, T.implicit_coeff(dt, k) + NU*kz**2,
                                 NU, mask_u, 6, nk, like=mask_u)
           for k in range(T.NSTAGE)]
    # exact Poiseuille: u = y(2-y)*3/2*Ub with Ub=2/3 -> u_max=1
    uex = Y*(2.0 - Y)/1.0          # u_max = 1 at centreline
    up = np.zeros((m.nelem, n, n, 3, NZ))
    up[..., 0, :] = uex[..., None]
    Uex = FR.to_modes(up)[..., :nk]
    # start from the EXACT solution; the gate is that it STAYS there
    Uc = Uex.copy()
    if '--vortex' in sys.argv:
        # GATE 2: superimpose a compact vortex; it must ADVECT OUT cleanly.
        # Solenoidal by construction (curl of a Gaussian streamfunction),
        # centred mid-channel at x0 = pi/2, radius 0.35.
        x0, y0, r0, amp = np.pi/2, 1.0, 0.35, 0.4
        g = amp*np.exp(-((X - x0)**2 + (Y - y0)**2)/(2*r0**2))
        dgdx = -(X - x0)/r0**2*g
        dgdy = -(Y - y0)/r0**2*g
        vp = np.zeros((m.nelem, n, n, 3, NZ))
        vp[..., 0, :] = dgdy[..., None]          # u' = dpsi/dy
        vp[..., 1, :] = -dgdx[..., None]         # v' = -dpsi/dx
        Uc = Uc + FR.to_modes(vp)[..., :nk]
        Uc = PJ._join(PJ._split(Uc)*mask_u + PJ._split(Uex)*(1 - mask_u))
    # inflow lifting: the prescribed values on masked dofs (walls carry 0,
    # inflow carries the parabola).  build as full field * (1 - mask):
    lift = PJ._split(Uex)*(1.0 - mask_u)
    s['ubc_in'] = lift
    s['ubc'] = lift
    pc = np.zeros((m.nelem, n, n, 1, nk), dtype=complex)
    Nprev = np.zeros_like(Uc)
    nrm = lambda a: float(np.sqrt((np.abs(a)**2).sum()))
    tend = 12.0 if '--vortex' in sys.argv else 5.0
    nstep = int(round(tend/dt))
    w0 = time.time()
    for i in range(nstep):
        tot = 0
        for k in range(T.NSTAGE):
            s['Mu'] = pre[k]
            Nk = -CV.convective(Uc, D, m.facx, m.facy, kz, NZ, skew=True)
            Uc, pc, inf = PJ.substage(s, Uc, pc, Nk, Nprev, k, dt)
            Nprev = Nk
            tot += inf[0] + inf[2]
        if '--snapshots' in sys.argv and abs((i+1)*dt - round((i+1)*dt/3)*3) < dt/2                 and round((i+1)*dt/3)*3 <= 9.1 and (i+1) % 10 == 0:
            pass
        if '--snapshots' in sys.argv and (i+1) in (1, 1500, 3000, 4500):
            np.savez(f'scratch/_obc_snap_{i+1}.npz', U=Uc, t=(i+1)*dt)
        if i % 500 == 499 or i == nstep - 1:
            err = nrm(Uc - Uex)/nrm(Uex)
            print(f't={(i+1)*dt:6.3f}  rel err vs exact {err:.3e}  CG {tot}  '
                  f'[{time.time()-w0:.0f}s]', flush=True)
            if not np.isfinite(err):
                print('BLEW UP', flush=True); return

if __name__ == '__main__':
    main()
