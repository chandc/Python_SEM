"""Preconditioner scaling on a REAL problem: laminar channel, N swept.

WHY THIS AND NOT THE EARLIER SCAN.  The previous scans built the right-hand side
as b = A*x with x a RANDOM continuous field, and concluded Jacobi's iteration
count is resolution-independent.  That was an artefact: a random field has more
grid-scale content at higher N, so the problem got ROUGHER as it got finer --
a different problem at every N, dominated by the well-conditioned high end of
the spectrum, which masks the growing low end.

The dense spectrum settles it -- cond(M^-1 A) on 2x2 elements, k_z = 0:

    N      4        6        8       10
    cond   4.3e5    5.6e6    4.1e7   2.0e8      (~N^6)
    sqrt   653      2376     6421    14094      (the CG estimate)

So the operator DOES degrade with polynomial order, as it must for SEM.

Here the right-hand side is a genuine RKW3 stage RHS from the laminar channel
driver -- Poiseuille plus a decaying roll perturbation -- so its physical
content is fixed and only the resolution changes.  dt is held fixed too.
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
import channel3d as C
from lssem3d import operator as OP, solver3d as S3, convect as CV, timestep as T
from lssem3d import precond as PC

RE, NZ, EX, DT, TOL, AMP = 100.0, 16, 3, 0.01, 1e-6, 0.05


def stage_rhs(s, U, k, dt, rw):
    m, D, kz, mask, nu, nz = s['m'], s['D'], s['kz'], s['mask'], s['nu'], s['nz']
    c = T.implicit_coeff(dt, k)
    Uc = OP.to_complex(U)
    Nk = -CV.convective(Uc, D, m.facx, m.facy, kz, nz)
    Nk[..., 0, 0] += s['fx']*nz
    R0 = OP.apply_L0_complex(Uc, D, m.facx, m.facy, kz, nu, 0.0, 0.0)
    Lk = -R0[..., 4:7, :]
    fc = np.zeros(Uc.shape[:-2] + (OP.NROW, Uc.shape[-1]), dtype=complex)
    for row, fld in ((4, OP.U_), (5, OP.V_), (6, OP.W_)):
        i = row - 4
        fc[..., row, :] = c*(Uc[..., fld, :] + dt*(T.GAMMA[k]*Nk[..., i, :]
                                                   + T.ALPHA[k]*Lk[..., i, :]))
    f = np.concatenate([fc.real, fc.imag], axis=-2)
    wqR = m.wq[..., None, None]
    r = OP.apply_LT(OP.apply_L(U, D, m.facx, m.facy, kz, nu, c, m.wq, 0.0, rw)
                    - f*wqR*C._fw(rw, f.shape[-2]),
                    D, m.facx, m.facy, kz, nu, c, 0.0)
    return -S3.gs(m, r)*mask, c


if __name__ == '__main__':
    print(f'Laminar channel Re={RE:g}, {EX}x{EX} elements FIXED, Nz={NZ}, '
          f'dt={DT} FIXED, tol={TOL:g}')
    print('RHS is a real RKW3 stage RHS (not a random field).\n')
    print(f"{'N':>4}{'planeDOF':>10}{'jac its':>9}{'jac s':>8}"
          f"{'PMG its':>9}{'PMG s':>8}{'it ratio':>10}{'TIME':>8}")
    out = []
    for N in (6, 8, 10, 12, 14):
        s = C.setup(N=N, ex=EX, ey=EX, nz=NZ, re=RE)
        U = C.initial_state(s, amp=AMP)
        rw = OP.momentum_row_weights(T.implicit_coeff(DT, 0))
        b, c = stage_rhs(s, U, 0, DT, rw)
        m, D, kz, mask, nu = s['m'], s['D'], s['kz'], s['mask'], s['nu']
        shape = (m.nelem, N+1, N+1, OP.NVAR_R, s['nk'])
        kw = dict(mesh=m, mask=mask, wq=m.wq, kap=0.0, rw=rw)
        Mi = S3.jacobi_inverse(S3.jacobi_diagonal_analytic(
            shape, D, m.facx, m.facy, kz, nu, c, **kw), mask)
        t0 = time.perf_counter()
        _, itj, _ = S3.pcg(b, D, m.facx, m.facy, kz, nu, c, m, mask, Mi, TOL,
                           60000, None, m.wq, 0.0, rw)
        tj = time.perf_counter()-t0
        h = (N, N//2, 2) if N//2 > 2 else (N, 2)
        t0 = time.perf_counter()
        P = PC.PMG(m, s['nk'], NZ, nu, c, kz, kap=0.0, rw=rw, orders=h,
                   deg=4, coarse_deg=4)
        tb = time.perf_counter()-t0
        t0 = time.perf_counter()
        _, itp, _ = S3.pcg(b, D, m.facx, m.facy, kz, nu, c, m, mask, P, TOL,
                           60000, None, m.wq, 0.0, rw)
        tp = time.perf_counter()-t0 + tb
        dof = m.nelem*(N+1)**2*OP.NVAR_R
        out.append(dict(N=N, dof=dof, jac_its=int(itj), jac_s=tj,
                        pmg_its=int(itp), pmg_s=tp, gain=tj/tp))
        print(f'{N:>4}{dof:>10,}{itj:>9}{tj:>8.1f}{itp:>9}{tp:>8.1f}'
              f'{itj/max(itp,1):>9.1f}x{tj/tp:>7.2f}x', flush=True)
    json.dump(out, open('scratch/precond_channel.json', 'w'), indent=1)
