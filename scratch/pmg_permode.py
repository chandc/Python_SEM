"""Which Fourier mode is setting the channel's iteration count?

pcg is BATCHED over modes and runs until the WORST has converged, so a single
bad mode sets the count for all 17.  The h-sweep at k_z=5.88 needs 11-41
iterations; the channel needs 402 and the k_z=0 cavity 204.  If k_z=0 is the
outlier, the channel is not a p-multigrid problem at all -- it is one bad mode,
and the row-7 work already showed k_z=0's softest mode is 100% PRESSURE
(cond 9.0e4) while k_z!=0 is vorticity-dominated (4.1e3).
"""
import os, sys
for _v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '8')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R,'scratch')); os.chdir(_R)
os.environ.setdefault('LSSEM3D_BACKEND', 'numba')
import numpy as np
TOL, CAP = 1e-8, 30000

def main():
    import lssem3d; lssem3d.set_backend(os.environ['LSSEM3D_BACKEND'])
    from lssem3d import operator as OP, precond as P3, solver3d as S3, timestep as T
    import channel3d as C, minchan as MC
    s = MC.setup(); dt = 8e-4
    cc = T.implicit_coeff(dt, 0); rw = OP.momentum_row_weights(cc)
    m, D = s['m'], s['D']
    print(f'channel 6x18 N=8 c={cc:.0f}: CG per SINGLE mode (nk=1 each)\n', flush=True)
    print(f'{"k":>3} {"k_z":>8} | {"jacobi":>7} {"pmg":>6} {"ratio":>6}')
    tot_j = tot_p = 0
    from lssem3d import bc as BC
    # sliced from the full-nk mask -- see the note in mode_group_solve.py; passing
    # a subset to build_mask masks the WRONG modes' imaginary halves.
    mask_full = BC.build_mask(m, s['nk'], pin_p=False, nz=s['nz'])
    BC.pin_dof(m, mask_full, OP.P_, 0)
    BC.pin_dof(m, mask_full, OP.NVAR+OP.P_, 0)
    for k in range(s['nk']):
        kz = np.array([float(s['kz'][k])])
        mask = np.ascontiguousarray(mask_full[..., k:k+1])
        shape = (m.nelem, s['N']+1, s['N']+1, OP.NVAR_R, 1)
        kwn = dict(mesh=m, mask=mask, wq=m.wq, kap=0.0, rw=rw)
        A = lambda x: S3.normal_op(x, D, m.facx, m.facy, kz, s['nu'], cc, **kwn)
        rng = np.random.default_rng(0)
        b = A(rng.standard_normal(shape)*mask)
        nb = np.linalg.norm(b)
        if nb < 1e-300:
            print(f'{k:3d} {kz[0]:8.2f} |  (empty mode)'); continue
        b /= nb
        Mi = S3.jacobi_inverse(S3.jacobi_diagonal_analytic(shape, D, m.facx,
                               m.facy, kz, s['nu'], cc, **kwn), mask)
        _, itj, _ = S3.pcg(b, D, m.facx, m.facy, kz, s['nu'], cc, m, mask, Mi,
                           TOL, CAP, None, m.wq, 0.0, rw)
        M = P3.PMG(m, 1, s['nz'], s['nu'], cc, kz, kap=0.0, rw=rw,
                   orders=(8,4,2), deg=6, pin_p=True, direct_coarse='element',
                   mask=mask)
        _, itp, _ = S3.pcg(b, D, m.facx, m.facy, kz, s['nu'], cc, m, mask, M,
                           TOL, CAP, None, m.wq, 0.0, rw)
        itj, itp = int(np.max(itj)), int(np.max(itp))
        tot_j, tot_p = max(tot_j, itj), max(tot_p, itp)
        flag = '   <-- BOTTLENECK' if itp > 100 else ''
        print(f'{k:3d} {kz[0]:8.2f} | {itj:7d} {itp:6d} {itj/max(itp,1):5.1f}x{flag}',
              flush=True)
    print(f'\nworst mode sets the batched count: jacobi {tot_j}, pmg {tot_p}')

if __name__ == '__main__':
    main()
