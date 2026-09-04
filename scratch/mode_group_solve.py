"""Measured: does splitting the batched solve into mode GROUPS actually pay?

scratch/pmg_permode.py showed k_z=0 needs 3442 Jacobi iterations while k_z=94
needs 50 -- but pcg applies A to all 17 modes every iteration and breaks only
when the WORST has converged, so the channel pays 3442*17 = 58514 mode-applies
where 5750 would do.  That is a 10.2x arithmetic waste and it needs no
preconditioner change to fix.

Arithmetic is not wall clock: a 1-mode batch vectorises worse than a 17-mode
batch, so the realised gain will be smaller.  This measures it.
"""
import os, sys, time
for _v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '8')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R,'scratch')); os.chdir(_R)
os.environ.setdefault('LSSEM3D_BACKEND', 'numba')
import numpy as np
TOL, CAP = 1e-8, 30000

def main():
    import lssem3d; lssem3d.set_backend(os.environ['LSSEM3D_BACKEND'])
    from lssem3d import operator as OP, solver3d as S3, timestep as T, bc as BC
    import channel3d as C, minchan as MC
    s = MC.setup(); dt = 8e-4
    cc = T.implicit_coeff(dt, 0); rw = OP.momentum_row_weights(cc)
    m, D, N, nk = s['m'], s['D'], s['N'], s['nk']
    full_shape = (m.nelem, N+1, N+1, OP.NVAR_R, nk)
    rng = np.random.default_rng(0)

    # THE MASK MUST BE BUILT ONCE FOR ALL nk MODES AND THEN SLICED.
    # build_mask(m, nmode, nz=nz) zeroes the whole imaginary half of COLUMN 0 of
    # the array it is handed (and of the Nyquist column), because at k=0 those
    # components are unphysical.  Handing it a SUBSET makes it do that to the
    # wrong modes -- for [9..16] it would zero mode 9's imaginary half, and every
    # single-mode solve at k>=1 would be a halved, wrong problem.  Measured: that
    # error inflated group counts to 1960-2402 where the true worst member needs
    # 84-600, which is what exposed it.
    mask_full = BC.build_mask(m, nk, pin_p=False, nz=s['nz'])
    BC.pin_dof(m, mask_full, OP.P_, 0)
    BC.pin_dof(m, mask_full, OP.NVAR+OP.P_, 0)

    def solve(kidx, label):
        """Solve the sub-batch of modes `kidx` and return (its, wall)."""
        kz = np.ascontiguousarray(s['kz'][kidx])
        nkl = len(kidx)
        mask = np.ascontiguousarray(mask_full[..., kidx])
        sh = (m.nelem, N+1, N+1, OP.NVAR_R, nkl)
        kwn = dict(mesh=m, mask=mask, wq=m.wq, kap=0.0, rw=rw)
        A = lambda x: S3.normal_op(x, D, m.facx, m.facy, kz, s['nu'], cc, **kwn)
        b = A(rng.standard_normal(sh)*mask)
        nb = np.linalg.norm(b)
        if nb < 1e-300: return 0, 0.0, nkl
        b /= nb
        Mi = S3.jacobi_inverse(S3.jacobi_diagonal_analytic(sh, D, m.facx, m.facy,
                               kz, s['nu'], cc, **kwn), mask)
        t0 = time.perf_counter()
        _, it, _ = S3.pcg(b, D, m.facx, m.facy, kz, s['nu'], cc, m, mask, Mi,
                          TOL, CAP, None, m.wq, 0.0, rw)
        return int(np.max(it)), time.perf_counter()-t0, nkl

    print(f'channel 6x18 N=8, c={cc:.0f}, Jacobi, tol={TOL:g}\n', flush=True)
    it, tw, _ = solve(np.arange(nk), 'all')
    print(f'{"ONE BATCH (production)":34s} its {it:5d}  wall {tw:7.1f}s', flush=True)
    base = tw

    for name, groups in (
        ('2 groups: [0] [1..16]', [[0], list(range(1, nk))]),
        ('4 groups: [0][1,2][3-8][9-16]',
         [[0], [1, 2], list(range(3, 9)), list(range(9, nk))]),
        ('per mode (17 solves)', [[k] for k in range(nk)]),
    ):
        tot_t, parts = 0.0, []
        for g in groups:
            it, tw, nkl = solve(np.array(g), str(g))
            tot_t += tw; parts.append(f'{it}')
        print(f'{name:34s} its {"/".join(parts):22s} wall {tot_t:7.1f}s   '
              f'{base/max(tot_t,1e-9):5.2f}x', flush=True)

if __name__ == '__main__':
    main()
