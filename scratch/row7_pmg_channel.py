"""Is point Jacobi really what the FOSLS channel should be running?

run01 -- the DNS demonstration -- is preconditioned by jacobi_diagonal_analytic +
jacobi_inverse.  Point Jacobi.  Neither minchan.py nor channel3d.py mentions pmg
or multigrid anywhere, yet lssem3d/precond.py HAS a PMG class that takes the
FOSLS operator directly (nk, nz, kz, rw, mask) and pcg already accepts a callable
M_inv.  6000 CG iterations in the worst stage of ONE step is what point Jacobi on
this operator costs; the question is what the V-cycle costs instead.

sec 7J found the PMG V-cycle "stalled at reduction factor exactly 1.0000" on the
w7 = 1 near-null cluster -- but that was AT w7 = 1.  At 1e-4 the cluster is gone,
so the stall may be gone with it, and PMG has never been re-tried since.

direct_coarse=False: the channel's coarse level is 13x37 nodes x 14 fields x 17
modes ~ 114k dof, far too large to assemble by probing.  Chebyshev coarsest.
"""
import os, sys, time
for _v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '8')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R,'scratch')); os.chdir(_R)
os.environ.setdefault('LSSEM3D_BACKEND', 'numba')
import numpy as np

SEED = os.path.join(_R,'scratch','fs_seed','seed_ckpt.npz')

def main():
    import lssem3d; lssem3d.set_backend(os.environ['LSSEM3D_BACKEND'])
    from lssem3d import operator as OP, precond as P3, timestep as T
    import channel3d as C, minchan as MC

    # channel3d.stage picks PAR.pcg (thread-parallel over Fourier modes) on
    # numpy/numba, and PAR slices the preconditioner as M_inv[..., s] -- fine for
    # a diagonal, impossible for a callable V-cycle.  On a device backend it
    # already goes straight to the serial S3.pcg, so PMG needs no change THERE.
    # Force serial here so Jacobi and PMG are timed on the same path.
    from lssem3d import parallel as PAR, solver3d as S3
    PAR.pcg = lambda *a, **kw: S3.pcg(*a, **{k: v for k, v in kw.items()
                                             if k != 'workers'})
    s = MC.setup(); d = np.load(SEED); U0 = d['U']; dt = 8.0e-4
    print(f'channel N={s["N"]} {s["m"].nelem} elems nk={s["nk"]} nz={s["nz"]}, dt={dt:g}',
          flush=True)

    def run(tag, mk):
        t0 = time.perf_counter()
        try:
            Minv = mk()
        except Exception as e:
            print(f'{tag:22s} BUILD FAILED: {type(e).__name__}: {str(e)[:90]}', flush=True)
            return
        tb = time.perf_counter()-t0
        Nprev = np.zeros(OP.to_complex(U0).shape[:-2] + (3, s['nk']), dtype=complex)
        t0 = time.perf_counter()
        try:
            U1, _, it = MC.advance(s, U0.copy(), Nprev, dt, Minv, tol=1e-8,
                                   max_iter=20000)
        except Exception as e:
            print(f'{tag:22s} SOLVE FAILED: {type(e).__name__}: {str(e)[:90]}', flush=True)
            return
        tw = time.perf_counter()-t0
        cap = '  CAPPED' if it >= 20000 else ''
        print(f'{tag:22s} build {tb:7.1f}s   worst-stage CG {it:6d}   solve {tw:7.1f}s{cap}',
              flush=True)
        return U1, it, tw

    run('point Jacobi', lambda: C.make_precond(s, dt, 0.0, rowweight=True))

    for orders in ((8,4,2), (8,4)):
        def mk(orders=orders):
            out = []
            for k in range(T.NSTAGE):
                cc = T.implicit_coeff(dt, k)
                out.append(P3.PMG(s['m'], s['nk'], s['nz'], s['nu'], cc, s['kz'],
                                  kap=0.0, rw=OP.momentum_row_weights(cc),
                                  orders=orders, deg=6, pin_p=True,
                                  direct_coarse=False, mask=s['mask']))
            return out
        run(f'PMG {orders}', mk)

if __name__ == '__main__':
    main()
