"""Week 1 items a1 and a2 on rig 2 (the 3D rig): what the coarse level costs.

    uv run --quiet python scratch/week1_coarse3d.py

a2  pc = 1 INSTEAD OF pc = 2.  On the production channel the dense p = 2 coarse
    inverse is 5.3 GB and its read is half the A100 preconditioner apply.  At
    p = 1 the coarse space is 11x smaller.  The question is the iteration cost,
    and it has to be asked in 3D: the 2D harness cannot build a p = 1 level at
    all (lgl_nodes has no 2-node case).

a1  COARSE FACTOR IN fp32.  Halving the read exactly would need a batched
    symmetric matvec, which torch does not have.  Storage precision is the
    lever that exists: the preconditioner does not enter the answer -- CG needs
    it only fixed and SPD -- so the coarse inverse can be fp32 while the state,
    the operator and the solution stay fp64.  Measured here as iterations and
    as agreement with the fp64 preconditioner; the channel's step-matches-Jacobi
    gate is the acceptance test.

Both are reported against the same baseline so the trade is explicit: bytes
against iterations.
"""
import os
import sys
import time

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '4')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R)
sys.path.insert(0, os.path.join(_R, 'scratch'))
os.chdir(_R)
os.environ.setdefault('LSSEM3D_BACKEND', 'numpy')

import numpy as np

import lssem3d
lssem3d.set_backend('numpy')
from lssem3d import operator as OP, solver3d as S3
from lssem3d.precond import VertexSchwarzBatched3D, _Level
from vs3d_check import setup


def run(N=6, ex=4, ey=4, nz=8, c=5405.4, weighting='legacy', tol=1e-10):
    nu = 1/180.
    m, nk, mask, kz = setup(N, ex, ey, nz)
    shape = (m.nelem, N+1, N+1, OP.NVAR_R, nk)
    rw = OP.momentum_row_weights(c, weighting=weighting)
    lev = _Level(m, nk, nz, nu, c, kz, 0.0, rw, False, mask=mask)
    rng = np.random.default_rng(0)
    x_ex = S3.gs(m, rng.standard_normal(shape))*mask
    bb = lev.A(x_ex)

    def cg(Minv):
        t0 = time.perf_counter()
        _, it, _ = S3.pcg(bb, lev.D, m.facx, m.facy, kz, nu, c, mesh=m, mask=mask,
                          M_inv=Minv, tol=tol, max_iter=20000, wq=m.wq, rw=rw)
        return int(it), time.perf_counter() - t0

    print(f'{ex}x{ey} N={N} nz={nz} nk={nk} c={c:g} {weighting}')
    ij, tj = cg(lev.M_inv)
    print(f'  jacobi                      {ij:6d} it {tj:6.1f}s')
    base = None
    zref = None
    rr = S3.gs(m, rng.standard_normal(shape))*mask
    for pc in (1, 2, 3):
        for fp32 in (False, True):
            try:
                t0 = time.perf_counter()
                M = VertexSchwarzBatched3D(m, nk, nz, nu, c, kz, 0.0, rw, mask=mask,
                                           pc=pc, coarse_dense=True, coarse_fp32=fp32)
                tb = time.perf_counter() - t0
                it, t = cg(M)
                z = M(rr)
                if not fp32:
                    zref = z
                    agree = ''
                else:
                    agree = f' | vs fp64 {np.abs(z-zref).max()/np.abs(zref).max():.1e}'
                cb = getattr(getattr(M, 'coarse_dev', None), 'bytes', 0)/1e6
                if base is None:
                    base = (it, cb)
                print(f'  patch + p={pc} coarse {"fp32" if fp32 else "fp64"}      '
                      f'{it:6d} it {t:6.1f}s | coarse {cb:8.1f} MB '
                      f'| vs p=2 fp64 baseline: {it/base[0]:.2f}x it, {cb/max(base[1],1e-9):.2f}x bytes '
                      f'| build {tb:.1f}s{agree}', flush=True)
            except Exception as e:
                print(f'  patch + p={pc} coarse {"fp32" if fp32 else "fp64"}      '
                      f'FAILED: {type(e).__name__}: {str(e).splitlines()[0][:70]}', flush=True)


if __name__ == '__main__':
    for N in (int(x) for x in os.environ.get('NS', '6').split(',')):
        run(N=N)
        print()
