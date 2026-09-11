"""Coarse order pc=1 vs pc=2 for the batched patch: iterations on a 4x4 N=8 channel mesh."""
import os, sys, time
for _v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ.setdefault(_v, '8')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, _R); os.chdir(_R)
os.environ.setdefault('LSSEM3D_BACKEND', 'numpy'); os.environ.setdefault('LSSEM3D_DEVICE', 'cpu')
import numpy as np
import lssem3d; lssem3d.set_backend(os.environ['LSSEM3D_BACKEND'])
from lssem2d.mesh import build_channel
from lssem3d import bc as BC, operator as OP, fourier as FR, solver3d as S3
from lssem3d.precond import VertexSchwarzBatched3D, _Level
if __name__ == '__main__':
    N, ex, ey, nz = int(os.environ.get('N', 8)), int(os.environ.get('EX', 4)), int(os.environ.get('EY', 4)), int(os.environ.get('NZ', 8))
    nu, c = 1/180., 5405.4
    m = build_channel(np.pi, 2.0, ex, ey, N, bcs=(0, 0, 1, 1)); m.periodic_x = np.pi; m.compute_global_indices()
    nk = nz//2 + 1
    mask = BC.build_mask(m, nk, pin_p=False, nz=nz); BC.pin_dof(m, mask, OP.P_, 0); BC.pin_dof(m, mask, OP.NVAR + OP.P_, 0)
    kz = FR.wavenumbers(nz, 0.34*np.pi); rw = OP.momentum_row_weights(c)
    shape = (m.nelem, N+1, N+1, OP.NVAR_R, nk)
    lev = _Level(m, nk, nz, nu, c, kz, 0.0, rw, False, mask=mask)
    b = lev.A(S3.gs(m, np.random.default_rng(1).standard_normal(shape))*mask)
    for pc, cd in ((2, False), (2, True), (1, False), (1, True), (None, False)):
        M = VertexSchwarzBatched3D(m, nk, nz, nu, c, kz, 0.0, rw, mask=mask, coarse=('element' if pc else None), pc=pc or 2, coarse_dense=cd)
        t0 = time.perf_counter(); x, it, _ = S3.pcg(b, lev.D, m.facx, m.facy, kz, nu, c, mesh=m, mask=mask, M_inv=M, tol=1e-10, max_iter=3000, wq=m.wq, rw=rw); tp = time.perf_counter() - t0
        print(f'{ex}x{ey} N={N}: coarse pc={pc} dense={cd}: {it} iterations, {tp:.1f}s, factors {M.bytes/1e6:.0f} MB', flush=True)
