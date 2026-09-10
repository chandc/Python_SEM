"""Batched/shared vertex patch vs the reference VertexSchwarz3D: equality, types, timing."""
import os, sys, time
for _v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ.setdefault(_v, '8')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, _R); os.chdir(_R)
os.environ.setdefault('LSSEM3D_BACKEND', 'numpy')
import numpy as np
import lssem3d; lssem3d.set_backend(os.environ['LSSEM3D_BACKEND'])
from lssem2d.mesh import build_channel
from lssem3d import bc as BC, operator as OP, fourier as FR, solver3d as S3
from lssem3d.precond import VertexSchwarz3D, VertexSchwarzBatched3D, _Level

if __name__ == '__main__':
    N, ex, ey, nz = int(os.environ.get('N', 4)), int(os.environ.get('EX', 2)), int(os.environ.get('EY', 2)), int(os.environ.get('NZ', 8))
    nu, c = 1/180., float(os.environ.get('C', 5405.4))
    m = build_channel(np.pi, 2.0, ex, ey, N, bcs=(0, 0, 1, 1)); m.periodic_x = np.pi; m.compute_global_indices()
    nk = nz//2 + 1
    mask = BC.build_mask(m, nk, pin_p=False, nz=nz); BC.pin_dof(m, mask, OP.P_, 0); BC.pin_dof(m, mask, OP.NVAR + OP.P_, 0)
    kz = FR.wavenumbers(nz, 0.34*np.pi)
    shape = (m.nelem, N+1, N+1, OP.NVAR_R, nk)
    rw = OP.momentum_row_weights(c)
    ref = None
    if os.environ.get('REF', '1') == '1':
        t0 = time.perf_counter(); ref = VertexSchwarz3D(m, nk, nz, nu, c, kz, 0.0, rw, mask=mask); tr = time.perf_counter() - t0
    t0 = time.perf_counter(); bat = VertexSchwarzBatched3D(m, nk, nz, nu, c, kz, 0.0, rw, mask=mask, verbose=True); tb = time.perf_counter() - t0
    rng = np.random.default_rng(0)
    r = S3.gs(m, rng.standard_normal(shape))*mask
    t0 = time.perf_counter(); zb = bat(r); ta_b = time.perf_counter() - t0
    t0 = time.perf_counter(); zb = bat(r); ta_b = time.perf_counter() - t0
    if ref is not None:
        t0 = time.perf_counter(); zr = ref(r); ta_r = time.perf_counter() - t0
        print(f'{ex}x{ey} N={N} nk={nk}: |batched - reference| = {np.abs(zb-zr).max()/np.abs(zr).max():.2e}  build ref {tr:.1f}s ({ref.bytes/1e6:.0f} MB) batched {tb:.1f}s ({bat.bytes/1e6:.1f} MB)  apply ref {ta_r*1e3:.0f} ms batched {ta_b*1e3:.0f} ms')
    else:
        print(f'{ex}x{ey} N={N} nk={nk}: build batched {tb:.1f}s ({bat.bytes/1e6:.1f} MB)  apply {ta_b*1e3:.0f} ms')
    lev = _Level(m, nk, nz, nu, c, kz, 0.0, rw, False, mask=mask)
    b = lev.A(S3.gs(m, rng.standard_normal(shape))*mask)
    t0 = time.perf_counter(); x, it, _ = S3.pcg(b, lev.D, m.facx, m.facy, kz, nu, c, mesh=m, mask=mask, M_inv=bat, tol=1e-10, max_iter=2000, wq=m.wq, rw=rw); tp = time.perf_counter() - t0
    print(f'  pcg with batched patch: {it} iterations, {tp:.1f}s')
