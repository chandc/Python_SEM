"""VertexSchwarz3D gate on a small periodic channel: condensed == dense,
symmetry, CG iterations vs Jacobi at c = 1 and 5405, legacy vs balanced."""
import os, sys, time
for _v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ.setdefault(_v, '4')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, _R); os.chdir(_R)
os.environ.setdefault('LSSEM3D_BACKEND', 'numpy')
import numpy as np
import lssem3d; lssem3d.set_backend('numpy')
from lssem2d.mesh import build_channel
from lssem3d import bc as BC, operator as OP, fourier as FR, solver3d as S3
from lssem3d.precond import VertexSchwarz3D, _Level


def setup(N, ex, ey, nz, lz=0.34*np.pi):
    m = build_channel(np.pi, 2.0, ex, ey, N, bcs=(0, 0, 1, 1)); m.periodic_x = np.pi; m.compute_global_indices()
    nk = nz//2 + 1
    mask = BC.build_mask(m, nk, pin_p=False, nz=nz)
    BC.pin_dof(m, mask, OP.P_, 0); BC.pin_dof(m, mask, OP.NVAR + OP.P_, 0)
    return m, nk, mask, FR.wavenumbers(nz, lz)


if __name__ == '__main__':
    N, ex, ey, nz = int(os.environ.get('N', 6)), int(os.environ.get('EX', 2)), int(os.environ.get('EY', 2)), int(os.environ.get('NZ', 8))
    nu = 1/180.
    m, nk, mask, kz = setup(N, ex, ey, nz)
    shape = (m.nelem, N+1, N+1, OP.NVAR_R, nk)
    rng = np.random.default_rng(0)
    print(f'{ex}x{ey} N={N} nz={nz} nk={nk} kz={np.round(kz,2)}')
    for weighting in os.environ.get('W', 'legacy,balanced').split(','):
        for c in [float(x) for x in os.environ.get('C', '1,5405.4').split(',')]:
            rw = OP.momentum_row_weights(c, weighting=weighting)
            lev = _Level(m, nk, nz, nu, c, kz, 0.0, rw, False, mask=mask)
            t0 = time.perf_counter(); Mc = VertexSchwarz3D(m, nk, nz, nu, c, kz, 0.0, rw, mask=mask, coarse='element', condense=True); tc = time.perf_counter()-t0
            t0 = time.perf_counter(); Md = VertexSchwarz3D(m, nk, nz, nu, c, kz, 0.0, rw, mask=mask, coarse='element', condense=False); td = time.perf_counter()-t0
            r = S3.gs(m, rng.standard_normal(shape))*mask
            zc, zd = Mc(r), Md(r)
            agree = np.abs(zc - zd).max()/np.abs(zd).max()
            # symmetry in the multiplicity-weighted inner product
            a, b = (S3.gs(m, rng.standard_normal(shape))*mask for _ in range(2))
            s1 = np.sum(b*Mc(a)*Mc.mw); s2 = np.sum(a*Mc(b)*Mc.mw); sym = abs(s1-s2)/abs(s1)
            # CG: b = A x_exact
            x_ex = S3.gs(m, rng.standard_normal(shape))*mask
            bb = lev.A(x_ex)
            res = {}
            for name, Minv in (('jacobi', lev.M_inv), ('vschwarz', Mc), ('vschwarz-1lev', VertexSchwarz3D(m, nk, nz, nu, c, kz, 0.0, rw, mask=mask, coarse=None))):
                t0 = time.perf_counter()
                x, it, rn = S3.pcg(bb, lev.D, m.facx, m.facy, kz, nu, c, mesh=m, mask=mask, M_inv=Minv, tol=1e-10, max_iter=20000, wq=m.wq, rw=rw)
                res[name] = (it, time.perf_counter()-t0)
            print(f'  {weighting:8s} c={c:7.1f}: cond==dense {agree:.1e}  sym {sym:.1e}  stored {Mc.bytes/1e6:.1f} MB (dense {Md.bytes/1e6:.1f})  build {tc:.1f}s | ' +
                  '  '.join(f'{k} {v[0]} it {v[1]:.1f}s' for k, v in res.items()), flush=True)
