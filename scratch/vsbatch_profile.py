"""Where does the batched apply spend its time on the channel mesh?"""
import os, sys, time
for _v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ.setdefault(_v, '8')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, _R); os.chdir(_R)
os.environ.setdefault('LSSEM3D_BACKEND', 'numpy'); os.environ.setdefault('LSSEM3D_DEVICE', 'cpu')
import numpy as np, torch
import lssem3d; lssem3d.set_backend('numpy')
from lssem2d.mesh import build_channel
from lssem3d import bc as BC, operator as OP, fourier as FR, solver3d as S3
from lssem3d.precond import VertexSchwarzBatched3D

if __name__ == '__main__':
    N, ex, ey, nz = int(os.environ.get('N', 8)), int(os.environ.get('EX', 6)), int(os.environ.get('EY', 18)), int(os.environ.get('NZ', 32))
    nu, c = 1/180., 5405.4
    m = build_channel(np.pi, 2.0, ex, ey, N, bcs=(0, 0, 1, 1)); m.periodic_x = np.pi; m.compute_global_indices()
    nk = nz//2 + 1
    mask = BC.build_mask(m, nk, pin_p=False, nz=nz); BC.pin_dof(m, mask, OP.P_, 0); BC.pin_dof(m, mask, OP.NVAR + OP.P_, 0)
    kz = FR.wavenumbers(nz, 0.34*np.pi); rw = OP.momentum_row_weights(c)
    shape = (m.nelem, N+1, N+1, OP.NVAR_R, nk)
    bat = VertexSchwarzBatched3D(m, nk, nz, nu, c, kz, 0.0, rw, mask=mask, verbose=True)
    md = bat.modes[1]
    print('mode 1 element types (count):', [int(p['n']) for p in md['eplan']], ' patch types (count, nb):', [(int(p['n']), int(p['nb'])) for p in md['pplan']])
    r = S3.gs(m, np.random.default_rng(0).standard_normal(shape))*mask
    rt = torch.as_tensor(r)
    def tm(f, n=3):
        f(); t0 = time.perf_counter()
        for _ in range(n): f()
        return (time.perf_counter() - t0)/n
    print(f'full apply (numpy in/out):      {tm(lambda: bat(r))*1e3:7.1f} ms')
    print(f'patch part only (torch in/out): {tm(lambda: bat._apply_patches(rt))*1e3:7.1f} ms')
    print(f'coarse restrict:                {tm(lambda: bat._restrict(r))*1e3:7.1f} ms')
    rc = bat._restrict(r)
    print(f'coarse solve (DirectCoarseE):   {tm(lambda: bat.coarse(rc))*1e3:7.1f} ms')
    zc = bat.coarse(rc)
    print(f'coarse prolong:                 {tm(lambda: bat._prolong(zc))*1e3:7.1f} ms')
    # inside the patch part: solves only
    def solves_only():
        for k, md in enumerate(bat.modes):
            if md is None: continue
            for t, pl in enumerate(md['eplan']):
                sc, L, KEI, KIE = md['efac'][t]
                B = torch.zeros((bat.nI, pl['n']), dtype=torch.float64); torch.cholesky_solve(B, L); torch.cholesky_solve(B, L); KEI @ B; KIE @ torch.zeros((bat.nE, pl['n']), dtype=torch.float64)
            for t, pl in enumerate(md['pplan']):
                sc, L = md['pfac'][t]
                torch.cholesky_solve(torch.zeros((pl['nb'], pl['n']), dtype=torch.float64), L)
    print(f'batched solves + matmuls only:  {tm(solves_only)*1e3:7.1f} ms   (torch threads {torch.get_num_threads()})')
