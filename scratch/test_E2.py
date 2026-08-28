import os, sys
for v in ('OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ[v]='8'
sys.path.insert(0,'.'); sys.path.insert(0,'scratch')
import numpy as np
import lssem3d
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem3d import project as PJ, solver3d as S3, fourier as FR, helmholtz as HH
np.random.seed(0)
N, EX, EY, NZ = 6, 2, 5, 8
LX, LZ = np.pi, 0.34*np.pi
m = build_channel(LX, 2.0, EX, EY, N, bcs=(0,0,1,1))
m.periodic_x = LX; m.compute_global_indices()
nk = NZ//2+1; kz = FR.wavenumbers(NZ, LZ)
mask_u = PJ.build_masks(m, nk, NZ, 3, wall=True)
mask_p = PJ.build_masks(m, nk, NZ, 1, wall=False)
ind = np.zeros(mask_p.shape); ind[0,0,0,0,0]=1.0
mask_p[...,0,0] *= (S3.gs(m, ind)[...,0,0] < 0.5)
D = diff_matrix(N); wq3 = m.wq[...,None,None]
Mginv = 1.0/S3.gs(m, wq3 + np.zeros_like(wq3))
mw = S3.multiplicity_weight(m, mask_p.shape)
E = lambda p: PJ.apply_E(p, D, m.facx, m.facy, wq3, kz, m, mask_p, mask_u, Mginv)
def c0(msk):
    return S3.gs(m, np.random.standard_normal(msk.shape))*msk
# spectrum probe: smallest Rayleigh quotients via inverse-iteration-ish sampling
# cheaper: assemble E on the small kz=0 block explicitly and eig it
# count masked pressure dofs at k=0 (real part, field 0)
sel = np.flatnonzero(mask_p[...,0,0].ravel())
# unique global columns
cols, seen = [], set()
sh = mask_p.shape
for flat in sel:
    oh = np.zeros(sh); oh.ravel()[flat] = 1.0
    g = S3.gs(m, oh)
    key = tuple(np.flatnonzero(np.abs(g[...,0,0].ravel())>.5))
    if key in seen: continue
    seen.add(key); cols.append((g*mask_p)[...,0,0].ravel())
B = np.stack(cols, axis=1)
print('kz=0 masked global dofs:', B.shape[1])
A = np.empty((B.shape[1],)*2)
for j in range(B.shape[1]):
    q = np.zeros(sh); q[...,0,0] = B[:,j].reshape(sh[:3])
    Aq = E(q)[...,0,0].ravel()
    A[:,j] = B.T @ (Aq*mw[...,0,0].ravel())
A = 0.5*(A+A.T)
w = np.linalg.eigvalsh(A)
print('smallest 8 eigenvalues of E (kz=0 block):', np.array2string(w[:8], precision=2))
print('largest:', f'{w[-1]:.1f}', ' null dim (<1e-8*max):', int((w < 1e-8*w[-1]).sum()))
