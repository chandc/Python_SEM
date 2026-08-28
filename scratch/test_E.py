"""Gates for the consistent P_N-P_N pressure operator."""
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
nk = NZ//2+1
kz = FR.wavenumbers(NZ, LZ)
mask_u = PJ.build_masks(m, nk, NZ, 3, wall=True)
mask_p = PJ.build_masks(m, nk, NZ, 1, wall=False)
D = diff_matrix(N)
wq3 = m.wq[..., None, None]
Mginv = 1.0/S3.gs(m, wq3 + np.zeros_like(wq3))
mw = S3.multiplicity_weight(m, mask_p.shape)
dot = lambda a,b: float((a*b*mw).sum())
E = lambda p: PJ.apply_E(p, D, m.facx, m.facy, wq3, kz, m, mask_p, mask_u, Mginv)
def c0(shape_mask):
    r = np.random.standard_normal(shape_mask.shape)
    return S3.gs(m, r)*shape_mask     # C0-consistent, masked
# 1. symmetry
p1, p2 = c0(mask_p), c0(mask_p)
a, b = dot(p1, E(p2)), dot(E(p1), p2)
print(f'symmetry: <p1,Ep2>={a:.6e}  <Ep1,p2>={b:.6e}  rel {abs(a-b)/abs(a):.2e}')
# 2. PSD
worst = min(dot(q, E(q))/max(dot(q,q),1e-30) for q in (c0(mask_p) for _ in range(6)))
print(f'PSD: min Rayleigh = {worst:.3e}')
# 3. projection zeroes the weak divergence
s = dict(m=m, Dg=D, fxg=m.facx, fyg=m.facy, kzg=kz, wq3=wq3, mask_p=mask_p,
         mask_u=mask_u, Mginv=Mginv, mw1=mw[...,0:1,0:1], tol=1e-10,
         null_kz0=np.ones(mask_p[...,0:1,0:1].shape)*mask_p[...,0:1,0:1],
         check_every=1)
s['null_norm'] = float((s['null_kz0']**2*s['mw1']).sum())
import time
variants = {}
variants['identity'] = lambda r: r
dj = HH.jacobi_diagonal_analytic(m, N, m.wq, kz**2, 1.0, 2, nk, mask=None)
ji = HH.jacobi_inverse(dj, mask_p)
variants['jacobi-K'] = lambda r: r*ji
fdm_s = HH.fdm_preconditioner(m, N, kz**2 + 1e-3, 1.0, mask_p, 2, nk, like=mask_p)
variants['fdm-shift'] = fdm_s
from lssem3d import hpmg
t0 = time.time()
variants['pmg-K'] = hpmg.HelmholtzPMG(m, N, kz**2, 1.0, 1, nk, NZ,
                                      wall=False, pin_kz0=False, deg=6,
                                      like=mask_p)
print(f'(pmg setup {time.time()-t0:.1f}s)')
uh = PJ._join(c0(mask_u))
def weakdiv(uc):
    from lssem3d import deriv as DV
    z = (DV.ddxT(wq3*uc[...,0:1,:], D, m.facx)
         + DV.ddyT(wq3*uc[...,1:2,:], D, m.facy) - 1j*kz*(wq3*uc[...,2:3,:]))
    return S3.gs(m, PJ._split(z))*mask_p
b0 = np.sqrt((weakdiv(uh)**2).sum())
for nm, M in variants.items():
    s['Mp'] = M
    u2, phi, it, res = PJ.project_consistent(s, uh, 0.01)
    b1 = np.sqrt((weakdiv(u2)**2).sum())
    print(f'{nm:>10}: CG {it:>4}  weak div -> {b1/b0:.1e} rel  res {res:.1e}')
