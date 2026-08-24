import sys, time
sys.path.insert(0,'.'); sys.path.insert(0,'scratch')
import numpy as np, lssem3d; lssem3d.set_backend('cupy')
import cupy as cp
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem3d import helmholtz as HH, hpmg, solver3d as S3, fourier as FR, project as PJ
N,NZ=8,32; nk=NZ//2+1
m=build_channel(np.pi,2.0,6,18,N,bcs=(0,0,1,1)); m.periodic_x=np.pi
m.compute_global_indices()
mask=PJ.build_masks(m,nk,NZ,1,wall=False)
ind=np.zeros(mask.shape); ind[0,0,0,0,0]=1.0
mask[...,0,0]*=(S3.gs(m,ind)[...,0,0]<0.5)
kz=FR.wavenumbers(NZ,0.34*np.pi); D=diff_matrix(N)
g=lambda a: cp.asarray(np.ascontiguousarray(a))
maskd=g(mask)
P=hpmg.HelmholtzPMG(m,N,kz**2,1.0,1,nk,NZ,wall=False,pin_kz0=True,deg=6)
r=g(np.random.default_rng(0).standard_normal(mask.shape))*maskd
A=lambda v: HH.apply(v,g(D),g(m.facx),g(m.facy),g(m.wq),g(kz**2),1.0,m,maskd)
for f,nm in ((A,'matvec (GPU)'),(P,'PMG V-cycle')):
    f(r); cp.cuda.Stream.null.synchronize()
    t0=time.perf_counter()
    for _ in range(5): f(r)
    cp.cuda.Stream.null.synchronize()
    print(f'  {nm:<16} {(time.perf_counter()-t0)/5*1e3:8.2f} ms')
print(f'\n  pressure field: {r.nbytes/2**20:.1f} MiB per host round trip')
