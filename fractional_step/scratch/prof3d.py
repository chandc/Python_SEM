"""Where does the 3D solver actually spend its time, and does threading help?

    uv run --quiet python scratch/prof3d.py [nmode] [threads]

3D_DEVELOPMENT_PLAN.md sec 4 says profile before optimising and expects
PCG matvec >> FFT > convection.  This measures it rather than assuming, and
checks whether BLAS threading does anything for the batched contractions --
every driver so far has pinned OMP_NUM_THREADS=1, so that has never been tested.

The two embarrassingly-parallel axes are DIFFERENT and this distinguishes them:

    convection   parallel over z-PLANES  (physical space, all modes coupled)
    the solve    parallel over k_z MODES (spectral space, modes independent)

with FFTs between.  Whichever dominates decides which axis is worth exploiting.
"""
import os, sys, time
_TH = sys.argv[2] if len(sys.argv) > 2 else '1'
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = _TH
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem3d import operator as OP, solver3d as S3, bc as BC, convect as CV
from lssem3d import fourier as FR

RE, EX, N = 1000.0, 6, 10
NU = 1.0/RE


def bench(nz, reps=6):
    n = N+1
    mesh = build_channel(1.0, 1.0, EX, EX, N, bcs=(1, 1, 1, 2))
    D = diff_matrix(N)
    nk = nz//2 + 1
    kz = FR.wavenumbers(nz, 2*np.pi)
    c, kap = 2500.0, 2500.0
    shape = (mesh.nelem, n, n, OP.NVAR_R, nk)
    mask = BC.build_mask(mesh, nk, pin_p=True)
    rng = np.random.default_rng(0)
    U = rng.standard_normal(shape)*mask
    Uc = OP.to_complex(U)

    def t(fn, r=reps):
        fn()                                   # warm
        t0 = time.perf_counter()
        for _ in range(r):
            fn()
        return (time.perf_counter()-t0)/r

    t_mat = t(lambda: S3.normal_op(U, D, mesh.facx, mesh.facy, kz, NU, c,
                                   mesh, mask, mesh.wq, kap))
    t_con = t(lambda: CV.convective(Uc, D, mesh.facx, mesh.facy, kz, nz))
    # the pipeline's actual round trip: modes -> physical -> modes
    _sl = np.ascontiguousarray(Uc[..., OP.U_, :])
    t_fft = t(lambda: FR.to_modes(FR.to_physical(_sl, nz)), r=reps*4)
    t_gs = t(lambda: S3.gs(mesh, U))
    return dict(nz=nz, nk=nk, matvec=t_mat, conv=t_con, fft=t_fft, gs=t_gs)


if __name__ == '__main__':
    nz = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    r = bench(nz)
    # a solve is ~45 CG iterations => 45 matvecs, three stages per step
    per_step = 3*(45*r['matvec'] + r['conv'])
    print(f"threads={_TH}  Nz={r['nz']}  modes={r['nk']}  "
          f"({EX}x{EX} elements, N={N})")
    print(f"  normal_op (1 matvec) : {r['matvec']*1e3:8.2f} ms")
    print(f"  convective (dealias) : {r['conv']*1e3:8.2f} ms")
    print(f"  gather-scatter alone : {r['gs']*1e3:8.2f} ms")
    print(f"  one rfft+irfft pair  : {r['fft']*1e3:8.2f} ms")
    print(f"  -> per RKW3 step (3 stages x [45 matvec + 1 convective]):")
    print(f"       matvecs   {3*45*r['matvec']:7.3f} s  "
          f"({100*45*r['matvec']/(45*r['matvec']+r['conv']):.1f}%)")
    print(f"       convection{3*r['conv']:7.3f} s  "
          f"({100*r['conv']/(45*r['matvec']+r['conv']):.1f}%)")
    print(f"       total     {per_step:7.3f} s")
