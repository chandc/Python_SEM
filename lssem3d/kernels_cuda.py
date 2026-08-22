"""Fused CUDA kernels for the 3D VVP operator — one pass, one launch.

WHY THIS EXISTS, and it is a measured reason rather than an aspiration.
`kernels_torch.py` is correct and 4.2× a serial CPU path, but on the Spark it
LOSES to 20-core mode-parallel numba (129.5 s vs 75.7 s on a minimal-channel
step).  Profiling says exactly where:

    convection (FFT + dealias)   18.2x faster on the GPU
    apply_L0_complex             23.2x
    to_complex / to_real          9.3x
    THE MATVEC                    3.17x     <-- the outlier

Everything the GPU touches as one large operation is 9-23x.  The matvec is ~30
separate kernel launches over a 2.08M-dof state, so it is LAUNCH-BOUND, not
bandwidth-bound, and that is what this fixes: one thread per (element, node,
mode), all 28 derivative sums and all 16 rows accumulated in registers, one
launch instead of thirty.

§7M got **7.6x** from exactly this fusion on the CPU.  The arithmetic below is a
transcription of `kernels_numba._kernel_L` / `_kernel_LT`, which is why that
module was written in explicit scalar form in the first place.

THREAD MAPPING IS THE ONE DESIGN CHOICE.  The mode index `k` is the fastest-
varying axis of the (nel, n, n, 14, nk) layout, so it maps to the fastest-varying
thread index: adjacent threads then read adjacent addresses and the loads
coalesce.  Mapping `e` to threadIdx would stride by n*n*14*nk and waste most of
every transaction.

Compiled on first use with `torch.utils.cpp_extension.load_inline` and cached by
torch under TORCH_EXTENSIONS_DIR.
"""
import os

import numpy as np

try:
    import torch
except Exception:
    torch = None

_MOD = None

_DECL = r"""
torch::Tensor apply_L(torch::Tensor U, torch::Tensor D, torch::Tensor facx,
                      torch::Tensor facy, torch::Tensor kz, double nu, double c,
                      torch::Tensor wq, double kap, torch::Tensor rw);
torch::Tensor apply_LT(torch::Tensor R, torch::Tensor D, torch::Tensor facx,
                       torch::Tensor facy, torch::Tensor kz, double nu,
                       double c, double kap);
"""

_SRC = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define NV 7
#define NVR 14
#define NRR 16

// U index for field f at (e,i,j,k):  ((((e*n+i)*n+j)*NVR)+f)*nk + k
__global__ void fused_L(const double* __restrict__ U, const double* __restrict__ D,
                        const double* __restrict__ facx, const double* __restrict__ facy,
                        const double* __restrict__ kz, double nu, double c,
                        const double* __restrict__ wq, double kap,
                        const double* __restrict__ rw, double* __restrict__ R,
                        int nel, int n, int nk)
{
    long t = (long)blockIdx.x*blockDim.x + threadIdx.x;
    long total = (long)nel*n*n*nk;
    if (t >= total) return;
    int k = (int)(t % nk);
    int j = (int)((t / nk) % n);
    int i = (int)((t / ((long)nk*n)) % n);
    int e = (int)(t / ((long)nk*n*n));

    double dxr[NV], dxi[NV], dyr[NV], dyi[NV];
#pragma unroll
    for (int f = 0; f < NV; ++f) { dxr[f]=0.0; dxi[f]=0.0; dyr[f]=0.0; dyi[f]=0.0; }

    const double fx = facx[e], fy = facy[e];
    for (int a = 0; a < n; ++a) {
        const double dx = D[i*n + a]*fx;
        const double dy = D[j*n + a]*fy;
        const long ax = ((((long)e*n + a)*n + j)*NVR)*nk + k;   // U[e,a,j,*,k]
        const long ay = ((((long)e*n + i)*n + a)*NVR)*nk + k;   // U[e,i,a,*,k]
#pragma unroll
        for (int f = 0; f < NV; ++f) {
            dxr[f] += dx*U[ax + (long)f*nk];
            dxi[f] += dx*U[ax + (long)(NV+f)*nk];
            dyr[f] += dy*U[ay + (long)f*nk];
            dyi[f] += dy*U[ay + (long)(NV+f)*nk];
        }
    }
    const long b = ((((long)e*n + i)*n + j)*NVR)*nk + k;
    double vr[NV], vi[NV];
#pragma unroll
    for (int f = 0; f < NV; ++f) { vr[f]=U[b + (long)f*nk]; vi[f]=U[b + (long)(NV+f)*nk]; }

    const double kk = kz[k];
    const double q  = wq[(((long)e*n + i)*n + j)];
    const long rb = ((((long)e*n + i)*n + j)*NRR)*nk + k;

    // fields: 0 u, 1 v, 2 w, 3 ox, 4 oy, 5 oz, 6 p
    double s;
    s = rw[0]*q; R[rb+ 0*nk] = s*(kap*vr[6] + dxr[0] + dyr[1] - kk*vi[2]);
                 R[rb+ 8*nk] = s*(kap*vi[6] + dxi[0] + dyi[1] + kk*vr[2]);
    s = rw[1]*q; R[rb+ 1*nk] = s*(dyr[2] + kk*vi[1] - vr[3]);
                 R[rb+ 9*nk] = s*(dyi[2] - kk*vr[1] - vi[3]);
    s = rw[2]*q; R[rb+ 2*nk] = s*(-kk*vi[0] - dxr[2] - vr[4]);
                 R[rb+10*nk] = s*( kk*vr[0] - dxi[2] - vi[4]);
    s = rw[3]*q; R[rb+ 3*nk] = s*(dxr[1] - dyr[0] - vr[5]);
                 R[rb+11*nk] = s*(dxi[1] - dyi[0] - vi[5]);
    s = rw[4]*q; R[rb+ 4*nk] = s*(c*vr[0] + dxr[6] + nu*(dyr[5] + kk*vi[4]));
                 R[rb+12*nk] = s*(c*vi[0] + dxi[6] + nu*(dyi[5] - kk*vr[4]));
    s = rw[5]*q; R[rb+ 5*nk] = s*(c*vr[1] + dyr[6] + nu*(-kk*vi[3] - dxr[5]));
                 R[rb+13*nk] = s*(c*vi[1] + dyi[6] + nu*( kk*vr[3] - dxi[5]));
    s = rw[6]*q; R[rb+ 6*nk] = s*(c*vr[2] - kk*vi[6] + nu*(dxr[4] - dyr[3]));
                 R[rb+14*nk] = s*(c*vi[2] + kk*vr[6] + nu*(dxi[4] - dyi[3]));
    s = rw[7]*q; R[rb+ 7*nk] = s*(dxr[3] + dyr[4] - kk*vi[5]);
                 R[rb+15*nk] = s*(dxi[3] + dyi[4] + kk*vr[5]);
}

// Transpose: D[a*n+i] (not D[i*n+a]), SAME sign as the forward term.
__global__ void fused_LT(const double* __restrict__ Rr, const double* __restrict__ D,
                         const double* __restrict__ facx, const double* __restrict__ facy,
                         const double* __restrict__ kz, double nu, double c, double kap,
                         double* __restrict__ C, int nel, int n, int nk)
{
    long t = (long)blockIdx.x*blockDim.x + threadIdx.x;
    long total = (long)nel*n*n*nk;
    if (t >= total) return;
    int k = (int)(t % nk);
    int j = (int)((t / nk) % n);
    int i = (int)((t / ((long)nk*n)) % n);
    int e = (int)(t / ((long)nk*n*n));

    double txr[8], txi[8], tyr[8], tyi[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) { txr[r]=0.0; txi[r]=0.0; tyr[r]=0.0; tyi[r]=0.0; }

    const double fx = facx[e], fy = facy[e];
    for (int a = 0; a < n; ++a) {
        const double dx = D[a*n + i]*fx;
        const double dy = D[a*n + j]*fy;
        const long ax = ((((long)e*n + a)*n + j)*NRR)*nk + k;
        const long ay = ((((long)e*n + i)*n + a)*NRR)*nk + k;
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            txr[r] += dx*Rr[ax + (long)r*nk];
            txi[r] += dx*Rr[ax + (long)(8+r)*nk];
            tyr[r] += dy*Rr[ay + (long)r*nk];
            tyi[r] += dy*Rr[ay + (long)(8+r)*nk];
        }
    }
    const long b = ((((long)e*n + i)*n + j)*NRR)*nk + k;
    double rr[8], ri[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) { rr[r]=Rr[b + (long)r*nk]; ri[r]=Rr[b + (long)(8+r)*nk]; }

    const double kk = kz[k];
    const long cb = ((((long)e*n + i)*n + j)*NVR)*nk + k;

    C[cb+ 0*nk] = txr[0] + kk*ri[2] - tyr[3] + c*rr[4];
    C[cb+ 7*nk] = txi[0] - kk*rr[2] - tyi[3] + c*ri[4];
    C[cb+ 1*nk] = tyr[0] - kk*ri[1] + txr[3] + c*rr[5];
    C[cb+ 8*nk] = tyi[0] + kk*rr[1] + txi[3] + c*ri[5];
    C[cb+ 2*nk] = kk*ri[0] + tyr[1] - txr[2] + c*rr[6];
    C[cb+ 9*nk] = -kk*rr[0] + tyi[1] - txi[2] + c*ri[6];
    C[cb+ 3*nk] = -rr[1] + nu*kk*ri[5] - nu*tyr[6] + txr[7];
    C[cb+10*nk] = -ri[1] - nu*kk*rr[5] - nu*tyi[6] + txi[7];
    C[cb+ 4*nk] = -rr[2] - nu*kk*ri[4] + nu*txr[6] + tyr[7];
    C[cb+11*nk] = -ri[2] + nu*kk*rr[4] + nu*txi[6] + tyi[7];
    C[cb+ 5*nk] = -rr[3] + nu*tyr[4] - nu*txr[5] + kk*ri[7];
    C[cb+12*nk] = -ri[3] + nu*tyi[4] - nu*txi[5] - kk*rr[7];
    C[cb+ 6*nk] = txr[4] + tyr[5] + kk*ri[6] + kap*rr[0];
    C[cb+13*nk] = txi[4] + tyi[5] - kk*rr[6] + kap*ri[0];
}

torch::Tensor apply_L(torch::Tensor U, torch::Tensor D, torch::Tensor facx,
                           torch::Tensor facy, torch::Tensor kz, double nu, double c,
                           torch::Tensor wq, double kap, torch::Tensor rw)
{
    int nel = U.size(0), n = U.size(1), nk = U.size(4);
    auto R = torch::empty({nel, n, n, NRR, nk}, U.options());
    long total = (long)nel*n*n*nk;
    int threads = 256, blocks = (int)((total + threads - 1)/threads);
    fused_L<<<blocks, threads>>>(U.data_ptr<double>(), D.data_ptr<double>(),
        facx.data_ptr<double>(), facy.data_ptr<double>(), kz.data_ptr<double>(),
        nu, c, wq.data_ptr<double>(), kap, rw.data_ptr<double>(),
        R.data_ptr<double>(), nel, n, nk);
    return R;
}

torch::Tensor apply_LT(torch::Tensor R, torch::Tensor D, torch::Tensor facx,
                            torch::Tensor facy, torch::Tensor kz, double nu,
                            double c, double kap)
{
    int nel = R.size(0), n = R.size(1), nk = R.size(4);
    auto C = torch::empty({nel, n, n, NVR, nk}, R.options());
    long total = (long)nel*n*n*nk;
    int threads = 256, blocks = (int)((total + threads - 1)/threads);
    fused_LT<<<blocks, threads>>>(R.data_ptr<double>(), D.data_ptr<double>(),
        facx.data_ptr<double>(), facy.data_ptr<double>(), kz.data_ptr<double>(),
        nu, c, kap, C.data_ptr<double>(), nel, n, nk);
    return C;
}
'''


def device():
    from .kernels_torch import device as _d
    return _d()


def available():
    return torch is not None and torch.cuda.is_available()


def module():
    """Compile on first use; torch caches the build under TORCH_EXTENSIONS_DIR."""
    global _MOD
    if _MOD is None:
        from torch.utils.cpp_extension import load_inline
        os.environ.setdefault('TORCH_EXTENSIONS_DIR',
                              os.environ.get('HOME', '/tmp') + '/.lssem3d_cuda')
        # DECLARATIONS go in cpp_sources.  With `functions=`, load_inline
        # GENERATES its own PYBIND11_MODULE in main.cpp, so supplying one in the
        # CUDA source collides -- and main.cpp cannot see the definitions unless
        # they are declared here.
        #
        # TORCH_CUDA_ARCH_LIST: torch's default list for this build stops at
        # sm_120, and GB10 is sm_121.  Without this it would fall back to PTX
        # JIT of compute_120 -- functional, slower, and silent about it.
        os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '12.1')
        _MOD = load_inline(name='lssem3d_fused', cpp_sources=_DECL,
                           cuda_sources=_SRC,
                           functions=['apply_L', 'apply_LT'], verbose=False)
    return _MOD


def _c(a):
    return a if a.is_contiguous() else a.contiguous()


def apply_L(Ur, D, facx, facy, kz, nu, c, wq=None, kap=0.0, rw=None):
    U = _c(Ur)
    nk = U.shape[-1]
    dev, dt = U.device, U.dtype
    one = torch.ones
    wqt = _c(wq) if wq is not None else one(U.shape[0], U.shape[1], U.shape[2],
                                            dtype=dt, device=dev)
    rwt = _c(rw) if rw is not None else one(8, dtype=dt, device=dev)
    kzt = kz if kz.numel() == nk else kz.expand(nk)
    return module().apply_L(U, _c(D), _c(facx), _c(facy), _c(kzt).to(dt),
                            float(nu), float(c), wqt, float(kap), rwt)


def apply_LT(Rr, D, facx, facy, kz, nu, c, kap=0.0):
    R = _c(Rr)
    nk = R.shape[-1]
    kzt = kz if kz.numel() == nk else kz.expand(nk)
    return module().apply_LT(R, _c(D), _c(facx), _c(facy), _c(kzt).to(R.dtype),
                             float(nu), float(c), float(kap))
