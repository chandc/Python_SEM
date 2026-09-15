"""Is a remote GPU worth porting this solver to?  A standalone FP64 probe.

    scp scratch/gpu_fp64_probe.py spark-b85b:~/
    ssh spark-b85b 'python3 gpu_fp64_probe.py'

STANDALONE BY DESIGN -- imports nothing from this repo, so it can be dropped on
any machine.  numpy only; torch or cupy used if present.

THE QUESTION, and it is narrower than "is the GPU fast".  The 3D VVP matvec is
MEMORY-BANDWIDTH bound in DOUBLE precision (3D_STATUS.md sec 3.3: threads tie
processes; sec 7M: the numba win came from fewer passes, not faster arithmetic).
So exactly two numbers decide whether a port is worth it:

  1. ACHIEVED FP64 STREAMING BANDWIDTH.  Not the spec sheet -- the measured
     triad.  If it does not beat the local M3 Max's 315 GB/s (measured, MLX,
     scratch/mlx_bandwidth_probe.py) there is nothing to win, however many
     TFLOPs the part advertises.

  2. THE FP32:FP64 RATIO.  This is the decisive one and it is why this script
     exists.  Datacenter parts (A100, H100) run FP64 at ~1:2 of FP32.  Parts
     built for low-precision AI inference throttle it to ~1:32 or 1:64.  This
     solver is double precision throughout, and the least-squares normal
     equations SQUARE the condition number (kappa ~ 1e4 after the row-7 fix,
     sec 7J), so dropping to FP32 is not a free option -- it would leave ~3
     significant digits.  A throttled part could easily run this workload
     SLOWER than the Mac despite winning every headline benchmark.

  A ratio near 1:2  -> port is worth costing out.
  A ratio near 1:32 -> the GPU is the wrong tool for THIS solver; say so and
                       stay on local MLX (11x at M7 scale, already measured).

THE PROBE SELF-CALIBRATES.  Run on a CPU -- hardware with full-rate FP64 -- it
reports a ratio of 2.0x, which is pure bandwidth (FP64 moves twice the bytes)
and nothing else.  That anchors the thresholds below: ~2x means bandwidth-
limited and healthy, ~32x means the arithmetic units are throttled.  Measured on
the local M3 Max CPU: 2.0x.  If a run reports something wildly different on a
device you trust, suspect the probe, not the device.

Also measured: the ddx-shaped batched small-matrix contraction, at the real
element/mode counts, because that is the access pattern the operator is built
from and it behaves nothing like a big dense GEMM.  And the host CPU, since a
Grace-class ARM host with high-bandwidth memory may itself beat the Mac even if
the GPU disappoints.
"""
import platform
import sys
import time

import numpy as np

# Local M3 Max baselines to compare against (scratch/mlx_bandwidth_probe.py,
# 2026-08-22).  Quoted so the remote output is self-interpreting.
LOCAL = {
    'numpy_triad_GBs': 47.8,      # 512 MB triad, CPU
    'mlx_triad_GBs': 314.8,       # 512 MB triad, M3 Max GPU via MLX
    'mlx_ddx_240e_ms': 2.97,      # 240 elem, N=8, nk=65 -- full M7 shape
    'numpy_ddx_240e_ms': 32.68,
}

# (nelem, n, nk, label) -- n = N+1.  The last row is full M7's shape.
SHAPES = [(9, 7, 9, 'Stage 5 channel'),
          (144, 9, 16, 'minimal channel-ish'),
          (240, 9, 65, 'FULL M7')]


def _bench(fn, sync, reps):
    fn(); sync()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    sync()
    return (time.perf_counter() - t)/reps


# --------------------------------------------------------------- backends

class NumpyBackend:
    name = 'numpy (host CPU)'
    def __init__(self): self.xp = np
    def array(self, a, dtype): return np.ascontiguousarray(a, dtype=dtype)
    def sync(self): pass
    def triad(self, a, b): return a + 2.0*b
    def ddx(self, D, U): return np.einsum('pi,eijvk->epjvk', D, U)
    def info(self): return f'{platform.processor() or platform.machine()}'


class TorchBackend:
    name = 'torch (GPU)'
    def __init__(self):
        import torch
        self.torch = torch
        if not torch.cuda.is_available():
            raise RuntimeError('torch present but no CUDA device')
        self.dev = torch.device('cuda')
    def array(self, a, dtype):
        dt = {np.float64: self.torch.float64, np.float32: self.torch.float32}[dtype]
        return self.torch.as_tensor(np.ascontiguousarray(a), dtype=dt, device=self.dev)
    def sync(self): self.torch.cuda.synchronize()
    def triad(self, a, b): return a + 2.0*b
    def ddx(self, D, U): return self.torch.einsum('pi,eijvk->epjvk', D, U)
    def info(self):
        p = self.torch.cuda.get_device_properties(0)
        return (f'{p.name}, {p.total_memory/2**30:.0f} GiB, cc {p.major}.{p.minor}, '
                f'{p.multi_processor_count} SMs')


class CupyBackend:
    name = 'cupy (GPU)'
    def __init__(self):
        import cupy
        self.cp = cupy
        cupy.cuda.runtime.getDeviceCount()
    def array(self, a, dtype): return self.cp.asarray(a, dtype=dtype)
    def sync(self): self.cp.cuda.Stream.null.synchronize()
    def triad(self, a, b): return a + 2.0*b
    def ddx(self, D, U): return self.cp.einsum('pi,eijvk->epjvk', D, U)
    def info(self):
        p = self.cp.cuda.runtime.getDeviceProperties(0)
        return f"{p['name'].decode()}, {p['totalGlobalMem']/2**30:.0f} GiB"


def backends():
    out = [NumpyBackend()]
    for cls in (TorchBackend, CupyBackend):
        try:
            out.append(cls())
            break                      # one GPU backend is enough
        except Exception as e:
            print(f'  [{cls.name}: unavailable -- {type(e).__name__}: {e}]')
    return out


# ----------------------------------------------------------------- probes

def triad(be, dtype, mb=512, reps=20):
    """a + 2*b over `mb` megabytes: 3 arrays touched, so 3*n*itemsize moved."""
    itemsize = np.dtype(dtype).itemsize
    n = mb*1024*1024//itemsize
    h = np.random.default_rng(0).standard_normal(n)
    a, b = be.array(h, dtype), be.array(h, dtype)
    t = _bench(lambda: be.triad(a, b), be.sync, reps)
    return 3*n*itemsize/1e9/t


def ddx(be, dtype, nel, n, nk, reps=10):
    U = np.random.default_rng(0).standard_normal((nel, n, n, 14, nk))
    D = np.random.default_rng(1).standard_normal((n, n))
    Ud, Dd = be.array(U, dtype), be.array(D, dtype)
    return _bench(lambda: be.ddx(Dd, Ud), be.sync, reps)


def main():
    print(f'host   : {platform.node()}  {platform.system()} {platform.machine()}')
    print(f'python : {sys.version.split()[0]}   numpy {np.__version__}\n')

    for be in backends():
        print(f'--- {be.name}: {be.info()}')
        try:
            b64 = triad(be, np.float64)
            b32 = triad(be, np.float32)
        except Exception as e:
            print(f'    triad failed: {e}\n'); continue
        print(f'    triad  FP64 {b64:8.1f} GB/s     FP32 {b32:8.1f} GB/s')

        print(f'    {"ddx shape":<38} {"FP64":>11} {"FP32":>10} {"ratio":>10}')
        ratios = []
        for nel, n, nk, label in SHAPES:
            try:
                t64, t32 = ddx(be, np.float64, nel, n, nk), ddx(be, np.float32, nel, n, nk)
            except Exception as e:
                print(f'    {label}: failed -- {e}'); continue
            ratios.append(t64/t32)
            tag = f'{label} ({nel}e n={n} nk={nk})'
            print(f'    {tag:<38} {t64*1e3:9.2f}ms {t32*1e3:9.2f}ms {t64/t32:9.2f}x')

        if ratios:
            r = float(np.median(ratios))
            print(f'\n    FP64 penalty (median over shapes): {r:.1f}x slower than FP32')
            if r < 3:
                verdict = ('FP64 is NOT throttled. Worth costing out a port -- '
                           'compare the bandwidth above against MLX 314.8 GB/s.')
            elif r < 10:
                verdict = ('FP64 is partially throttled. Marginal; only worth it '
                           'if the bandwidth win is large.')
            else:
                verdict = ('FP64 IS THROTTLED. This is a low-precision AI part; '
                           'wrong tool for a double-precision LS solver. Stay on '
                           'local MLX.')
            print(f'    VERDICT: {verdict}')
        print()

    print('Local M3 Max baselines to beat (measured 2026-08-22):')
    print(f'  triad FP64  : numpy {LOCAL["numpy_triad_GBs"]:.1f} GB/s, '
          f'MLX GPU {LOCAL["mlx_triad_GBs"]:.1f} GB/s')
    print(f'  ddx FULL M7 : numpy {LOCAL["numpy_ddx_240e_ms"]:.2f} ms, '
          f'MLX GPU {LOCAL["mlx_ddx_240e_ms"]:.2f} ms')
    print('\nA port is worth costing out only if the remote GPU beats 314.8 GB/s')
    print('in FP64 AND its FP64 penalty is small. Either one failing kills it.')


if __name__ == '__main__':
    main()
