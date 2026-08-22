"""What can this Mac ACTUALLY do in float64?  Dtypes ASSERTED, not assumed.

WHY THIS EXISTS.  mlx_bandwidth_probe.py fed float64 numpy arrays to mx.array()
and reported the result as FP64.  MLX SILENTLY DOWNCASTS: float64 -> float32,
complex128 -> complex64.  So it measured FP32 and labelled it FP64, and sec 7N
drew a conclusion from it that was exactly backwards.

    >>> mx.array(np.zeros(4, dtype=np.float64)).dtype
    mlx.core.float32
    >>> mx.array(x, dtype=mx.float64)      # on GPU
    ValueError: float64 is not supported on the GPU

Metal has no double precision.  MLX is therefore UNUSABLE for this solver on the
GPU, whatever its float32 numbers look like -- the LS normal equations square the
condition number (kappa ~ 1e4 post-row-7), so float32 would leave ~3 digits.

Every measurement below asserts its dtype before timing.  A benchmark that does
not check what it is benchmarking is not a measurement.
"""
import time

import numpy as np

SHAPES = [(9, 7, 9, 'Stage 5 channel'), (144, 9, 16, 'minimal channel-ish'),
          (240, 9, 65, 'FULL M7')]


def _t(f, reps):
    f(); t = time.perf_counter()
    for _ in range(reps):
        f()
    return (time.perf_counter() - t)/reps


def main():
    import mlx.core as mx
    n = 64*1024*1024//8
    h = np.random.default_rng(0).standard_normal(n)

    print('=== streaming triad, FP64 ONLY (dtype asserted) ===')
    a = np.ascontiguousarray(h); b = a.copy()
    assert a.dtype == np.float64
    t = _t(lambda: a + 2.0*b, 20)
    print(f'  numpy  CPU float64 : {3*n*8/1e9/t:7.1f} GB/s')

    with mx.stream(mx.cpu):
        A = mx.array(h, dtype=mx.float64); B = mx.array(h, dtype=mx.float64)
        mx.eval(A, B)
        assert A.dtype == mx.float64, A.dtype
        t = _t(lambda: mx.eval(A + 2.0*B), 20)
        print(f'  MLX    CPU float64 : {3*n*8/1e9/t:7.1f} GB/s')
    # MLX is LAZY: creating a float64 array on the gpu stream succeeds; it is
    # the OPERATION that raises.  Testing creation alone reports a false pass --
    # which this script did on its first run.
    try:
        with mx.stream(mx.gpu):
            G = mx.array(h[:1024], dtype=mx.float64)
            mx.eval(G + 2.0*G)
        print('  MLX    GPU float64 : ???  unexpectedly worked -- recheck')
    except Exception as e:
        print(f'  MLX    GPU float64 : IMPOSSIBLE -- {e}')

    print('\n=== ddx-shaped batched contraction, FP64 (dtype asserted) ===')
    print(f'  {"shape":<34} {"numpy":>10} {"MLX cpu":>10}     GB10 torch')
    gb10 = {'Stage 5 channel': 0.04, 'minimal channel-ish': 0.98, 'FULL M7': 6.68}
    for nel, nn, nk, label in SHAPES:
        U = np.random.default_rng(0).standard_normal((nel, nn, nn, 14, nk))
        D = np.random.default_rng(1).standard_normal((nn, nn))
        assert U.dtype == np.float64 and D.dtype == np.float64
        tn = min(_t(lambda: np.einsum('pi,eijvk->epjvk', D, U), 10) for _ in range(3))
        with mx.stream(mx.cpu):
            Um = mx.array(U, dtype=mx.float64); Dm = mx.array(D, dtype=mx.float64)
            mx.eval(Um, Dm)
            assert Um.dtype == mx.float64
            tm = min(_t(lambda: mx.eval(mx.einsum('pi,eijvk->epjvk', Dm, Um)), 10) for _ in range(3))
        tag = f'{label} ({nel}e n={nn} nk={nk})'
        print(f'  {tag:<34} {tn*1e3:9.2f}ms {tm*1e3:9.2f}ms {gb10[label]:9.2f}ms'
              f'   -> GB10 {tn*1e3/gb10[label]:.1f}x vs numpy')


if __name__ == '__main__':
    main()
