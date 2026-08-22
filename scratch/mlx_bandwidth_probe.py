"""Is the GPU worth it for a BANDWIDTH-bound operator?

The 3D matvec is memory-bandwidth bound (prof3d_procs.py: threads tie
processes), and numba's win came from making fewer passes over the data, not
from arithmetic.  That makes the relevant question for MLX narrow and concrete:
how much more bandwidth does the M3 Max GPU actually deliver than the CPU side,
on arrays the size of our state?

This measures a streaming triad (a = b + s*c, 3 arrays touched) and a batched
small-matrix contraction shaped like ddx -- the two access patterns the operator
is built from.  It does NOT port the operator; it prices the ceiling first.
"""
import os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np


def bench(f, reps=20, sync=None):
    f()
    if sync: sync()
    t = time.perf_counter()
    for _ in range(reps):
        r = f()
    if sync: sync()
    return (time.perf_counter() - t)/reps


def main():
    import mlx.core as mx
    print(f'mlx {mx.__version__}, default device {mx.default_device()}')
    print(f'\n{"MB":>8} {"numpy GB/s":>12} {"mlx GB/s":>12} {"ratio":>8}   streaming triad')
    for mb in (4, 32, 128, 512):
        n = mb*1024*1024//8
        a = np.random.default_rng(0).standard_normal(n)
        b = a.copy()
        A, B = mx.array(a), mx.array(b)
        tn = bench(lambda: a + 2.0*b)
        tm = bench(lambda: mx.eval(A + 2.0*B), sync=lambda: mx.eval(A))
        gb = 3*n*8/1e9
        print(f'{mb:8d} {gb/tn:12.1f} {gb/tm:12.1f} {tn/tm:7.2f}x')

    # ddx-shaped: (nelem*nmode) independent (n x n) x (n x n) products
    print(f'\n{"case":>22} {"numpy":>10} {"mlx":>10} {"ratio":>8}   batched small matmul')
    for nel, n, nk in ((9, 7, 9), (144, 9, 16), (240, 9, 65)):
        U = np.random.default_rng(0).standard_normal((nel, n, n, 14, nk))
        D = np.random.default_rng(1).standard_normal((n, n))
        Um, Dm = mx.array(U), mx.array(D)
        tn = bench(lambda: np.einsum('pi,eij...->epj...', D, U), reps=10)
        tm = bench(lambda: mx.eval(mx.einsum('pi,eijvk->epjvk', Dm, Um)),
                   reps=10, sync=lambda: mx.eval(Dm))
        dof = nel*n*n*14*nk
        print(f'{f"{nel}e N={n-1} nk={nk}":>22} {tn*1e3:9.2f}ms {tm*1e3:9.2f}ms {tn/tm:7.2f}x'
              f'   ({dof/1e6:.2f}M dof)')


if __name__ == '__main__':
    main()
