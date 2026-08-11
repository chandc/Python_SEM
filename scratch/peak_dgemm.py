import os, time
import numpy as np
print("threads env:", {k: os.environ.get(k) for k in
      ('OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OPENBLAS_NUM_THREADS') if os.environ.get(k)})
print(f"numpy {np.__version__}")
for n in (1000, 2000, 4000):
    A = np.random.rand(n, n); B = np.random.rand(n, n)
    A @ B                                   # warm up
    t = 1e30
    for _ in range(3):
        t0 = time.perf_counter(); A @ B; t = min(t, time.perf_counter()-t0)
    print(f"  dgemm n={n:5d}  {2.0*n**3/t/1e9:8.1f} GFLOP/s  ({t*1e3:.1f} ms)")
# small-matrix regime: what the SEM derivative kernel actually does
for n in (5, 10, 15):
    A = np.random.rand(n, n); B = np.random.rand(n, n)
    reps = 200000
    A @ B
    t0 = time.perf_counter()
    for _ in range(reps): A @ B
    t = time.perf_counter()-t0
    print(f"  dgemm n={n:5d}  {2.0*n**3*reps/t/1e9:8.2f} GFLOP/s  "
          f"({t/reps*1e9:.0f} ns/call)  <- SEM block size")
