import numpy as np
import time

nelem = 400
N = 8
k = 4
ndof = nelem * N * N # approximate

U = np.random.randn(nelem, N+1, N+1, k)
idx = np.random.randint(0, ndof, size=(nelem, N+1, N+1))

# approach 1: add.at
global_U1 = np.zeros((ndof, k))
t0 = time.perf_counter()
np.add.at(global_U1, idx, U)
res1 = global_U1[idx]
t1 = time.perf_counter()

# approach 2: bincount
flat_idx = idx.ravel()
global_U2 = np.zeros((ndof, k))
t2 = time.perf_counter()
for i in range(k):
    global_U2[:, i] = np.bincount(flat_idx, weights=U[..., i].ravel(), minlength=ndof)
res2 = global_U2[idx]
t3 = time.perf_counter()

print(f"add.at: {(t1-t0)*1000:.3f} ms")
print(f"bincount: {(t3-t2)*1000:.3f} ms")
