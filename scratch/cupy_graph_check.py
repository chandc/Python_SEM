"""Does the graph-captured CG agree with the reference, and what does it save?

    docker run --rm --gpus all -v "$PWD":/work -w /work lssem-cupy:latest \
           python scratch/cupy_graph_check.py

Three columns, and the middle one is the one that matters: a graph that is
fast and wrong is worse than no graph at all.
"""
import sys, time
sys.path.insert(0, '.')
import numpy as np, cupy as cp
import lssem3d
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem3d import operator as OP, solver3d as S3, bc as BC, fourier as FR
from lssem3d.cupy_graph import pcg_graph

L = 2*np.pi
g = cp.asarray


def run(N=8, ex=6, nz=24, nu=1/180., c=525.0, tol=1e-8, batch=20):
    m = build_channel(L, L, ex, ex, N, bcs=(0, 0, 0, 0))
    m.periodic_x = L; m.periodic_y = L; m.compute_global_indices()
    nk = nz//2 + 1
    mask = BC.build_mask(m, nk, pin_p=False, nz=nz)
    BC.pin_dof(m, mask, OP.P_, 0)
    D = diff_matrix(N); kz = FR.wavenumbers(nz, L)
    rw = OP.momentum_row_weights(c)
    shape = (m.nelem, N+1, N+1, OP.NVAR_R, nk)
    lssem3d.set_backend('numpy')
    xt = S3.make_continuous(m, np.random.default_rng(1).standard_normal(shape))*mask
    b = S3.normal_op(xt, D, m.facx, m.facy, kz, nu, c, mesh=m, mask=mask,
                     wq=m.wq, kap=0.0, rw=rw)
    Mi = S3.jacobi_inverse(S3.jacobi_diagonal_analytic(
        shape, D, m.facx, m.facy, kz, nu, c, m, mask, m.wq, 0.0, rw=rw), mask)
    lssem3d.set_backend('cupy')
    kw = dict(mesh=m, mask=g(mask), M_inv=g(Mi), tol=tol, max_iter=4000,
              wq=g(m.wq), rw=g(rw))
    args = (g(b), g(D), g(m.facx), g(m.facy), g(kz), nu, c)

    def timed(f):
        t0 = time.perf_counter(); out = f(); cp.cuda.Stream.null.synchronize()
        return out, time.perf_counter()-t0
    (x_ref, it_ref, _), t_ref = timed(lambda: S3.pcg(*args, **kw))
    (x_nog, it_nog, _), t_nog = timed(
        lambda: pcg_graph(*args, batch=batch, capture=False, **kw))
    (x_gph, it_gph, _), t_gph = timed(
        lambda: pcg_graph(*args, batch=batch, capture=True, **kw))
    sc = float(cp.abs(x_ref).max())
    d_nog = float(cp.abs(x_nog - x_ref).max())/sc
    d_gph = float(cp.abs(x_gph - x_ref).max())/sc
    # the floor: the reference solver's own run-to-run spread (atomics)
    (x_ref2, _, _), _ = timed(lambda: S3.pcg(*args, **kw))
    floor = float(cp.abs(x_ref2 - x_ref).max())/sc
    print(f'  dof {b.size/1e6:.2f} M, tol {tol:g}, batch {batch}')
    print(f'  {"reference pcg":<24}{it_ref:6d} its{t_ref:9.2f} s')
    print(f'  {"batched, no graph":<24}{it_nog:6d} its{t_nog:9.2f} s'
          f'   diff {d_nog:.2e}')
    print(f'  {"GRAPH-CAPTURED":<24}{it_gph:6d} its{t_gph:9.2f} s'
          f'   diff {d_gph:.2e}   speedup {t_ref/t_gph:.2f}x')
    print(f'  reference self-spread (the floor): {floor:.2e}')
    ok = d_gph < max(10*floor, 1e-9)
    print(f'  CORRECTNESS: {"PASS" if ok else "FAIL"}'
          f' -- graph differs from reference by {d_gph/max(floor,1e-30):.1f}x the floor')
    return ok


if __name__ == '__main__':
    print('CUDA graph capture on the CuPy PCG\n')
    ok = run()
    sys.exit(0 if ok else 1)
