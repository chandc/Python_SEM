"""Solve an exported FOSLS matrix with AmgX on the GB10, via the C API.

    python3 amgx_run.py <matrix.npz> <config-name> [--block 4]

WHY ctypes AND NOT amgx_capi.  The whole point of the test is point-block
aggregation with block_dimx=4 -- AmgX has no user near-null-space API, so
aggregating NODES rather than DOFs is its only route to the field-constant
coarse space that F2 showed is what makes this operator h-independent.  The
MatrixMarket reader gives no way to declare that; AMGX_matrix_upload_all does.

MATCHED TO F2.  tol=1e-8 relative, max_iters=20000, b = A @ x_rand with the same
seed count_cg() used, so the iteration counts are directly comparable.

The residual reported is recomputed on the HOST from the downloaded solution --
||b - Ax||/||b|| -- not read from AmgX's own monitor.  A solver reporting its own
convergence is not evidence that it converged.
"""
import ctypes
import sys
import time

import numpy as np
import scipy.sparse as sp

LIB = '/tmp/AMGX/build/libamgxsh.so'
MODE_dDDI = 8193                      # include/amgx_config.h:109

_COMMON = ('config_version=2, determinism_flag=1, '
           'solver(main)=PCG, main:max_iters=20000, main:tolerance=1e-8, '
           'main:convergence=RELATIVE_INI, main:norm=L2, '
           'main:monitor_residual=1, main:obtain_timings=1, '
           'main:print_solve_stats=0, ')

_AMG = ('main:preconditioner(amg)=AMG, amg:max_iters=1, amg:cycle=V, '
        'amg:max_levels=50, amg:monitor_residual=0, amg:print_grid_stats=1, '
        'amg:smoother(sm)=BLOCK_JACOBI, sm:relaxation_factor=0.8, '
        'amg:presweeps=0, amg:postsweeps=3, amg:coarse_solver=NOSOLVER, ')

CONFIGS = {
    # the compiled-Jacobi baseline -- F2's Jacobi column, but in CUDA
    # A preconditioner must be applied EXACTLY ONCE.  Without max_iters=1 AmgX
    # runs BLOCK_JACOBI as a multi-sweep solver, which is not a fixed SPD
    # operator, and PCG stagnates (measured: 20000 its, residual 0.80).
    'jacobi':    _COMMON + 'main:preconditioner(pj)=BLOCK_JACOBI, '
                           'pj:relaxation_factor=1.0, pj:max_iters=1, '
                           'pj:monitor_residual=0',
    # aggregation AMG -- the analogue of pyamg smoothed_aggregation
    'agg':       _COMMON + _AMG + 'amg:algorithm=AGGREGATION, '
                                  'amg:selector=SIZE_2',
    'agg8':      _COMMON + _AMG + 'amg:algorithm=AGGREGATION, '
                                  'amg:selector=SIZE_8',
    # classical Ruge-Stuben, for contrast
    'classical': _COMMON + _AMG + 'amg:algorithm=CLASSICAL, '
                                  'amg:interpolator=D2, amg:selector=PMIS',
}


def _chk(rc, what):
    if rc != 0:
        raise RuntimeError(f'AmgX {what} returned rc={rc}')


def main():
    npz, cfgname = sys.argv[1], sys.argv[2]
    blk = 1
    if '--block' in sys.argv:
        blk = int(sys.argv[sys.argv.index('--block') + 1])

    d = np.load(npz, allow_pickle=True)
    A = sp.csr_matrix((d['data'], d['indices'], d['indptr']),
                      shape=(int(d['n']), int(d['n'])))
    b = d['b'].astype(np.float64)

    if blk > 1:
        B = A.tobsr(blocksize=(blk, blk))
        B.sort_indices()
        nrow, nnz = B.shape[0]//blk, B.indptr[-1]
        ptr, idx, val = B.indptr, B.indices, np.ascontiguousarray(B.data)
    else:
        nrow, nnz = A.shape[0], A.nnz
        ptr, idx, val = A.indptr, A.indices, A.data

    ptr = np.ascontiguousarray(ptr, dtype=np.int32)
    idx = np.ascontiguousarray(idx, dtype=np.int32)
    val = np.ascontiguousarray(val, dtype=np.float64).ravel()

    lib = ctypes.CDLL(LIB)
    lib.AMGX_initialize()
    cfg = ctypes.c_void_p(); rsc = ctypes.c_void_p()
    mtx = ctypes.c_void_p(); vx = ctypes.c_void_p(); vb = ctypes.c_void_p()
    slv = ctypes.c_void_p()
    _chk(lib.AMGX_config_create(ctypes.byref(cfg),
                                CONFIGS[cfgname].encode()), 'config_create')
    _chk(lib.AMGX_resources_create_simple(ctypes.byref(rsc), cfg), 'resources')
    for h in (mtx,):
        _chk(lib.AMGX_matrix_create(ctypes.byref(h), rsc, MODE_dDDI), 'mat')
    for h in (vx, vb):
        _chk(lib.AMGX_vector_create(ctypes.byref(h), rsc, MODE_dDDI), 'vec')
    _chk(lib.AMGX_solver_create(ctypes.byref(slv), rsc, MODE_dDDI, cfg), 'solver')

    _chk(lib.AMGX_matrix_upload_all(
        mtx, ctypes.c_int(nrow), ctypes.c_int(int(nnz)),
        ctypes.c_int(blk), ctypes.c_int(blk),
        ptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        idx.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        val.ctypes.data_as(ctypes.c_void_p), None), 'upload_all')
    _chk(lib.AMGX_vector_upload(vb, ctypes.c_int(nrow), ctypes.c_int(blk),
                                b.ctypes.data_as(ctypes.c_void_p)), 'vec_upload')
    _chk(lib.AMGX_vector_set_zero(vx, ctypes.c_int(nrow), ctypes.c_int(blk)),
         'set_zero')

    t0 = time.perf_counter(); _chk(lib.AMGX_solver_setup(slv, mtx), 'setup')
    t1 = time.perf_counter(); _chk(lib.AMGX_solver_solve(slv, vb, vx), 'solve')
    t2 = time.perf_counter()

    its = ctypes.c_int()
    lib.AMGX_solver_get_iterations_number(slv, ctypes.byref(its))
    x = np.zeros(A.shape[0], dtype=np.float64)
    lib.AMGX_vector_download(vx, x.ctypes.data_as(ctypes.c_void_p))
    true_res = np.linalg.norm(b - A @ x)/np.linalg.norm(b)

    print(f'RESULT {npz.split("/")[-1]} cfg={cfgname} blk={blk} '
          f'n={A.shape[0]} nnz={A.nnz} its={its.value} '
          f'setup={t1-t0:.4f}s solve={t2-t1:.4f}s total={t2-t0:.4f}s '
          f'res={true_res:.3e}')

    for f, h in ((lib.AMGX_solver_destroy, slv), (lib.AMGX_vector_destroy, vx),
                 (lib.AMGX_vector_destroy, vb), (lib.AMGX_matrix_destroy, mtx),
                 (lib.AMGX_resources_destroy, rsc)):
        f(h)
    lib.AMGX_config_destroy(cfg)
    lib.AMGX_finalize()


if __name__ == '__main__':
    main()
