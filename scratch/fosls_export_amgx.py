"""Export the 2D FOSLS matrices for AmgX, in BOTH forms the comparison needs.

    uv run --quiet python scratch/fosls_export_amgx.py

WHY TWO FORMS.  F2 measured pyamg on `A[free][:,free]` -- constrained DOFs
deleted INDIVIDUALLY.  That is correct for a scalar solver and fatal for a block
one: it removes single DOFs from 4-DOF nodes, so the matrix is no longer
4-blockable and AmgX's point-block aggregation cannot be applied.  (Exactly the
`blocksize and A.shape must be compatible` failure the pyamg block-smoother test
hit.)

  scalar  A[free][:,free]                -- reproduces the pyamg numbers exactly
  block   full matrix, identity rows on  -- 4-DOF blocks intact, node-major
          constrained DOFs                  ordering DOF = node*4 + field

The block form adds unit eigenvalues for the constrained DOFs.  They converge
immediately and do not flatter the iteration count.

RHS is built as b = A @ x_rand with the SAME seed count_cg() uses, so AmgX's
iteration counts are directly comparable to F2's.
"""
import os
import sys

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R)
sys.path.insert(0, os.path.join(_R, 'scratch'))

import numpy as np
import scipy.sparse as sp

import fosls_amg as FG

OUT = os.path.join(_R, 'scratch', 'amgx_mats')
NV = FG.NV


def block_form(A_full, free):
    """Full matrix with identity on constrained DOFs -- 4-DOF blocks preserved."""
    n = A_full.shape[0]
    assert n % NV == 0, f'{n} not divisible by {NV} -- ordering assumption broken'
    con = ~free
    A = A_full.tolil(copy=True)
    for c in np.nonzero(con)[0]:
        A.rows[c] = [int(c)]
        A.data[c] = [1.0]
    A = A.tocsr()
    # zero the constrained COLUMNS too, so the block matrix stays symmetric
    A = A.T.tolil()
    for c in np.nonzero(con)[0]:
        A.rows[c] = [int(c)]
        A.data[c] = [1.0]
    return A.tocsr().T.tocsr()


def one(tag, N, ex, ey, seed=0):
    m = FG.build_channel(2.0, 1.0, ex, ey, N, bcs=(1, 1, 1, 2))
    m.compute_global_indices()
    st = FG.SolverState(m, FG.diff_matrix(N), nu=1/100., dt=1e4,
                        fac1=1.0, w_mom=1.0)
    n = N + 1
    z = np.zeros((m.nelem, n, n))
    st.update_linearisation(z, z.copy())
    A_full, free, _ = FG.FA.assemble(st, z, z.copy(), pin_p=True)
    A_full = A_full.tocsr()

    for form, A in (('scalar', A_full[free][:, free].tocsr()),
                    ('block', block_form(A_full, free))):
        A = A.astype(np.float64)
        A.sort_indices()
        rng = np.random.default_rng(seed)
        b = A @ rng.standard_normal(A.shape[0])
        f = os.path.join(OUT, f'{tag}_{form}.npz')
        np.savez_compressed(f, indptr=A.indptr.astype(np.int32),
                            indices=A.indices.astype(np.int32),
                            data=A.data, b=b, n=A.shape[0], nnz=A.nnz,
                            N=N, ex=ex, ey=ey, form=form)
        print(f'  {tag:16s} {form:6s} n={A.shape[0]:7d} nnz={A.nnz:9d} '
              f'blk4={"yes" if A.shape[0] % NV == 0 else "NO"}  -> {os.path.basename(f)}')


def main():
    os.makedirs(OUT, exist_ok=True)
    print('h-sweep (N=4) -- matches F2 gate')
    for ex, ey in ((4, 4), (8, 8), (12, 12), (16, 16)):
        one(f'h_N4_{ex}x{ey}', 4, ex, ey)
    # Extended past F2e's N=8.  The LOR argument this plan raised rests on the
    # element block being DENSE, and it grows as (N+1)^4: 100^2 at N=4, 324^2 at
    # N=8, 676^2 at N=12.  If AMG degrades with order, THIS is where it shows.
    print('\np-sweep (6x6) -- F2e range plus higher p')
    for N in (4, 6, 8, 10, 12):
        one(f'p_N{N}_6x6', N, 6, 6)
    print(f'\nwritten to {OUT}')


if __name__ == '__main__':
    main()
