"""F0 -- assemble the 2D LSSEM operator as a sparse matrix, and prove it is THE operator.

    uv run --quiet python scratch/fosls_assemble.py

WHY ASSEMBLE AT ALL.  lssem2d is matrix-free, which is right for production.  But
AMG needs a matrix, an ellipticity constant needs a generalised eigenproblem, and
every preconditioner claim in sec 7I-7K was made without either.  See
FOSLS_2D_PLAN.md.

HOW, AND WHY NOT THE OBVIOUS WAY.

  * NOT column-by-column probing of the global operator: that is O(ndof) matvecs.
  * NOT a hand-written element matrix: reimplementing L0 is exactly the mistake
    L1 warns about -- "tests that compare the operator to itself cannot find a
    wrong operator".  Four missing factors in A survived a full symmetry and
    convergence suite that way (sec 2.1).

INSTEAD: probe ELEMENT-LOCALLY.  L0 is element-local before gather-scatter, so a
unit vector placed at the same local index (i, j, var) in EVERY element produces,
in one call, that column of every element's block at once.  n^2 * 4 probes total
-- 324 at N = 8 -- rather than nelem * n^2 * 4.

That probe field is DISCONTINUOUS across element boundaries, which is fine here
and is the whole point: we want the unassembled blocks A_e, and the assembly
Q^T (+_e A_e) Q is then done explicitly through mesh.gidx.  sec 7G records the
same trick, and the trap of forgetting that discontinuity when gather-scatter IS
wanted.

The gate at the bottom is the real deliverable: the assembled matrix must
reproduce solver.apply_A to round-off on a CONTINUOUS field.  Anything less and
this is a lookalike, not the operator.
"""
import os
import sys

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np
import scipy.sparse as sp

from lssem2d.assembly import gather_scatter
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState, apply_L, apply_LT
from lssem2d.mesh import build_channel
from lssem2d.solver import apply_A

NV = 4                                   # u, v, p, omega


def element_blocks(state, fu, fv):
    """The unassembled per-element matrices A_e = L0_e^T (rho W) L0_e.

    Returns (nelem, ndof_e, ndof_e) with ndof_e = n*n*NV, ordered so that
    local index (i, j, var) maps to ((i*n) + j)*NV + var.
    """
    m = state.mesh
    n = m.N + 1
    nde = n*n*NV
    blocks = np.zeros((m.nelem, nde, nde))
    U = np.zeros((m.nelem, n, n, NV))
    for i in range(n):
        for j in range(n):
            for v in range(NV):
                col = ((i*n) + j)*NV + v
                U[:] = 0.0
                U[:, i, j, v] = 1.0          # same local index in EVERY element
                c = apply_LT(state, apply_L(state, U, fu, fv), fu, fv)
                blocks[:, :, col] = c.reshape(m.nelem, nde)
    return blocks


def assemble(state, fu, fv, pin_p=False):
    """Q^T (+_e A_e) Q with the Dirichlet mask applied on both sides.

    Prescribed dofs get a unit diagonal so the matrix stays non-singular and SPD;
    the free block is what matters and is what `free` indexes.
    """
    m = state.mesh
    n = m.N + 1
    blocks = element_blocks(state, fu, fv)
    gid = m.gidx                                     # (nelem, n, n) global node id
    ng = int(gid.max()) + 1
    ndof = ng*NV

    # local (e, i, j, var) -> global dof
    g = np.empty((m.nelem, n, n, NV), dtype=np.int64)
    for v in range(NV):
        g[..., v] = gid*NV + v
    gflat = g.reshape(m.nelem, -1)

    rows = np.repeat(gflat[:, :, None], gflat.shape[1], axis=2).ravel()
    cols = np.repeat(gflat[:, None, :], gflat.shape[1], axis=1).ravel()
    A = sp.coo_matrix((blocks.ravel(), (rows, cols)), shape=(ndof, ndof)).tocsr()

    mask = state.get_global_mask(pin_p=pin_p)         # (nelem, n, n, NV), 0 = fixed
    free = np.zeros(ndof, dtype=bool)
    np.logical_or.at(free, gflat.ravel(), (mask.reshape(m.nelem, -1) > 0.5).ravel())

    d = sp.diags(free.astype(float))
    A = d @ A @ d + sp.diags((~free).astype(float))
    return A.tocsr(), free, g


def _to_global(g, ng, local):
    """Read a local array at one representative copy of each global dof."""
    out = np.zeros(ng*NV)
    out[g.ravel()] = local.ravel()
    return out


def gate(N=6, ex=3, ey=2, nu=1/100., dt=1.0, seed=0, pin_p=True, verbose=True):
    """The deliverable: assembled A must BE solver.apply_A, not resemble it."""
    m = build_channel(2.0, 1.0, ex, ey, N, bcs=(1, 1, 1, 2))
    m.compute_global_indices()
    st = SolverState(m, diff_matrix(N), nu=nu, dt=dt, fac1=1.0)
    n = N + 1
    fu = np.zeros((m.nelem, n, n))
    fv = np.zeros((m.nelem, n, n))        # Stokes limit: no convection
    st.update_linearisation(fu, fv)       # primes dfu_dx etc; apply_L needs them

    A, free, g = assemble(st, fu, fv, pin_p=pin_p)
    ng = int(m.gidx.max()) + 1

    rng = np.random.default_rng(seed)
    xg = rng.standard_normal(ng*NV)*free          # global, free dofs only
    xl = xg[g]                                    # scatter to local -- CONTINUOUS
    yl = apply_A(st, xl, fu, fv, pin_p=pin_p)     # matrix-free reference
    yg_ref = _to_global(g, ng, yl)
    yg = A @ xg

    scale = np.abs(yg_ref[free]).max()
    err = np.abs((yg - yg_ref)[free]).max()/scale
    asym = abs(A - A.T).max()/abs(A).max()
    if verbose:
        print(f'  N={N} {ex}x{ey}  ndof={ng*NV}  free={int(free.sum())}  '
              f'nnz={A.nnz}  density={A.nnz/(ng*NV)**2:.4f}')
        print(f'    matvec vs apply_A : {err:.3e}')
        print(f'    asymmetry         : {asym:.3e}')
    return A, free, err, asym


if __name__ == '__main__':
    print('F0 gate -- assembled A vs matrix-free apply_A\n')
    ok = True
    for N, ex, ey in ((4, 2, 2), (6, 3, 2), (8, 3, 3)):
        A, free, err, asym = gate(N=N, ex=ex, ey=ey)
        # SPD via DENSE eigenvalues.  eigsh(which='SA') does not converge on an
        # operator this ill-conditioned without shift-invert, and these gate
        # cases are small enough that dense is both cheaper and definitive --
        # and it hands over kappa, which F1 needs anyway.
        Af = np.asarray(A[free][:, free].todense())
        ev = np.linalg.eigvalsh(Af)
        lo, hi = ev[0], ev[-1]
        print(f'    lambda_min/max    : {lo:.3e} / {hi:.3e}   kappa = {hi/lo:.2e}'
              f'   {"SPD" if lo > 0 else "NOT SPD"}')
        ok &= (err < 1e-12) and (asym < 1e-12) and (lo > 0)
    print(f'\nF0 GATE: {"PASS" if ok else "FAIL"}   (err<1e-12, asym<1e-12, SPD)')
