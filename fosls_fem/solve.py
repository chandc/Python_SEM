"""Solvers for the assembled FOSLS normal equations.

L^T L is symmetric positive definite by construction, so no pivoting is needed
and CG is the natural iterative choice.  Two paths: a sparse direct factorisation
for 2D problems that fit, and CG with smoothed-aggregation AMG beyond that.

The AMG path is not just for size.  The FOSLS claim under test here is
Cai-Manteuffel-McCormick H^1 norm-equivalence, whose practical consequence is
that the discrete operator conditions like a Laplacian -- so multigrid should be
optimal, with an iteration count independent of h.  Measuring that iteration
count IS part of testing the theory, not merely an implementation detail.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def solve_direct(A, b):
    return spla.spsolve(A.tocsc(), b)


def solve_amg(A, b, tol=1e-10, maxiter=500, x0=None):
    """CG preconditioned by smoothed-aggregation AMG.  Returns (x, iters)."""
    import pyamg
    ml = pyamg.smoothed_aggregation_solver(A.tocsr(), max_coarse=50)
    it = [0]
    def cb(_):
        it[0] += 1
    x = ml.solve(b, x0=x0, tol=tol, maxiter=maxiter, accel='cg', callback=cb)
    return x, it[0]


def condition_estimate(A, k=1):
    """Extremal eigenvalues of an SPD sparse matrix, for the h-scaling study."""
    lo = spla.eigsh(A, k=k, which='SA', return_eigenvectors=False,
                    tol=1e-6, maxiter=5000)
    hi = spla.eigsh(A, k=k, which='LA', return_eigenvectors=False,
                    tol=1e-6, maxiter=5000)
    return float(lo.min()), float(hi.max())
