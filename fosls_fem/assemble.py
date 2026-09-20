"""Assemble the sparse normal equations L^T L and L^T f, and apply Dirichlet data.

Global dof numbering is node-major: dof `NF*node + field`, matching the
element ordering in `element_q1`.  Assembly is COO triplets into
`scipy.sparse`; nothing clever, because the clever part of this code is that
the element matrices are exact and the whole system is SPD by construction.

DIRICHLET BY ELIMINATION, not penalty.  A penalty leaves the matrix SPD but
ill-conditioned by the penalty factor, which would confound the conditioning
measurements this code exists to make.  Elimination moves the known values to
the right-hand side and replaces the row and column with the identity, keeping
symmetry exactly.
"""
import numpy as np
import scipy.sparse as sp

from .element_q1 import element_matrix, element_newton, element_residual, NF, NN


def assemble(mesh, flin, coef, rhs=None, element=element_matrix, **kw):
    """Global (L^T L, L^T f) for a QuadMesh.

    flin : (nnode, 2) linearisation velocities
    rhs  : (nnode, NF) nodal row right-hand sides, or None
    """
    nd = mesh.ndof
    rows, cols, vals = [], [], []
    b = np.zeros(nd)
    flin = np.asarray(flin, float)
    rhs = None if rhs is None else np.asarray(rhs, float)
    for q in mesh.quads:
        Ae, be = element(mesh.xy[q], flin[q], coef,
                         None if rhs is None else rhs[q], **kw)
        g = (NF*q[:, None] + np.arange(NF)[None, :]).ravel()    # 16 global dofs
        rows.append(np.repeat(g, len(g)))
        cols.append(np.tile(g, len(g)))
        vals.append(Ae.ravel())
        b[g] += be
    A = sp.coo_matrix((np.concatenate(vals),
                       (np.concatenate(rows), np.concatenate(cols))),
                      shape=(nd, nd)).tocsr()
    return A, b


def dof_index(nodes, field):
    return NF*np.asarray(nodes, int) + field


def apply_dirichlet(A, b, fixed, values):
    """Eliminate constrained dofs, preserving symmetry exactly.

    `fixed` is an array of global dof indices, `values` their prescribed data.
    """
    fixed = np.asarray(fixed, int)
    values = np.asarray(values, float)
    A = A.tolil(copy=True)
    b = b.copy()
    free = np.setdiff1d(np.arange(A.shape[0]), fixed)
    # move the known columns to the right-hand side, then clear them
    b[free] -= np.asarray(A[free][:, fixed] @ values).ravel()
    for d in fixed:
        A.rows[d], A.data[d] = [d], [1.0]
    A = A.tocsc()
    for d in fixed:                       # clear the columns too, for symmetry
        s, e = A.indptr[d], A.indptr[d+1]
        keep = A.indices[s:e] == d
        A.data[s:e] = np.where(keep, 1.0, 0.0)
    A.eliminate_zeros()
    b[fixed] = values
    return A.tocsr(), b


def assemble_newton(mesh, U, coef, f=None):
    """Global Newton step: (J^T J) dU = -J^T R, with R the TRUE nonlinear residual.

    `element_newton` builds both from one routine, so the residual and the
    Jacobian cannot drift out of step -- the usual failure mode of a hand-rolled
    Newton.  Returns (A, b) with A dU = b.
    """
    nd = mesh.ndof
    rows, cols, vals = [], [], []
    b = np.zeros(nd)
    U = np.asarray(U, float)
    f = None if f is None else np.asarray(f, float)
    for q in mesh.quads:
        g = (NF*q[:, None] + np.arange(NF)[None, :]).ravel()
        Ae, be = element_newton(mesh.xy[q], U[g], coef,
                                None if f is None else f[q])
        rows.append(np.repeat(g, len(g)))
        cols.append(np.tile(g, len(g)))
        vals.append(Ae.ravel())
        b[g] += be
    A = sp.coo_matrix((np.concatenate(vals),
                       (np.concatenate(rows), np.concatenate(cols))),
                      shape=(nd, nd)).tocsr()
    return A, b


def functional(mesh, U, coef, f=None):
    """J(U_h) = sum over elements of int |L(U) - f|^2, the FOSLS functional."""
    U = np.asarray(U, float)
    f = None if f is None else np.asarray(f, float)
    tot = 0.0
    for q in mesh.quads:
        g = (NF*q[:, None] + np.arange(NF)[None, :]).ravel()
        tot += element_residual(mesh.xy[q], U[g], coef,
                                None if f is None else f[q])
    return tot
