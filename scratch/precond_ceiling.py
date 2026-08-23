"""DOES NOT WORK -- kept as a record of an approach that fails, and why.

Intent: measure the ceiling a block preconditioner could reach, by comparing
condition numbers of the dense element-local operator under point-Jacobi, a
14x14 nodal field-block, and an exact element inverse.  If the exact version
barely beat point-Jacobi, FDM (an approximation to it) could not, and the idea
would be dead for minutes of CPU instead of days of work.

WHY IT FAILS.  A single element with periodic wrap and no boundary mask is
nearly singular: kappa(A) ~ 1e12.  At that conditioning the smallest
"eigenvalue" is set by the null-space cutoff rather than by the
discretisation, and np.linalg.pinv is numerical noise.  The output says so
plainly if read carefully -- the exact element inverse should give kappa = 1
BY CONSTRUCTION and instead reports 8.8e12; kappa under Jacobi DECREASES with
N, which is backwards; and the field-block "gain" jumps 230x, 40x, 79x with no
pattern.  Every one of those is the test measuring its own tolerance.

THE SOUND VERSION measures CG ITERATIONS, not condition numbers, on the real
assembled operator with real boundary conditions -- iterations are the
quantity that matters and they are robust where eigenvalues of a 1e12-
conditioned matrix are not.  That needs the exact 14x14 nodal blocks, which
cannot be probed cheaply (a spectral element's stencil spans the element, so
colouring needs (N+1)^2 * 14 probe vectors) and should instead be assembled
analytically, the way jacobi_diagonal_analytic already does for the diagonal.
That is real work, not a quick screen.

Original docstring follows.

What is the BEST any block preconditioner could do on this operator?

    python scratch/precond_ceiling.py [Nlist] [nz]

Before building a fast-diagonalisation preconditioner, measure the ceiling it
is aiming at.  FDM is an APPROXIMATION to an exact element-local inverse, so
whatever the exact inverse achieves is an upper bound on what FDM can.  If the
exact version barely beats point-Jacobi, the approximation certainly will not,
and the whole idea is dead for a few minutes of CPU rather than days of work.

Four preconditioners on the dense element-local operator A = L^T W L, one
element, no assembly (normal_op with mesh=None is the unassembled operator, as
its docstring says -- for single-element tests exactly like this one):

  none          kappa(A)                    the raw problem
  point-Jacobi  kappa(D^-1 A)               what the solver uses today
  field-block   kappa(F^-1 A)               14x14 block per node -- the
                                            nodal block-Jacobi idea
  element-block kappa(E^-1 A)               the EXACT element inverse, which
                                            is what FDM approximates: the
                                            ceiling for FDM, Schwarz, and
                                            element block-Jacobi alike

CG iterations scale like sqrt(kappa), so the useful comparison is the ratio of
square roots, and how it CHANGES with N -- a preconditioner that wins a
constant factor is worth little, one whose advantage grows with N is worth
building, because that growth is the whole objection to point-Jacobi.
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import lssem3d; lssem3d.set_backend('numpy')
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem3d import operator as OP, solver3d as S3, bc as BC, fourier as FR
from lssem3d import timestep as T

Ns = [int(v) for v in (sys.argv[1].split(',') if len(sys.argv) > 1
                       else ['4', '6', '8'])]
nz = int(sys.argv[2]) if len(sys.argv) > 2 else 2
L = 2*np.pi
dt = 0.0039
c = T.implicit_coeff(dt, 0)
print(f'one element, Nz={nz}, dt={dt} (c={c:.0f})\n')
print(f'{"N":>3} {"dof":>6} {"none":>11} {"Jacobi":>11} {"field-blk":>11} '
      f'{"elem-blk":>11}   {"sqrt gain vs Jacobi":>22}')
prev = None
for N in Ns:
    m = build_channel(L, L, 1, 1, N, bcs=(0, 0, 0, 0))
    m.periodic_x = L; m.periodic_y = L; m.compute_global_indices()
    nk = nz//2 + 1
    D = diff_matrix(N)
    kz = FR.wavenumbers(nz, L)
    shape = (1, N+1, N+1, OP.NVAR_R, nk)
    n = int(np.prod(shape))
    rw = OP.momentum_row_weights(c)
    # dense A, column by column
    A = np.empty((n, n))
    e = np.zeros(shape)
    for j in range(n):
        e.flat[j] = 1.0
        A[:, j] = S3.normal_op(e, D, m.facx, m.facy, kz, 6.25e-4, c,
                               None, None, m.wq, 0.0, rw).reshape(-1)
        e.flat[j] = 0.0
    A = 0.5*(A + A.T)                       # symmetrise the round-off

    def kappa(M_inv_A):
        w = np.linalg.eigvalsh(0.5*(M_inv_A + M_inv_A.T))
        w = w[w > w.max()*1e-13]            # drop the operator's null space
        return w.max()/w.min()

    k_none = kappa(A)
    d = np.diag(A).copy(); d[d <= 0] = 1.0
    k_jac = kappa(A/d[:, None])
    # field-block: 14x14 coupling among fields at each (node, mode)
    Af = A.reshape(shape + shape)
    P = np.zeros_like(A)
    Pv = P.reshape(shape + shape)
    idx = np.ndindex(1, N+1, N+1, nk)
    for (a, b, cc, k) in [(0, i, j, kk) for i in range(N+1)
                          for j in range(N+1) for kk in range(nk)]:
        blk = Af[a, b, cc, :, k, a, b, cc, :, k]
        Pv[a, b, cc, :, k, a, b, cc, :, k] = np.linalg.pinv(blk)
    k_fld = kappa(P @ A)
    k_ele = kappa(np.linalg.pinv(A) @ A)    # exact: kappa -> 1 by construction
    # the honest element-block number: exact inverse gives 1, so instead report
    # what a SINGLE element block buys in a multi-element setting -- here, with
    # one element, it IS the exact inverse.  Kept to show the bound is 1.
    g = np.sqrt(k_jac/k_fld)
    print(f'{N:>3} {n:>6} {k_none:>11.3e} {k_jac:>11.3e} {k_fld:>11.3e} '
          f'{k_ele:>11.3e}   field-block {g:>6.2f}x fewer CG its')
    if prev is not None:
        print(f'    kappa(Jacobi) grew {k_jac/prev[0]:>5.2f}x, '
              f'kappa(field-block) grew {k_fld/prev[1]:>5.2f}x  '
              f'-- {"field-block is winning MORE as N rises" if k_fld/prev[1] < k_jac/prev[0] else "no relative gain with N"}')
    prev = (k_jac, k_fld)
print('\n  elem-blk is 1.0 by construction here (one element = exact inverse);'
      '\n  it is shown to make the ceiling explicit.  The number that decides'
      '\n  whether to build FDM is whether the field-block advantage GROWS'
      '\n  with N -- a constant factor is not worth the work.')
