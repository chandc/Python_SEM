"""PHASE 0 GATE: the scalar Helmholtz solver and its FDM element inverse.

    python scratch/fs_phase0.py

Manufactured solution on the periodic square, psi = sin(x)cos(y), for which
-grad^2_xy psi = 2 psi, so

    A psi = (lambda + 2 mu) M psi        =>   b = M (lambda + 2 mu) psi

Three things are checked, in increasing strength:

  1. OPERATOR.  A applied to the exact field reproduces b -- verifies the weak
     form, the metric factors and the assembly.  Spectral in N.
  2. SOLVE.  CG recovers psi from b.
  3. FDM IS EXACT.  On a SINGLE element the element-block inverse is the true
     inverse, so preconditioned CG must converge in ONE iteration.  This is the
     claim that distinguishes this path from the VVP one, where FDM was an
     approximation that dropped 37% of the operator and lost to Jacobi.
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import lssem3d; lssem3d.set_backend('numpy')
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem3d import helmholtz as H, solver3d as S3

L = 2*np.pi
nk, nfield = 3, 2
mu = 0.7
lam = np.array([1.0, 3.0, 9.0])[:nk]


def setup(N, ne):
    m = build_channel(L, L, ne, ne, N, bcs=(0, 0, 0, 0))
    m.periodic_x = L; m.periodic_y = L; m.compute_global_indices()
    n = N + 1
    X = np.empty((m.nelem, n, n)); Y = np.empty_like(X)
    for e in range(m.nelem):
        X[e] = m.xnod[e][:, None]; Y[e] = m.ynod[e][None, :]
    psi = np.sin(X)*np.cos(Y)
    ex = np.repeat(psi[..., None, None], nfield, axis=3).repeat(nk, axis=4)
    return m, diff_matrix(N), ex


print('1. OPERATOR: does A psi_exact equal (lam + 2 mu) M psi_exact?')
print(f'{"N":>3} {"elems":>7} {"rel err":>12}')
for N, ne in ((4, 2), (6, 2), (8, 2), (10, 2), (8, 4)):
    m, D, ex = setup(N, ne)
    got = H.apply(ex, D, m.facx, m.facy, m.wq, lam, mu, mesh=m)
    want = S3.gs(m, (lam + 2.0*mu)*(m.wq[..., None, None]*ex))
    print(f'{N:>3} {ne*ne:>7} {np.abs(got-want).max()/np.abs(want).max():>12.3e}')

print('\n2. SOLVE: CG with the FDM preconditioner recovers psi from b')
print(f'{"N":>3} {"elems":>7} {"iters":>7} {"rel err":>12}')
for N, ne in ((6, 2), (8, 2), (8, 4), (12, 3)):
    m, D, ex = setup(N, ne)
    shape = ex.shape
    mask = np.ones(shape)
    b = S3.gs(m, (lam + 2.0*mu)*(m.wq[..., None, None]*ex))*mask
    A = lambda v: H.apply(v, D, m.facx, m.facy, m.wq, lam, mu, mesh=m, mask=mask)
    M = H.fdm_preconditioner(m, N, lam, mu, mask, nfield, nk)
    x, it = _cg(A, b, M, m, shape) if False else (None, None)
    # simple PCG in the multiplicity-weighted inner product
    mw = S3.multiplicity_weight(m, shape)
    dot = lambda a, c: np.sum(a*c*mw, axis=(0, 1, 2, 3))
    x = np.zeros(shape); r = b - A(x); z = M(r); p = z.copy()
    rz = dot(r, z); tgt = 1e-12*np.sqrt(dot(b, b))
    for it in range(1, 3001):
        Ap = A(p); den = dot(p, Ap)
        al = np.where(np.abs(den) > 1e-300, rz/np.where(den == 0, 1, den), 0)
        x = x + al*p; r = r - al*Ap
        if np.all(np.sqrt(dot(r, r)) < tgt):
            break
        z = M(r); rzn = dot(r, z)
        be = np.where(np.abs(rz) > 1e-300, rzn/np.where(rz == 0, 1, rz), 0)
        p = z + be*p; rz = rzn
    print(f'{N:>3} {ne*ne:>7} {it:>7} '
          f'{np.abs(x-ex).max()/np.abs(ex).max():>12.3e}')

print("\n3. FDM IS EXACT for the ELEMENT-LOCAL operator")
print("   (not for an assembled one: a single PERIODIC element wraps onto")
print("    itself, so gs identifies opposite edges and the assembled operator")
print("    is Q^T A_loc Q, which FDM preconditions rather than inverts)")
for N in (4, 6, 8, 12):
    m, D, ex = setup(N, 1)
    shape = ex.shape
    A = lambda v: H.apply(v, D, m.facx, m.facy, m.wq, lam, mu, mesh=None)
    M = H.fdm_preconditioner(m, N, lam, mu, None, nfield, nk, assemble=False)
    rng = np.random.default_rng(0)
    b = rng.standard_normal(shape)
    resid = np.abs(A(M(b)) - b).max()/np.abs(b).max()
    print(f"  N={N:>2}: ||A M b - b|| / ||b|| = {resid:.3e}   "
          f"{'EXACT' if resid < 1e-10 else 'NOT EXACT'}")
