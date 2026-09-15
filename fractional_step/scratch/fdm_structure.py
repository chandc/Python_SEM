"""Is the VVP least-squares operator separable field-block by field-block?

    python scratch/fdm_structure.py [N] [nz]

Fast diagonalisation requires the operator to be a SUM OF TENSOR PRODUCTS.
Classical FDM does not apply to the full VVP operator, which couples all seven
fields at every node -- but it does not have to.  A field-block-diagonal
preconditioner needs only each DIAGONAL block to be separable, leaving the
field coupling to CG.

Worked out from the row definitions, u appears in exactly four rows -- ux
(continuity), ik*u (vorticity-y), -uy (vorticity-z) and c*u (momentum-x) -- so
with the row weights, and writing K1 = D^T diag(w) D and M1 = diag(w):

    A_uu = (fx/fy) K1 (x) M1  +  (fy/fx) M1 (x) K1
                              +  ((1 + kz^2)/(fx*fy)) M1 (x) M1

a Helmholtz operator.  Note the momentum row weight 1/c^2 cancels the c^2 from
c*u, leaving a clean unit mass term -- which is why the LEGACY SCALING is what
makes this separable at all.

This script does not trust that derivation: it builds the block numerically by
applying the real operator to unit vectors and compares entrywise.  If they
agree to machine precision, FDM applies EXACTLY to the diagonal blocks and the
only approximation in the preconditioner is dropping the field coupling.
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import lssem3d; lssem3d.set_backend('numpy')
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix, lgl_weights
from lssem3d import operator as OP, solver3d as S3, fourier as FR, timestep as T

N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
nz = int(sys.argv[2]) if len(sys.argv) > 2 else 4
L = 2*np.pi
m = build_channel(L, L, 1, 1, N, bcs=(0, 0, 0, 0))
m.periodic_x = L; m.periodic_y = L; m.compute_global_indices()
nk = nz//2 + 1
D, w = diff_matrix(N), lgl_weights(N)
kz = FR.wavenumbers(nz, L)
dt = 0.0039
c = T.implicit_coeff(dt, 0)
nu = 6.25e-4
rw = OP.momentum_row_weights(c)
fx, fy = float(m.facx[0]), float(m.facy[0])
n1 = N + 1
shape = (1, n1, n1, OP.NVAR_R, nk)
print(f'N={N}, Nz={nz} (nk={nk}), fx={fx:.4f}, fy={fy:.4f}, c={c:.1f}\n')

# ---- numerically extract the REAL-part diagonal blocks, per field, per mode
ntot = int(np.prod(shape))
A = np.empty((ntot, ntot))
e = np.zeros(shape)
for j in range(ntot):
    e.flat[j] = 1.0
    A[:, j] = S3.normal_op(e, D, m.facx, m.facy, kz, nu, c, None, None,
                           m.wq, 0.0, rw).reshape(-1)
    e.flat[j] = 0.0
A = A.reshape(shape + shape)

K1 = D.T @ np.diag(w) @ D
M1 = np.diag(w)

def predicted(field, k):
    """The tensor-product form each diagonal block should have."""
    kk = kz[k]**2
    if field in (OP.U_, OP.V_, OP.W_):
        return (fx/fy)*np.kron(K1, M1) + (fy/fx)*np.kron(M1, K1) \
               + ((1.0 + kk)/(fx*fy))*np.kron(M1, M1)
    if field in (OP.OX_, OP.OY_, OP.OZ_):
        # Row 7 is oxx + oyy + ik*oz, so ox picks up ONLY d/dx, oy only d/dy,
        # and oz only kz^2 -- each vorticity component sees one direction, not
        # both.  The vorticity rows contribute the unit mass, and the nu*ik
        # terms in the momentum rows carry the 1/c^2 momentum weight.
        w7 = OP.ROW7_WEIGHT
        stiff = {OP.OX_: w7*(fx/fy)*np.kron(K1, M1),
                 OP.OY_: w7*(fy/fx)*np.kron(M1, K1),
                 OP.OZ_: w7*kk/(fx*fy)*np.kron(M1, M1)}[field]
        return stiff + ((1.0 + nu*nu*kk/(c*c))/(fx*fy))*np.kron(M1, M1)
    return (1.0/(c*c))*((fx/fy)*np.kron(K1, M1) + (fy/fx)*np.kron(M1, K1)
                        + (kk/(fx*fy))*np.kron(M1, M1))

names = {OP.U_: 'u', OP.V_: 'v', OP.W_: 'w', OP.OX_: 'ox', OP.OY_: 'oy',
         OP.OZ_: 'oz', OP.P_: 'p'}
print(f'{"field":>6} {"mode":>5} {"||block||":>12} {"rel err vs tensor-product form":>32}')
for f in (OP.U_, OP.V_, OP.W_, OP.OX_, OP.OY_, OP.OZ_, OP.P_):
    for k in range(nk):
        blk = A[0, :, :, f, k, 0, :, :, f, k].reshape(n1*n1, n1*n1)
        pred = predicted(f, k)
        rel = np.abs(blk - pred).max()/max(np.abs(blk).max(), 1e-300)
        flag = '  EXACT' if rel < 1e-12 else ('  close' if rel < 1e-3 else '  NO')
        print(f'{names[f]:>6} {k:>5} {np.abs(blk).max():>12.4e} {rel:>32.3e}{flag}')

print('\n  How much of the operator is the field COUPLING that FDM would drop?')
for k in range(nk):
    full = A[0, :, :, :, k, 0, :, :, :, k].reshape(n1*n1*OP.NVAR_R//1, -1) \
        if False else None
Ar = A.reshape(ntot, ntot)
Bd = np.zeros_like(Ar)
Bv = Bd.reshape(shape + shape)
for f in range(OP.NVAR_R):
    for k in range(nk):
        Bv[0, :, :, f, k, 0, :, :, f, k] = A[0, :, :, f, k, 0, :, :, f, k]
off = np.abs(Ar - Bd).max()/np.abs(Ar).max()
fro = np.linalg.norm(Ar - Bd)/np.linalg.norm(Ar)
print(f'   off-block max  {off:.3f}   off-block Frobenius fraction {fro:.3f}')
print('   (that is what CG must still handle; a preconditioner only has to\n'
      '    cluster the spectrum, not invert the operator)')
