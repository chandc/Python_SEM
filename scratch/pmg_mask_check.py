"""Does PMG's finest level use the SAME mask the outer solve uses?

A preconditioner must be defined on exactly the space the operator is.  If a
level pins a dof the solve leaves free, M returns zero there, M is singular on
the space CG searches, and the V-cycle under-performs for a reason no amount of
smoother or coarse-grid tuning can fix.

build_mask(pin_p=True) -- PMG's default -- pins pressure at EVERY Fourier mode.
The physics needs it pinned only at k = 0: for k != 0 the ik*p term in the
z-momentum row already determines pressure uniquely.  A driver that pins at
mode 0 only therefore disagrees with PMG's levels, and PMG is the one that is
wrong.
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import lssem3d; lssem3d.set_backend('numpy')
from lssem2d.mesh import build_channel
from lssem3d import bc as BC, operator as OP, precond as PC, fourier as FR
from lssem3d import timestep as T

N, ne, nz = 8, 3, 16
L = 2*np.pi
nk = nz//2 + 1
m = build_channel(L, L, ne, ne, N, bcs=(0, 0, 0, 0))
m.periodic_x = L; m.periodic_y = L; m.compute_global_indices()
sm = BC.build_mask(m, nk, pin_p=False, nz=nz)
BC.pin_dof(m, sm, OP.P_, 0)
kz = FR.wavenumbers(nz, L)
c = T.implicit_coeff(0.0039, 0)
rw = OP.momentum_row_weights(c)

old = PC.PMG(m, nk, nz, 6.25e-4, c, kz, rw=rw, orders=(8, 4, 2), deg=6)
new = PC.PMG(m, nk, nz, 6.25e-4, c, kz, rw=rw, orders=(8, 4, 2), deg=6, mask=sm)
for nm, P in (('pin_p=True (old default)', old), ('mask= passed (fixed)', new)):
    same = np.array_equal(P.levels[0].mask, sm)
    bad = int(((sm == 1) & (P.levels[0].mask == 0)).sum())
    print(f'  {nm:<26} level-0 mask == solve mask: {same}'
          f'   dofs free in solve but pinned by PMG: {bad}')

mw = PC.S3.multiplicity_weight(m, sm.shape)
rng = np.random.default_rng(0)
x = PC.S3.gs(m, rng.standard_normal(sm.shape))*sm
y = PC.S3.gs(m, rng.standard_normal(sm.shape))*sm
print()
for nm, P in (('old', old), ('new', new)):
    s = abs(np.sum(x*P(y)*mw) - np.sum(y*P(x)*mw))/abs(np.sum(x*P(x)*mw))
    print(f'  {nm}: V-cycle runs, symmetry (continuous, mw-weighted) {s:.2e}')

lev = new.levels[-1]
dc = new.coarse
B = dc.B[0]
g = np.ones(B.shape[1])
out = dc.lu[0](g)
A = np.empty((B.shape[1], B.shape[1]))
mwr = lev.mw.ravel()
for a in range(B.shape[1]):
    A[:, a] = B.T @ (lev.A(B[:, a].reshape(lev.shape)).ravel()*mwr)
A = 0.5*(A + A.T)
print(f'\n  Cholesky coarse solve residual ||A x - b||: '
      f'{float(np.abs(A @ out - g).max()):.2e}')
