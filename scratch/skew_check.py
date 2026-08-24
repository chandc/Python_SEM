"""Does the skew form conserve energy when div u is NOT zero?

That is the property in question.  For a divergence-free field both forms
conserve; the test only means something on a field that carries divergence,
which is exactly the state the projection path is in (3.6e-04).
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import lssem3d; lssem3d.set_backend('numpy')
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem3d import convect as CV, fourier as FR, deriv as DV

L = 2*np.pi
N, ne, nz = 8, 3, 8
m = build_channel(L, L, ne, ne, N, bcs=(0, 0, 0, 0))
m.periodic_x = L; m.periodic_y = L; m.compute_global_indices()
nk = nz//2+1
kz = FR.wavenumbers(nz, L)
D, wq = diff_matrix(N), m.wq
n = N+1
X = np.empty((m.nelem, n, n)); Y = np.empty_like(X)
for e in range(m.nelem):
    X[e] = m.xnod[e][:, None]; Y[e] = m.ynod[e][None, :]
z = (L/nz)*np.arange(nz)

for label, mk in (('divergence-FREE (TGV)', 0), ('carrying divergence', 1)):
    up = np.zeros((m.nelem, n, n, 3, nz))
    up[..., 0, :] = np.sin(X)[..., None]*np.cos(Y)[..., None]*np.cos(z)
    up[..., 1, :] = -np.cos(X)[..., None]*np.sin(Y)[..., None]*np.cos(z)
    if mk:                                   # add a non-solenoidal part
        up[..., 0, :] += 0.05*np.sin(2*X)[..., None]*np.ones_like(z)
    Uc = FR.to_modes(up)
    u, v, w = (Uc[..., i:i+1, :] for i in range(3))
    dv = DV.ddx(u, D, m.facx) + DV.ddy(v, D, m.facy) + 1j*kz*w
    reldiv = np.sqrt(np.sum(np.abs(dv)**2*wq[..., None, None]) /
                     np.sum(np.abs(Uc)**2*wq[..., None, None]))
    print(f'{label}:  ||div u||/||u|| = {reldiv:.2e}')
    # PHYSICAL space: no Parseval factors to get wrong, and this is the
    # quantity that actually matters -- dE/dt from convection alone.
    up_ = FR.to_physical(Uc, nz)
    E = float(np.sum(np.sum(up_**2, axis=-2)*wq[..., None]))
    for form in (False, True):
        H = -CV.convective(Uc, D, m.facx, m.facy, kz, nz, skew=form)
        Hp = FR.to_physical(H, nz)
        pr = float(np.sum(np.sum(up_*Hp, axis=-2)*wq[..., None]))
        print(f'    {"skew " if form else "advec"}:  '
              f'2<u,H>/E = {2*pr/E:+.4e}')
    print()
