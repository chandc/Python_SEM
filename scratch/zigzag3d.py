"""Is there a 2D-style node-to-node zigzag in the 3D channel fields?

Metrics on the physical-space velocity of (a) FOSLS run01 checkpoints (legacy
weighting, dt=8e-4) and (b) the fractional-step state on the same mesh:
  * sign-change fraction of consecutive GLL-node differences of u along y
    (wall-normal lines at every x node and z plane), near-wall elements and all
    -- a smooth profile gives few alternations, a zigzag gives ~1;
  * fraction of modal (Legendre) energy in the top two polynomial modes per
    element, along y and along x, for u, v, w -- the checkerboard signature.
"""
import os, sys, glob
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, _R); os.chdir(_R)
import numpy as np
from lssem2d.lgl import lgl_nodes, lgl_weights
from lssem3d import fourier as FR

N = 8; NZ = 32


def legendre_matrix(N):
    x = np.asarray(lgl_nodes(N)); w = np.asarray(lgl_weights(N))
    P = np.polynomial.legendre.legvander(x, N)                 # (N+1 nodes, N+1 modes)
    gam = 2.0/(2*np.arange(N+1) + 1.0); gam[N] = 2.0/N
    return (P*w[:, None]).T/gam[:, None]                     # modal = M @ nodal


def metrics(U):
    """U physical (nelem, n, n, 3, nz)."""
    M = legendre_matrix(N)
    out = {}
    u = U[..., 0, :]
    d = np.diff(u, axis=2)                                    # along y within each element
    sc = np.sign(d[:, :, 1:, :])*np.sign(d[:, :, :-1, :]) < 0
    # wall elements: y-element index from the element numbering (6 x 18 mesh: e = ix*18 + iy ?) -> use ynod
    out['signchg_all'] = sc.mean()
    # modal energy fractions
    for f, nm in enumerate('uvw'):
        q = U[..., f, :]
        ay = np.einsum('ki,eijz->ekjz', M, q)                 # modes along i? careful: axis1 = x-index i, axis2 = y-index j
        ax_ = np.einsum('ki,eijz->ekjz', M, q)                # modes along x (axis 1)
        ay_ = np.einsum('kj,eijz->eikz', M, q)                # modes along y (axis 2)
        ex = (ax_**2).sum(axis=(0, 2, 3)); ey = (ay_**2).sum(axis=(0, 1, 3))
        out[f'top2_x_{nm}'] = ex[-2:].sum()/ex.sum(); out[f'top2_y_{nm}'] = ey[-2:].sum()/ey.sum()
        out[f'top1_y_{nm}'] = ey[-1]/ey.sum()
    return out


if __name__ == '__main__':
    from lssem2d.mesh import build_channel
    m = build_channel(np.pi, 2.0, 6, 18, N, bcs=(0, 0, 1, 1)); m.periodic_x = np.pi; m.compute_global_indices()
    ywall = (m.ynod.min(axis=1) < 0.12) | (m.ynod.max(axis=1) > 1.88)      # first element off each wall
    cases = [('FS state t=15.95', 'results/minchan_re180_E/state_t15.95.npz', 'U')]
    for ck in os.environ.get('CKS', '0000100,0000300,0001000,0003000,0005000,0006200').split(','):
        cases.append((f'FOSLS run01 ckpt {ck}', f'scratch/run01_ck/checkpoint_{ck}.npz', 'U'))
    print(f'{"case":26s} {"t":>6} {"signchg all":>11} {"signchg wall":>12} | top-2-mode energy fraction along y: {"u":>8} {"v":>8} {"w":>8} | along x: {"u":>8} {"v":>8} {"w":>8}')
    for lab, f, key in cases:
        if not os.path.exists(f): print(lab, 'missing'); continue
        z = np.load(f); A = z[key]
        if A.shape[-2] == 14:                                     # split-real FOSLS state -> complex -> velocity
            Uc = A[..., :7, :] + 1j*A[..., 7:, :]; Uc = Uc[..., :3, :]
        else:
            Uc = A
        U = FR.to_physical(np.ascontiguousarray(Uc), NZ)
        mt = metrics(U)
        dw = np.diff(U[ywall][..., 0, :], axis=2); scw = (np.sign(dw[:, :, 1:, :])*np.sign(dw[:, :, :-1, :]) < 0).mean()
        t = float(z['t']) if 't' in z.files else float('nan')
        print(f'{lab:26s} {t:6.2f} {mt["signchg_all"]:11.3f} {scw:12.3f} | {mt["top2_y_u"]:8.1e} {mt["top2_y_v"]:8.1e} {mt["top2_y_w"]:8.1e} | {mt["top2_x_u"]:8.1e} {mt["top2_x_v"]:8.1e} {mt["top2_x_w"]:8.1e}', flush=True)
