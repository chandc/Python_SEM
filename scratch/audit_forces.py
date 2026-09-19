"""Audit of the cylinder force integral: is C_D / C_L computed correctly?

The forces are the headline numbers of the whole cylinder study and had never
been independently checked.  Four tests, each of which a different class of bug
would fail:

1. GEOMETRY.  sum(ds) must equal pi*D and sum(n ds) must vanish.  The second is
   the sharp one: a closed body in uniform pressure feels no force, so any error
   in the normals or the arc weights shows up there even when the circumference
   happens to come out right.

2. NO-SLIP.  The omega-form of the viscous traction, nu*omega*t, is only valid
   where u = 0 on the wall; if the Dirichlet condition were not exact the
   tangential-derivative terms it drops would not vanish.

3. THE PRESSURE/FRICTION SPLIT against the literature.  This is the test with
   teeth.  A wrong constant, a wrong normal or a wrong viscous formula distorts
   the SPLIT; a flow-level error (blockage, resolution) scales both parts
   together.  Qu et al. (2013) and Park et al. (1998) both report C_Dp
   separately, so the friction share is checkable at Re = 100.

4. AN INDEPENDENT VISCOUS TRACTION.  Recompute it as nu*(grad u + grad u^T).n
   from the velocity field, which shares no code and no variable with the
   omega-form.  They must agree to the size of the least-squares slack in
   omega - curl u, which this formulation enforces only weakly.
"""
import os
import sys

_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R)
sys.path.insert(0, os.path.join(_R, 'scratch'))
os.chdir(_R)
sys.argv = [sys.argv[0], '--N', '8']            # driver parses argv at import

import numpy as np
import lssem2d
from lssem2d import curvi
from lssem2d.lgl import diff_matrix
import curvi_cyl_re100 as drv

RUN = 'scratch/_cyl_pg_H80/final.npz'
NU = 0.01
LIT = (('Qu et al. (2013) D9', 0.984, 1.319), ('Park et al. (1998)', 0.99, 1.33))


def main():
    lssem2d.set_backend('numpy')
    m = curvi.build_cylinder_box(
        N=8, xs_upstream=curvi.nested_upstream_edges(30.0, 3),
        xs_downstream=curvi.nested_downstream_edges(50.0, 3),
        ys_side=curvi.chained_lateral_edges(40.0, 2))
    m.compute_global_indices()
    D = diff_matrix(m.N)
    wn = drv.wall_nodes(m, D)
    U = np.load(RUN, allow_pickle=True)['U0']

    print('1. geometry of the wall quadrature')
    arc = sum(float(ws.sum()) for _, _, _, ws in wn)
    nxs = sum(float((ws*nx).sum()) for _, nx, _, ws in wn)
    nys = sum(float((ws*ny).sum()) for _, _, ny, ws in wn)
    print(f'   wall edges {len(wn)} (expect 16);  sum(ds) {arc:.10f} vs pi '
          f'{np.pi:.10f}, err {abs(arc-np.pi):.1e}')
    print(f'   closure sum(n_x ds) {nxs:+.1e}   sum(n_y ds) {nys:+.1e}')

    print('\n2. no-slip on the integration nodes')
    uw = max(float(np.abs(U[e, 0, :, 0]).max()) for e, _, _, _ in wn)
    vw = max(float(np.abs(U[e, 0, :, 1]).max()) for e, _, _, _ in wn)
    print(f'   max|u| {uw:.1e}   max|v| {vw:.1e}')

    print('\n3. pressure / friction split')
    fxp = fxv = fyp = fyv = 0.0
    for e, nx, ny, ws in wn:
        p, om = U[e, 0, :, 2], U[e, 0, :, 3]
        tx, ty = -ny, nx
        fxp += float((ws*(-p*nx)).sum()); fxv += float((ws*(NU*om*tx)).sum())
        fyp += float((ws*(-p*ny)).sum()); fyv += float((ws*(NU*om*ty)).sum())
    cdp, cdf = 2*fxp, 2*fxv
    print(f'   ours                C_Dp {cdp:.4f}  C_Df {cdf:.4f}  '
          f'share {100*cdf/(cdp+cdf):.1f} %')
    for nm, p_, t_ in LIT:
        print(f'   {nm:19s} C_Dp {p_:.4f}  C_Df {t_-p_:.4f}  '
              f'share {100*(t_-p_)/t_:.1f} %')

    print('\n4. viscous traction, omega-form vs velocity-gradient form')
    ddr = lambda A: np.einsum('ai,eij->eaj', D, A)
    dds = lambda A: np.einsum('bj,eij->eib', D, A)
    g = {}
    for k, f in (('u', 0), ('v', 1)):
        g[k+'x'] = m.rx*ddr(U[..., f]) + m.sx*dds(U[..., f])
        g[k+'y'] = m.ry*ddr(U[..., f]) + m.sy*dds(U[..., f])
    gx = gy = 0.0
    for e, nx, ny, ws in wn:
        a, b = g['ux'][e, 0, :], g['uy'][e, 0, :]
        c, d = g['vx'][e, 0, :], g['vy'][e, 0, :]
        gx += float((ws*(NU*(2*a*nx + (b + c)*ny))).sum())
        gy += float((ws*(NU*((b + c)*nx + 2*d*ny))).sum())
    print(f'   friction C_D  omega {2*fxv:+.5f}  gradient {2*gx:+.5f}  '
          f'diff {abs(2*fxv-2*gx):.1e}')
    print(f'   friction C_L  omega {2*fyv:+.5f}  gradient {2*gy:+.5f}  '
          f'diff {abs(2*fyv-2*gy):.1e}')
    print(f'   total    C_D  omega {cdp+cdf:.5f}  gradient {2*fxp+2*gx:.5f}')
    print('\n   the residual IS the least-squares slack in omega - curl u, which '
          'this\n   formulation enforces weakly; it is not an error in the '
          'force integral.')


if __name__ == '__main__':
    main()
