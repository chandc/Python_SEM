"""Boundary-condition masking and values for the 7-field 3D system.

NEW CODE.  lssem2d is not modified.

BC CODES follow lssem2d.mesh: per element, `mesh.bc[e]` is (W, E, S, N) with

    1, 2, 3  wall / lid / prescribed velocity  -> velocity is Dirichlet
    4        outflow, p = 0                    -> pressure is Dirichlet
    5        symmetry                          -> v and omega Dirichlet

The 3D extension is the obvious one and is stated rather than assumed: where 2D
freezes (u, v), 3D freezes (u, v, **w**), because a no-slip wall in a
z-periodic geometry admits no spanwise slip either.  Vorticity is left free
everywhere -- as in 2D, the least-squares system determines it.

TWO MASKS, ONE RULE.  `lssem2d` carries a warning that `SolverState.get_global_mask`
and `bc.apply_mask` duplicate the same logic and once disagreed on `bc == 4`,
with the result that a p = 0 outflow imposed nothing at all (OUTFLOW_BC_STUDY.md
sec 3, measured max|p| = 4.87e-01 on a plane where p = 0 was claimed).  This
module deliberately has ONE function that builds the mask and ONE that writes
the values, and `apply_values` is written to touch exactly the entries
`build_mask` zeroes.  `test_bc.py` asserts that correspondence directly so the
2D failure cannot recur here.

THE NULL SPACE.  Without a pressure condition L^T L is singular: a constant p at
k_z = 0 is annihilated (solver3d tests assert this).  `pin_p` freezes a single
pressure degree of freedom, which is what lssem2d does for closed domains.  A
domain with a `bc == 4` outflow already fixes p and needs no pin.
"""
import numpy as np
from . import operator as OP

VEL = (OP.U_, OP.V_, OP.W_)
WALLISH = (1, 2, 3)


def _edges(mesh, e):
    """(bc code, index expression) for the four edges of element e."""
    return ((mesh.bc[e, 0], (e, 0, slice(None))),        # W
            (mesh.bc[e, 1], (e, -1, slice(None))),       # E
            (mesh.bc[e, 2], (e, slice(None), 0)),        # S
            (mesh.bc[e, 3], (e, slice(None), -1)))       # N


def build_mask(mesh, nmode, pin_p=False):
    """(nelem, n, n, 14, nmode) with 1 on free DOFs and 0 on prescribed ones.

    Both the real and imaginary halves of a prescribed field are frozen: a
    Dirichlet condition applies to the complex coefficient, not to its real part
    alone.  Forgetting the imaginary half leaves half the boundary condition
    unimposed at every non-zero mode, and is invisible at k_z = 0 where the
    imaginary part is zero anyway -- so it would pass a Stage 1 test and fail
    only in 3D.
    """
    n = mesh.N + 1
    mask = np.ones((mesh.nelem, n, n, OP.NVAR_R, nmode))

    def freeze(idx, fields):
        for f in fields:
            mask[idx + (f,)] = 0.0                    # real part
            mask[idx + (OP.NVAR + f,)] = 0.0          # imaginary part

    for e in range(mesh.nelem):
        for code, idx in _edges(mesh, e):
            if code in WALLISH:
                freeze(idx, VEL)
            elif code == 4:
                freeze(idx, (OP.P_,))
            elif code == 5:
                freeze(idx, (OP.V_, OP.OZ_))
    if pin_p:
        mask[0, 0, 0, OP.P_, :] = 0.0
        mask[0, 0, 0, OP.NVAR + OP.P_, :] = 0.0
    return mask


def apply_values(mesh, U, nmode, lid_speed=0.0, pin_p=False):
    """Write prescribed values into U, in place, and return it.

    Touches exactly the entries build_mask zeroes -- see the module docstring on
    why that correspondence is enforced by test rather than by convention.

    lid_speed applies to `bc == 2` edges (the moving lid); all other wall-like
    edges get zero velocity.  Only the k_z = 0 mode carries a steady lid: a
    z-uniform boundary value has no content in any other mode, and writing it
    into every mode would impose a lid that oscillates in z.
    """
    def zero(idx, fields):
        for f in fields:
            U[idx + (f,)] = 0.0
            U[idx + (OP.NVAR + f,)] = 0.0

    for e in range(mesh.nelem):
        for code, idx in _edges(mesh, e):
            if code in WALLISH:
                zero(idx, VEL)
                if code == 2 and lid_speed:
                    # k_z = 0 only; the imaginary half stays zero
                    U[idx + (OP.U_,)][..., 0] = lid_speed
            elif code == 4:
                zero(idx, (OP.P_,))
            elif code == 5:
                zero(idx, (OP.V_, OP.OZ_))
    if pin_p:
        U[0, 0, 0, OP.P_, :] = 0.0
        U[0, 0, 0, OP.NVAR + OP.P_, :] = 0.0
    return U


def prescribed_entries(mesh, nmode, pin_p=False):
    """Boolean array, True where a DOF is prescribed.  = (build_mask == 0)."""
    return build_mask(mesh, nmode, pin_p) == 0.0
