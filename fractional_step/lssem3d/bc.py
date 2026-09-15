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
from . import fourier as FR
from lssem2d.assembly import gather_scatter

VEL = (OP.U_, OP.V_, OP.W_)
WALLISH = (1, 2, 3)


def _edges(mesh, e):
    """(bc code, index expression) for the four edges of element e."""
    return ((mesh.bc[e, 0], (e, 0, slice(None))),        # W
            (mesh.bc[e, 1], (e, -1, slice(None))),       # E
            (mesh.bc[e, 2], (e, slice(None), 0)),        # S
            (mesh.bc[e, 3], (e, slice(None), -1)))       # N


def pin_dof(mesh, mask, field, mode=0, elem=0, i=0, j=0):
    """Prescribe ONE GLOBAL dof by zeroing EVERY LOCAL COPY of it.

    Setting `mask[elem, i, j, field, mode] = 0` prescribes a single local copy.
    On a mesh where that node is shared -- an element interface, and in
    particular a PERIODIC seam -- its siblings are still free, so:

      * the dof is not actually pinned (the siblings carry it), and
      * the mask is INCONSISTENT across copies of one global node, which breaks
        the symmetry of A = M Q^T Q L^T W L M.  M is then not well defined on
        the global space, and CG is being run on a non-symmetric operator.

    Measured on the continuous subspace (the space the assembled operator acts
    on -- random LOCAL vectors are discontinuous and cannot test this):

        cavity, no periodicity, node multiplicity 1   sym err 1.1e-15
        channel, periodic x,    multiplicity 2        sym err 1.5e-07
        Taylor-Green, periodic x and y, mult 4        sym err 5.9e-05
        Taylor-Green, all copies pinned               sym err 0.0e+00

    `gs` of a one-hot array marks exactly the copies of that global node, which
    is why it is the right instrument here.
    """
    ind = np.zeros(mask.shape)
    ind[elem, i, j, field, mode] = 1.0
    nel, n, _, nv, nk = ind.shape
    spread = gather_scatter(mesh, ind.reshape(nel, n, n, nv*nk)).reshape(ind.shape)
    mask[spread > 0.5] = 0.0
    return mask


def real_mode_columns(nmode, nz=None):
    """Mode columns whose coefficients must be real, clipped to `nmode`.

    Single source of truth for `build_mask`, `apply_values` and
    `prescribed_entries`: those three must agree on which DOFs are prescribed,
    and the 2D code's original bug was exactly a disagreement between the mask
    and the value-writer.

    nz defaults to the even count implied by nmode (`2*(nmode-1)`), the standard
    rfft layout.  nmode == 1 means k = 0 alone, where the entire imaginary half
    is unphysical.
    """
    if nmode == 1:
        return (0,)
    if nz is None:
        nz = 2*(nmode - 1)
    return tuple(k for k in FR.real_mode_indices(nz) if k < nmode)


def build_mask(mesh, nmode, pin_p=False, nz=None):
    """(nelem, n, n, 14, nmode) with 1 on free DOFs and 0 on prescribed ones.

    Both the real and imaginary halves of a prescribed field are frozen: a
    Dirichlet condition applies to the complex coefficient, not to its real part
    alone.  Forgetting the imaginary half leaves half the boundary condition
    unimposed at every non-zero mode, and is invisible at k_z = 0 where the
    imaginary part is zero anyway -- so it would pass a Stage 1 test and fail
    only in 3D.

    THE REAL MODES ARE ALSO CONSTRAINED, everywhere, not just on boundaries.
    For real physical data `fourier.real_mode_indices` gives the modes whose
    coefficients must be real: k = 0 always, and the Nyquist mode when nz is
    even.  `irfft` DISCARDS the imaginary part of those modes, so any content
    the solver puts there is invisible in physical space -- an unconstrained,
    non-physical direction that CG will happily fill.  Measured in
    `test_integration_multimode.py`: the Nyquist imaginary part reached 1.5e-03
    against a real part of 6.1e-03 after three steps, i.e. comparable size, and
    `fourier.assert_hermitian_ok` -- the library's own stated invariant --
    would have failed on the solver's state.

    Freezing them is what makes the state and the physical field the same
    object.  With nmode = 1 the entire imaginary half is prescribed, which is
    exactly right, since at k = 0 every imaginary component is unphysical.

    This is a CORRECTNESS fix, not the k_z = 0 fast path.  It removes those DOFs
    from the solve, but the arrays keep their full width, so the matvec costs
    what it did before.  Actually skipping the zero block needs the operator to
    branch on it -- still open (3D_STATUS.md sec 5).

    nz: physical z-point count, needed only to know whether a Nyquist mode
        exists (it does iff nz is even).  Defaults to the even value implied by
        nmode, `2*(nmode-1)`, which is the standard rfft layout.
    """
    n = mesh.N + 1
    mask = np.ones((mesh.nelem, n, n, OP.NVAR_R, nmode))

    for k in real_mode_columns(nmode, nz):
        mask[..., OP.NVAR:, k] = 0.0                  # whole imaginary half

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
        # EVERY local copy of the pinned node, not just one -- see pin_dof.
        # Harmless where the node is unshared (the cavity corner, multiplicity
        # 1); essential on a periodic mesh, where it is shared 2 or 4 ways.
        for _k in range(nmode):
            pin_dof(mesh, mask, OP.P_, _k)
            pin_dof(mesh, mask, OP.NVAR + OP.P_, _k)
    return mask


def apply_values(mesh, U, nmode, lid_speed=0.0, pin_p=False, nz=None):
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
    # The modes that must be real are prescribed to zero in their imaginary
    # half EVERYWHERE, not just on boundaries -- see build_mask.  Written after
    # the edge loop; the lid lives in the real half, so nothing is undone.
    for k in real_mode_columns(nmode, nz):
        U[..., OP.NVAR:, k] = 0.0
    return U


def prescribed_entries(mesh, nmode, pin_p=False, nz=None):
    """Boolean array, True where a DOF is prescribed.  = (build_mask == 0)."""
    return build_mask(mesh, nmode, pin_p, nz) == 0.0
