"""Fused numba kernels for the VVP operator.

Each kernel performs the tensor-product contractions AND the elementwise algebra
in a single pass, writing directly into the native (nelem, n, n, 4) layout.  That
removes the per-field strided reads, the BLAS call overhead on tiny (n x n)
blocks, and every intermediate temporary.

Serial @njit only -- prange measured 0.82x (slower) at this problem size, and
numexpr 0.15-0.49x.  See NUMBA_INTEGRATION_PROPOSAL.md section 2.

Index conventions, from lssem2d/operators.py:
    dUdx(U)[e,i,j] = facx[e] * sum_k D[i,k] U[e,k,j]
    dUdy(U)[e,i,j] = facy[e] * sum_k D[j,k] U[e,i,k]
    DxT (S)[e,i,j] = facx[e] * sum_k D[k,i] S[e,k,j]
    DyT (S)[e,i,j] = facy[e] * sum_k D[k,j] S[e,i,k]

LEAST-SQUARES ROW WEIGHTING -- read before editing.  The momentum rows are
    fac1*u + dt*(...)                     NOT   (fac1/dt)*u + (...)
They are the same equation but differ by a factor dt AS LEAST-SQUARES ROWS, so
the (fac1/dt) form over-weights momentum by 1/dt against continuity and the
vorticity definition.  It is harmless when the residual is ~0 (cavity,
Poiseuille) and diverges on under-resolved cases such as the BFS.  These kernels
therefore carry f1 and dtl SEPARATELY rather than a single idt, and _kernel_LT
scales the two momentum components of `su` by dtl on read -- that is the exact
transpose, since the new operator is R_new = S R_old with S = diag(dt,dt,1,1).
An earlier draft of these kernels (in the proposal document) used idt=fac1/dt
and would silently reintroduce the divergence.  See lssem_baseline.f90 rhs().
"""
import os

import numpy as np

_FASTMATH = os.environ.get("LSSEM_FASTMATH", "1") == "1"

# numba's on-disk cache does NOT key on njit flags such as fastmath: a cache
# written with fastmath=True is silently reused when fastmath=False is asked
# for, so LSSEM_FASTMATH=0 would appear to do nothing.  Verified -- whichever
# flavour compiled first determined the result for both (checksums differed in
# the 13th digit, so the flag does change the arithmetic).  Give each flavour
# its own cache directory.  Must happen before `from numba import njit`, since
# numba reads its configuration at import time.
if "NUMBA_CACHE_DIR" not in os.environ:
    os.environ["NUMBA_CACHE_DIR"] = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "__nbcache__",
        "fastmath" if _FASTMATH else "strict",
    )

from numba import njit  # noqa: E402  (must follow the NUMBA_CACHE_DIR setup)


@njit(fastmath=_FASTMATH, boundscheck=False, cache=True)
def _kernel_L(U, D, facx, facy, wq, fu, fv, dfux, dfuy, dfvx, dfvy,
              nu, f1, dtl, out):
    NE, n = U.shape[0], U.shape[1]
    for e in range(NE):
        fx = facx[e]
        fy = facy[e]
        for i in range(n):
            for j in range(n):
                ux = 0.0; vx = 0.0; px = 0.0; ox = 0.0
                uy = 0.0; vy = 0.0; py = 0.0; oy = 0.0
                for k in range(n):
                    dik = D[i, k]; djk = D[j, k]
                    ux += dik*U[e, k, j, 0]; vx += dik*U[e, k, j, 1]
                    px += dik*U[e, k, j, 2]; ox += dik*U[e, k, j, 3]
                    uy += djk*U[e, i, k, 0]; vy += djk*U[e, i, k, 1]
                    py += djk*U[e, i, k, 2]; oy += djk*U[e, i, k, 3]
                ux *= fx; vx *= fx; px *= fx; ox *= fx
                uy *= fy; vy *= fy; py *= fy; oy *= fy
                u = U[e, i, j, 0]; v = U[e, i, j, 1]; om = U[e, i, j, 3]
                w = wq[e, i, j]; a = fu[e, i, j]; b = fv[e, i, j]
                out[e, i, j, 0] = (f1*u + dtl*(a*ux + b*uy
                                               + u*dfux[e, i, j] + v*dfuy[e, i, j]
                                               + px + nu*oy))*w
                out[e, i, j, 1] = (f1*v + dtl*(a*vx + b*vy
                                               + u*dfvx[e, i, j] + v*dfvy[e, i, j]
                                               + py - nu*ox))*w
                out[e, i, j, 2] = (ux + vy)*w
                out[e, i, j, 3] = (om + uy - vx)*w


@njit(fastmath=_FASTMATH, boundscheck=False, cache=True)
def _kernel_LT(su, D, facx, facy, fu, fv, dfux, dfuy, dfvx, dfvy,
               nu, idt, dtl, out):
    NE, n = su.shape[0], su.shape[1]
    for e in range(NE):
        fx = facx[e]
        fy = facy[e]
        for i in range(n):
            for j in range(n):
                tx1 = 0.0; tx2 = 0.0; tx3 = 0.0; tx4 = 0.0; txg1 = 0.0; txg3 = 0.0
                ty1 = 0.0; ty2 = 0.0; ty3 = 0.0; ty4 = 0.0; tyg2 = 0.0; tyg4 = 0.0
                for k in range(n):
                    dki = D[k, i]; dkj = D[k, j]
                    # momentum components carry the dt row weighting
                    s1x = dtl*su[e, k, j, 0]; s2x = dtl*su[e, k, j, 1]
                    tx1 += dki*s1x
                    tx2 += dki*s2x
                    tx3 += dki*su[e, k, j, 2]          # continuity: unweighted
                    tx4 += dki*su[e, k, j, 3]          # vorticity:  unweighted
                    txg1 += dki*fu[e, k, j]*s1x        # Dx^T(fu*su1)
                    txg3 += dki*fu[e, k, j]*s2x        # Dx^T(fu*su2)
                    s1y = dtl*su[e, i, k, 0]; s2y = dtl*su[e, i, k, 1]
                    ty1 += dkj*s1y
                    ty2 += dkj*s2y
                    ty3 += dkj*su[e, i, k, 2]
                    ty4 += dkj*su[e, i, k, 3]
                    tyg2 += dkj*fv[e, i, k]*s1y        # Dy^T(fv*su1)
                    tyg4 += dkj*fv[e, i, k]*s2y        # Dy^T(fv*su2)
                tx1 *= fx; tx2 *= fx; tx3 *= fx; tx4 *= fx; txg1 *= fx; txg3 *= fx
                ty1 *= fy; ty2 *= fy; ty3 *= fy; ty4 *= fy; tyg2 *= fy; tyg4 *= fy
                s1 = dtl*su[e, i, j, 0]; s2 = dtl*su[e, i, j, 1]
                s4 = su[e, i, j, 3]
                out[e, i, j, 0] = (idt*s1 + dfux[e, i, j]*s1 + dfvx[e, i, j]*s2
                                   + tx3 + txg1 + ty4 + tyg2)
                out[e, i, j, 1] = (idt*s2 + dfuy[e, i, j]*s1 + dfvy[e, i, j]*s2
                                   - tx4 + txg3 + ty3 + tyg4)
                out[e, i, j, 2] = tx1 + ty2
                out[e, i, j, 3] = s4 - nu*tx2 + nu*ty1


def _C(a):
    """numba accepts strided arrays but compiles a slower path for them.

    newton_step passes fu = U[..., 0], which IS strided, so this guard is not
    cosmetic.
    """
    return a if a.flags.c_contiguous else np.ascontiguousarray(a)


def _bufs(state):
    """Allocate the numba-only work buffers on first use.

    Done lazily rather than in SolverState.__init__ so selecting the NumPy
    backend costs no extra memory.
    """
    if not hasattr(state, '_nb_su'):
        m = state.mesh
        n = m.N + 1
        state._nb_D = np.ascontiguousarray(state.D)
        state._nb_su = np.empty((m.nelem, n, n, 4))
        state._nb_c = np.empty((m.nelem, n, n, 4))
    return state


def apply_L(state, U, fu, fv):
    """Fused numba apply_L.  Signature identical to the NumPy reference.

    The kernel's `f1`/`dtl` arguments are exactly (a_mass, a_flux), so the
    least-squares weight decoupling needs no kernel change -- only these
    coefficients, taken from the one shared definition in lssem.ls_coeffs.
    """
    from .lssem import ls_coeffs
    _bufs(state)
    m = state.mesh
    a_mass, a_flux, _ = ls_coeffs(state)
    _kernel_L(_C(U), state._nb_D, m.facx, m.facy, m.wq, _C(fu), _C(fv),
              _C(state.dfu_dx), _C(state.dfu_dy), _C(state.dfv_dx), _C(state.dfv_dy),
              state.nu, a_mass, a_flux, state._nb_su)
    return state._nb_su


def apply_LT(state, su, fu, fv):
    """Fused numba apply_LT.  Signature identical to the NumPy reference."""
    from .lssem import ls_coeffs
    _bufs(state)
    m = state.mesh
    a_mass, a_flux, _ = ls_coeffs(state)
    idt = a_mass / a_flux if a_flux != 0.0 else 0.0
    _kernel_LT(_C(su), state._nb_D, m.facx, m.facy, _C(fu), _C(fv),
               _C(state.dfu_dx), _C(state.dfu_dy), _C(state.dfv_dx), _C(state.dfv_dy),
               state.nu, idt, a_flux, state._nb_c)
    return state._nb_c


def warmup(state):
    """Pay JIT compilation once at setup, not inside a timed loop or step 1.

    Requires state.update_linearisation() to have been called.
    """
    m = state.mesh
    n = m.N + 1
    Z = np.zeros((m.nelem, n, n, 4))
    z = np.zeros((m.nelem, n, n))
    if state.dfu_dx is None:
        state.update_linearisation(z, z)
    apply_LT(state, apply_L(state, Z, z, z), z, z)
    return state
