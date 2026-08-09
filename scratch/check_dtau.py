"""Verification ladder for the dtau implementation (PSEUDO_TIME_DESIGN.md sec 6).

1. dtau=None is a no-op: kappa == 0 and every operator is bit-identical.
2. The RESIDUAL is unchanged when dtau is set -- this is the trap the design note
   flags.  apply_L gains kappa*u but _drop_pseudo removes it, so su_nl must come
   back bit-identical.  A control with the cancellation disabled must FAIL, or
   the check proves nothing.
3. The OPERATOR does change: apply_A must gain exactly kappa on the momentum
   diagonal, checked against a finite-difference-free identity --
   A_dtau(dU) - A_none(dU) must equal the extra terms, verified by building the
   augmented L by hand.
4. compute_jacobi tracks apply_L: the Jacobi diagonal must equal the true
   diagonal of A, extracted column by column on a small mesh.
"""
import os, sys
import numpy as np
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState, apply_L, apply_LT, ls_coeffs, ls_pseudo
import lssem2d.solver as S
from lssem2d.assembly import gather_scatter

RE = 100.0
N = 4


def mk(dtau=None, **kw):
    m = build_channel(1.0, 1.0, 2, 2, N)
    st = SolverState(m, diff_matrix(N), nu=1.0/RE, dt=0.5, fac1=1.0, dtau=dtau, **kw)
    rng = np.random.default_rng(7)
    U = rng.standard_normal((m.nelem, N+1, N+1, 4))
    fu = np.ascontiguousarray(U[..., 0])
    fv = np.ascontiguousarray(U[..., 1])
    st.update_linearisation(fu, fv)
    return st, U, fu, fv


print("=" * 74)
print("1. dtau = None is a no-op")
print("=" * 74)
st0, U, fu, fv = mk(None)
print(f"   ls_pseudo(dtau=None)      = {ls_pseudo(st0):.1f}")
for kw in ({}, dict(w_mom=0.1, w_mass=0.0), dict(w_mom=2.0, w_mass=1.0)):
    s, _, _, _ = mk(None, **kw)
    print(f"   ls_pseudo{str(kw):<34} = {ls_pseudo(s):.1f}   ls_coeffs = "
          f"{tuple(round(x, 6) for x in ls_coeffs(s))}")

print()
print("=" * 74)
print("2. the RESIDUAL is unchanged when dtau is set")
print("=" * 74)


def residual(dtau, cancel=True):
    st, U, fu, fv = mk(dtau)
    su_hist = np.zeros_like(U)
    f = np.ascontiguousarray(U[..., 0]/2.0); g = np.ascontiguousarray(U[..., 1]/2.0)
    st.update_linearisation(f, g)
    r = apply_L(st, U, f, g) - su_hist
    if cancel:
        S._drop_pseudo(st, r, U)
    return r


r_none = residual(None)
for dtau in (10.0, 1.0, 0.1):
    r = residual(dtau)
    print(f"   dtau = {dtau:>5}  max|residual - residual(None)| = "
          f"{np.abs(r - r_none).max():.3e}"
          f"    {'OK (roundoff)' if np.abs(r - r_none).max() < 1e-14 else 'DIFFERS'}")
r_bad = residual(1.0, cancel=False)
print(f"   control, cancellation DISABLED             = "
      f"{np.abs(r_bad - r_none).max():.3e}"
      f"    {'(check is live)' if not np.allclose(r_bad, r_none) else '(CHECK IS DEAD)'}")

print()
print("=" * 74)
print("3. the OPERATOR does change, by exactly kappa on the momentum rows")
print("=" * 74)
# L_dtau(U) - L_none(U) must be kappa*u*wq and kappa*v*wq on rows 0,1 and 0 elsewhere
for dtau in (10.0, 1.0, 0.1):
    sa, U, fu, fv = mk(dtau)
    sb, _, _, _ = mk(None)
    sa.update_linearisation(fu, fv); sb.update_linearisation(fu, fv)
    La = apply_L(sa, U, fu, fv).copy()
    Lb = apply_L(sb, U, fu, fv).copy()
    kap = ls_pseudo(sa)
    wq = sa.mesh.wq
    exp = np.zeros_like(La)
    exp[..., 0] = kap * U[..., 0] * wq
    exp[..., 1] = kap * U[..., 1] * wq
    err = np.abs((La - Lb) - exp).max()
    # transpose: the collocation coefficient must move by kappa/a_flux on su1,su2
    _, a_flux, _ = ls_coeffs(sa)
    su = np.ascontiguousarray(np.random.default_rng(3).standard_normal(La.shape))
    Ta = apply_LT(sa, su.copy(), fu, fv).copy()
    Tb = apply_LT(sb, su.copy(), fu, fv).copy()
    expT = np.zeros_like(Ta)
    expT[..., 0] = kap * su[..., 0]
    expT[..., 1] = kap * su[..., 1]
    errT = np.abs((Ta - Tb) - expT).max()
    print(f"   dtau = {dtau:>5}  kappa = {kap:.4f}   "
          f"max|dL - kappa*u*wq| = {err:.3e}   max|dL^T - kappa*su| = {errT:.3e}")

print()
print("=" * 74)
print("4. compute_jacobi still equals the true diagonal of A")
print("=" * 74)
for dtau in (None, 1.0, 0.1):
    st, U, fu, fv = mk(dtau)
    st.update_linearisation(fu, fv)
    M_inv = S.compute_jacobi(st, fu, fv)
    m = st.mesh
    shape = (m.nelem, N+1, N+1, 4)
    # true diagonal: e_i^T A e_i on the assembled system
    mask = st.get_global_mask()
    diag_true = np.zeros(shape)
    ncheck = 0
    rng = np.random.default_rng(11)
    # ELEMENT-INTERIOR nodes only.  apply_A gather-scatters, so a single local
    # node set to 1 is not a global unit vector at a node shared between
    # elements -- comparing there measures my test, not the preconditioner.
    idx = [tuple(i) for i in rng.integers([0, 1, 1, 0], [m.nelem, N, N, 4], size=(25, 4))]
    errs = []
    for ij in idx:
        if mask[ij] == 0.0:
            continue
        e = np.zeros(shape); e[ij] = 1.0
        Ae = S.apply_A(st, e, fu, fv)
        d_true = Ae[ij]
        d_jac = 1.0/M_inv[ij] if M_inv[ij] != 0 else np.nan
        if np.isfinite(d_jac):
            errs.append(abs(d_true - d_jac)/max(abs(d_true), 1e-30))
            ncheck += 1
    print(f"   dtau = {str(dtau):>5}   {ncheck} interior dofs   "
          f"max relative error vs true diagonal = {max(errs):.3e}")
