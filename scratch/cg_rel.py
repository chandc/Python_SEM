"""CG with RELATIVE breakdown guards (see KOVASZNAY_VALIDATION.md sec 4).

pcg_solve guards with absolute thresholds (|alpha_denom| < 1e-20).  A = L^T L
squares the scale, so p.Ap ~ ||r||^2 and the guard trips on a healthy iteration
once ||b|| ~ 1e-10.  Identical algorithm, guards made relative to their operands.
"""
import numpy as np
from lssem2d.solver import apply_A


def pcg_rel(state, b, fu, fv, M_inv, multiplicity_weight, pin_p=False,
            max_iter=5000, tol=1e-6, cgsfac=0.0, precond=None):
    _M = precond if precond is not None else (lambda _r: M_inv * _r)
    mw = multiplicity_weight
    b_norm = np.sqrt(np.sum(b*b*mw))
    if b_norm == 0.0:
        return np.zeros_like(b), 0
    target = max(cgsfac*b_norm, tol) if cgsfac > 0 else tol
    x = np.zeros_like(b); r = b.copy(); z = _M(r); p = z.copy()
    rho_prev = np.sum(r*z*mw)
    for i in range(max_iter):
        Ap = apply_A(state, p, fu, fv, pin_p=pin_p)
        alpha_denom = np.sum(p*Ap*mw)
        pn = np.sqrt(np.sum(p*p*mw)); an = np.sqrt(np.sum(Ap*Ap*mw))
        if pn == 0.0 or abs(alpha_denom) <= 1e-15*pn*an:
            return x, i+1
        alpha = rho_prev/alpha_denom
        x = x + alpha*p
        r = r - alpha*Ap
        if np.sqrt(np.sum(r*r*mw)) < target:
            true_r = b - apply_A(state, x, fu, fv, pin_p=pin_p)
            if np.sqrt(np.sum(true_r*true_r*mw)) < target:
                return x, i+1
            r = true_r; z = _M(r); p = z.copy(); rho_prev = np.sum(r*z*mw)
            continue
        z = _M(r); rho = np.sum(r*z*mw)
        rn = np.sqrt(np.sum(r*r*mw)); zn = np.sqrt(np.sum(z*z*mw))
        if abs(rho) <= 1e-15*rn*zn:
            return x, i+1
        p = z + (rho/rho_prev)*p
        rho_prev = rho
    return x, max_iter
