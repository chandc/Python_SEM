"""Does a RELATIVE CG breakdown guard recover Chan's N=14 accuracy?

pcg_solve guards breakdown with ABSOLUTE thresholds:
    if abs(alpha_denom) < 1e-20: return
    if abs(rho_prev)    < 1e-20: return
A = L^T L squares the scale, so p.Ap ~ ||r||^2 and once ||b|| ~ 1e-10 the guard
trips on a perfectly healthy iteration.  Here the same CG is re-run with guards
made relative to the operands, everything else identical.
"""
import sys, os, time
import numpy as np
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC); sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import kov
import lssem2d.solver as S
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
        if abs(alpha_denom) <= 1e-15*pn*an or pn == 0.0:      # RELATIVE
            return x, i+1
        alpha = rho_prev/alpha_denom
        x = x + alpha*p
        r = r - alpha*Ap
        r_norm = np.sqrt(np.sum(r*r*mw))
        if r_norm < target:
            true_r = b - apply_A(state, x, fu, fv, pin_p=pin_p)
            if np.sqrt(np.sum(true_r*true_r*mw)) < target:
                return x, i+1
            r = true_r; z = _M(r); p = z.copy(); rho_prev = np.sum(r*z*mw)
            continue
        z = _M(r); rho = np.sum(r*z*mw)
        rn = np.sqrt(np.sum(r*r*mw)); zn = np.sqrt(np.sum(z*z*mw))
        if abs(rho) <= 1e-15*rn*zn:                            # RELATIVE
            return x, i+1
        p = z + (rho/rho_prev)*p
        rho_prev = rho
    return x, max_iter


CHAN = {9: (1.56e-6, 3.58e-7, 3.76e-6), 14: (9.22e-13, 4.72e-13, 1.47e-11)}
print(f"{'N':>4}{'guard':>10}{'steps':>7}{'CG':>8}{'wall':>8}{'|dU|':>10}"
      f"{'res':>10}{'eps_u':>12}{'eps_v':>12}{'eps_p':>12}{'Chan eps_u':>12}")
_orig = S.pcg_solve
for N in (9, 14):
    for tag in ('absolute', 'relative'):
        S.pcg_solve = _orig if tag == 'absolute' else pcg_rel
        try:
            r = kov.run(4, 2, N, 1e-13, cap=60, cg_tol=1e-14)
        finally:
            S.pcg_solve = _orig
        print(f"{N:>4}{tag:>10}{r['steps']:>7}{r['cg']:>8}{r['wall']:>7.1f}s"
              f"{r['dU']:>10.1e}{r['res']:>10.1e}{r['eu']:>12.3e}{r['ev']:>12.3e}"
              f"{r['ep']:>12.3e}{CHAN[N][0]:>12.2e}", flush=True)
