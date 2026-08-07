"""Preconditioner comparison on the Ghia lid-driven cavity, Re=1000.

Two measurements:
  A. one real Newton solve   -- CG iterations, operator applies, wall time
  B. 20 actual BDF steps     -- total CG iterations and wall time to advance

(B) is the one that matters practically: it includes preconditioner setup cost,
which is paid once per solve and is NOT free for Chebyshev/p-MG (a 20-step power
iteration for lambda_max, plus coarse-mesh assembly for p-MG).
"""
import os
import sys, time, numpy as np
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState, apply_L, apply_LT
from lssem2d.solver import compute_jacobi, pcg_solve
from lssem2d.assembly import gather_scatter
from lssem2d.bc import apply_bc
import lssem2d.solver as S
from lssem2d import precond as P

SC = os.path.dirname(os.path.abspath(__file__))
RE, DT = 1000.0, 0.1
d = np.load('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo/cavity_re1000_data.npz')
U0 = d['U_steady'].copy()
NE, n = U0.shape[0], U0.shape[1]
N = n - 1
EX = int(round(NE**0.5))
mesh = build_channel(1.0, 1.0, EX, EX, N, bcs=(1, 1, 1, 2))
st = SolverState(mesh, diff_matrix(N), nu=1.0/RE, dt=DT, fac1=1.0)
PIN = True

# --- sanity: does the state we are linearising about actually match Ghia? ----
from lssem2d.lgl import lgl_nodes
def centreline_u(U):
    ys, us = [], []
    for e in range(NE):
        xs = mesh.xnod[e]
        if xs[0] - 1e-9 <= 0.5 <= xs[-1] + 1e-9:
            w = np.ones(n)
            for i in range(n):
                for j in range(n):
                    if i != j: w[i] /= (xs[i]-xs[j])
            dd = 0.5 - xs
            if np.any(np.abs(dd) < 1e-13):
                L = np.zeros(n); L[np.argmin(np.abs(dd))] = 1.0
            else:
                num = w/dd; L = num/num.sum()
            for j in range(n):
                ys.append(mesh.ynod[e, j]); us.append(np.dot(L, U[e, :, j, 0]))
    o = np.argsort(ys); ys, us = np.array(ys)[o], np.array(us)[o]
    k = np.concatenate(([True], np.diff(ys) > 1e-9))
    return ys[k], us[k]

yc, uc = centreline_u(U0)
gu, gy = d['ghia_u'], d['ghia_y']
rms = np.sqrt(np.mean((np.interp(gy, yc, uc) - gu)**2))
print(f"cavity Re={RE:.0f}  {NE} elem ({EX}x{EX}) order {N},  dt={DT}")
print(f"linearisation state vs Ghia 1982 centreline u: RMS {rms:.4f} "
      f"({100*rms/max(abs(gu.max()), abs(gu.min())):.1f}% of |u|max)\n")

# --------------------------------------------------------- A. one Newton RHS
U = apply_bc(mesh, U0.copy(), time=0.0, pin_p=PIN)
su_hist = np.zeros_like(U)
su_hist[..., 0] += 1.0*U0[..., 0]*mesh.wq
su_hist[..., 1] += 1.0*U0[..., 1]*mesh.wq
st.update_linearisation(U[..., 0]/2.0, U[..., 1]/2.0)
su_nl = apply_L(st, U, U[..., 0]/2.0, U[..., 1]/2.0) - su_hist
fu = np.ascontiguousarray(U[..., 0]); fv = np.ascontiguousarray(U[..., 1])
st.update_linearisation(fu, fv)
gm = st.get_global_mask(pin_p=PIN)
b = -gather_scatter(mesh, apply_LT(st, su_nl, fu, fv))*gm
Mi = compute_jacobi(st, fu, fv, pin_p=PIN)
mult = gather_scatter(mesh, np.ones_like(U))
mw = 1.0/np.where(mult < 1e-10, 1.0, mult)
print(f"A. ONE Newton solve   ||b|| = {np.sqrt(np.sum(b*b*mw)):.4e}, target cgsfac=1e-3")
print(f"   {'preconditioner':<20}{'setup s':>9}{'CG it':>8}{'A-appl':>9}{'A/it':>7}{'solve s':>9}{'speedup':>9}")

KINDS = [('jacobi', {}), ('chebyshev4', dict(deg=2)), ('chebyshev4', dict(deg=4)),
         ('chebyshev4', dict(deg=6)),
         ('pmg2', dict(pc=4, deg=2, coarse_deg=6)),
         ('pmg2', dict(pc=2, deg=2, coarse_deg=6))]
base_t = None
for kind, kw in KINDS:
    t0 = time.perf_counter()
    M = P.make(kind, st, fu, fv, Mi, PIN, **kw)
    ts = time.perf_counter()-t0
    cnt = [0]; orig = S.apply_A
    def counted(*A, **K):
        cnt[0] += 1; return orig(*A, **K)
    S.apply_A = counted; P.apply_A = counted
    t1 = time.perf_counter()
    try:
        _, it = pcg_solve(st, b, fu, fv, Mi, mw, pin_p=PIN, max_iter=20000,
                          tol=1e-6, cgsfac=1e-3, precond=M)
    finally:
        S.apply_A = orig; P.apply_A = orig
    tw = time.perf_counter()-t1
    if base_t is None: base_t = tw
    tag = kind + ('' if not kw else f" pc={kw['pc']}" if kind == 'pmg2' else f" d={kw['deg']}")
    print(f"   {tag:<20}{ts:>9.3f}{it:>8}{cnt[0]:>9}{cnt[0]/max(it,1):>7.1f}{tw:>9.3f}{base_t/tw:>8.2f}x")

# ------------------------------------------------- B. 20 real BDF time steps
NSTEP = 20
print(f"\nB. {NSTEP} BDF steps from the converged state (setup cost included)")
print(f"   {'preconditioner':<20}{'tot CG it':>11}{'A-appl':>10}{'wall s':>9}{'speedup':>9}")
base_t = None
for kind, kw in KINDS:
    tot = [0]; cnt = [0]
    _p = S.pcg_solve; origA = S.apply_A
    def counted(*A, **K):
        cnt[0] += 1; return origA(*A, **K)
    def pcg(state, bb, ffu, ffv, MM, mmw, pin_p=False, max_iter=5000, tol=1e-6, cgsfac=0.0, precond=None):
        pre = None if kind == 'jacobi' else P.make(kind, state, ffu, ffv, MM, pin_p, **kw)
        x, it = _p(state, bb, ffu, ffv, MM, mmw, pin_p=pin_p, max_iter=20000,
                   tol=1e-6, cgsfac=1e-3, precond=pre)
        tot[0] += it; return x, it
    S.pcg_solve = pcg; S.apply_A = counted; P.apply_A = counted
    hist = [U0.copy()]
    t0 = time.perf_counter()
    try:
        for s in range(NSTEP):
            Un = S.step_bdf(st, hist, time=s*DT, max_newton=1, newton_tol=1e-10,
                            newton_factor=0.0, pin_p=PIN, cgsfac=1e-3,
                            cg_max_iter=20000, verbose=False)
    finally:
        S.pcg_solve = _p; S.apply_A = origA; P.apply_A = origA
    tw = time.perf_counter()-t0
    if base_t is None: base_t = tw
    tag = kind + ('' if not kw else f" pc={kw['pc']}" if kind == 'pmg2' else f" d={kw['deg']}")
    print(f"   {tag:<20}{tot[0]:>11}{cnt[0]:>10}{tw:>9.2f}{base_t/tw:>8.2f}x")
