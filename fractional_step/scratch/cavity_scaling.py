"""Does the preconditioner ranking flip with resolution on the Ghia cavity?

Jacobi wins at 4x4 order 8.  Small problems favour cheap preconditioners: the
matvec is cheap and the spectral spread is modest.  If p-MG is ever going to
pay for its 13 matvecs per application, it has to be at higher resolution.

For each mesh: spin up a developed state from rest, then solve one real Newton
system with each preconditioner and record iterations / applies / wall time.
"""
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

RE, DT, PIN, SPINUP = 1000.0, 0.1, True, 40
CASES = [(4, 8), (6, 10), (8, 12), (10, 12)]

print(f"Ghia cavity Re={RE:.0f}, dt={DT}, one real Newton solve, cgsfac=1e-3")
print(f"spin-up: {SPINUP} BDF steps from rest (Jacobi) to get a representative state\n")
print(f"{'mesh':<16}{'DOF':>8}  {'preconditioner':<18}{'CG it':>7}{'A-appl':>9}"
      f"{'wall s':>9}{'vs jacobi':>11}")

for EX, N in CASES:
    n = N+1
    mesh = build_channel(1.0, 1.0, EX, EX, N, bcs=(1, 1, 1, 2))
    st = SolverState(mesh, diff_matrix(N), nu=1.0/RE, dt=DT, fac1=1.0)
    ndof = mesh.nelem*n*n*4
    U0 = np.zeros((mesh.nelem, n, n, 4))
    hist = [U0]
    for s in range(SPINUP):
        U0 = S.step_bdf(st, hist, time=s*DT, max_newton=1, newton_tol=1e-10,
                        newton_factor=0.0, pin_p=PIN, cgsfac=1e-3,
                        cg_max_iter=20000, verbose=False)
    if not np.all(np.isfinite(U0)):
        print(f"{EX}x{EX} order {N:<4}  spin-up went non-finite -- skipped"); continue

    U = apply_bc(mesh, U0.copy(), time=0.0, pin_p=PIN)
    sh = np.zeros_like(U)
    sh[..., 0] += U0[..., 0]*mesh.wq; sh[..., 1] += U0[..., 1]*mesh.wq
    st.update_linearisation(U[..., 0]/2.0, U[..., 1]/2.0)
    su_nl = apply_L(st, U, U[..., 0]/2.0, U[..., 1]/2.0) - sh
    fu = np.ascontiguousarray(U[..., 0]); fv = np.ascontiguousarray(U[..., 1])
    st.update_linearisation(fu, fv)
    gm = st.get_global_mask(pin_p=PIN)
    b = -gather_scatter(mesh, apply_LT(st, su_nl, fu, fv))*gm
    Mi = compute_jacobi(st, fu, fv, pin_p=PIN)
    mult = gather_scatter(mesh, np.ones_like(U))
    mw = 1.0/np.where(mult < 1e-10, 1.0, mult)

    kinds = [('jacobi', {}), ('chebyshev4', dict(deg=6)), ('chebyshev4', dict(deg=10)),
             ('pmg2', dict(pc=max(2, N//3), deg=2, coarse_deg=6)),
             ('pmg2', dict(pc=max(2, N//2), deg=4, coarse_deg=10))]
    base = None
    for k, (kind, kw) in enumerate(kinds):
        t0 = time.perf_counter()
        M = P.make(kind, st, fu, fv, Mi, PIN, **kw)
        cnt = [0]; orig = S.apply_A
        def counted(*A, **K):
            cnt[0] += 1; return orig(*A, **K)
        S.apply_A = counted; P.apply_A = counted
        try:
            _, it = pcg_solve(st, b, fu, fv, Mi, mw, pin_p=PIN, max_iter=30000,
                              tol=1e-6, cgsfac=1e-3, precond=M)
        finally:
            S.apply_A = orig; P.apply_A = orig
        tw = time.perf_counter()-t0
        if base is None: base = tw
        tag = kind + ('' if not kw else f" pc={kw['pc']} d={kw['deg']}" if kind == 'pmg2'
                      else f" d={kw['deg']}")
        lead = f"{EX}x{EX} order {N:<4}{ndof:>8}  " if k == 0 else " "*26
        print(f"{lead}{tag:<18}{it:>7}{cnt[0]:>9}{tw:>9.3f}{base/tw:>10.2f}x")
    print()
