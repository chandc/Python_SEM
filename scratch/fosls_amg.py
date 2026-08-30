"""F2 -- can AMG deliver h-independent iterations on the FOSLS operator?

    uv run --quiet python scratch/fosls_amg.py

SCOPE, set by F1's retraction.  At production dt the mass term dominates, the
problem is ill-conditioned regardless of preconditioner, and Jacobi is
near-optimal -- so AMG has little to offer the TIME-STEPPER.  Its payoff is
confined to the STEADY solver, which is where w_mom=1 is already used.  Everything
here runs steady (dt=1e4) with w_mom=1, the regime F1 measured.

MUST BE RUN AT >= 6x6 ELEMENTS.  Jacobi's condition number and the H^1 ceiling
CROSS at 4x4: below that Jacobi wins outright and an AMG test would show it
losing for reasons that have nothing to do with AMG.  Testing at 2x2, which is
what a quick check would naturally do, would be actively misleading.

THE GATE, stated in advance: iterations FLAT WITHIN 20% across a 4x h-refinement.
A constant-factor reduction is the result sec 7K already rejected (p-MG: 7.4x
fewer iterations, 0.28x wall) and must be recorded as such, not reported as a win.

The predicted ceiling is sqrt(c2/c1) ~ 124 iterations at nu=1e-2, flat.  Jacobi
grows 2-3x per refinement level, so the two diverge without bound.

STEP 1 HERE IS THE BASELINE THE PLAN PREDICTED WOULD FAIL: AMG on the SEM matrix
directly.  Its element blocks are DENSE (324x324 at N=8), and AMG's premise is
strong connections in a sparse graph, so it should coarsen badly.  Measuring it
gives the contrast for the LOR operator and costs nothing -- the matrix exists.
"""
import os
import sys
import time

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R)
sys.path.insert(0, os.path.join(_R, 'scratch'))

import numpy as np
import pyamg
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, cg

from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState
from lssem2d.mesh import build_channel

import fosls_assemble as FA

NV = FA.NV


def case(N, ex, ey, nu=1/100., dt=1e4):
    m = build_channel(2.0, 1.0, ex, ey, N, bcs=(1, 1, 1, 2))
    m.compute_global_indices()
    st = SolverState(m, diff_matrix(N), nu=nu, dt=dt, fac1=1.0, w_mom=1.0)
    n = N + 1
    fu = np.zeros((m.nelem, n, n)); fv = np.zeros((m.nelem, n, n))
    st.update_linearisation(fu, fv)
    A, free, g = FA.assemble(st, fu, fv, pin_p=True)
    return A[free][:, free].tocsr(), free


def count_cg(A, M=None, tol=1e-8, maxit=20000, seed=0):
    """CG iterations to tol.  Same RHS construction for every preconditioner."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(A.shape[0])
    b = A @ x
    it = [0]

    def cb(_xk):
        it[0] += 1
    try:
        cg(A, b, rtol=tol, maxiter=maxit, M=M, callback=cb)
    except TypeError:                       # older scipy
        cg(A, b, tol=tol, maxiter=maxit, M=M, callback=cb)
    return it[0]


def near_null(nfree, free):
    """One constant per FIELD -- the natural near-kernel of a 4-field system.

    pyamg defaults to a single all-ones vector, which for a coupled system with
    four unknowns per node is wrong: it cannot represent a mode constant in one
    field and zero in the others.
    """
    B = np.zeros((nfree, NV))
    idx = np.arange(free.size)[free]
    for v in range(NV):
        B[idx % NV == v, v] = 1.0
    return B


def _hsweep():
    print("F2 -- AMG variants on the SEM matrix, h-refinement at N=4\n")
    print("  step 1 used B = ones (a SCALAR near-null space) on a 4-FIELD system,")
    print("  and max_coarse=200 gave only TWO levels.  Both are fixed below.\n")

    variants = [
        ('scalar B, 2 lvl ', dict(B=None, max_coarse=200)),
        ('scalar B, deep  ', dict(B=None, max_coarse=20)),
        ('4-field B, deep ', dict(B='block', max_coarse=20)),
        ('4-field B, energy', dict(B='block', max_coarse=20, smooth='energy')),
    ]
    meshes = ((4, 4), (6, 6), (8, 8), (12, 12))
    print(f'{"variant":>18} ' + ' '.join(f'{f"{a}x{b}":>8}' for a, b in meshes)
          + f' {"growth":>9}')

    cache = {}
    for m in meshes:
        cache[m] = case(4, *m)

    # Jacobi reference
    row = []
    for m in meshes:
        Af, free = cache[m]
        row.append(count_cg(Af, sp.diags(1.0/Af.diagonal())))
    print(f'{"JACOBI":>18} ' + ' '.join(f'{r:8d}' for r in row)
          + f' {row[-1]/row[0]:8.2f}x')

    for lbl, kw in variants:
        row, ok = [], True
        for m in meshes:
            Af, free = cache[m]
            n = Af.shape[0]
            B = (near_null(n, free) if kw.get('B') == 'block'
                 else np.ones((n, 1)))
            args = {k: v for k, v in kw.items() if k != 'B'}
            args.setdefault('smooth', 'jacobi')
            try:
                ml = pyamg.smoothed_aggregation_solver(Af, B=B, **args)
                M = LinearOperator((n, n), matvec=ml.aspreconditioner().matvec)
                row.append(count_cg(Af, M))
            except Exception as e:
                row.append(-1); ok = False
        g = f'{row[-1]/row[0]:8.2f}x' if ok and row[0] > 0 else '     --'
        print(f'{lbl:>18} ' + ' '.join(f'{r:8d}' for r in row) + f' {g}')

    print('\n  growth = iterations at 12x12 / at 4x4, i.e. over a 3x refinement.')
    print('  GATE: flat within 20% (growth < 1.2x).  Jacobi is the thing to beat;')
    print('  a constant factor is the sec 7K result already rejected.')


def order_and_walltime():
    """Does AMG-on-SEM survive p-refinement, and does it pay in WALL TIME?

    Two things the h-sweep cannot answer.

    (1) ORDER.  The plan argued AMG must go on a low-order refined operator
        because SEM element blocks are DENSE -- 324x324 at N=8, 1156x1156 at
        N=16.  At N=4 the block is only 100x100 and AMG coped fine, so that
        argument is untested where it actually bites.

    (2) WALL TIME.  sec 7K rejected p-multigrid at 0.28x wall DESPITE 7.4x fewer
        iterations.  AMG's setup is dearer than p-MG's.  Iterations are necessary,
        not sufficient, and this project has already been burned by quoting the
        iteration column alone.
    """
    print('\n\nF2b -- p-refinement and wall time (6x6 elements, fixed)\n')
    print(f'{"N":>4} {"ndof":>7} {"blk":>7} {"dens":>7} {"jac it":>8} {"AMG it":>7} '
          f'{"setup":>8} {"jac s":>8} {"AMG s":>8} {"speedup":>8}')
    for N in (4, 6, 8, 10):
        Af, free = case(N, 6, 6)
        n = Af.shape[0]
        B = near_null(n, free)
        Mj = sp.diags(1.0/Af.diagonal())
        t0 = time.perf_counter(); ij = count_cg(Af, Mj); tj = time.perf_counter()-t0
        t0 = time.perf_counter()
        ml = pyamg.smoothed_aggregation_solver(Af, B=B, max_coarse=20,
                                               smooth='energy')
        ts = time.perf_counter()-t0
        M = LinearOperator((n, n), matvec=ml.aspreconditioner().matvec)
        t0 = time.perf_counter(); ia = count_cg(Af, M); ta = time.perf_counter()-t0
        blk = (N+1)**2*NV
        print(f'{N:4d} {n:7d} {blk:6d}^2 {Af.nnz/n**2:7.4f} {ij:8d} {ia:7d} '
              f'{ts:7.2f}s {tj:7.2f}s {ta:7.2f}s {tj/(ta+ts):7.2f}x')
    print('\n  speedup counts AMG SETUP + solve against Jacobi solve -- the honest')
    print('  comparison, and the one sec 7K applied when it rejected p-MG.')


def h_walltime():
    """Wall time under h-REFINEMENT -- where the flat iteration count should pay.

    The p-sweep showed AMG only breaking even: it wins 7-11x on iterations and
    loses ~7-10x on cost per iteration.  But p-refinement is not where the
    argument lives.  Under h-refinement Jacobi's count GROWS while AMG's is flat,
    so the ratio should widen and the wall time should follow.

    CAVEAT THAT LIMITS ALL OF THESE NUMBERS.  This compares two PYTHON
    implementations: scipy CG with a Python callback, against pyamg's
    Python-level V-cycle.  Production runs a fused numba/CUDA matvec for the
    Jacobi path and would need an equally tuned V-cycle to be comparable.  The
    ITERATION counts transfer; the wall times do not.  sec 7K's 0.28x was measured
    the same way and carries the same caveat.
    """
    print('\n\nF2c -- wall time under h-refinement (N=4)\n')
    print(f'{"mesh":>7} {"ndof":>7} {"jac it":>8} {"AMG it":>7} {"ratio":>7} '
          f'{"jac s":>8} {"AMG s":>8} {"speedup":>8}')
    for ex, ey in ((4, 4), (8, 8), (12, 12), (16, 16)):
        Af, free = case(4, ex, ey)
        n = Af.shape[0]
        B = near_null(n, free)
        t0 = time.perf_counter()
        ij = count_cg(Af, sp.diags(1.0/Af.diagonal()))
        tj = time.perf_counter()-t0
        t0 = time.perf_counter()
        ml = pyamg.smoothed_aggregation_solver(Af, B=B, max_coarse=20,
                                               smooth='energy')
        M = LinearOperator((n, n), matvec=ml.aspreconditioner().matvec)
        ia = count_cg(Af, M)
        ta = time.perf_counter()-t0
        print(f'{f"{ex}x{ey}":>7} {n:7d} {ij:8d} {ia:7d} {ij/ia:6.1f}x '
              f'{tj:7.2f}s {ta:7.2f}s {tj/ta:7.2f}x')
    print('\n  AMG s INCLUDES setup.  Iteration counts transfer to production;')
    print('  wall times do not -- both sides here are Python.')


def standalone_vcycle():
    """AMG as a SOLVER, not a preconditioner -- the more demanding test.

    Every F2 number so far is preconditioned CG.  But CG can carry a mediocre
    preconditioner: it compensates by building a Krylov space, so the iteration
    count can look healthy while the V-cycle itself barely converges.

    sec 7K found exactly that for p-multigrid -- "the V-cycle stalled with
    reduction factor exactly 1.0000 on those modes" -- and that stall is what
    exposed the row-7 problem.  So the standalone convergence factor

        rho = ||r_{k+1}|| / ||r_k||

    is the honest measure of whether AMG is SOLVING this operator.  Textbook
    multigrid gives rho ~ 0.1-0.3, h-independent.  rho close to 1 would mean CG
    was carrying a preconditioner that does not itself converge.
    """
    print('\n\nF2d -- AMG as a STANDALONE solver: V-cycle convergence factor\n')
    print(f'{"mesh":>7} {"ndof":>7} {"lvls":>5} {"op cx":>7} {"rho":>8} '
          f'{"V-cyc to 1e-8":>14} {"CG its":>7}')
    for ex, ey in ((4, 4), (8, 8), (12, 12), (16, 16)):
        Af, free = case(4, ex, ey)
        n = Af.shape[0]
        B = near_null(n, free)
        ml = pyamg.smoothed_aggregation_solver(Af, B=B, max_coarse=20,
                                               smooth='energy')
        rng = np.random.default_rng(0)
        b = Af @ rng.standard_normal(n)
        res = []
        ml.solve(b, x0=np.zeros(n), tol=1e-10, maxiter=60, residuals=res,
                 accel=None)
        r = np.asarray(res)
        k = max(2, len(r)//2)
        rho = float((r[-1]/r[k])**(1.0/(len(r)-1-k))) if len(r) > k+1 else float('nan')
        nv = int(np.ceil(np.log(1e-8)/np.log(rho))) if 0 < rho < 1 else -1
        M = LinearOperator((n, n), matvec=ml.aspreconditioner().matvec)
        print(f'{f"{ex}x{ey}":>7} {n:7d} {len(ml.levels):5d} '
              f'{ml.operator_complexity():7.3f} {rho:8.4f} {nv:14d} '
              f'{count_cg(Af, M):7d}')
    print('\n  rho ~ 0.1-0.3 and flat is textbook.  rho -> 1 would mean CG was')
    print('  carrying a preconditioner that does not itself converge (sec 7K).')



def near_null_computed(Af, k):
    """The operator's ACTUAL softest modes, via shift-invert."""
    from scipy.sparse.linalg import eigsh
    w, V = eigsh(Af.tocsc(), k=k, sigma=0.0, which='LM')
    return w, V


def p_sweep_diagnostics():
    """Does the N=4 story survive to PRODUCTION order?

    Every headline F2 number -- h-independence, rho ~ 0.97, the omega-dominated
    near-kernel, 138 -> 52 -- was measured at N=4, the LOWEST order this code
    supports.  Production is N=8.  That matters for one specific reason already
    written into this plan: the LOR argument was rejected on N=4 evidence, and
    the dense-block concern it rested on bites hardest at high N.  The element
    block is 100x100 at N=4 and 324x324 at N=8.

    So this re-runs the four diagnostics across p on a FIXED 6x6 mesh:
      rho          -- does the V-cycle degrade with order?
      CG its       -- does the field-constant coarse space still work?
      omega energy -- is the softest mode still omega-dominated?
      computed B   -- does supplying the true modes still pay?
    """
    print('\n\nF2e -- do the N=4 conclusions survive p-refinement?  (6x6 mesh)\n')
    print(f'{"N":>3} {"blk":>9} {"ndof":>7} {"jac":>6} {"AMG":>5} {"rho":>7} '
          f'{"om frac":>8} {"CG cmp":>7} {"gain":>6}')
    rows = []
    for N in (4, 6, 8):
        Af, free = case(N, 6, 6)
        n = Af.shape[0]
        B = near_null(n, free)
        ij = count_cg(Af, sp.diags(1.0/Af.diagonal()))
        ml = pyamg.smoothed_aggregation_solver(Af, B=B, max_coarse=20,
                                               smooth='energy')
        ia = count_cg(Af, LinearOperator((n, n),
                                         matvec=ml.aspreconditioner().matvec))
        rng = np.random.default_rng(0)
        b = Af @ rng.standard_normal(n)
        res = []
        ml.solve(b, x0=np.zeros(n), tol=1e-10, maxiter=60, residuals=res,
                 accel=None)
        r = np.asarray(res); k = max(2, len(r)//2)
        rho = float((r[-1]/r[k])**(1.0/(len(r)-1-k))) if len(r) > k+1 else float('nan')

        # softest mode: how much of its energy sits in omega?
        w, V = near_null_computed(Af, 8)
        idx = np.arange(free.size)[free]
        v0 = V[:, 0]
        frac = [float((v0[idx % NV == f]**2).sum()) for f in range(NV)]
        omf = frac[3]/sum(frac)

        # does supplying the TRUE modes still pay at this order?
        mlc = pyamg.smoothed_aggregation_solver(Af, B=V, max_coarse=20,
                                                smooth='energy')
        ic = count_cg(Af, LinearOperator((n, n),
                                         matvec=mlc.aspreconditioner().matvec))
        blk = (N+1)**2*NV
        print(f'{N:3d} {f"{blk}^2":>9} {n:7d} {ij:6d} {ia:5d} {rho:7.4f} '
              f'{omf:8.3f} {ic:7d} {ia/max(ic,1):5.2f}x')
        rows.append((N, n, ij, ia, rho, omf, ic))
    print('\n  om frac = fraction of the SOFTEST mode energy in omega.')
    print('  CG cmp  = CG its with 8 COMPUTED near-null modes as B.')
    print('  If rho and om frac are flat in N, the N=4 story is the N=8 story.')
    return rows


if __name__ == "__main__":
    p_sweep_diagnostics()
