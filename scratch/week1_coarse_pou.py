"""Week 1 of COARSE_AND_ITERATIONS_PLAN.md, on rig 1 (the 2D harness).

    uv run --quiet python scratch/week1_coarse_pou.py

Three of the four week-1 items are measured here, because all three are
iteration-count questions and the 2D cavity operator answers those in seconds:

  b1  PARTITION-OF-UNITY WEIGHTING.  The patch contributions are currently summed
      plainly, so a dof inside the mesh is corrected once per patch containing it
      -- four times in 2D -- and the additive sum over-corrects by that factor.
      Weighting by the inverse patch count, symmetrically so the preconditioner
      stays SPD, is standard for overlapping Schwarz on spectral elements and is
      reported to buy 1.5-3x.  `pou=True` on the 2D classes.

  a2  pc = 1 INSTEAD OF pc = 2.  An 11x smaller coarse space in 3D (133 nodes per
      mode against 481), so the 5.3 GB dense coarse inverse that dominates the
      A100 apply becomes 0.47 GB.  The only question is what it costs in
      iterations.

  b3  RICHER COARSE SPACE, pc = 3 or 4.  The opposite trade: more iterations
      saved, a coarse solve too big for a dense inverse, which is what would
      justify building the sparse device solver (a3).  Measured here with the
      host direct solver, so the case for a3 is quantified before it is built.

Protocol is vs2d_bench.py's, unchanged so the numbers are comparable: Re = 1000
cavity operator, cold random right-hand side normalised in the gather-scatter
norm, matrix-free production `pcg_solve`, tol 1e-8, pressure pinned.  c = 5405
is the channel's value (dt = 1.85e-4); c = 1 is the steady regime, included
because §7 of the paper claims the patch method is flat across both and the
weighting must not break that.
"""
import os
import sys
import time

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '8')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R)
sys.path.insert(0, os.path.join(_R, 'scratch'))
os.chdir(_R)

import numpy as np

import lssem2d
from lssem2d import solver as S
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState
from lssem2d.mesh import build_channel
from vertex_schwarz2d import VertexSchwarzCondensed2D, make_coarse

RE, TOL = 1000.0, 1e-8
CAP = 4000


def case(N, dt, ex):
    m = build_channel(1.0, 1.0, ex, ex, N, bcs=(1, 1, 1, 2))
    m.compute_global_indices()
    st = SolverState(m, diff_matrix(N), nu=1.0/RE, dt=dt, fac1=1.0)
    n = N + 1
    fu = np.zeros((m.nelem, n, n)); fv = np.zeros_like(fu)
    st.update_linearisation(fu, fv)
    mult = S.gather_scatter(m, np.ones((m.nelem, n, n, 4)))
    mw = 1.0/np.where(mult < 1e-10, 1.0, mult)
    Mi = S.compute_jacobi(st, fu, fv, pin_p=True)
    x = np.random.default_rng(0).standard_normal((m.nelem, n, n, 4))
    b = S.apply_A(st, x, fu, fv, pin_p=True)
    b /= np.sqrt(np.sum(b*b*mw))

    def run(pre, cap=CAP):
        t0 = time.perf_counter()
        _, it = S.pcg_solve(st, b, fu, fv, Mi, mw, pin_p=True, max_iter=cap,
                            tol=TOL, precond=pre)
        return int(it), time.perf_counter() - t0

    return st, fu, fv, Mi, run, x.size


def coarse_dofs(st, pc):
    """Unknowns in the pc-level coarse space, i.e. what a dense inverse would hold."""
    m = st.mesh
    return (m.nelemx*pc + 1)*(m.nelemy*pc + 1)*4 if hasattr(m, 'nelemx') else None


def main():
    lssem2d.set_backend('numba')
    print(f'2D cavity Re={RE:.0f}, cold RHS, tol={TOL:g}, production pcg_solve\n')
    for dt, lab in ((1.85e-4, 'c = 5405 (the channel)'), (1.0, 'c = 1 (steady)')):
        for ex, N in ((4, 8), (4, 12), (8, 8)):
            st, fu, fv, Mi, run, nd = case(N, dt, ex)
            ij, tj = run(None, cap=40000)
            print(f'--- {lab}   {ex}x{ex} elements, N={N}, {nd} dofs '
                  f'| Jacobi {ij} it, {tj:.1f}s')
            print(f'{"coarse":>8} | {"plain sum":>18} | {"partition of unity":>18} | gain')
            for pc in (None, 1, 2, 3, 4):
                row = []
                for pou in (False, True):
                    try:
                        co = None if pc is None else make_coarse(st, fu, fv, Mi, True, pc=pc)
                        pre = VertexSchwarzCondensed2D(st, fu, fv, pin_p=True,
                                                       coarse=co, pou=pou)
                        it, t = run(pre)
                        row.append((it, t))
                    except Exception as e:
                        row.append((None, str(e).split('\n')[0][:40]))
                lab_pc = 'none' if pc is None else f'p={pc}'
                cells = []
                for it, t in row:
                    cells.append(f'{it:6d} it {t:7.1f}s' if it is not None else f'{"FAILED":>18}')
                gain = ''
                if row[0][0] and row[1][0]:
                    gain = f'{row[0][0]/row[1][0]:.2f}x'
                print(f'{lab_pc:>8} | {cells[0]} | {cells[1]} | {gain}', flush=True)
            print()


if __name__ == '__main__':
    main()
