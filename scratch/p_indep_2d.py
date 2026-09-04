"""Is the 2D ladder p-independent under the SAME protocol as the 3D sweep?

sec 6.9 reports 2D ladder growth 1.05x over N=8..24 -- but those are AVERAGE CG
iterations per BDF step while driving the cavity to steady state (hence the
non-integer 605.8, 32.4).  Near steady the right-hand side is a small, smooth
physical increment lying mostly in well-conditioned directions.  The 3D sweeps
solve a COLD manufactured RHS b = A x_rand, which excites the whole spectrum.

Those are different experiments, and the difference favours 2D.  So run 2D the
3D way: cold, manufactured RHS, same tolerance, halving ladder, direct coarse.

If 2D also grows ~5x, the "1.05x" is a protocol artifact and the 3D ladder was
never broken.  If 2D stays flat, the difference is real and lives in the
formulation.
"""
import os, sys
for _v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '8')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); os.chdir(_R)
import numpy as np
import lssem2d
from lssem2d import precond as P, solver as S
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState
from lssem2d.mesh import build_channel

# dt=1e30 was a BUG: it makes |b| ~ 1e62 and CG's convergence test degenerate,
# so every case "converged" in 1 iteration.  DT=1.0 is what sec 6.9's own script
# uses; 1.85e-4 matches the 3D channel's mass coefficient c = 1/(beta*dt) = 5405,
# since the 2D analogue is fac1/dt.
RE, EX, TOL = 1000.0, 4, 1e-8
DTS = (1.0, 1.85e-4)


def ladder(N):
    seq, p = [N], N
    while p > 2:
        p = max(2, p//2); seq.append(p)
    return tuple(seq[1:])          # PMG2 wants the COARSE sequence


def main():
    lssem2d.set_backend('numba')
    print(f'2D cavity operator, Re={RE:.0f}, {EX}x{EX} elements, '
          f'COLD manufactured RHS, tol={TOL:g}\n', flush=True)
    for DT in DTS:
      print(f'\n--- dt={DT:g}  (2D mass coeff fac1/dt = {1/DT:.0f}) ---')
      print(f'{"N":>3} {"gDOF":>8} {"jacobi":>8} {"ladder+direct":>14} {"ratio":>7}')
      rows = []
      for N in (8, 12, 16, 20, 24):
          m = build_channel(1.0, 1.0, EX, EX, N, bcs=(1, 1, 1, 2))
          m.compute_global_indices()
          st = SolverState(m, diff_matrix(N), nu=1.0/RE, dt=DT, fac1=1.0)
          n = N + 1
          fu = np.zeros((m.nelem, n, n)); fv = np.zeros_like(fu)
          st.update_linearisation(fu, fv)
          mult = S.gather_scatter(m, np.ones((m.nelem, n, n, 4)))
          mw = 1.0/np.where(mult < 1e-10, 1.0, mult)
          Mi = S.compute_jacobi(st, fu, fv, pin_p=True)
          rng = np.random.default_rng(0)
          x = rng.standard_normal((m.nelem, n, n, 4))
          b = S.apply_A(st, x, fu, fv, pin_p=True)
          b /= np.linalg.norm(b)

          def run(pre):
              out = S.pcg_solve(st, b, fu, fv, Mi, mw, pin_p=True,
                                max_iter=40000, tol=TOL, precond=pre)
              it = out[1] if isinstance(out, tuple) else 0
              return int(it[0] if isinstance(it, (list, tuple)) else it)

          itj = run(None)
          pmg = P.make('pmg2', st, fu, fv, Mi, True, pc=ladder(N), deg=6,
                       coarse_solver='direct')
          itp = run(pmg)
          print(f'{N:3d} {x.size:8d} {itj:8d} {itp:14d} {itj/max(itp,1):6.1f}x', flush=True)
          rows.append((itj, itp))
      r = np.array(rows, float)
      print(f'  growth N=8->24:  jacobi {r[-1,0]/r[0,0]:.2f}x   '
            f'ladder {r[-1,1]/r[0,1]:.2f}x')
    print('  sec 6.9 (per-BDF-step averages): jacobi 4.00x, ladder 1.05x')
    print('  3D, same cold protocol:          jacobi 5.12x, ladder 5.56x')


if __name__ == '__main__':
    main()
