"""Is the legacy rise at small dt the DISCRETISATION or the SOLVER?

At N = 12, dt = 9.77e-5 the legacy error rose to 2.40e-9 while the balanced
error fell to 8.25e-11.  Two mechanisms predict a rise:

  (a) the fixed point degenerates -- momentum weighted dt^2 below the
      constraints (ZIGZAG_CURE_RESEARCH.md sec 4.9): a property of the discrete
      problem, which no solver can repair;
  (b) the legacy operator's condition number grows like 1/dt^2, so a
      double-precision solve cannot deliver the answer.

The discriminator is the SOLVER TOLERANCE, and it must be the tolerance that
actually binds.  pcg_solve stops at max(cgsfac*|b|, cg_tol); in the sweep the
relative part fell below the absolute floor, so varying cgsfac alone changed
NOTHING (three runs identical to five digits -- a no-op test).  Here cgsfac = 0
so cg_tol binds, swept over eight orders, with the CG iteration counts reported
as evidence that the solves really are converging.

Insensitive error => (a).  Error tracking the tolerance => (b).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mms2d_temporal import march

if __name__ == '__main__':
    N, E, dt = 12, 2, 9.765625e-05
    print(f'N={N} {E}x{E} dt={dt:.4e} ({int(round(0.2/dt))} steps), cgsfac=0 so cg_tol binds')
    print(f'{"cg_tol":>9} | {"legacy":>12} {"CG/solve":>9} {"max":>5} | {"balanced":>12} {"CG/solve":>9} {"max":>5} | {"ratio":>7}')
    for tol in (1e-8, 1e-12, 1e-16):
        e = {w: march(N, E, dt, w, cgsfac=0.0, cg_tol=tol, report_cg=True)
             for w in ('legacy', 'balanced')}
        print(f'{tol:9.0e} | {e["legacy"]["uv"]:12.4e} {e["legacy"]["cg"]:9.1f} {e["legacy"]["cgmax"]:5.0f} |'
              f' {e["balanced"]["uv"]:12.4e} {e["balanced"]["cg"]:9.1f} {e["balanced"]["cgmax"]:5.0f} |'
              f' {e["legacy"]["uv"]/e["balanced"]["uv"]:7.2f}', flush=True)
