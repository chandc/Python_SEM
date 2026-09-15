"""The missing factor of 2: profile a WHOLE STEP of the channel under numba.

numba_where_now.py accounts for 8.4x on the kernels dropping to a predicted 5.0x
per CG iteration (34% of an iteration is now outside the fused code).  The
channel measured 2.36-2.49x.  So roughly another factor of 2 is unexplained, and
it must live in per-STEP work rather than per-iteration work: convection, the
RHS/defect assembly, the preconditioner build, or the thread pool.

cProfile on two steps, sorted by cumulative time, says which.
"""
import os, sys, cProfile, pstats, io as _io
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
SC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SC)
sys.path.insert(0, ROOT); sys.path.insert(0, SC); os.chdir(ROOT)
from lssem3d import backend
import channel3d_stage5 as S5


def main(nstep=2):
    backend.set_backend('numba')
    S5.run_case(0.01, 'perturbed', False, nstep=1, verbose=False, rowweight=True)
    pr = cProfile.Profile(); pr.enable()
    S5.run_case(0.01, 'perturbed', False, nstep=nstep, verbose=False, rowweight=True)
    pr.disable()
    backend.set_backend('numpy')
    s = _io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats('cumulative').print_stats(22)
    print(s.getvalue()[:4200])


if __name__ == '__main__':
    main()
