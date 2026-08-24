"""GPU parity for the projection path: does cupy reproduce numpy?

    python scratch/fs_parity.py [N ne nz nsteps]

The acceptance criterion is SELF-CALIBRATING, for a reason learned the hard way
on the least-squares port (CUPY_BACKEND.md).  An absolute threshold like 1e-8
demands the two backends agree more closely than either agrees with truth, and
a fraction-of-solver-error criterion fails because scatter-add atomics make the
GPU path non-deterministic run to run.  So: measure cupy's own re-run spread
first, then require the cross-backend difference to sit within a small multiple
of it.  Anything tighter is measuring noise.
"""
import sys, time
sys.path.insert(0, '.'); sys.path.insert(0, 'scratch')
import numpy as np
import lssem3d; lssem3d.set_backend('numpy')
from lssem3d import project as PJ, helmholtz as HH, convect as CV, timestep as T
import fs_phase2 as F2

a = [int(v) for v in sys.argv[1:5]]
N, ne, nz, nsteps = (a + [8, 3, 8, 3][len(a):])[:4]
NU, DT, TOL = 0.01, 0.01, 1e-10


def run(backend, seed_shift=0):
    lssem3d.set_backend(backend)
    s = F2.build(N=N, ne=ne, nz=nz, nu=NU, tol=TOL, backend=backend)
    # Match the CONVERGENCE TEST INTERVAL across backends.  With check_every
    # defaulting to 1 on numpy and 10 on cupy the two stop at different
    # iterations, so they differ at solver-tolerance level (3.8e-10 against a
    # 1e-10 tolerance) -- a real difference, but between two different
    # algorithms rather than between two backends running the same one.
    s['check_every'] = 1
    Uc = F2.ic_tgv(s)
    xp = np
    if backend == 'cupy':
        import cupy as xp
    pc = xp.zeros((s['m'].nelem, N+1, N+1, 1, s['nk']), dtype=complex)
    Nprev = xp.zeros((s['m'].nelem, N+1, N+1, 3, s['nk']), dtype=complex)
    pre = [HH.fdm_preconditioner(s['m'], N,
                                 T.implicit_coeff(DT, k) + NU*(s['kz']**2),
                                 NU, s['mask_u'], 6, s['nk'],
                                 like=s['mask_u'])
           for k in range(T.NSTAGE)]
    t0 = time.perf_counter()
    for i in range(nsteps):
        for k in range(T.NSTAGE):
            s['Mu'] = pre[k]
            Nk = -CV.convective(Uc, s['Dg'], s['fxg'], s['fyg'], s['kzg'],
                                s['nz'])
            Uc, pc, _ = PJ.substage(s, Uc, pc, Nk, Nprev, k, DT)
            Nprev = Nk
    if backend == 'cupy':
        xp.cuda.Stream.null.synchronize()
        el = time.perf_counter() - t0
        return xp.asnumpy(Uc), el
    return Uc, time.perf_counter() - t0


print(f'{ne}x{ne} elements, N={N}, Nz={nz}, {nsteps} steps, tol={TOL:.0e}\n')
un, tn = run('numpy')
print(f'  numpy  {tn:7.3f} s')
uc1, tc1 = run('cupy')
uc2, _ = run('cupy')
print(f'  cupy   {tc1:7.3f} s   ({tn/tc1:.1f}x)')
scale = np.abs(un).max()
spread = float(np.abs(uc1 - uc2).max()/scale)
cross = float(np.abs(uc1 - un).max()/scale)
print(f'\n  cupy re-run spread (its own non-determinism): {spread:.3e}')
print(f'  cupy vs numpy                                : {cross:.3e}')
tol_ = max(20*spread, 1e-12)
print(f'  criterion: within 20x the spread, i.e. < {tol_:.3e}')
print(f'  {"PASS" if cross < tol_ else "FAIL"}')
