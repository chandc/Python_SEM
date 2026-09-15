"""INCONCLUSIVE -- kept as a record of a measurement that does not work yet.

THE CONTROL FAILS.  This reports the VVP least-squares path at temporal order
~0.97, and Gate 1 verifies that path at 2.00.  A method that gets a known
answer wrong cannot be trusted for the unknown ones, so none of the constants
below should be quoted.

WHAT WAS TRIED, in order, and what each ruled out:

  reference 4x finer than the smallest dt   orders 2.65 / 0.58 / 1.16
  reference 16x finer                        orders 2.58 / 0.37 / 1.04
      -> the reference was not the problem.
  error in ENERGY                            LS errors reached 5.8e-11 against
      a CG tolerance of 1e-11 -- at the solver floor, and energy over this
      window changes only in the 6th digit, so it was measuring cancellation.
  error in the VELOCITY FIELD                orders 0.97 / 0.39 / 0.32
      -> better conditioned, still wrong.

WHAT I SUSPECT AND HAVE NOT TESTED.  At dt = 0.02 with TEND = 0.2 the coarse
run is only TEN steps, so what is being measured may be startup behaviour
rather than asymptotic accumulation, while the 320-step reference is in a
different regime entirely.  A max-norm over the field can also be dominated by
a handful of points.

WHAT WOULD MAKE IT WORK: many more steps at every dt (longer TEND, or a dt
ladder that starts finer), an L2 norm rather than max, and -- first -- a run
that REPRODUCES 2.00 for the least-squares path before any comparison is drawn
from it.  Until that control passes, the accuracy-constant question is open.

Original docstring: Accuracy CONSTANTS on TGV: at equal dt, how much temporal error does each path incur?

    python scratch/fs_constants.py

Both paths are second order.  Order is not the question -- the CONSTANT is, and
on the channel they differ by ~10x (1.055e-4 against 1.045e-5 at dt = 0.0025),
which means matching LSSEM's accuracy needs dt about 3.2x smaller and eats into
the ~18x speedup.  The TGV comparison never checked this.

METHOD: temporal SELF-convergence, each path against its own fine-dt limit.
The reference must be MUCH finer than the smallest measured dt -- a first
attempt used 4x and produced orders of 2.65, 0.58 and 1.16, none of them clean,
because the reference carried error comparable to what it was measuring.  16x
here.
A shared reference would be wrong: the two paths have DIFFERENT spatial
discretisations -- 14-field least-squares against velocity-plus-pressure -- so
they converge to the same continuous solution but not to the same discrete one.
Differencing them would measure that gap, not the temporal error.  Self-
convergence isolates the temporal term, which is what the constant means.
"""
import sys, time
sys.path.insert(0, '.'); sys.path.insert(0, 'scratch')
import numpy as np
import lssem3d; lssem3d.set_backend('numpy')
from lssem3d import (project as PJ, helmholtz as HH, convect as CV,
                     operator as OP, solver3d as S3, timestep as T)
import fs_phase2 as F2
import tgv_gpu_run as TG

N, NE, NZ, NU, TEND = 8, 2, 8, 0.01, 0.2
DTS = (0.02, 0.01, 0.005)
DT_REF = 0.000625    # 8x finer than the smallest measured dt
TOL = 1e-11


def run_fs(dt, km):
    s = F2.build(N=N, ne=NE, nz=NZ, nu=NU, tol=TOL)
    Uc = F2.ic_tgv(s)
    pc = np.zeros((s['m'].nelem, N+1, N+1, 1, s['nk']), dtype=complex)
    Np_ = np.zeros((s['m'].nelem, N+1, N+1, 3, s['nk']), dtype=complex)
    nstep = int(round(TEND/dt))
    if km:
        s['Mu'] = HH.fdm_preconditioner(s['m'], N, 2.0/dt + NU*(s['kz']**2),
                                        NU, s['mask_u'], 6, s['nk'])
        s['force'] = None
        phi = None
        for _ in range(nstep):
            Uc, phi, _ = PJ.step_kim_moin(s, Uc, phi, dt)
    else:
        pre = [HH.fdm_preconditioner(s['m'], N,
                                     T.implicit_coeff(dt, k) + NU*(s['kz']**2),
                                     NU, s['mask_u'], 6, s['nk'])
               for k in range(T.NSTAGE)]
        for _ in range(nstep):
            for k in range(T.NSTAGE):
                s['Mu'] = pre[k]
                Nk = -CV.convective(Uc, s['Dg'], s['fxg'], s['fyg'], s['kzg'],
                                    NZ)
                Uc, pc, _ = PJ.substage(s, Uc, pc, Nk, Np_, k, dt)
                Np_ = Nk
    return np.asarray(Uc)


def run_ls(dt):
    ops = TG.Ops('numpy')
    cfg = dict(nu=NU, N=N, ex=NE, ey=NE, nz=NZ, tend=1.0, snap=1.0, cfl=1.0,
               tol=TOL)
    s = TG.setup(cfg, np, ops)
    U = TG.ic_tgv(s)
    Np_ = np.zeros(OP.to_complex(U).shape[:-2] + (3, s['nk']), dtype=complex)
    Minv, rws = TG.precond(s, dt)
    for _ in range(int(round(TEND/dt))):
        for k in range(T.NSTAGE):
            U, Np_, _ = TG.stage(s, U, Np_, k, dt, Minv, rws[k], TOL,
                                 check_every=1)
    # VELOCITY only: the two paths carry different state (14 fields against
    # velocity-plus-pressure), and u, v, w is what they share.
    Uc = OP.to_complex(U)
    return np.ascontiguousarray(Uc[..., [OP.U_, OP.V_, OP.W_], :])


for name, fn in (('VVP LSSEM', lambda d: run_ls(d)),
                 ('FS per-substage', lambda d: run_fs(d, False)),
                 ('FS Kim-Moin', lambda d: run_fs(d, True))):
    t0 = time.perf_counter()
    ref = fn(DT_REF)
    errs = []
    for dt in DTS:
        # FIELD difference, not energy.  Energy over this window changes in
        # the 6th digit, so differencing it measured cancellation and the CG
        # tolerance floor (LS errors came out at 5.8e-11 against tol 1e-11)
        # rather than the temporal term.
        e = float(np.abs(fn(dt) - ref).max()/np.abs(ref).max())
        errs.append(e)
    o = [np.log2(errs[i]/errs[i+1]) for i in range(len(errs)-1)]
    C = [errs[i]/DTS[i]**2 for i in range(len(DTS))]
    print(f'{name:<16} ' + '  '.join(f'{e:.2e}' for e in errs)
          + f'   order {np.mean(o):.2f}   C={np.mean(C):.2e}'
          + f'   [{time.perf_counter()-t0:.0f}s]', flush=True)
print(f'\n  columns: dt = {", ".join(str(d) for d in DTS)}')
print(f'  reference: each path at dt = {DT_REF} (its OWN fine-dt limit)')
print('  C = err/dt^2, the temporal error CONSTANT.  Lower is better.')
