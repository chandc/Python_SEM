"""Does the minimal-channel DNS depend on its time step?

run01 ran at dt = 8e-4 with the legacy weighting, which puts the momentum
equation at m*a = 1/c = 1.3e-4 against the constraints -- four orders below the
safe value, and a hundred times deeper into the affected regime than the
Orr-Sommerfeld case where the same weighting cost 3-6% of a growth rate
(ZIGZAG_CURE_RESEARCH.md sec 4.5).  The criterion flags the production run.
Whether it MATTERS is a different question, because the 3D stage is a Stokes
projection with explicit convection and the 2D mechanism demonstrably does not
fire there (no mesh-scale mode in run01's fields, sec 4.8).

THE TEST.  Restart the same state at several dt, integrate to the SAME physical
time, and compare.  Richardson on the production configuration:

    || U(dt) - U(dt/2) ||  ~  C dt^2   for a clean 2nd-order scheme (RKW3/CN is
    2nd order overall -- Crank-Nicolson limits it, lssem3d/tests/test_stage4).

A larger difference, or an order below 2, says the trajectory carries a
time-step bias at production settings.

TWO THINGS THAT WOULD OTHERWISE CONTAMINATE IT.

  * The solver.  minchan's production tolerance is 1e-6 per stage; over 100
    steps that noise (~1e-5) would swamp a second-order signal of ~1e-8.  So the
    study runs at 1e-12 and asks the discretisation question.  Whether the
    PRODUCTION tolerance adds its own dt-dependence is a separate run.
  * The restart.  Changing dt across a restart is exact here: RKW3 has
    ZETA[0] = 0, so the carried convective history is never used at stage 0 and
    is overwritten within the step.  No BDF-style history mismatch (contrast the
    2D driver, where it cost a 4e-3 transient).

    python scratch/minchan_dtsens.py --dt 8e-4 --horizon 0.08 --out /runs/dt8e-4
"""
import argparse
import os
import sys
import time

import numpy as np

_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R)
sys.path.insert(0, os.path.join(_R, 'scratch'))
os.chdir(_R)

from lssem3d import backend, operator as OP, device as DEV
import minchan as MC


def march(ckpt, dt, horizon, out, precond='vsbatch', weighting='legacy',
          tol=1e-12, coarse_dense=1, N=8, ex=6, ey=18, nz=32):
    nstep = int(round(horizon/dt))
    s = MC.setup(N, ex, ey, nz)
    z = np.load(ckpt)
    U = z['U']
    t0_phys = float(z['t'])
    print(f'dt={dt:.3e}  {nstep} steps to t = {t0_phys:.4f} + {horizon:g}'
          f'   precond={precond} weighting={weighting} tol={tol:g}', flush=True)
    dev = backend.get_backend() in ('torch', 'cuda', 'cupy')
    Minv = MC._precond(s, dt, precond=precond, weighting=weighting,
                       coarse_dense=bool(coarse_dense))
    if dev:
        s_run, U = MC.to_device(s, U)
        Minv = [DEV.to_device(q, U) if not callable(q) else q for q in Minv]
    else:
        s_run = s
    Nprev = DEV.to_device(
        np.zeros(OP.to_complex(np.asarray(z['U'])).shape[:-2] + (3, s['nk']), dtype=complex), U) \
        if dev else np.zeros(OP.to_complex(z['U']).shape[:-2] + (3, s['nk']), dtype=complex)
    t0 = time.perf_counter()
    for i in range(nstep):
        U, Nprev, it = MC.advance(s_run, U, Nprev, dt, Minv, tol=tol,
                                  weighting=weighting)
        if (i + 1) % 10 == 0 or i == 0:
            Uh = DEV.to_host(U)
            print(f'  step {i+1:5d}/{nstep}  u_tau={MC.u_tau(s, Uh):.6f} '
                  f'rms_w={MC.rms_w(s, Uh):.6f} div={MC.momentum_budget(s, Uh)[3]:.3e} '
                  f'CG={it}  [{time.perf_counter()-t0:.0f}s]', flush=True)
    Uh = DEV.to_host(U)
    ke, eps = MC.energy(s, Uh)
    np.savez_compressed(out, U=Uh, dt=dt, nstep=nstep, horizon=horizon,
                        t=t0_phys + horizon, tol=tol, weighting=weighting,
                        u_tau=MC.u_tau(s, Uh), bulk=MC.bulk(s, Uh),
                        rms_w=MC.rms_w(s, Uh), ke=ke, eps=eps,
                        div=MC.momentum_budget(s, Uh)[3])
    print(f'  DONE dt={dt:.3e}: u_tau={MC.u_tau(s, Uh):.6f} E={ke:.4f} '
          f'eps={eps:.4f} div={MC.momentum_budget(s, Uh)[3]:.4e} '
          f'[{time.perf_counter()-t0:.0f}s] -> {out}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--dt', type=float, required=True)
    ap.add_argument('--horizon', type=float, default=0.08)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tol', type=float, default=1e-12)
    ap.add_argument('--precond', default='vsbatch')
    ap.add_argument('--weighting', default='legacy')
    a = ap.parse_args()
    backend.set_backend(os.environ.get('LSSEM3D_BACKEND', 'cuda'))
    march(a.ckpt, a.dt, a.horizon, a.out, precond=a.precond,
          weighting=a.weighting, tol=a.tol)
