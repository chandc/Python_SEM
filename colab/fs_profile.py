"""Where does a fractional-step time step actually go?  Phase profile on the GPU.

    python colab/fs_profile.py --steps 20 --backend cupy --consistent \
           --restart results/minchan_re180_E/state_t15.95.npz

WHY.  The least-squares path was optimised hard -- 25.4 s/step to 1.62 s on an
A100, mostly by finding that 84 % of the time was the preconditioner apply and
half of that was streaming one dense factor.  Comparing that against a
fractional-step code nobody has profiled would not be a comparison of
formulations, it would be a comparison of effort.  This measures the same thing
for the projection path so the optimisation can be done on both sides before
either number goes in the paper.

HOW IT AVOIDS MEASURING THE WRONG CODE.  It does not reimplement the setup: it
wraps three functions and then runs the production driver
(`fractional_step/scratch/fs_minchan_stats.py`) unchanged through runpy.  The
driver reaches those functions through module attributes -- `HH.solve`,
`CV.convective`, and `project_consistent` via `lssem3d.project`'s own globals --
so patching the module attribute intercepts the real call.  Whatever the driver
does, that is what is timed.

PHASES.  Per RKW3 substage, three substages per step:

  convective   CV.convective   explicit term, FFT-heavy, dealiased in z
  helmholtz    HH.solve        (c_k + nu k_z^2)M + nu K on the three velocities
  pressure     the projection: project_consistent (E-path) or HH.solve (K-path)

What is left over -- the weak assembly, the gradient, the rotational pressure
update, the statistics -- is reported as `other`, so the columns sum to the step.

GPU TIMING REQUIRES SYNCHRONISATION.  cupy queues kernels asynchronously, so an
unsynchronised timer measures launch overhead and attributes the real work to
whatever happens to synchronise next.  Every phase boundary synchronises here;
the cost of that is included in the total, which is the honest place for it.
"""
import argparse
import os
import runpy
import sys
import time

_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FS = os.path.join(_R, 'fractional_step')
if not os.path.isdir(FS):
    raise SystemExit(f'no vendored fractional-step tree at {FS}')
sys.path.insert(0, FS)
sys.path.insert(0, os.path.join(FS, 'scratch'))
os.chdir(FS)

import numpy as np


class Phase:
    """Accumulates wall time and solver iterations for one phase."""

    def __init__(self, name):
        self.name, self.t, self.n, self.it = name, [], 0, []

    def add(self, dt, it=None):
        self.t.append(dt)
        self.n += 1
        if it is not None:
            self.it.append(it)

    def tail(self, skip):
        """Totals excluding the first `skip` calls (warm-up, kernel compilation)."""
        return self.t[skip:], self.it[skip:] if self.it else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=20)
    ap.add_argument('--backend', default='cupy')
    ap.add_argument('--consistent', action='store_true',
                    help='the E path (consistent P_N-P_N); default is the K path')
    ap.add_argument('--restart', default=os.path.join(
        _R, 'results/minchan_re180_E/state_t15.95.npz'))
    ap.add_argument('--dt', type=float, default=3.5e-4)
    ap.add_argument('--tolp', default='1e-4')
    ap.add_argument('--outdir', default='/tmp/fs_profile')
    ap.add_argument('--graph', action='store_true',
                    help='replay the V-cycle from a CUDA graph (check it first '
                         'with colab/fs_graph_check.py)')
    a = ap.parse_args()

    import lssem3d
    lssem3d.set_backend(a.backend)
    from lssem3d import project as PJ, helmholtz as HH, convect as CV

    if a.backend == 'cupy':
        import cupy as xp
        sync = xp.cuda.runtime.deviceSynchronize
        dev = xp.cuda.runtime.getDeviceProperties(0)['name'].decode()
    else:
        sync = lambda: None
        dev = 'host (%s)' % a.backend

    P = {k: Phase(k) for k in ('convective', 'helmholtz', 'pressure')}

    def wrap(mod, name, phase, it_index=None):
        orig = getattr(mod, name)

        def timed(*args, **kw):
            sync(); t0 = time.perf_counter()
            out = orig(*args, **kw)
            sync()
            it = None
            if it_index is not None and isinstance(out, tuple) and len(out) > it_index:
                v = out[it_index]
                it = int(v) if np.isscalar(v) else None
            phase.add(time.perf_counter() - t0, it)
            return out
        setattr(mod, name, timed)
        return orig

    wrap(CV, 'convective', P['convective'])
    wrap(HH, 'solve', P['helmholtz'], it_index=1)
    wrap(PJ, 'project_consistent', P['pressure'], it_index=2)
    # The K path takes its pressure through HH.solve as well; it is separated
    # below by call order (velocity first, pressure second, per substage).
    k_path = not a.consistent

    t0 = float(np.load(a.restart)['t'])
    tend = t0 + a.steps*a.dt
    argv = ['fs_minchan_stats.py', '--restart', a.restart, '--backend', a.backend,
            '--dt', repr(a.dt), '--tend', repr(tend), '--outdir', a.outdir,
            '--tolp', a.tolp]
    if a.consistent:
        argv.append('--consistent')
    if a.graph:
        argv.append('--graph')
    sys.argv = argv

    print(f'device: {dev}\npath:   {"E (consistent P_N-P_N)" if a.consistent else "K (weak Laplacian)"}'
          f'\nsteps:  {a.steps} from t={t0:.4f}, dt={a.dt:g}\n', flush=True)

    wall0 = time.perf_counter()
    runpy.run_path(os.path.join('scratch', 'fs_minchan_stats.py'), run_name='__main__')
    wall = time.perf_counter() - wall0

    # ---- report ----
    skip_steps = 1                       # first step carries kernel compilation
    nsub = 3
    print('\n' + '='*72)
    print(f'{a.steps} steps in {wall:.1f} s wall (including setup and the first step)')
    tot_phase = 0.0
    rows = []
    for name in ('convective', 'helmholtz', 'pressure'):
        ph = P[name]
        if ph.n == 0:
            continue
        # the K path routes pressure through HH.solve: odd calls are velocity,
        # even are pressure, in call order within each substage
        ts, its = ph.tail(skip_steps*nsub*(2 if (k_path and name == 'helmholtz') else 1))
        if k_path and name == 'helmholtz':
            vel, pres = ts[0::2], ts[1::2]
            ivel, ipres = its[0::2], its[1::2]
            for lab, tt, ii in (('helmholtz (u)', vel, ivel), ('pressure (K)', pres, ipres)):
                rows.append((lab, tt, ii)); tot_phase += sum(tt)
        else:
            rows.append((name, ts, its)); tot_phase += sum(ts)
    nstep_eff = a.steps - skip_steps
    print(f'\n{"phase":16s} {"s/step":>9s} {"% of step":>10s} {"calls/step":>11s} '
          f'{"iterations/call":>16s}')
    for lab, ts, its in rows:
        sps = sum(ts)/max(nstep_eff, 1)
        it = f'{np.mean(its):.0f}' if its else '-'
        print(f'{lab:16s} {sps:9.4f} {100*sum(ts)/max(tot_phase,1e-30):9.1f}% '
              f'{len(ts)/max(nstep_eff,1):11.1f} {it:>16s}')
    print(f'{"":16s} {"-"*9}')
    print(f'{"phases total":16s} {tot_phase/max(nstep_eff,1):9.4f}')
    print('\nNote: setup (preconditioner build) and the first step are excluded from')
    print('the per-step figures; `wall` above includes them.')


if __name__ == '__main__':
    main()
