"""The matched fractional-step campaign, in resumable overnight sessions.

    python colab/run_fs_dns.py --hours 10 --drive /content/drive/MyDrive/lssem_fs

Marches the projection solver over the SAME span the least-squares run covered,
from the SAME field, on the same grid, so the only difference between the two
records is the discretisation.

WHAT MAKES IT A MATCHED COMPARISON, point by point:

  * the same starting field -- the least-squares run's own restart checkpoint,
    converted by `colab/fs_seed_from_fosls.py` (a slice, not an interpolation:
    identical mesh, identical Fourier modes, velocity and pressure lifted out of
    the seven-field state);
  * the same span -- t = 4.96 to 30, 25.04 eddy turnovers;
  * the same time step -- 8e-4, which the projection path was shown to tolerate
    (u_tau agreeing to 0.01 % against its usual 3.5e-4);
  * the same statistics machinery -- `fs_minchan_stats.py`'s collector is the
    direct ancestor of the least-squares one and writes the same schema, so
    `colab/section10.py` scores both without modification;
  * the same averaging window -- statistics from t = 5.2, discarding the same
    quarter turnover the least-squares run discarded.

Session handling follows `run_channel_dns.py`: find the newest checkpoint, march
to a self-imposed wall-clock deadline, sync to Drive as it goes, stop cleanly.
The projection driver now carries its accumulators across a restart, so a
campaign split over several nights averages exactly as one run would.
"""
import argparse
import glob
import os
import select
import shutil
import signal
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FS = os.path.join(ROOT, 'fractional_step')
os.chdir(FS)


def put(src, dst):
    if not os.path.exists(src):
        return False
    tmp = dst + '.part'
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)
    return True


def sync(out, drive, archive=2.0):
    """Never let a Drive fault end the run (see run_channel_dns.safe_sync)."""
    if not drive:
        return
    try:
        os.makedirs(drive, exist_ok=True)
        n = 0
        # final_state.npz / stats_final.npz ARE IN THIS LIST DELIBERATELY.  They
        # are written by the solver only when the target t is reached, i.e. on
        # the LAST session of the campaign -- and until 2026-09-16 they were not
        # synced, so the one run that actually finished would have left its final
        # field and its final accumulators on the VM to be destroyed with it.
        # The periodic chk_latest is up to --chkmin behind the end, so this is
        # not a duplicate of it: it is the only copy of the run's own endpoint.
        for f in ('chk_latest.npz', 'stats_latest.npz', 'stats_run.log',
                  'final_state.npz', 'stats_final.npz'):
            n += put(os.path.join(out, f), os.path.join(drive, f))
        # Numbered snapshots, which is what section10.py's window globs for.
        # stats_final must appear among them or the window's upper end stops
        # short of the target by up to one checkpoint interval.
        for base in ('stats_latest.npz', 'stats_final.npz'):
            st = os.path.join(out, base)
            if os.path.exists(st):
                with np.load(st, allow_pickle=True) as z:
                    t = float(z['t'])
                put(st, os.path.join(drive, f'stats_t{t:07.3f}.npz'))
        # ARCHIVE VELOCITY FIELDS, not only the accumulators.  The statistics
        # files carry U, uu, vv, ww, uv and nothing else, so vorticity rms,
        # pointwise divergence and the modal spectrum -- the quantities where
        # the two formulations are expected to differ most -- cannot be computed
        # from them at all.  Those need the field.  One snapshot per `archive`
        # turnovers, ~10 MB each, keeps that comparison possible; the
        # least-squares run kept its checkpoints on the same reasoning.
        for base in ('chk_latest.npz', 'final_state.npz'):
            ck = os.path.join(out, base)
            if not os.path.exists(ck):
                continue
            with np.load(ck) as z:
                tc = float(z['t'])
            slot = int(tc/archive)*archive
            dst = os.path.join(drive, f'field_t{slot:05.1f}.npz')
            if not os.path.exists(dst):
                put(ck, dst)
        return n
    except Exception as e:
        print(f'  [!] sync to Drive FAILED: {type(e).__name__}: {e}; '
              f'the run continues', flush=True)


def resolve(out, drive, seed):
    for cand, where in ((os.path.join(out, 'chk_latest.npz'), 'local'),
                        (os.path.join(drive, 'chk_latest.npz') if drive else '', 'Drive')):
        if cand and os.path.exists(cand):
            if where == 'Drive':
                os.makedirs(out, exist_ok=True)
                put(cand, os.path.join(out, 'chk_latest.npz'))
                put(os.path.join(drive, 'stats_latest.npz'),
                    os.path.join(out, 'stats_latest.npz'))
                cand = os.path.join(out, 'chk_latest.npz')
            with np.load(cand) as z:
                t = float(z['t'])
            print(f'resuming from the {where} checkpoint at t = {t:.4f}')
            return cand, t
    if seed and os.path.exists(seed):
        with np.load(seed) as z:
            t = float(z['t'])
        print(f'starting from the converted least-squares field at t = {t:.4f}')
        return seed, t
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hours', type=float, default=10.0)
    ap.add_argument('--target-t', type=float, default=30.0)
    ap.add_argument('--dt', type=float, default=8e-4)
    ap.add_argument('--chkmin', type=float, default=5.0)
    ap.add_argument('--sync-min', type=float, default=10.0)
    ap.add_argument('--out', default='/content/fs_run')
    ap.add_argument('--drive', default='')
    ap.add_argument('--seed', default='')
    ap.add_argument('--backend', default='torch')
    ap.add_argument('--no-graph', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    restart, t0 = resolve(a.out, a.drive, a.seed)
    if restart is None:
        raise SystemExit(
            f'no checkpoint in {a.out} or {a.drive}, and no seed at '
            f'{a.seed or "(none given)"}.\nRun colab/fs_seed_from_fosls.py first '
            f'to convert the least-squares restart.')
    if t0 >= a.target_t - 1e-9:
        print(f'target already reached: t = {t0:.3f} >= {a.target_t}')
        return 0

    cmd = [sys.executable, '-u', os.path.join('scratch', 'fs_minchan_stats.py'),
           '--restart', restart, '--backend', a.backend, '--consistent',
           '--dt', repr(a.dt), '--tend', repr(a.target_t), '--outdir', a.out,
           '--chkmin', repr(a.chkmin), '--resume-stats', 'auto']
    if not a.no_graph:
        cmd.append('--graph')

    print(f'\n{"="*72}\n$ ' + ' '.join(cmd))
    print(f'  t = {t0:.3f} -> {a.target_t:g}  ({(a.target_t-t0):.2f} turnovers, '
          f'{int((a.target_t-t0)/a.dt):,} steps)')
    print(f'  budget {a.hours:g} h, checkpoint every {a.chkmin:g} min\n{"="*72}\n',
          flush=True)
    if a.dry_run:
        return 0

    start = time.perf_counter()
    deadline = start + a.hours*3600
    child = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1)
    next_sync, stopping = start + a.sync_min*60, False
    try:
        while True:
            r, _, _ = select.select([child.stdout], [], [], 20.0)
            if r:
                line = child.stdout.readline()
                if not line:
                    break
                print(line.rstrip(), flush=True)
            elif child.poll() is not None:
                break
            now = time.perf_counter()
            if now >= next_sync:
                sync(a.out, a.drive)
                print(f'  [synced at t+{(now-start)/3600:.2f} h]', flush=True)
                next_sync = now + a.sync_min*60
            if now >= deadline and not stopping:
                stopping = True
                print(f'\n  >> {a.hours:g} h budget reached; stopping\n', flush=True)
                child.send_signal(signal.SIGINT)
            if stopping and now >= deadline + 180:
                child.kill(); break
    except KeyboardInterrupt:
        child.send_signal(signal.SIGINT)
    finally:
        try:
            child.wait(timeout=240)
        except Exception:
            child.kill()
        sync(a.out, a.drive)

    ck = os.path.join(a.out, 'chk_latest.npz')
    t = float(np.load(ck)['t']) if os.path.exists(ck) else t0
    el = (time.perf_counter() - start)/3600
    print(f'\n{"="*72}')
    print(f'session: {t-t0:.3f} turnovers in {el:.2f} h '
          f'({el/max(t-t0,1e-9):.2f} h per turnover)')
    print(f'state:   t = {t:.3f} of {a.target_t:g}'
          + ('  -- TARGET REACHED' if t >= a.target_t - 1e-9 else
             f'  ({a.target_t-t:.2f} to go, ~{(a.target_t-t)*el/max(t-t0,1e-9):.1f} h)'))
    print(f'{"="*72}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
