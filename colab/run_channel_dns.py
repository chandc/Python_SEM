"""Overnight minimal-channel DNS at Re_tau = 180 on a Colab A100, resumable.

    python colab/run_channel_dns.py --hours 10 --drive /content/drive/MyDrive/lssem_dns

One Colab session is shorter than this run, so the script is built around that
fact rather than against it: it finds the newest checkpoint (locally, then on
Drive, then the seed), marches until a wall-clock deadline it sets itself,
copies the state to Drive as it goes, and stops cleanly with a checkpoint on
Drive.  Running the same cell tomorrow night continues from there.  Nothing
about the physics changes between sessions -- `minchan.run` reloads the
convective history and the running statistics with the state, so a restart is
not a restart of the flow.

WHAT IT RUNS (the settings are not defaults, they are the validated ones):

  * `weighting=legacy`.  Convection is explicit RKW3, so each implicit stage is
    a Stokes PROJECTION with a non-solenoidal right-hand side; the constraint
    rows must dominate for the stage to return a divergence-free field.  The
    balanced weighting of the 2D work is WRONG here and drives the divergence
    from 8.6e-4 to 3.3e-1 in one step (BALANCED_CONDENSED_PLAN.md sec 1.5).
  * `precond=vsbatch share_precond=1`.  Condensed vertex-patch Schwarz with a
    p = 2 coarse space, batched, one preconditioner built at the middle stage's
    c and reused for all three: 72 CG per stage against Jacobi's 4675, step
    identical to Jacobi in every logged digit.
  * `dt=8e-4`, the value run01 used, CFL 1.11 against the RKW3 limit of 1.732.

STATISTICS AND THE TRANSIENT.  `PlaneStats` accumulates running SUMS from t = 0,
so the average carried in a checkpoint includes run01's start-up transient and
must not be used as it stands.  It does not need to be reset: sums are additive,
so the average over any window [a, b] is (sums_b - sums_a)/(n_b - n_a).  Each
sync therefore archives the small stats file under its step number, and
`colab/stats_window.py` differences two of them.  Choose the window after the
run, from the u_tau history, instead of guessing it now.
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
os.chdir(ROOT)

DEFAULT_SEED = 'checkpoint_0006200.npz'          # run01 at t = 4.96, 24 MB


def ckpt_step(path):
    """Step number from inside the file, not from its name (npz members are
    read lazily, so this does not pull the 24 MB of state)."""
    try:
        with np.load(path) as z:
            return int(z['step'])
    except Exception:
        return -1


def newest_ckpt(d):
    best, bstep = None, -1
    for p in sorted(glob.glob(os.path.join(d, 'checkpoint_*.npz'))):
        st = ckpt_step(p)
        if st > bstep:
            best, bstep = p, st
    return best, bstep


def put(src, dst):
    """Copy through a temp name so a reader (or a Drive sync) never sees a
    half-written file, the same reason minchan writes checkpoints that way."""
    if not os.path.exists(src):
        return False
    tmp = dst + '.part'
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)
    return True


def sync(out, drive, keep=3, archive=2500):
    """Push the run to Drive: log and diagnostics always, the newest checkpoint,
    and a stats snapshot under its step number (a few kB, kept forever -- these
    are what the averaging window is built from)."""
    if not drive:
        return
    os.makedirs(drive, exist_ok=True)
    n = 0
    for f in ('run.log', 'diag.npz', 'config.json'):
        n += put(os.path.join(out, f), os.path.join(drive, f))
    ck, st = newest_ckpt(out)
    if ck:
        n += put(ck, os.path.join(drive, os.path.basename(ck)))
        if os.path.exists(os.path.join(out, 'stats.npz')):
            n += put(os.path.join(out, 'stats.npz'),
                     os.path.join(drive, f'stats_{st:07d}.npz'))
    # Keep the newest few checkpoints plus a coarse archive; 24 MB each adds up.
    have = sorted((ckpt_step(p), p) for p in glob.glob(os.path.join(drive, 'checkpoint_*.npz')))
    for i, (st_, p) in enumerate(have):
        if i >= len(have) - keep or st_ % archive == 0:
            continue
        os.remove(p)
    return n


def safe_sync(out, drive, tag=''):
    """A sync failure must never end the run.  Colab's Drive FUSE mount can go
    unresponsive or drop in a long session, and shutil.copy2 then raises; the
    previous form let that exception out of the loop and take a ten-hour run
    with it.  The local checkpoints keep accumulating either way, so a failed
    sync costs nothing but the copy -- the next one, ten minutes later, picks
    up the newest file."""
    try:
        sync(out, drive)
        return True
    except Exception as e:
        print(f'  [!] sync to Drive FAILED{tag}: {type(e).__name__}: {e}\n'
              f'      the run continues and local checkpoints are intact; '
              f'if this repeats, remount Drive from another cell', flush=True)
        return False


def resolve_start(out, drive, seed):
    """Local checkpoint, else Drive, else the seed.  Returns (resume_path, step)."""
    ck, st = newest_ckpt(out)
    if ck:
        print(f'resuming from the local checkpoint {ck} (step {st})')
        return ck, st
    if drive:
        ck, st = newest_ckpt(drive)
        if ck:
            local = os.path.join(out, os.path.basename(ck))
            os.makedirs(out, exist_ok=True)
            put(ck, local)
            print(f'resuming from Drive: {ck} (step {st}) -> {local}')
            return local, st
    if seed and os.path.exists(seed):
        local = os.path.join(out, os.path.basename(seed))
        os.makedirs(out, exist_ok=True)
        put(seed, local)
        st = ckpt_step(local)
        print(f'starting from the seed checkpoint {seed} (step {st})')
        return local, st
    return None, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hours', type=float, default=10.0,
                    help='wall-clock budget; the run stops cleanly and syncs before this')
    ap.add_argument('--target-t', type=float, default=30.0,
                    help='stop for good at this simulated time (eddy turnovers, since delta/u_tau = 1)')
    ap.add_argument('--dt', type=float, default=8e-4)
    ap.add_argument('--every', type=int, default=100, help='steps between checkpoints')
    ap.add_argument('--sync-min', type=float, default=10.0, help='minutes between Drive syncs')
    ap.add_argument('--out', default='/content/run02')
    ap.add_argument('--drive', default='', help='Drive directory for the run; empty disables syncing')
    ap.add_argument('--seed', default='', help=f'first-session checkpoint (default <drive>/../lssem_data/{DEFAULT_SEED})')
    ap.add_argument('--backend', default='torch')
    ap.add_argument('--precond', default='vsbatch')
    ap.add_argument('--weighting', default='legacy')
    ap.add_argument('--share', type=int, default=1)
    ap.add_argument('--coarse-dense', default='', help='1 device dense coarse, 0 host sparse LU; empty = automatic')
    ap.add_argument('--pc', type=int, default=2, help='coarse level degree (2 validated; 1 is 13x smaller for +5%% iterations)')
    ap.add_argument('--coarse-fp32', type=int, default=0, help='hold the coarse inverse in fp32 (half the read); state and solution stay fp64')
    ap.add_argument('--from-scratch', action='store_true',
                    help='allow starting from the tripped initial condition when no '
                         'checkpoint is found (a fresh transition, ~10 turnovers of '
                         'transient before any statistics are worth keeping)')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    seed = a.seed or (os.path.join(os.path.dirname(a.drive.rstrip('/')), 'lssem_data', DEFAULT_SEED)
                      if a.drive else '')
    resume, step0 = resolve_start(a.out, a.drive, seed)
    if resume is None and not a.from_scratch:
        # Silently starting a new transition would burn the session and produce
        # nothing comparable with run01, which is exactly the failure a forgotten
        # upload causes.  Make it an explicit choice.
        raise SystemExit(
            f'no checkpoint found in {a.out}'
            + (f' or {a.drive}' if a.drive else '')
            + f' and no seed at {seed or "(none given)"}.\n'
              'Upload run01\'s checkpoint_0006200.npz to Drive (MyDrive/lssem_data/), '
              'or pass --from-scratch\nto start a fresh transition from the tripped '
              'initial condition.')
    nstep = int(round(a.target_t/a.dt))
    if step0 >= nstep:
        print(f'target already reached: step {step0} is t = {step0*a.dt:.2f} >= {a.target_t}')
        return 0

    cmd = [sys.executable, '-u', 'scratch/minchan.py', 'run',
           f'out={a.out}', f'nstep={nstep}', f'dt={a.dt:g}', f'every={a.every}',
           f'backend_name={a.backend}', f'precond={a.precond}',
           f'weighting={a.weighting}', f'share_precond={a.share}',
           f'pc={a.pc}', f'coarse_fp32={a.coarse_fp32}']
    if a.coarse_dense != '':
        cmd.append(f'coarse_dense={int(a.coarse_dense)}')
    if resume:
        cmd.append(f'resume={resume}')
    env = dict(os.environ, LSSEM3D_BACKEND=a.backend, LSSEM3D_DEVICE='cuda')

    print(f'\n{"="*72}\n$ ' + ' '.join(cmd))
    print(f'  from step {step0} (t = {step0*a.dt:.3f}) to step {nstep} (t = {a.target_t:g})')
    print(f'  budget {a.hours:g} h, checkpoint every {a.every} steps, sync every {a.sync_min:g} min')
    print(f'  Drive: {a.drive or "(none -- results die with this VM)"}\n{"="*72}\n', flush=True)
    if a.dry_run:
        return 0

    t0 = time.perf_counter()
    deadline = t0 + a.hours*3600
    child = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
    next_sync = t0 + a.sync_min*60
    step, rate_from, announced = step0, None, False
    stopping = False
    try:
        while True:
            r, _, _ = select.select([child.stdout], [], [], 20.0)
            if r:
                line = child.stdout.readline()
                if not line:
                    break
                print(line.rstrip(), flush=True)
                if line.startswith('t='):
                    try:
                        step = int(round(float(line.split('t=')[1].split()[0])/a.dt))
                    except Exception:
                        pass
                    # Start timing after the first reported line: everything
                    # before it is the preconditioner build, which is paid once.
                    if rate_from is None:
                        rate_from = (time.perf_counter(), step)
                    elif not announced and step - rate_from[1] >= 30:
                        el, s0 = rate_from
                        sps = (time.perf_counter() - el)/(step - s0)
                        left = (deadline - time.perf_counter())/sps
                        announced = True
                        print(f'\n  >> {sps:.2f} s/step; {left:.0f} steps fit in the remaining budget '
                              f'= t + {left*a.dt:.1f} (target t = {a.target_t:g}, '
                              f'{max(0.0, (nstep - step - left)*sps/3600):.1f} h beyond tonight)\n', flush=True)
            elif child.poll() is not None:
                break
            now = time.perf_counter()
            if now >= next_sync:
                if safe_sync(a.out, a.drive):
                    print(f'  [synced to Drive at t+{(now-t0)/3600:.2f} h]', flush=True)
                next_sync = now + a.sync_min*60
            if now >= deadline and not stopping:
                stopping = True
                print(f'\n  >> wall-clock budget reached ({a.hours:g} h); stopping at the next '
                      f'opportunity so the checkpoint is clean\n', flush=True)
                child.send_signal(signal.SIGINT)
            if stopping and now >= deadline + 180:
                child.kill()
                break
    except KeyboardInterrupt:
        print('\n  >> interrupted; stopping the run and syncing', flush=True)
        child.send_signal(signal.SIGINT)
    except Exception as e:                       # a supervisor fault must still sync
        print(f'\n  >> supervisor error: {type(e).__name__}: {e}; syncing what exists',
              flush=True)
        try:
            child.send_signal(signal.SIGINT)
        except Exception:
            pass
    finally:
        try:
            child.wait(timeout=240)
        except Exception:
            child.kill()
        safe_sync(a.out, a.drive, tag=' (final)')

    ck, st = newest_ckpt(a.out)
    el = (time.perf_counter() - t0)/3600
    done = st - step0
    print(f'\n{"="*72}')
    print(f'session: {done} steps in {el:.2f} h'
          + (f' = {el*3600/done:.2f} s/step' if done > 0 else ''))
    print(f'state:   step {st}, t = {st*a.dt:.3f} of {a.target_t:g}'
          + (f'  ({(nstep-st)*a.dt:.2f} to go'
             + (f', ~{(nstep-st)*(el*3600/done)/3600:.1f} h at tonight\'s rate)' if done > 0 else ')')
             if st < nstep else '  -- TARGET REACHED'))
    print(f'files:   {ck}' + (f'\n         synced to {a.drive}' if a.drive else ''))
    print(f'{"="*72}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
