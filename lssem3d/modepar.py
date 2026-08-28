"""Mode-space multiprocessing for the numpy channel path.

The implicit solves are block-diagonal across Fourier modes.  This pool gives
each worker process a CONTIGUOUS block of modes; per solve the parent writes
the full RHS into shared memory, wakes the workers, and each runs the WHOLE
CG for its block -- no per-iteration communication, two barrier crossings per
solve.  Workers build their own sliced preconditioners at startup (the
subset() machinery from mode-adaptive freezing), so nothing unpicklable
crosses the process boundary.

Partition: k = 0 (the all-Neumann Poisson, worst-conditioned) gets a worker
to itself; remaining modes split into contiguous blocks weighted toward
low k.  Each worker caps its BLAS threads so workers*threads ~ P-cores.

macOS: spawn start method (fork + Accelerate is unsafe).
"""
import multiprocessing as mp
import os
import sys

import numpy as np
from multiprocessing import shared_memory


def _partition(nk, nworkers):
    """[0] alone, then contiguous blocks, smaller at low k."""
    if nworkers <= 1 or nk <= 2:
        return [list(range(nk))]
    parts = [[0]]
    rest = list(range(1, nk))
    nb = nworkers - 1
    # geometric-ish split: low-k blocks smaller (they iterate more)
    sizes = np.diff(np.unique(np.round(
        np.geomspace(1, len(rest) + 1, nb + 1)).astype(int)))
    sizes = list(sizes) if len(sizes) == nb else \
        [len(rest)//nb + (1 if i < len(rest) % nb else 0) for i in range(nb)]
    at = 0
    for sz in sizes:
        if sz <= 0:
            continue
        parts.append(rest[at:at+sz]); at += sz
    if at < len(rest):
        parts[-1].extend(rest[at:])
    return parts


def _worker(wid, cfg, idx, shm_names, conn):
    # cfg['geom']: 'channel' (walled minimal-channel rig, the original) or
    # 'periodic' (doubly-periodic square, the TGV/Kim-Moin rig).  cfg
    # 'with_e' adds the consistent-E solve (kind 2), serving both the
    # stage-pressure and projection systems; worker 0 computes the kz=0
    # kernel basis and purges it, other workers have no singular modes.
    for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS'):
        os.environ[v] = str(cfg['blas_threads'])
    sys.path.insert(0, cfg['root'])
    sys.path.insert(0, os.path.join(cfg['root'], 'scratch'))
    import lssem3d
    from lssem2d.mesh import build_channel
    from lssem2d.lgl import diff_matrix
    from lssem3d import (project as PJ, helmholtz as HH, fourier as FR,
                         solver3d as S3, timestep as T, hpmg)
    N, EX, EY, NZ = cfg['N'], cfg['ex'], cfg['ey'], cfg['nz']
    NU, DT, TOL = cfg['nu'], cfg['dt'], cfg['tol']
    LX = cfg['lx']
    geom = cfg.get('geom', 'channel')
    wall = geom == 'channel'
    if wall:
        m = build_channel(LX, 2.0, EX, EY, N, bcs=(0, 0, 1, 1))
        m.periodic_x = LX
    else:
        m = build_channel(LX, LX, EX, EY, N, bcs=(0, 0, 0, 0))
        m.periodic_x = LX; m.periodic_y = LX
    m.compute_global_indices()
    nk = NZ//2 + 1
    kz = FR.wavenumbers(NZ, cfg['lz'])
    mask_u = PJ.build_masks(m, nk, NZ, 3, wall=wall)
    mask_p = PJ.build_masks(m, nk, NZ, 1, wall=False)
    idx = np.asarray(idx)
    D = diff_matrix(N)
    if cfg.get('km'):
        mu_full = [HH.fdm_preconditioner(m, N, 2.0/DT + NU*(kz**2), NU,
                                         mask_u, 6, nk)]
    else:
        mu_full = [HH.fdm_preconditioner(m, N,
                   T.implicit_coeff(DT, k) + NU*(kz**2), NU, mask_u, 6, nk)
                   for k in range(T.NSTAGE)]
    mu_sub = [M.subset(idx) for M in mu_full]
    mp_sub = None
    if not cfg.get('with_e'):
        mask_pk = mask_p.copy()
        ind = np.zeros(mask_pk.shape); ind[0, 0, 0, 0, 0] = 1.0
        mask_pk[..., 0, 0] *= (S3.gs(m, ind)[..., 0, 0] < 0.5)
        mask_p = mask_pk
        pmg = hpmg.HelmholtzPMG(m, N, kz**2, 1.0, 1, nk, NZ, wall=False,
                                pin_kz0=True, deg=6, like=None)
        mp_sub = pmg.subset(idx)
    e_state = None
    if cfg.get('with_e'):
        from . import epmg
        wq3 = m.wq[..., None, None]
        Mginv = 1.0/S3.gs(m, wq3 + np.zeros_like(wq3))
        pmg_e = epmg.ConsistentPMG(m, N, kz, nk, NZ, deg=6, wall=wall)
        pe_sub = pmg_e.subset(idx)
        kz_a = kz[idx]
        mp_a, mu_a = mask_p[..., idx], mask_u[..., idx]
        A_e = lambda p_: PJ.apply_E(p_, D, m.facx, m.facy, wq3, kz_a, m,
                                    mp_a, mu_a, Mginv)
        purge_e = None
        # a worker owning ANY real Fourier mode (k = 0 or the Nyquist
        # k = nk-1) must purge the shared 2-D kernel at those lane positions
        real_globals = [g for g in (0, nk - 1) if g in idx]
        if real_globals:
            nb = epmg.kz0_null_basis(m, N, kz, nk, NZ, mask_p, mask_u)
            mwl = np.asarray(S3.multiplicity_weight(m, mask_p.shape))
            mw0 = mwl[..., 0, 0]
            pos = [int(np.flatnonzero(idx == g)[0]) for g in real_globals]
            def purge_e(z, nb=nb, mw0=mw0, pos=tuple(pos)):
                z = z.copy()
                for kl in pos:
                    for q in nb:
                        num = (z[..., 0, kl]*q*mw0).sum()
                        z[..., 0, kl] -= num*q
                return z
        e_state = (A_e, pe_sub, purge_e)
    if cfg.get('km'):
        lam_u = [(2.0/DT + NU*(kz**2))[idx]]
    else:
        lam_u = [(T.implicit_coeff(DT, k) + NU*(kz**2))[idx]
                 for k in range(T.NSTAGE)]
    lam_p = (kz**2)[idx]
    musk_u, musk_p = mask_u[..., idx], mask_p[..., idx]
    n1 = N + 1
    shp_u = (m.nelem, n1, n1, 6, nk)
    shp_p = (m.nelem, n1, n1, 2, nk)
    shm = {k: shared_memory.SharedMemory(name=v) for k, v in shm_names.items()}
    bu = np.ndarray(shp_u, dtype=np.float64, buffer=shm['bu'].buf)
    xu = np.ndarray(shp_u, dtype=np.float64, buffer=shm['xu'].buf)
    bp = np.ndarray(shp_p, dtype=np.float64, buffer=shm['bp'].buf)
    xp_ = np.ndarray(shp_p, dtype=np.float64, buffer=shm['xp'].buf)
    info = np.ndarray((cfg['nworkers'], 2), dtype=np.float64,
                      buffer=shm['info'].buf)
    gmax = np.ndarray((1,), dtype=np.float64, buffer=shm['gmax'].buf)
    while True:
        msg = conn.recv()
        if msg is None:
            break
        kind, stage = msg
        if kind == 2:      # consistent-E solve (stage pressure / projection)
            A_e, pe_sub, purge_e = e_state
            b = bp[..., idx].copy()
            if purge_e is not None:
                b = purge_e(b)
            x, it, res = PJ._pcg(A_e, b, pe_sub, m, cfg['tol_p'], 1,
                                 purge=purge_e, bmax_global=float(gmax[0]))
            xp_[..., idx] = x
        elif kind == 0:    # velocity
            b = bu[..., idx].copy()
            x, it, res = HH.solve(b, D, m.facx, m.facy, m.wq, lam_u[stage],
                                  NU, m, musk_u, mu_sub[stage], tol=TOL,
                                  check_every=1)
            xu[..., idx] = x
        else:              # pressure
            b = bp[..., idx].copy()
            x, it, res = HH.solve(b, D, m.facx, m.facy, m.wq, lam_p,
                                  1.0, m, musk_p, mp_sub, tol=TOL,
                                  check_every=1)
            xp_[..., idx] = x
        conn.send((it, res))


class ModePool:
    def __init__(self, cfg, nworkers=4, blas_threads=3):
        cfg = dict(cfg, nworkers=nworkers, blas_threads=blas_threads,
                   root=os.getcwd())
        cfg.setdefault('tol_p', cfg.get('tol', 1e-6))
        self.with_e = bool(cfg.get('with_e'))
        nk = cfg['nz']//2 + 1
        n1 = cfg['N'] + 1
        import lssem3d  # parent side shapes
        nelem = cfg['ex']*cfg['ey']
        self.shp_u = (nelem, n1, n1, 6, nk)
        self.shp_p = (nelem, n1, n1, 2, nk)
        mkshm = lambda shape: shared_memory.SharedMemory(
            create=True, size=int(np.prod(shape))*8)
        self.shm = dict(bu=mkshm(self.shp_u), xu=mkshm(self.shp_u),
                        bp=mkshm(self.shp_p), xp=mkshm(self.shp_p),
                        info=mkshm((nworkers, 2)), gmax=mkshm((1,)))
        self.bu = np.ndarray(self.shp_u, np.float64, self.shm['bu'].buf)
        self.xu = np.ndarray(self.shp_u, np.float64, self.shm['xu'].buf)
        self.bp = np.ndarray(self.shp_p, np.float64, self.shm['bp'].buf)
        self.xp = np.ndarray(self.shp_p, np.float64, self.shm['xp'].buf)
        self.info = np.ndarray((nworkers, 2), np.float64,
                               self.shm['info'].buf)
        self.gmax = np.ndarray((1,), np.float64, self.shm['gmax'].buf)
        ctx = mp.get_context('spawn')
        self.parts = _partition(nk, nworkers)
        names = {k: v.name for k, v in self.shm.items()}
        # PIPES, NOT EVENT PAIRS.  The event handshake had a race: the
        # parent's done-clear could overlap a slow worker's done-set from the
        # PREVIOUS task, desynchronising the protocol -- production deadlocked
        # at step 21 while the short bench never tripped it.  A pipe is an
        # ordered channel; send task, recv ack, no shared flags.
        self.pipes = []
        self.procs = []
        for i, p in enumerate(self.parts):
            parent_c, child_c = ctx.Pipe()
            pr = ctx.Process(target=_worker, args=(i, cfg, p, names, child_c),
                             daemon=True)
            pr.start()
            self.pipes.append(parent_c)
            self.procs.append(pr)

    def solve(self, kind, stage, b):
        tgt = self.bu if kind == 'u' else self.bp
        tgt[...] = b
        # global per-mode scale for the workers' dead-lane threshold
        # (unweighted proxy; order of magnitude is what the threshold needs)
        self.gmax[0] = float(np.sqrt((b*b).sum(axis=(0, 1, 2, 3))).max())
        code = {'u': 0, 'p': 1, 'e': 2}[kind]
        for c in self.pipes:
            c.send((code, stage))
        acks = [c.recv() for c in self.pipes]
        out = (self.xu if kind == 'u' else self.xp).copy()
        its = int(max(a[0] for a in acks))
        res = float(max(a[1] for a in acks))
        return out, its, res

    def close(self):
        for c in self.pipes:
            c.send(None)
        for pr in self.procs:
            pr.join(timeout=5)
        for v in self.shm.values():
            v.close(); v.unlink()
