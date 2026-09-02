"""Plane-averaged Reynolds-stress statistics for the FOSLS minimal channel.

`minchan.py` logs SCALARS only -- u_tau, U_bulk, rms_w, energy, momentum budget.
To demonstrate that VVP-FOSLS can do DNS the deliverable is PROFILES: U+(y+) and
the u/v/w rms and <uv> profiles against Kim, Moin & Moser.  Without this a
multi-day run produces no comparable output, which is lesson L16 in its most
expensive form.

Ported from the fractional-step collector (`fs_minchan_stats.py`) so the two
codes' statistics are directly comparable -- same grouping by y, same
quadrature-weighted plane average, same `sums`/`nsamp` convention.

    sums[0] = <u>         sums[1] = <uu>
    sums[2] = <vv>        sums[3] = <ww>        sums[4] = <uv>

all plane means; divide by `nsamp`.  Fluctuation rms are recovered as
sqrt(<uu> - <u>^2) etc, so the mean is subtracted at REDUCTION time, not
accumulation time -- which keeps the accumulator a running sum and lets samples
be added from restarts without bias.
"""
import numpy as np

from lssem3d import fourier as FR, operator as OP


class PlaneStats:
    """Running sums of plane-averaged quantities, restart-safe."""

    def __init__(self, s, nz):
        m = s['m']
        n = m.N + 1
        self.nz = nz
        Y = np.empty((m.nelem, n, n))
        for e in range(m.nelem):
            Y[e] = m.ynod[e][None, :]
        yk = np.round(Y, 10).ravel()
        self.order = np.argsort(yk, kind='stable')
        splits = np.flatnonzero(np.diff(yk[self.order]) > 1e-9) + 1
        self.groups = np.split(np.arange(yk.size), splits)
        self.y = np.array([yk[self.order][g[0]] for g in self.groups])
        wcol = m.wq.ravel()[self.order]
        self.gw = [wcol[g] for g in self.groups]
        self.gwsum = np.array([w.sum() for w in self.gw])
        self.sums = np.zeros((5, len(self.groups)))
        self.nsamp = 0
        self.series = []                      # (t, u_tau) pairs

    def accumulate(self, Uh, t=None, utau=None):
        """Uh is the SPLIT-REAL host state (nelem, n, n, 14, nk)."""
        Uc = OP.to_complex(Uh)
        P = FR.to_physical(Uc[..., (OP.U_, OP.V_, OP.W_), :], self.nz)
        flat = P.reshape(-1, 3, self.nz)[self.order]
        quants = (lambda f: f[:, 0, :],
                  lambda f: f[:, 0, :]**2,
                  lambda f: f[:, 1, :]**2,
                  lambda f: f[:, 2, :]**2,
                  lambda f: f[:, 0, :]*f[:, 1, :])
        for iq, q in enumerate(quants):
            for j, g in enumerate(self.groups):
                v = q(flat[g])
                self.sums[iq, j] += float((v.mean(axis=1)*self.gw[j]).sum()
                                          / self.gwsum[j])
        self.nsamp += 1
        if t is not None and utau is not None:
            self.series.append((float(t), float(utau)))

    def save(self, path, nu, dt, t):
        np.savez_compressed(
            path, y=self.y, sums=self.sums, nsamp=self.nsamp, t=t, nu=nu, dt=dt,
            utau_series=np.array(self.series, dtype=float),
            note='sums = plane means of U,uu,vv,ww,uv; divide by nsamp')

    def state(self):
        return dict(stats_sums=self.sums, stats_nsamp=self.nsamp,
                    stats_series=np.array(self.series, dtype=float))

    def load(self, z):
        """Restore from a checkpoint so a restart does not reset the average."""
        if 'stats_sums' in z:
            self.sums = np.asarray(z['stats_sums']).copy()
            self.nsamp = int(z['stats_nsamp'])
            self.series = [tuple(r) for r in np.asarray(z['stats_series'])]


def profiles(path):
    """Reduce a saved stats file to U+, rms and <uv> profiles in wall units."""
    z = np.load(path, allow_pickle=True)
    n, nu, y = int(z['nsamp']), float(z['nu']), z['y']
    U, uu, vv, ww, uv = z['sums']/n
    ut = np.asarray(z['utau_series'])
    utau = float(ut[-min(len(ut), 2000):, 1].mean()) if ut.size else 1.0
    return dict(y=y, yp=y*utau/nu, Up=U/utau,
                urms=np.sqrt(np.maximum(uu - U**2, 0))/utau,
                vrms=np.sqrt(np.maximum(vv, 0))/utau,
                wrms=np.sqrt(np.maximum(ww, 0))/utau,
                uv=uv/utau**2, utau=utau, nsamp=n, Re_tau=utau/nu)
