"""FOSLS run01 and fractional-step E-run vs the published Re_tau=180 databases.

Velocity statistics: accumulated plane averages (FOSLS: checkpoint-subtraction
window; FS: the archived t=3..15.95 window).  Vorticity and pressure rms are
NOT in the accumulators, so they come from snapshots: every FOSLS checkpoint
with t >= T0 (omega is a PRIMARY unknown there, and is also recomputed as
curl u for a consistency check) and the single archived FS state (omega by
spectral/SEM differentiation).  Everything is scaled by the MEASURED u_tau.
"""
import os, sys, glob
for _v in ('OMP_NUM_THREADS',): os.environ.setdefault(_v, '4')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, 'scratch')); os.chdir(_R)
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ref180

OUT = os.path.join(_R, 'scratch', 'compare_ref180.png')
T0 = 1.0          # FOSLS window start (first checkpoint with t >= T0)
YV = 60.0         # box-validity limit, y+ ~ Lz/3
FS_STATS = 'results/minchan_re180_E/stats_E.npz'
FS_STATE = 'results/minchan_re180_E/state_t15.95.npz'


def main():
    import lssem3d; lssem3d.set_backend('numpy')
    from lssem3d import operator as OP, fourier as FR, deriv as DV
    import minchan as MC
    from minchan_stats import PlaneStats
    s = MC.setup(); m = s['m']; nz = s['nz']; kz = s['kz']; nu = s['nu']; D = s['D']
    ps = PlaneStats(s, nz); y = ps.y

    def pmean(f):                       # (nelem,n,n,nz) physical -> (145,)
        flat = f.reshape(-1, nz)[ps.order]
        return np.array([(flat[g].mean(axis=1)*ps.gw[j]).sum()/ps.gwsum[j]
                         for j, g in enumerate(ps.groups)])

    def fold(a, yp, odd=False):
        """Fold a 145-node profile onto the lower half by channel symmetry."""
        ypf = np.round(np.minimum(y, 2 - y), 8)
        u = np.unique(ypf)
        out = np.array([np.mean(np.where(y[ypf == v] > 1.0, (-a if odd else a)[ypf == v], a[ypf == v])) for v in u])
        return u, out

    def curl(Uc):                       # Uc (...,3,nk) complex -> (...,3,nk)
        u, v, w = (np.ascontiguousarray(Uc[..., i:i+1, :]) for i in range(3))
        ddx = lambda f: DV.ddx(f, D, m.facx); ddy = lambda f: DV.ddy(f, D, m.facy)
        ox = ddy(w) - 1j*kz*v
        oy = 1j*kz*u - ddx(w)
        oz = ddx(v) - ddy(u)
        return np.concatenate([ox, oy, oz], axis=-2)

    class Acc:                          # rms accumulator over snapshots
        def __init__(self, nq): self.s1 = np.zeros((nq, len(y))); self.s2 = np.zeros((nq, len(y))); self.n = 0
        def add(self, phys, gauge=False):   # phys (nelem,n,n,nq,nz)
            # gauge=True: pressure is defined up to a constant PER SNAPSHOT
            # (pin_p=False), so remove the instantaneous volume mean first or the
            # gauge drift between snapshots masquerades as p' (it doubled p'_wall).
            if gauge:
                phys = phys - (pmean(phys[..., 0, :])*ps.gwsum).sum()/ps.gwsum.sum()
            for q in range(phys.shape[-2]):
                self.s1[q] += pmean(phys[..., q, :]); self.s2[q] += pmean(phys[..., q, :]**2)
            self.n += 1
        def rms(self): return np.sqrt(np.maximum(self.s2/self.n - (self.s1/self.n)**2, 0))

    # ---------------- FOSLS ----------------
    cks = sorted(glob.glob('scratch/run01_ck/checkpoint_*.npz'))
    zs = [(f, np.load(f)) for f in cks]
    ia = next(i for i, (f, z) in enumerate(zs) if float(z['t']) >= T0)
    fa, za = zs[ia]; fb, zb = zs[-1]
    na, nb = int(za['stats_nsamp']), int(zb['stats_nsamp'])
    S = (zb['stats_sums'] - za['stats_sums'])/(nb - na)
    ser = np.asarray(zb['stats_series']); ta, tb = float(za['t']), float(zb['t'])
    utF = ser[(ser[:, 0] > ta) & (ser[:, 0] <= tb), 1].mean()
    ReF = utF/nu
    U, uu, vv, ww, uv = S
    prim, diff, pacc = Acc(3), Acc(3), Acc(1)
    nsnap = 0
    for f, z in zs[ia:]:
        Uc = OP.to_complex(z['U'])
        prim.add(FR.to_physical(np.ascontiguousarray(Uc[..., (OP.OX_, OP.OY_, OP.OZ_), :]), nz))
        diff.add(FR.to_physical(curl(Uc[..., (OP.U_, OP.V_, OP.W_), :]), nz))
        pacc.add(FR.to_physical(np.ascontiguousarray(Uc[..., OP.P_:OP.P_+1, :]), nz), gauge=True)
        nsnap += 1
    F = dict(name=f'FOSLS run01, t∈[{ta:.2f},{tb:.2f}] ({nb-na} samples, {nsnap} snapshots)',
             Re_tau=ReF, utau=utF)
    ypF = np.minimum(y, 2 - y)*utF/nu
    F['yp'], F['U'] = fold(U/utF, ypF)
    _, F['urms'] = fold(np.sqrt(np.maximum(uu - U**2, 0))/utF, ypF)
    _, F['vrms'] = fold(np.sqrt(np.maximum(vv, 0))/utF, ypF)
    _, F['wrms'] = fold(np.sqrt(np.maximum(ww, 0))/utF, ypF)
    _, F['uv'] = fold(-uv/utF**2, ypF, odd=True)
    om = prim.rms()*nu/utF**2; omd = diff.rms()*nu/utF**2
    for k, i in (('oxrms', 0), ('oyrms', 1), ('ozrms', 2)):
        _, F[k] = fold(om[i], ypF); _, F[k+'_curl'] = fold(omd[i], ypF)
    _, F['prms'] = fold(pacc.rms()[0]/utF**2, ypF)
    F['yp'] = F['yp']*utF/nu   # fold returned y (not y+); rescale

    # ---------------- fractional step ----------------
    z = np.load(FS_STATS, allow_pickle=True)
    n = int(z['nsamp']); U, uu, vv, ww, uv = z['sums']/n
    ser = np.asarray(z['utau_series']); utS = ser[:, 1].mean(); ReS = utS/nu
    G = dict(name=f'fractional step (E), t∈[3,{float(z["t"]):.2f}] ({n} samples, 1 snapshot)',
             Re_tau=ReS, utau=utS)
    ypS = np.minimum(y, 2 - y)*utS/nu
    G['yp'], G['U'] = fold(U/utS, ypS)
    _, G['urms'] = fold(np.sqrt(np.maximum(uu - U**2, 0))/utS, ypS)
    _, G['vrms'] = fold(np.sqrt(np.maximum(vv, 0))/utS, ypS)
    _, G['wrms'] = fold(np.sqrt(np.maximum(ww, 0))/utS, ypS)
    _, G['uv'] = fold(-uv/utS**2, ypS, odd=True)
    d = np.load(FS_STATE); Uc = d['U']; pc = d['p']
    a3, a1 = Acc(3), Acc(1)
    a3.add(FR.to_physical(curl(Uc), nz)); a1.add(FR.to_physical(np.ascontiguousarray(pc), nz), gauge=True)
    om = a3.rms()*nu/utS**2
    for k, i in (('oxrms', 0), ('oyrms', 1), ('ozrms', 2)):
        _, G[k] = fold(om[i], ypS)
    _, G['prms'] = fold(a1.rms()[0]/utS**2, ypS)
    G['yp'] = G['yp']*utS/nu

    # ---------------- references ----------------
    refs = {k: fn() for k, fn in ref180.ALL.items()}
    V = refs['vk']
    quants = [('U', '$U^+$'), ('urms', "$u'^+$"), ('vrms', "$v'^+$"), ('wrms', "$w'^+$"),
              ('uv', "$-\\langle u'v'\\rangle^+$"), ('oxrms', "$\\omega_x'^+$"),
              ('oyrms', "$\\omega_y'^+$"), ('ozrms', "$\\omega_z'^+$"), ('prms', "$p'^+$")]

    # ---------------- table ----------------
    def at(dd, q, yq): return np.interp(yq, dd['yp'], dd[q])
    print(f"FOSLS: {F['name']}  u_tau={utF:.4f}  Re_tau={ReF:.1f}")
    print(f"FS   : {G['name']}  u_tau={utS:.4f}  Re_tau={ReS:.1f}")
    print(f"\n{'quantity':10s} {'ref (V&K S4_B3)':>22s} {'ref spread':>10s} | {'FOSLS':>16s} {'dev':>7s} | {'FS':>16s} {'dev':>7s} | {'mean|dev| y+<60':>18s}")
    for q, _ in quants:
        yq = np.linspace(0.5, YV, 120)
        if q == 'U': yq = np.array([5, 15, 30, 50.])
        r = at(V, q, yq)
        avail = [d for d in refs.values() if q in d and d[q] is not None]
        spread = max(np.max(np.abs(at(d, q, yq) - r)/np.max(np.abs(r))) for d in avail)
        if q == 'U':
            for yy, rr in zip(yq, r):
                fF, fS = at(F, q, yy), at(G, q, yy)
                print(f"U+@{yy:<6.0f} {rr:22.3f} {spread*100:9.1f}% | {fF:16.3f} {100*(fF/rr-1):6.1f}% | {fS:16.3f} {100*(fS/rr-1):6.1f}% |")
            continue
        i = np.argmax(r); iF = np.argmax(at(F, q, yq)); iS = np.argmax(at(G, q, yq))
        pF, pS = at(F, q, yq)[iF], at(G, q, yq)[iS]
        mF = np.mean(np.abs(at(F, q, yq) - r))/np.max(r); mS = np.mean(np.abs(at(G, q, yq) - r))/np.max(r)
        print(f"{q:10s} {r[i]:12.3f} @{yq[i]:6.1f}   {spread*100:9.1f}% | {pF:8.3f} @{yq[iF]:5.1f} {100*(pF/r[i]-1):6.1f}% | {pS:8.3f} @{yq[iS]:5.1f} {100*(pS/r[i]-1):6.1f}% | F {mF*100:5.1f}%  S {mS*100:5.1f}%")
    print("\nwall values (y+=0):  ref ox'=%.3f oz'=%.3f p'=%.3f | FOSLS ox'=%.3f (curl %.3f) oz'=%.3f (curl %.3f) p'=%.3f | FS ox'=%.3f oz'=%.3f p'=%.3f" % (
        V['oxrms'][0], V['ozrms'][0], V['prms'][0], F['oxrms'][0], F['oxrms_curl'][0], F['ozrms'][0], F['ozrms_curl'][0], F['prms'][0],
        G['oxrms'][0], G['ozrms'][0], G['prms'][0]))
    print("FOSLS primary-vs-curl omega rms, max rel diff y+<60: ox %.1f%% oy %.1f%% oz %.1f%%" % tuple(
        100*np.max(np.abs(at(F, k, yq) - at(F, k+'_curl', yq)))/np.max(at(F, k, yq)) for k in ('oxrms', 'oyrms', 'ozrms')))

    # ---------------- figure ----------------
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    for ax, (q, lab) in zip(axes.ravel(), quants):
        lo = np.full(200, np.inf); hi = np.full(200, -np.inf); yg = np.linspace(0.1, 180, 200)
        for d in refs.values():
            if q in d and d[q] is not None:
                v = at(d, q, yg); lo = np.minimum(lo, v); hi = np.maximum(hi, v)
        ax.fill_between(yg, lo, hi, color='0.75', alpha=0.6, label='5 DNS databases (spread)')
        k = slice(1, None) if q == 'U' else slice(None)      # log axis: drop y+=0
        ax.plot(V['yp'], V[q], 'k-', lw=1.2, label='Vreman & Kuerten 2014 S4_B3')
        ax.plot(F['yp'][k], F[q][k], 'r-', lw=1.8, label='FOSLS run01')
        if q+'_curl' in F: ax.plot(F['yp'], F[q+'_curl'], 'r:', lw=1.2, label='FOSLS curl u')
        ax.plot(G['yp'][k], G[q][k], 'b--', lw=1.5, label='fractional step (E)')
        ax.axvspan(YV, 200, color='y', alpha=0.12)
        ax.set_xlim(0.1 if q == 'U' else 0, 180); ax.set_ylabel(lab); ax.set_xlabel('$y^+$')
        if q == 'U': ax.set_xscale('log'); ax.legend(fontsize=8, loc='upper left')
        ax.grid(alpha=0.3)
    fig.suptitle(f"minimal channel (Lx+=565, Lz+=192) vs full-channel DNS at Re_τ=180 — yellow: y+>{YV:.0f} outside box validity\n"
                 f"{F['name']}, Re_τ={ReF:.1f}   |   {G['name']}, Re_τ={ReS:.1f}", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT, dpi=130); print('wrote', OUT)


if __name__ == '__main__':
    main()
