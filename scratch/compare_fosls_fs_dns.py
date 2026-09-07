"""FOSLS vs fractional-step minimal-channel DNS against published data.

Curves: run01 (FOSLS, accumulated stats to t=4.96), the fractional-step seed run
(stats_latest.npz), the Abe-Antonia-Kawamura Re_tau=180 DNS (full-size box,
reference/akm_chan180/ch180.dat), and the Reichardt correlation, which is a fit
to pipe/channel EXPERIMENTS and stands in for the experimental mean profile.
Experimental fluctuation peaks (hot-film, Kreplin & Eckelmann 1979) are quoted
in the printed table rather than drawn: no digitized profile is on disk, and a
band pretending to be data would be worse than a cited number.

Wall units are exact for both runs (forcing fixes u_tau = 1, nu = 1/180);
nothing is rescaled and no constant is fitted.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RE_TAU, NU = 180.0, 1.0 / 180.0
OUT = os.path.join(_R, "scratch", "fosls_vs_fs_vs_published.png")


def load_run(sums, y):
    """(y+, U+, u'+, v'+, w'+, -uv+) from plane-averaged sums on nodes y."""
    U, uu, vv, ww, uv = sums
    up = np.sqrt(np.maximum(uu - U * U, 0.0))
    vp, wp = np.sqrt(np.maximum(vv, 0.0)), np.sqrt(np.maximum(ww, 0.0))
    # fold the two halves onto 0..1: statistics are symmetric (antisymmetric for
    # uv), and averaging the halves removes the odd half of the sampling noise
    ylo = np.unique(np.round(np.minimum(y, 2.0 - y), 12))
    def sym(a, odd=False):
        lo = np.interp(ylo, y, a)
        hi = np.interp(2.0 - ylo, y, a)
        return 0.5 * (lo + (-hi if odd else hi))
    return (ylo * RE_TAU, sym(U), sym(up), sym(vp), sym(wp), sym(-uv, odd=True))


def load_akm(path):
    rows1, rows2 = [], []
    block = 0
    for ln in open(path):
        s = ln.split()
        if "u_mean+" in ln:
            block = 1
            continue
        if "-uv+" in ln and "vv+" in ln:
            block = 2
            continue
        if block and len(s) >= 4:
            try:
                v = [float(x) for x in s]
            except ValueError:
                block = 0
                continue
            (rows1 if block == 1 else rows2).append(v)
    a1, a2 = np.array(rows1), np.array(rows2)
    return dict(y1=a1[:, 1], U=a1[:, 2], u=a1[:, 3], w=a1[:, 4],
                y2=a2[:, 1], v=a2[:, 2], uv=a2[:, 3])


def reichardt(yp, k=0.40):
    return (np.log(1 + k * yp) / k
            + 7.8 * (1 - np.exp(-yp / 11.0) - (yp / 11.0) * np.exp(-yp / 3.0)))


def peaks(yp, *arrs):
    return [(float(a.max()), float(yp[a.argmax()])) for a in arrs]


def main():
    import glob
    fsd = np.load(os.path.join(_R, "scratch/fs_seed/stats_latest.npz"))
    y = np.array(fsd["y"])  # identical mesh for both runs (run01 was seeded on it)
    fs = load_run(fsd["sums"] / int(fsd["nsamp"]), y)
    fs_n, fs_t = int(fsd["nsamp"]), float(fsd["t"])

    ck = sorted(glob.glob(os.path.join(_R, "scratch/run01_ck/checkpoint_*.npz")))[-1]
    d = np.load(ck)
    fo = load_run(d["stats_sums"] / int(d["stats_nsamp"]), y)
    fo_n, fo_t = int(d["stats_nsamp"]), float(d["t"])

    akm = load_akm(os.path.join(_R, "reference/akm_chan180/ch180.dat"))
    # the AKM file may store rms or variance; the u peak decides which
    if akm["u"].max() > 4.0:
        for k in ("u", "v", "w"):
            akm[k] = np.sqrt(akm[k])

    fig, ax = plt.subplots(1, 4, figsize=(21, 4.6))
    st = dict(fosls=dict(color="k", lw=2, label=f"FOSLS run01 ({fo_n} samp, t={fo_t:.2f})"),
              fs=dict(color="tab:blue", lw=1.6, label=f"fractional step ({fs_n} samp, t={fs_t:.2f})"),
              akm=dict(color="tab:red", ls="none", marker="o", ms=3, markevery=3,
                       label="AKM DNS Re$_\\tau$=180 (full box)"))

    yp = np.logspace(-0.5, np.log10(180), 200)
    a = ax[0]
    a.semilogx(fo[0][1:], fo[1][1:], **st["fosls"])
    a.semilogx(fs[0][1:], fs[1][1:], **st["fs"])
    a.semilogx(akm["y1"][1:], akm["U"][1:], **st["akm"])
    a.semilogx(yp, reichardt(yp), "g--", lw=1.2, label="Reichardt (expt. correlation)")
    a.semilogx(yp[yp < 12], yp[yp < 12], ":", color="gray", lw=1)
    a.set(xlabel="$y^+$", ylabel="$U^+$", title="(a) mean profile", ylim=(0, 21))
    a.legend(fontsize=7.5, loc="upper left")

    a = ax[1]
    for i, (nm, c) in enumerate((("u", "tab:blue"), ("w", "tab:green"), ("v", "tab:orange"))):
        j = {"u": 2, "v": 3, "w": 4}[nm]
        a.plot(fo[0], fo[j], color=c, lw=2)
        a.plot(fs[0], fs[j], color=c, lw=1.2, ls="--")
        ya = akm["y1"] if nm in ("u", "w") else akm["y2"]
        a.plot(ya, akm[nm], "o", color=c, ms=3, markevery=3, mfc="none")
        a.annotate(f"${nm}'^+$", (a.get_xlim()[1] * 0.75, fo[j][np.searchsorted(fo[0], 130)]),
                   color=c)
    a.plot([], [], "k-", lw=2, label="FOSLS")
    a.plot([], [], "k--", lw=1.2, label="fractional step")
    a.plot([], [], "ko", ms=3, mfc="none", label="AKM DNS")
    a.set(xlabel="$y^+$", ylabel="rms", title="(b) Reynolds normal stresses", xlim=(0, 180))
    a.legend(fontsize=8)

    a = ax[2]
    a.plot(fo[0], fo[5], **st["fosls"])
    a.plot(fs[0], fs[5], **st["fs"])
    a.plot(akm["y2"], akm["uv"], **st["akm"])
    a.set(xlabel="$y^+$", ylabel="$-\\langle u'v'\\rangle^+$",
          title="(c) Reynolds shear stress", xlim=(0, 180))
    a.legend(fontsize=8)

    a = ax[3]
    for nm, (ypn, U, *_ ), s in (("FOSLS", fo, "-"), ("frac. step", fs, "--")):
        dUdy = np.gradient(U, ypn)
        tot = dUdy + (fo[5] if nm == "FOSLS" else fs[5])
        a.plot(ypn, tot, s, color="tab:red", lw=1.5,
               label=f"{nm}: $dU^+/dy^+ - \\langle u'v'\\rangle^+$")
    a.plot([0, 180], [1, 0], "g--", lw=1.2, label="$1-y/\\delta$ (exact)")
    a.set(xlabel="$y^+$", ylabel="stress", title="(d) total-stress balance", xlim=(0, 180))
    a.legend(fontsize=8)

    fig.suptitle("Minimal-channel DNS at $Re_\\tau=180$: FOSLS vs fractional step "
                 "vs published DNS and experiment", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print("saved ->", OUT)

    # printed comparison table, experimental values included
    rows = []
    for nm, r in (("FOSLS", fo), ("frac step", fs)):
        (pu, yu), (pv, yv), (pw, yw), (puv, yuv) = peaks(r[0], r[2], r[3], r[4], r[5])
        rows.append((nm, pu, yu, pv, pw, puv, yuv))
    (au, ayu), = peaks(akm["y1"], akm["u"])
    (av, _), (auv, ayuv) = peaks(akm["y2"], akm["v"], akm["uv"])
    (aw, _), = peaks(akm["y1"], akm["w"])
    rows.append(("AKM DNS", au, ayu, av, aw, auv, ayuv))
    print(f"\n{'':12s}{'u`+ pk':>8s}{'@y+':>6s}{'v`+ pk':>8s}{'w`+ pk':>8s}"
          f"{'-uv+ pk':>9s}{'@y+':>6s}")
    for nm, pu, yu, pv, pw, puv, yuv in rows:
        print(f"{nm:12s}{pu:8.3f}{yu:6.1f}{pv:8.3f}{pw:8.3f}{puv:9.3f}{yuv:6.1f}")
    print(f"{'expt (K&E79)':12s}{'~2.6-2.9':>8s}{'~15':>6s}{'~0.9':>8s}{'~1.0-1.1':>8s}"
          f"{'~0.7':>9s}{'~30':>6s}")


if __name__ == "__main__":
    main()
