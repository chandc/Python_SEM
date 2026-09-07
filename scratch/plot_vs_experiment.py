"""FOSLS and fractional-step DNS against DIGITIZED experimental profiles.

Experimental sources, both digitized from the open-access KMM (1987) JFM paper
(400 dpi render, hole-based symbol detection + manual curation of symbols that
touched curves, decade-tick calibration; digitizer in the session scratchpad,
verification overlays eyeballed against the originals):

 * Eckelmann (1974), 'corrected' scaling of KMM fig. 5 -- mean profile,
   oil channel, Re_tau = 142.
 * Kreplin & Eckelmann (1979), renormalized per KMM fig. 7 -- u', v', w' rms,
   oil channel, Re_tau = 194 (hot-film).

Digitization accuracy is limited by the 1987 print quality: symbol radius maps
to about +-0.4 in y+ and +-0.015 in rms units. The v'/w' hot-film values carry
the cross-contamination caveat KMM discuss (Perry, Lim & Henbest 1985).

CSVs written next to the reference data: reference/eckelmann1974_U_digitized.csv,
reference/kreplin_eckelmann1979_rms_digitized.csv.
"""
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RE_TAU = 180.0
RAW = ("/private/tmp/claude-501/-Users-danielchan-Dropbox-Apple-MLX-CFD/"
       "5a6b9e55-8157-4098-a1f9-f0db86e57051/scratchpad/kmm_digitized_raw.npz")
OUT = os.path.join(_R, "scratch", "fosls_fs_vs_experiment.png")


def solver_profiles(sums, y):
    U, uu, vv, ww, uv = sums
    up = np.sqrt(np.maximum(uu - U * U, 0.0))
    vp, wp = np.sqrt(np.maximum(vv, 0.0)), np.sqrt(np.maximum(ww, 0.0))
    ylo = np.unique(np.round(np.minimum(y, 2.0 - y), 12))
    def sym(a):
        return 0.5 * (np.interp(ylo, y, a) + np.interp(2.0 - ylo, y, a))
    return ylo * RE_TAU, sym(U), sym(up), sym(vp), sym(wp)


def main():
    raw = np.load(RAW)
    # fig 5: log-x decades at cols 151/734/1316 (spacing 582.5), y 0..20 over
    # rows 877..46
    f5 = raw["f5"]
    ex_y = 10 ** ((f5[:, 0] - 151.0) / 582.5)
    ex_U = (877.0 - f5[:, 1]) / (877.0 - 46.0) * 20.0
    ke = {k: raw[f"f7_{k}"] for k in ("u", "v", "w")}

    os.makedirs(os.path.join(_R, "reference"), exist_ok=True)
    np.savetxt(os.path.join(_R, "reference", "eckelmann1974_U_digitized.csv"),
               np.column_stack([ex_y, ex_U]), fmt="%.4f", delimiter=",",
               header="y+,U+  (Eckelmann 1974 corrected, digitized from KMM 1987 fig.5)")
    with open(os.path.join(_R, "reference",
                           "kreplin_eckelmann1979_rms_digitized.csv"), "w") as f:
        f.write("# Kreplin & Eckelmann 1979, renormalized, digitized from KMM 1987 fig.7\n"
                "component,y+,rms+\n")
        for k, a in ke.items():
            for yp, v in a:
                f.write(f"{k},{yp:.2f},{v:.4f}\n")

    fsd = np.load(os.path.join(_R, "scratch/fs_seed/stats_latest.npz"))
    y = np.array(fsd["y"])
    fs = solver_profiles(fsd["sums"] / int(fsd["nsamp"]), y)
    ck = sorted(glob.glob(os.path.join(_R, "scratch/run01_ck/checkpoint_*.npz")))[-1]
    d = np.load(ck)
    fo = solver_profiles(d["stats_sums"] / int(d["stats_nsamp"]), y)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    a = ax[0]
    a.semilogx(fo[0][1:], fo[1][1:], "k-", lw=2, label="FOSLS run01 (t=4.96)")
    a.semilogx(fs[0][1:], fs[1][1:], "-", color="tab:blue", lw=1.6,
               label="fractional step (t=15.95)")
    a.semilogx(ex_y, ex_U, "o", ms=7, mfc="none", color="tab:red",
               label="Eckelmann (1974) expt, Re$_\\tau$=142")
    yy = np.logspace(0, np.log10(200), 100)
    a.semilogx(yy[yy < 13], yy[yy < 13], ":", color="gray", lw=1)
    a.semilogx(yy[yy > 8], 2.5 * np.log(yy[yy > 8]) + 5.5, ":", color="gray",
               lw=1, label="$u^+\\!=y^+$;  $2.5\\ln y^+ + 5.5$")
    a.set(xlabel="$y^+$", ylabel="$U^+$", title="(a) mean velocity vs experiment",
          xlim=(1, 200), ylim=(0, 21))
    a.legend(fontsize=8, loc="upper left")

    a = ax[1]
    comp = {"u": (2, "tab:blue", "o"), "w": (4, "tab:green", "+"),
            "v": (3, "tab:orange", "^")}
    for k, (j, c, mk) in comp.items():
        a.plot(fo[0], fo[j], "-", color=c, lw=2)
        a.plot(fs[0], fs[j], "--", color=c, lw=1.2)
        a.plot(ke[k][:, 0], ke[k][:, 1], mk, color=c, ms=7,
               mfc="none", mew=1.4)
        a.annotate(f"${k}'^+$", (76, fo[j][np.searchsorted(fo[0], 74)] + 0.06),
                   color=c, fontsize=11)
    a.plot([], [], "k-", lw=2, label="FOSLS")
    a.plot([], [], "k--", lw=1.2, label="fractional step")
    a.plot([], [], "ks", ms=6, mfc="none",
           label="Kreplin & Eckelmann (1979) expt, Re$_\\tau$=194")
    a.set(xlabel="$y^+$", ylabel="rms / $u_\\tau$",
          title="(b) turbulence intensities vs experiment", xlim=(0, 80),
          ylim=(0, 3.0))
    a.legend(fontsize=8, loc="upper right")

    fig.suptitle("Minimal-channel DNS vs digitized experimental profiles "
                 "(hot-film / oil channel, via KMM 1987 figs. 5 & 7)")
    fig.tight_layout()
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
