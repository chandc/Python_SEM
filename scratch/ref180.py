"""Uniform loaders for the published Re_tau=180 channel datasets in reference/.

Every loader returns a dict of 1-D arrays on the lower half-channel (y+ from wall
to centreline), in wall units:
    yp, U, urms, vrms, wrms, uv (= -<u'v'>+), and where available
    oxrms, oyrms, ozrms (vorticity rms, nu/u_tau^2), prms.
"""
import numpy as np, os
ROOT = os.path.join(os.path.dirname(__file__), "..", "reference")

def _half(yp, *cols):
    """Fold a full-channel profile (y from -1..1 or 0..2) onto the lower half."""
    n = len(yp)
    m = (n + 1) // 2
    return [c[:m] for c in (yp,) + cols]

def mkm1999():
    d = os.path.join(ROOT, "mkm1999_chan180")
    m = np.loadtxt(f"{d}/chan180.means")      # y y+ U dU/dy W dW/dy P
    r = np.loadtxt(f"{d}/chan180.reystress")  # y y+ Ruu Rvv Rww Ruv Ruw Rvw
    w = np.loadtxt(f"{d}/chan180.vortvar")    # y y+ Wxx Wyy Wzz ...
    p = np.loadtxt(f"{d}/chan180.velp")       # y y+ Rup Rvp Rwp Rpp
    # files are already wall -> centreline (129 points), no folding needed
    yp, U, uu, vv, ww, uv, wx, wy, wz, pp = (
        m[:, 1], m[:, 2], r[:, 2], r[:, 3], r[:, 4], r[:, 5],
        w[:, 2], w[:, 3], w[:, 4], p[:, 5])
    return dict(name="MKM 1999", Re_tau=178.12, yp=yp, U=U,
                urms=np.sqrt(uu), vrms=np.sqrt(vv), wrms=np.sqrt(ww), uv=-uv,
                oxrms=np.sqrt(wx), oyrms=np.sqrt(wy), ozrms=np.sqrt(wz),
                prms=np.sqrt(pp))

def lm2015():
    d = os.path.join(ROOT, "lm2015_chan180")
    m = np.loadtxt(f"{d}/LM_Channel_0180_mean_prof.dat", comments="%")
    r = np.loadtxt(f"{d}/LM_Channel_0180_vel_fluc_prof.dat", comments="%")
    w = np.loadtxt(f"{d}/LM_Channel_0180_vor_pres_fluc_prof.dat", comments="%")
    yp = m[:, 1]
    return dict(name="Lee & Moser 2015", Re_tau=182.088, yp=yp, U=m[:, 2],
                urms=np.sqrt(r[:, 2]), vrms=np.sqrt(r[:, 3]),
                wrms=np.sqrt(r[:, 4]), uv=-r[:, 5],
                oxrms=np.sqrt(w[:, 2]), oyrms=np.sqrt(w[:, 3]),
                ozrms=np.sqrt(w[:, 4]), prms=np.sqrt(w[:, -1]))

def vreman2014(case="S4_B3"):
    d = os.path.join(ROOT, "vreman2014_chan180")
    m = np.loadtxt(f"{d}/Chan180_{case}_basic_mean.txt", comments="%")
    r = np.loadtxt(f"{d}/Chan180_{case}_basic_rms.txt", comments="%")
    # cols (1-based): 1 y+, 2-4 u1..u3, 5 p, 6-14 du_i/dx_j, then vorticity
    # components and Reynolds stresses follow — locate by header scan.
    with open(f"{d}/Chan180_{case}_basic_rms.txt") as f:
        hdr = [l for l in f if l.lstrip().startswith("%")]
    def col(tag):
        for l in hdr:
            if "Column" in l and tag in l:
                return int(l.split("Column")[1].split(":")[0]) - 1
        raise KeyError(tag)
    yp = m[:, 0]
    out = dict(name="Vreman & Kuerten 2014 " + case, Re_tau=180.0, yp=yp,
               U=m[:, 1], urms=r[:, 1], vrms=r[:, 2], wrms=r[:, 3], prms=r[:, 4])
    # vorticity rms from velocity-gradient rms is not available (needs covariances);
    # use the vor columns if present
    for k, tag in (("oxrms", "vor1"), ("oyrms", "vor2"), ("ozrms", "vor3")):
        out[k] = r[:, col(tag)] / 180.0          # u_tau/H -> nu/u_tau^2
    with open(f"{d}/Chan180_{case}_basic_mean.txt") as f:
        hdrm = [l for l in f if l.lstrip().startswith("%")]
    out["uv"] = -m[:, col_mean(m, hdrm, "u1*u2")]
    return out

def col_mean(m, hdr, tag):
    for l in hdr:
        if "Column" in l and tag in l:
            return int(l.split("Column")[1].split(":")[0]) - 1
    raise KeyError(tag)

def torroja():
    d = os.path.join(ROOT, "torroja_chan180")
    a = np.loadtxt(f"{d}/Re180.prof", comments="%")
    # y/h y+ U+ u'+ v'+ w'+ -Om_z+ om_x'+ om_y'+ om_z'+ uv+ ...
    return dict(name="Torroja (del Alamo & Jimenez 2003)", Re_tau=180.0,
                yp=a[:, 1], U=a[:, 2], urms=a[:, 3], vrms=a[:, 4], wrms=a[:, 5],
                oxrms=a[:, 7], oyrms=a[:, 8], ozrms=a[:, 9],
                uv=-a[:, 10], prms=a[:, 16])   # col 16 = total p rms (13,14 = rapid, slow)

def akm2001():
    d = os.path.join(ROOT, "akm_chan180")
    lines = open(f"{d}/ch180.dat").read().splitlines()
    def block(start_tag, ncol):
        i = next(k for k, l in enumerate(lines) if l.strip().startswith("j") and start_tag in l)
        rows = []
        for l in lines[i + 1:]:
            s = l.split()
            if len(s) != ncol + 1: break
            rows.append([float(x) for x in s[1:]])
        return np.array(rows)
    a = block("u_mean+", 4)   # y+ U+ uu+ ww+
    b = block("vv+", 3)       # y+ vv+ -uv+
    return dict(name="Abe, Kawamura & Matsuo 2001 (JAXA)", Re_tau=180.0,
                yp=a[:, 0], U=a[:, 1], urms=np.sqrt(a[:, 2]), wrms=np.sqrt(a[:, 3]),
                vrms=np.sqrt(np.interp(a[:, 0], b[:, 0], b[:, 1])),
                uv=np.interp(a[:, 0], b[:, 0], b[:, 2]))

ALL = dict(mkm=mkm1999, lm=lm2015, vk=vreman2014, torroja=torroja, akm=akm2001)

if __name__ == "__main__":
    for k, fn in ALL.items():
        try:
            d = fn()
        except Exception as e:
            print(f"{k:8s} FAILED: {e}"); continue
        def pk(q):
            if q not in d or d[q] is None: return "   --  "
            i = np.argmax(d[q]); return f"{d[q][i]:.3f}@{d[q.replace('rms','') and 'yp'][i]:5.1f}"
        Uc = d["U"][-1] if len(d["U"]) else np.nan
        print(f"{d['name']:36s} Re_tau={d['Re_tau']:.1f} Uc={Uc:6.2f}  "
              f"u'={pk('urms')} v'={pk('vrms')} w'={pk('wrms')} -uv={pk('uv')} "
              f"wx'={pk('oxrms')} wz'={pk('ozrms')} p'={pk('prms')}")
