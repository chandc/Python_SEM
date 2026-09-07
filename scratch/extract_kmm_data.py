"""Convert the detected/curated KMM fig 5 & 7 pixel points to data units.

Curation of fig 7 (documented against the zoom crops):
 * auto circles minus two triangle-misclasses, plus two circles that sat on the
   steep curve and one on the u-curve at y+~24 that were auto-tagged 'tri';
 * triangles: auto minus the reclasses, plus the ones touching the long-dash
   curve at y+ >= 32 read manually from zoomA, plus (336,727) and (533,631);
 * plus signs: auto plus the curve-touching ones from zoomA and (335,567).
Fig 5: every hollow detection is an Eckelmann circle.
"""
import numpy as np


def fig7_points():
    circ = [(190, 778), (192, 753), (194, 728), (208, 721), (216, 671),
            (231, 574), (243, 476), (257, 404), (270, 332), (300, 223),
            (331, 182), (361, 158), (390, 156), (420, 162), (451, 179),
            (481, 195), (532, 235), (649, 293), (765, 330), (883, 369),
            (1057, 414), (1232, 448)]
    tri = [(230, 802), (246, 788), (259, 771), (275, 764), (305, 746),
           (336, 727), (425, 693), (455, 680), (485, 665), (533, 631),
           (652, 606), (768, 594), (885, 594), (1058, 598), (1232, 597)]
    plus = [(229, 707), (244, 667), (259, 635), (274, 611), (303, 580),
            (335, 567), (363, 552), (393, 550), (423, 547), (454, 547),
            (484, 549), (534, 551), (652, 555), (768, 581), (886, 579),
            (1060, 583), (1232, 585)]
    # frame (t,b,l,r) = (71, 846, 175, 1341); line ~8 px thick -> centres
    l, r, t, b = 179.0, 1337.0, 75.0, 842.0
    fx = lambda x: (x - l) / (r - l) * 80.0
    fy = lambda y: (b - y) / (b - t) * 3.0
    out = {}
    for nm, ps in (("u", circ), ("v", tri), ("w", plus)):
        a = np.array([(fx(x), fy(y)) for x, y in sorted(ps)])
        out[nm] = a[(a[:, 0] > 0.3) & (a[:, 0] < 80)]
    return out


def fig5_points():
    # all hollow detections were circles; frame (42,881,147,1567)
    pts = [(203, 833), (330, 803), (465, 734), (596, 649), (650, 621),
           (735, 528), (817, 439), (862, 411), (958, 313), (1040, 265),
           (1089, 250), (1191, 207), (1266, 179), (1350, 152), (1424, 139)]
    # log-x calibration from the decade ticks: 10^0 at the left frame centre,
    # and the decade spacing measured from the major ticks under the axis
    # (found below by tick detection; hard numbers pasted after running it).
    return np.array(pts, dtype=float)


def fig5_decades():
    from PIL import Image
    img = np.asarray(Image.open("fig5-10.pgm"))[1500:2650, 500:2250]
    dark = img < 128
    b = 877  # bottom frame line centre (frame b=881, thickness ~8)
    strip = dark[b - 40: b - 6, :]  # ticks extend up from the axis
    heights = strip.sum(axis=0)
    cols = np.flatnonzero(heights > 20)  # long (major) ticks only
    groups = np.split(cols, np.flatnonzero(np.diff(cols) > 3) + 1)
    return [int(g.mean()) for g in groups if len(g)]


if __name__ == "__main__":
    ticks = fig5_decades()
    print("fig5 major ticks at columns:", ticks)
    p7 = fig7_points()
    for nm, a in p7.items():
        print(f"fig7 {nm}: {len(a)} pts, y+ {a[0,0]:.1f}..{a[-1,0]:.1f}, "
              f"peak {a[:,1].max():.3f}")
    np.savez("kmm_digitized_raw.npz", **{f"f7_{k}": v for k, v in p7.items()},
             f5=fig5_points(), f5_ticks=np.array(ticks))
