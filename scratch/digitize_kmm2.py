"""Digitize KMM (1987) figs 5 & 7 experimental points, take 2.

Fixes over take 1:
 * frame = rows/cols whose LONGEST CONTIGUOUS dark run spans most of the crop
   (text lines have many dark pixels but short runs);
 * hollow symbols (circles, triangles) are found by their enclosed HOLES via
   binary_fill_holes on the whole mask, so a symbol touching a curve is still
   found;
 * plus signs among small hole-free components with a cross test.
"""
import numpy as np
from PIL import Image
from scipy import ndimage


def longest_run(b):
    """longest run of True per row of 2-D bool array."""
    out = np.zeros(b.shape[0], dtype=int)
    for i, row in enumerate(b):
        d = np.diff(np.concatenate(([0], row.view(np.int8), [0])))
        starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        if len(starts):
            out[i] = (ends - starts).max()
    return out


def frame(dark):
    """bounding box of the largest connected component: the axes frame (the
    curves touch it, which only helps -- the box is still the frame)."""
    lab, n = ndimage.label(dark)
    sizes = ndimage.sum_labels(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    big = int(np.argmax(sizes)) + 1
    sl = ndimage.find_objects(lab == big)[0]
    return sl[0].start, sl[0].stop - 1, sl[1].start, sl[1].stop - 1


def holes(dark):
    filled = ndimage.binary_fill_holes(dark)
    return filled & ~dark


def hollow_symbols(dark):
    """centres of enclosed holes with symbol-interior size; classify by shape."""
    hh = holes(dark)
    lab, n = ndimage.label(hh)
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        m = lab[sl] == i
        a = int(m.sum())
        if not (5 <= h <= 34 and 5 <= w <= 34 and a >= 25):
            continue
        rows_w = m.sum(axis=1).astype(float)
        k = max(2, h // 3)
        botw, topw = rows_w[-k:].mean(), rows_w[:k].mean()
        kind = "tri" if botw > 1.6 * max(topw, 1.0) else "circ"
        out.append((sl[1].start + w / 2, sl[0].start + h / 2, kind, a, h, w))
    return out


def plus_signs(dark):
    lab, n = ndimage.label(dark)
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        m = lab[sl] == i
        a = int(m.sum())
        if not (9 <= h <= 34 and 9 <= w <= 34 and 25 <= a):
            continue
        if int(ndimage.binary_fill_holes(m).sum()) - a > 0.1 * a:
            continue  # hollow -> handled elsewhere
        fill = a / (h * w)
        if fill > 0.5:
            continue  # blob / text
        # cross test: middle row and column carry most of the mass
        cm = m[h // 2 - 1: h // 2 + 2, :].sum() + m[:, w // 2 - 1: w // 2 + 2].sum()
        if cm < 0.75 * a:
            continue
        if not (0.6 < h / w < 1.6):
            continue
        out.append((sl[1].start + w / 2, sl[0].start + h / 2, "plus", a, h, w))
    return out


def overlay(img, fr, pts, out, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.imshow(img, cmap="gray")
    cols = dict(circ="red", tri="lime", plus="blue")
    for (cx, cy, k, *_ ) in pts:
        ax.plot(cx, cy, "x", color=cols[k], ms=8, mew=1.5)
    t, b, l, r = fr
    for v in (t, b):
        ax.axhline(v, color="orange", lw=0.6)
    for v in (l, r):
        ax.axvline(v, color="orange", lw=0.6)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def run(figfile, crop, name):
    img = np.asarray(Image.open(figfile), dtype=np.uint8)[
        crop[0]:crop[1], crop[2]:crop[3]]
    dark = img < 128
    fr = frame(dark)
    t, b, l, r = fr
    inner = dark[t + 4: b - 3, l + 4: r - 3]
    pts = []
    for (cx, cy, k, a, h, w) in hollow_symbols(inner) + plus_signs(inner):
        pts.append((cx + l + 4, cy + t + 4, k, a, h, w))
    overlay(img, fr, pts, f"{name}_verify.png",
            f"{name}: red=circ green=tri blue=plus  frame={fr}")
    print(name, "frame(t,b,l,r)=", fr, " counts:",
          {k: sum(1 for p in pts if p[2] == k) for k in ("circ", "tri", "plus")})
    return fr, pts


if __name__ == "__main__":
    fr7, pts7 = run("fig7-12.pgm", (280, 1330, 600, 2200), "fig7")
    np.save("fig7_pts.npy", np.array([(x, y, {"circ": 0, "tri": 1, "plus": 2}[k])
                                      for x, y, k, *_ in pts7]))
    np.save("fig7_frame.npy", np.array(fr7))
    fr5, pts5 = run("fig5-10.pgm", (1500, 2650, 500, 2250), "fig5")
    np.save("fig5_pts.npy", np.array([(x, y, 0) for x, y, k, *_ in pts5 if k == "circ"]))
    np.save("fig5_frame.npy", np.array(fr5))
