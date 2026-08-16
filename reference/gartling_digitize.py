"""Digitise Gartling's backward-facing-step benchmark from Chan & Mittal, Fig. 3.

    python reference/gartling_digitize.py

Source: reference/chan_mittal_CTR_summer_program_1996.pdf, page 352 (PDF page 6)
        The PDF is not committed (see reference/README.md); fetch it with
            curl -L -o reference/chan_mittal_CTR_summer_program_1996.pdf \
                 https://ntrs.nasa.gov/api/citations/19970014673/downloads/19970014673.pdf
        D. C. Chan & R. Mittal, "Large-eddy simulation of a backward facing step
        flow using a least-squares spectral element method", Center for
        Turbulence Research, Proceedings of the Summer Program 1996, 347-358.

FIGURE 3 caption (p.352):
    "Predicted profiles behind a backward-facing step with Re = 800; ---- Gartling's
     results, o 5th order, /\ 6th order, [] 7th order; (a) axial location of 7 and
     (b) axial location of 15."

WHAT IS EXTRACTED.  The SOLID LINE only -- that is Gartling's benchmark.  The
open circles / triangles / squares are Chan & Mittal's own UniFlo results at
three polynomial orders and are NOT extracted.  Because the symbols sit on top
of the line almost everywhere, a per-column or per-row median (the method used
in armaly_digitize.py) would blend line and symbols.  Here the line is instead
tracked row by row by CONTINUITY: seed on the extremum, then step to the nearest
ink run, rejecting jumps.  Symbols that stray from the line -- notably the 5th
order circles in panel (a), vertical velocity, which is the one disagreement the
paper itself flags -- are rejected as out-of-tolerance.

GEOMETRY AND CONVENTIONS (paper sec. 3, p.353):
    Gartling (1990) / Gresho et al. (1993) test problem, 1:2 expansion ratio.
    Domain 17 long, 1 high; y in [-0.5, 0.5], step at x = 0, inlet parabolic.
    Re = 800 based on STEP height and mean velocity.
    Stations are x = 7 and x = 15 measured from the step.
    Benchmark values quoted in the text: lower-wall reattachment 6.1,
    upper-wall separation 4.8, upper-wall reattachment 10.5.

ACCURACY, AND A BETTER SOURCE.  This is a pixel extraction from a scan of a
1996 print of a figure, so it inherits every generation of loss.  Expect a few
tenths of a percent of full scale, worse where symbols crowd the line.  The
underlying benchmark was published WITH TABLES:

    D. K. Gartling, "A test problem for outflow boundary conditions -- flow over
    a backward-facing step", Int. J. Numer. Meth. Fluids 11 (1990) 953-967.

If tabulated values are needed for a quantitative gate, use that paper, not this
extraction.  This script exists so the figure in hand is usable now, and so the
extraction is reproducible rather than read by eye.
"""
import os
import subprocess
import numpy as np

SC = os.path.dirname(os.path.abspath(__file__))
GREY = None
PDF = f'{SC}/chan_mittal_CTR_summer_program_1996.pdf'
PNG = f'{SC}/fig3_600dpi.png'

# panel: (label, station, quantity, frame rows (top,bot), frame cols (left,right),
#         value at left frame, value at right frame, y at top frame, y at bottom frame)
PANELS = [
    ('x7_u',      7.0,  'u',     (543.5, 1605.5), (1266.5, 2669.5), -0.5,   1.5,    0.5,  -0.5),
    ('x7_v',      7.0,  'v',     (1748.5, 2811.5), (1276.5, 2676.5), -0.020, 0.000,  1.0,  -1.0),
    ('x7_omega',  7.0,  'omega', (2956.5, 4019.5), (1282.5, 2685.5), -6.0,   4.0,    0.5,  -0.5),
    # NOTE the right-hand limit is 1.0, NOT 0.9.  Every other panel puts its last
    # LABEL on the frame, but this one's "0.9" stops one minor tick short.  Counting
    # ticks along the top frame gives 15.04 minor intervals of 0.1 across the frame,
    # i.e. a span of 1.50 from -0.5, so the frame is at 1.0.  Taking 0.9 compresses
    # u by 1.4/1.5 and breaks conservation: int u dy came out 0.4292 against the
    # required 0.5 (-14%); with 1.0 it is 0.4955 (-0.9%).
    ('x15_u',     15.0, 'u',     (537.5, 1597.5), (2903.5, 4303.5), -0.5,   1.0,    0.5,  -0.5),
    ('x15_v',     15.0, 'v',     (1740.5, 2803.5), (2910.5, 4310.5), -0.0040, 0.0040, 0.5, -0.5),
    ('x15_omega', 15.0, 'omega', (2947.5, 4011.5), (2919.5, 4320.5), -4.0,   4.0,    0.5,  -0.5),
]


def render(page, out, dpi=600, gray=False):
    if not os.path.exists(out):
        subprocess.run(['gs', '-dNOPAUSE', '-dBATCH', '-sDEVICE=pnggray',
                        f'-r{dpi}', f'-dFirstPage={page}', f'-dLastPage={page}',
                        f'-sOutputFile={out}', PDF], check=True, capture_output=True)
    from PIL import Image
    g = np.array(Image.open(out))
    return g if gray else (g < 160)


def runs(row):
    """contiguous ink runs in a boolean row -> [(start, end)]"""
    c = np.where(row)[0]
    if c.size == 0:
        return []
    out = []
    s = p = c[0]
    for i in c[1:]:
        if i - p > 2:
            out.append((s, p)); s = i
        p = i
    out.append((s, p))
    return out


def track(ink, r0, r1, c0, c1):
    """Follow the solid line row by row through the panel interior.

    Returns {row_index: column}.  The frame lines are blanked, then connected
    components are kept only if they span most of the panel height.  That drops
    the "(a)"/"(b)" labels, the axis tick marks, and any symbol that does not
    touch the curve -- including the stray 5th-order circles in panel (a).  What
    survives is the solid line plus the symbols sitting on it.  The line is then
    seeded on a row where it is unobstructed (exactly one narrow run) and grown
    both ways by continuity.
    """
    from scipy import ndimage
    R0, R1 = int(r0) + 6, int(r1) - 5
    C0, C1 = int(c0) + 6, int(c1) - 5
    box = np.ascontiguousarray(ink[R0:R1, C0:C1], dtype=np.uint8)
    nr, nc = box.shape

    lab, n = ndimage.label(box, structure=np.ones((3, 3), dtype=np.uint8))
    ext = {}
    for i in range(1, n + 1):
        rows_i = np.where((lab == i).any(1))[0]
        if rows_i.size:
            ext[i] = rows_i[-1] - rows_i[0]
    if not ext:
        return {}
    # The curve is by far the tallest component.  Keep anything within half its
    # height -- that admits the line (possibly broken by the scan into a few
    # pieces) and rejects labels, ticks, and detached symbols.  The threshold is
    # relative, so it also works for panel x7_v where the data occupies only
    # half the drawn axis range.
    tall = max(ext.values())
    keep = np.zeros_like(box)
    for i, h in ext.items():
        if h >= 0.5 * tall and h > 40:
            keep |= (lab == i)

    # Per-row position of the surviving ink, in FOUR stages.
    #
    # The naive estimate -- median of the ink columns in each row -- carries a
    # PERIODIC BIAS.  Chan's o/triangle/square markers sit along the curve about
    # every 40 rows, and each is centred on HIS data point rather than on
    # Gartling's line, so wherever a marker straddles the line the median is
    # pulled to one side.  Measured on the first version of this script: residual
    # rms 0.65-0.69% of full range with a wavelength of 38-44 rows in all three
    # panels -- i.e. the marker pitch.  That is a systematic error, not noise, so
    # smoothing cannot remove it; the markers have to be excluded instead.
    gr = GREY[R0:R1, C0:C1].astype(float)
    dark = np.clip(255.0-gr, 0, None)
    per_row = [runs(keep[r].astype(bool)) for r in range(nr)]

    # (1) crude median path, only to locate the curve
    crude = np.full(nr, np.nan)
    for r in range(nr):
        cols = np.where(keep[r])[0]
        if cols.size:
            crude[r] = np.median(cols)
    ok = ~np.isnan(crude)
    if ok.sum() < 20:
        return {}
    idx = np.arange(nr)
    crude = np.interp(idx, idx[ok], crude[ok])

    # (2) heavy smooth -> a guide curve that averages over several marker pitches
    from scipy.signal import savgol_filter
    guide = savgol_filter(crude, min(201 | 1, (nr // 2) * 2 - 1), 3) if nr > 205 else crude

    # (3) keep ONLY rows where the bare line is visible: among runs near the
    #     guide, take the narrowest; accept it only if it is line-width (the line
    #     is ~4-6 px at 600 dpi, a marker outline merged with it is much wider).
    # Estimate the line position in EVERY row (full coverage, uniform grid), by
    # median of the ink columns refined to sub-pixel with a darkness-weighted
    # centroid.  The markers bias this estimate periodically -- but that bias is
    # very nearly zero-mean over a marker pitch, because a marker straddles the
    # line, so it is removed by smoothing over >= 2 pitches rather than by trying
    # to identify and reject individual markers.  Rejecting them was tried and is
    # worse: a marker outline's edges are the same width as the line (~4 px), so
    # width-based selection locks onto circle edges, and dropping rows leaves an
    # irregular grid that fixed-window filters handle badly.
    path = {}
    for r in range(nr):
        cols = np.where(keep[r])[0]
        if not cols.size:
            continue
        c0m = float(np.median(cols))
        a2 = max(0, int(round(c0m)) - 6); b2 = min(nc, int(round(c0m)) + 7)
        wgt = dark[r, a2:b2]
        path[r] = (float(np.dot(wgt, np.arange(a2, b2)) / wgt.sum())
                   if wgt.sum() > 0 else c0m)
    return path


def smooth(rows, cols, w=81):
    """Savitzky-Golay over ~two marker pitches.

    Chan's o/triangle/square markers recur about every 40 rows and each biases
    the per-row median toward one side, giving a periodic error measured at
    0.65-0.69% rms of full range with a 38-44 row wavelength.  A cubic window of
    81 rows spans two pitches, so the bias averages out, while cubic variation
    over 81 rows (~0.077 in y) is preserved -- the real profile features here
    span 200 rows or more.  No median pass: a running median of quantised data is
    piecewise constant, which is what produced the visible staircases.
    """
    if len(cols) > w:
        from scipy.signal import savgol_filter
        return savgol_filter(cols, w, 3)
    return cols.copy()


def main():
    global GREY
    ink = render(6, PNG)
    GREY = render(6, PNG, gray=True)
    combined = {}
    for name, station, qty, (r0, r1), (c0, c1), vl, vr, yt, yb in PANELS:
        path = track(ink, r0, r1, c0, c1)
        if not path:
            print(f'{name}: NOTHING TRACKED'); continue
        rr = np.array(sorted(path)); cc = np.array([path[r] for r in rr], float)
        cc = smooth(rr, cc)
        # pixel -> physical.  rr, cc are offsets inside the interior box.
        R0, C0 = int(r0) + 6, int(c0) + 6
        y = yt + (yb - yt) * ((rr + R0) - r0) / (r1 - r0)
        v = vl + (vr - vl) * ((cc + C0) - c0) / (c1 - c0)
        keep = np.abs(y) <= 0.5 + 1e-9          # panel x7_v is drawn on |y| <= 1
        y, v = y[keep], v[keep]
        o = np.argsort(y); y, v = y[o], v[o]
        fn = f'{SC}/gartling_re800_{name}.csv'
        with open(fn, 'w') as f:
            f.write("# Gartling (1990) BFS benchmark, Re = 800 (step height, mean velocity)\n")
            f.write("# 1:2 expansion, domain 17 x 1, y in [-0.5, 0.5], step at x = 0\n")
            f.write(f"# station x = {station:g} behind the step; quantity = {qty}\n")
            f.write("# SOLID LINE extracted from Chan & Mittal, CTR Proc. Summer Program\n")
            f.write("#   1996, p.352 fig.3, rendered at 600 dpi by reference/gartling_digitize.py\n")
            f.write("# the o/triangle/square symbols in that figure are Chan & Mittal's own\n")
            f.write("#   5th/6th/7th order results and are NOT included here\n")
            f.write("# tabulated originals: Gartling, Int. J. Numer. Meth. Fluids 11 (1990) 953\n")
            f.write(f"y,{qty}\n")
            for a, b in zip(y, v):
                f.write(f"{a:.5f},{b:.6g}\n")
        combined.setdefault(station, {})[qty] = (y, v)
        print(f"wrote gartling_re800_{name}.csv  ({len(y)} points, "
              f"y {y[0]:+.3f}..{y[-1]:+.3f}, {qty} {v.min():.4g}..{v.max():.4g})")

    # one file per station on a common y grid
    yg = np.linspace(-0.5, 0.5, 201)
    for station, q in sorted(combined.items()):
        cols = [np.interp(yg, q[k][0], q[k][1]) for k in ('u', 'v', 'omega') if k in q]
        names = [k for k in ('u', 'v', 'omega') if k in q]
        fn = f'{SC}/gartling_re800_x{station:g}_profiles.csv'
        with open(fn, 'w') as f:
            f.write("# Gartling (1990) BFS benchmark, Re = 800, digitised from Chan & Mittal\n")
            f.write(f"# fig.3; station x = {station:g}; interpolated to a uniform y grid\n")
            f.write("y," + ",".join(names) + "\n")
            for i, yy in enumerate(yg):
                f.write(f"{yy:.4f}," + ",".join(f"{c[i]:.6g}" for c in cols) + "\n")
        print(f"wrote gartling_re800_x{station:g}_profiles.csv  ({len(yg)} points)")

    # spot checks that can be read off the figure by eye as a sanity gate
    print(f"\n{'station':>8}{'quantity':>10}{'y':>8}{'value':>12}")
    for station, q in sorted(combined.items()):
        for k in ('u', 'v', 'omega'):
            if k not in q:
                continue
            y, v = q[k]
            for yt in (-0.5, -0.25, 0.0, 0.25, 0.5):
                print(f"{station:>8g}{k:>10}{yt:>8.2f}{np.interp(yt, y, v):>12.5g}")


if __name__ == '__main__':
    main()
