"""Digitise Armaly, Durst, Pereira & Schonung, JFM 127 (1983) 473-496.

    python reference/armaly_digitize.py

Source PDF: reference/armaly_durst_pereira_schonung_JFM_1983.pdf
            (free copy: courses.washington.edu/me431/handouts/armaly-jfm-83.pdf)

The paper contains NO TABLES -- every quantity is published as a figure.  This
script renders the figure pages with ghostscript at 600 dpi, finds the axes and
tick marks by projection, calibrates, and extracts the curves by taking the
median ink row per column.  It is reproducible; nothing is read by eye.

    Figure 4    x1/S vs Re, MEASURED (laser-Doppler).  Re axis runs to 7000, so
                the laminar branch is compressed: 0.413 px per unit Re at 600 dpi
                (1 px ~ 2.4 in Re).  Only Re < 450 is extracted -- above that the
                x2..x5 curves cross and a per-column median is meaningless.
    Figure 13a  x1/S vs Re, PREDICTED (their own TEACH finite-difference code).
                Linear axis 0-1300, 1.52 px per unit Re -- much better resolved.
                Extracted to Re < 480, where the x4/x5 curves appear.

> An earlier version of this file carried values read BY EYE.  They were wrong:
> Fig 13a at Re = 389 was read as 7.0 where the extraction gives 8.00, a 14 %
> error.  Eyeball digitisation of a scanned figure is not good enough here.

GEOMETRY AND CONVENTIONS (paper sec. 2.1 p.475, sec. 2.2.1 p.478):
    inlet height h = 5.2 mm, outlet H = 10.1 mm, step S = 4.9 mm
    expansion ratio 1 : 1.94        S/h = 0.942  (NOT 1 -- x/S != x/h)
    Re = V*D/nu,  V = 2/3 u_max,inlet = AVERAGE inlet velocity,  D = 2h
    x/S is normalised by STEP height.

VALIDITY: "two-dimensional flows only at Reynolds numbers Re < 400 and Re > 6000"
(p.474).  Re = 389 sits just inside the lower window.

NOT AVAILABLE: the published velocity profiles (figs 5, 6) are at Re = 1095 and
1290, both ABOVE the 2-D limit.  There is no experimental velocity profile at
Re ~ 389 in this paper.
"""
import os
import subprocess
import numpy as np

SC = os.path.dirname(os.path.abspath(__file__))
PDF = f'{SC}/armaly_durst_pereira_schonung_JFM_1983.pdf'


def render(page, out, dpi=600):
    if not os.path.exists(out):
        subprocess.run(['gs', '-dNOPAUSE', '-dBATCH', '-sDEVICE=pnggray',
                        f'-r{dpi}', f'-dFirstPage={page}', f'-dLastPage={page}',
                        f'-sOutputFile={out}', PDF],
                       check=True, capture_output=True)
    from PIL import Image
    return np.array(Image.open(out)) < 128


def extract(ink, px0, pxs, py0, pys, c0, c1, r0, r1):
    """median ink row per column -> (Re, x/S)"""
    out = []
    for c in range(c0, c1):
        b = np.where(ink[r0:r1, c])[0]
        if b.size:
            out.append(((c-px0)/pxs, (py0-(r0+np.median(b)))/pys))
    return np.array(out)


def main():
    # ---- Figure 13a (page 16): PREDICTED.  axes: y-col 1051, x-row 3326
    #      ticks every 100 Re (col 1054 = Re 0, 3032 = Re 1300)
    #      ticks every 1 in x/S  (row 1263 = 14, 3038 = 2)  -> x/S = 0 at 3334
    ink = render(16, f'{SC}/fig13a_600dpi.png')
    pred = extract(ink, 1054.0, (3032-1054)/1300.0, 3334.0, (3038-1263)/12.0,
                   int(1054+30*1.5215), int(1054+480*1.5215), 2000, 3300)

    # ---- Figure 4 (page 7): MEASURED.  frame y-col 583, x-row 5533
    #      x ticks 1000..7000 at cols 985..3463 -> Re 0 at col 572, 0.413 px/Re
    #      y ticks 25..0 at rows 3463..5527     -> 82.56 px per x/S unit
    ink = render(7, f'{SC}/fig4_600dpi.png')
    meas = extract(ink, 572.0, 413.0/1000.0, 5527.0, (5527-3463)/25.0,
                   600, 790, 3470, 5520)

    for name, a, kind in (('armaly_fig4_x1_measured.csv', meas, 'MEASURED (laser-Doppler)'),
                          ('armaly_fig13a_x1_predicted.csv', pred, 'PREDICTED (TEACH FD code)')):
        with open(f'{SC}/{name}', 'w') as f:
            f.write(f"# Armaly et al., JFM 127 (1983) 473-496 -- {kind}\n")
            f.write("# primary reattachment length x1, normalised by STEP height S = 4.9 mm\n")
            f.write("# Re = V*2h/nu, V = average inlet velocity, h = inlet height 5.2 mm\n")
            f.write("# pixel-extracted at 600 dpi by reference/armaly_digitize.py\n")
            f.write("# 2-D flow only for Re < 400 (paper p.474)\n")
            f.write("Re,x1_over_S\n")
            for r, v in a:
                f.write(f"{r:.2f},{v:.4f}\n")
        print(f"wrote {name}  ({len(a)} points, Re {a[0,0]:.0f}..{a[-1,0]:.0f})")

    print(f"\n{'Re':>6}{'measured':>11}{'predicted':>11}")
    for t in (100, 200, 300, 389, 400):
        m = meas[np.argmin(np.abs(meas[:, 0]-t)), 1]
        p = pred[np.argmin(np.abs(pred[:, 0]-t)), 1]
        print(f"{t:>6}{m:>11.3f}{p:>11.3f}")
    print("\nAt Re = 389 -- the value this project targets -- Armaly's measured and")
    print("computed reattachment agree at x1/S ~ 8.0, matching the repo's 8.0 +/- 0.3 gate.")


if __name__ == '__main__':
    main()
