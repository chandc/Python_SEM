"""Extract Fig. 3b of Gourianov et al. (2022) -- TGV dissipation at Re = 800.

    uv run --quiet python reference/gourianov_digitize.py

Companion to `gartling_digitize.py` / `armaly_digitize.py`, but this figure is
VECTOR art, so nothing is read by pixel analysis: the curve polylines and the
axis tick marks are pulled straight out of the PDF content stream with PyMuPDF
and mapped through the ticks.  Accuracy is therefore limited by the authors'
plotting, not by our extraction.

Writes reference/gourianov_fig3b_tgv_re800.csv:
    t/T0, then eps(t)/(E0/T0) for DNS (256^3) and MPS chi = 192/128/96.

THE t = 0 INTERCEPT IS A FREE CALIBRATION CHECK, and it settles an ambiguity
in the paper's text.  For the TGV initial condition Omega_0/V = 3/8 exactly, so

    eps(0)/(E0/T0) = 2 nu (3/8) / (E0/T0)

is 0.0471 if E0 is the ACTUAL mean kinetic energy density u0^2/8, but 0.0118 if
E0 = u0^2/2 as the Methods text states.  Extraction gives 0.0472 -- the actual
energy, to 0.2%.  So Table 1's e values are normalised by the true initial
kinetic energy and our e values are directly comparable to them, with no factor
of four (see TGV_VALIDATION.md sec 8).
"""
import numpy as np
import pymupdf

PDF = 'reference/2106.05782v3.pdf'
PAGE = 5                                   # 0-based; Fig. 3 is on page 6
COL = {(0., 0., 0.): 'DNS_256',
       (0., 0., 1.): 'MPS_chi192',
       (0., 0.501991331577301, 0.): 'MPS_chi128',
       (1., 0., 0.): 'MPS_chi96'}
INSETS = [(175.1, 241.4, 451.0, 484.2),    # zoom insets drawn over the panel
          (198.7, 221.6, 414.5, 430.8),
          (259.2, 289.5, 449.4, 485.9)]


def ticks(pg):
    """Axis calibration from the tick marks themselves."""
    xt, yt = set(), set()
    for it in pg.get_drawings():
        for s in it['items']:
            if s[0] != 'l':
                continue
            p, q = s[1], s[2]
            if abs(p.y - q.y) < .3 and 118 < min(p.x, q.x) < 122 \
               and 1 < abs(p.x - q.x) < 5 and 410 < p.y < 490:
                yt.add(round(p.y, 2))
            if abs(p.x - q.x) < .3 and 483 < min(p.y, q.y) < 488 \
               and 1 < abs(p.y - q.y) < 5 and 118 < p.x < 253:
                xt.add(round(p.x, 2))
    return sorted(xt), sorted(yt)


def main():
    pg = pymupdf.open(PDF)[PAGE]
    xt, yt = ticks(pg)
    x0, x1 = xt[0], xt[-1]                 # t/T0 = 0 .. 2  (17 ticks, 0.125 apart)
    yb, dy = yt[-1], (yt[-1] - yt[0])/(len(yt) - 1)
    sy = 0.05/dy                           # 14 ticks, 0.05 apart, bottom = 0
    assert len(xt) == 17 and len(yt) == 14, (len(xt), len(yt))

    def in_inset(r):
        return any(r.x0 >= a-1 and r.x1 <= b+1 and r.y0 >= c-1 and r.y1 <= e+1
                   for a, b, c, e in INSETS)

    pts = {v: [] for v in COL.values()}
    for it in pg.get_drawings():
        c = it.get('color')
        r = it['rect']
        if c not in COL or in_inset(r):
            continue
        if not (x0-2 <= r.x0 and r.x1 <= x1+2 and 410 <= r.y0 and r.y1 <= yb+2):
            continue
        for s in it['items']:
            if s[0] == 'l':
                pts[COL[c]] += [(s[1].x, s[1].y), (s[2].x, s[2].y)]

    tg = np.linspace(0, 2, 81)
    cols = {}
    for nm, P in pts.items():
        a = np.array(sorted(set(P)))
        t = (a[:, 0] - x0)/(x1 - x0)*2.0
        v = (yb - a[:, 1])*sy
        cols[nm] = np.array([np.median(v[np.abs(t - g) < 0.02])
                             if (np.abs(t - g) < 0.02).any() else np.nan
                             for g in tg])

    # calibration check (see module docstring)
    pred = 2*(1/800.)*(3/8.)/((1/8.)/(2*np.pi))
    got = cols['DNS_256'][0]
    print(f'eps(0)/(E0/T0):  extracted {got:.4f}   analytic (E0 = u0^2/8) {pred:.4f}'
          f'   -> {abs(got-pred)/pred*100:.1f}% ; E0 = u0^2/2 would give '
          f'{pred/4:.4f}')
    ip = int(np.nanargmax(cols['DNS_256']))
    print(f'DNS peak {cols["DNS_256"][ip]:.3f} at t/T0 = {tg[ip]:.2f}')

    out = 'reference/gourianov_fig3b_tgv_re800.csv'
    names = list(cols)
    with open(out, 'w') as f:
        f.write('# Gourianov et al. (2022) arXiv:2106.05782v3, Fig. 3b\n')
        f.write('# 3-D Taylor-Green vortex, Re = 800; DNS is 256^3, 8th-order FD\n')
        f.write('# + RK2 + Chorin projection.  Column values are the energy\n')
        f.write('# dissipation eps(t) normalised by E0/T0, E0 the initial kinetic\n')
        f.write('# energy and T0 = Lbox/u0.  Extracted from the PDF vector paths\n')
        f.write('# by reference/gourianov_digitize.py -- not by pixel analysis.\n')
        f.write('t_over_T0,' + ','.join(names) + '\n')
        for k, g in enumerate(tg):
            f.write(f'{g:.4f},' + ','.join(
                ('' if np.isnan(cols[n][k]) else f'{cols[n][k]:.4f}')
                for n in names) + '\n')
    print('wrote', out)


if __name__ == '__main__':
    main()
