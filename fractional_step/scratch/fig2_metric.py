"""Chan CTR Fig. 2's ACTUAL metric: u(y) and omega(y) at 4h and 5h, short vs long.

Everything reported in this session so far has been x_r/h, max|u|, exit pressure
and J.  None of those is what Chan plotted.  Fig. 2 compares the LONG and
TRUNCATED solutions profile-by-profile, and its claim is that the free outflow
"tolerates truncation without distorting the interior".

Reference points:
  * Chan's stated long-vs-short agreement: "less than 10 percent" (digitized 5.0%)
  * the July Fortran reproduction: interior preserved to ~1% through 4h,
    defect confined to the outlet plane  (project-uniflo-provenance)

h = 0.5 (inlet height), so 4h = x 2.0 and 5h = x 2.5.  5h IS the truncated
outlet plane, so it is the worst station by construction; 4h is the interior
test.  Velocity scale is the inlet CENTRELINE velocity U0 = 1.5.

This scores states we already have -- no solving.  A state that scores badly at
4h has distorted the interior, which is exactly what Fig. 2 says should not
happen.
"""
import os, sys
import numpy as np
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
sys.path.insert(0, SC)
from lssem2d.lgl import lgl_nodes, diff_matrix
from upper_wall import read_fortran

U0 = 1.5
H = 0.5
STATIONS = [(2.0, '4h'), (2.5, '5h')]

LONG_PY = 'bfswm_0.1.npz'
LONG_F = '/Users/danielchan/Dropbox/F90_SEM/pmg_clean/run_chan389_long/chan389_long.dat'

SHORT = [
    ('short: own converged state (the artifact)', 'bfsnp2_off_nopin.npz'),
    ('short: seeded + line search',               'bfsint3_spectralIC_ls.npz'),
    ('short: seeded + dtau=1 (kappa 0.1)',        'dtauA_dtau1.npz'),
    ('short: seeded + dtau=0.1 (kappa 1)',        'dtauA_dtau0p1.npz'),
    ('short: the interpolated target itself',     'bfsint3_IC.npz'),
    ('short: legacy dt=0.5 time-stepping (the IC)','dt_dt0p5_devc_short_state.npz'),
    ('OWN IC, steady, no dtau',                   'own_none.npz'),
    ('OWN IC, steady, kappa 0.03',                'own_k0p03.npz'),
    ('OWN IC, steady, kappa 0.1',                 'own_k0p1.npz'),
    ('OWN IC, steady, kappa 0.3',                 'own_k0p3.npz'),
    ('COLD developed profile (the IC itself)',    'cold_IC.npz'),
    ('COLD, steady, no dtau',                     'cold_none.npz'),
    ('COLD, steady, kappa 1 (capped)',            'cold_k1.npz'),
]


def bary_w(z):
    n = len(z); w = np.ones(n)
    for j in range(n):
        for k in range(n):
            if k != j:
                w[j] /= (z[j]-z[k])
    return w


def column(U, xn, yn, xq, comp):
    """Spectrally evaluate component `comp` along the line x = xq."""
    n = U.shape[1]
    z = lgl_nodes(n-1); wb = bary_w(z)

    def lag(x):
        d = x - z
        hit = np.where(np.abs(d) < 1e-13)[0]
        if hit.size:
            r = np.zeros(n); r[hit[0]] = 1.0; return r
        r = wb/d
        return r/r.sum()

    ys, vs = [], []
    for e in range(U.shape[0]):
        x0, x1 = xn[e, 0], xn[e, -1]
        if not (x0-1e-9 <= xq <= x1+1e-9):
            continue
        lx = lag(2.0*(xq-x0)/(x1-x0) - 1.0)
        for j in range(n):
            ys.append(yn[e, j]); vs.append(float(lx @ U[e, :, j, comp]))
    ys, vs = np.array(ys), np.array(vs)
    o = np.argsort(ys); ys, vs = ys[o], vs[o]
    k = np.concatenate(([True], np.diff(ys) > 1e-12))
    return ys[k], vs[k]


def load(f):
    d = np.load(f'{SC}/{f}')
    return d['U'], d['xnod'], d['ynod']


UL, xL, yL = load(LONG_PY)
_, XPf, YPf, Uf = read_fortran(LONG_F)

print("Chan CTR Fig. 2 metric -- rms(short - long) / U0,  U0 = 1.5")
print("h = 0.5, so 4h = x 2.0 (interior) and 5h = x 2.5 (the truncated outlet plane)")
print("targets: Chan long-vs-short 5.0% digitized ('less than 10 percent');")
print("         July Fortran reproduction ~1% through 4h\n")

# sanity: our Python long vs the Fortran long, at the same stations
print("control -- Python long vs FORTRAN long (should be small):")
for xq, lab in STATIONS:
    yp, up = column(UL, xL, yL, xq, 0)
    yf, uf = column(Uf, XPf, YPf, xq, 0)
    _, op = column(UL, xL, yL, xq, 3)
    _, of = column(Uf, XPf, YPf, xq, 3)
    du = np.sqrt(np.mean((up - np.interp(yp, yf, uf))**2))/U0
    do = np.sqrt(np.mean((op - np.interp(yp, yf, of))**2))/(U0/H)
    print(f"   {lab}:  u {100*du:6.2f}%    omega {100*do:6.2f}%")

print(f"\n{'state':<44}" + "".join(f"{lab+' u':>12}{lab+' om':>12}" for _, lab in STATIONS))
for tag, f in SHORT:
    try:
        US, xS, yS = load(f)
    except FileNotFoundError:
        print(f"{tag:<44}  (missing {f})")
        continue
    cells = []
    for xq, lab in STATIONS:
        ys, us = column(US, xS, yS, xq, 0)
        yl, ul = column(UL, xL, yL, xq, 0)
        _, os_ = column(US, xS, yS, xq, 3)
        _, ol = column(UL, xL, yL, xq, 3)
        du = np.sqrt(np.mean((us - np.interp(ys, yl, ul))**2))/U0
        do = np.sqrt(np.mean((os_ - np.interp(ys, yl, ol))**2))/(U0/H)
        cells.append(f"{100*du:>11.2f}%{100*do:>11.2f}%")
    print(f"{tag:<44}" + "".join(cells))

print("\nomega normalised by U0/h = 3.0.  4h is the interior test; 5h is the outlet")
print("plane itself, where Fig. 2 expects the defect to be confined.")
