"""Every cylinder Re=100 run, measured the same way, in one table.

ONE METHOD FOR ALL ROWS.  The numbers quoted through the study were produced at
different times with different window choices, and a table assembled from those
is not a comparison -- it is a collection.  Here every run is reduced by the
same code: the saturated portion of the record, truncated to a whole number of
shedding periods, St by both estimators, C_D and C_L over the same window.

WHOLE CYCLES MATTER for the means.  A partial cycle biases mean C_D by roughly
the fluctuation amplitude times the fraction left over -- order 0.3 %, which is
the size of the domain effects this study is trying to resolve.

THE HARMONIC RATIO IS THE VALIDITY CHECK.  C_D is forced at twice the shedding
frequency, so its spectrum must peak at 2*St.  That is physics, not a property
of either estimator; a row that misses 2.000 is not a limit cycle and its other
columns should not be read.
"""
import os
import sys

_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R)
sys.path.insert(0, os.path.join(_R, 'scratch'))
os.chdir(_R)

import numpy as np
from curvi_cyl_st_fft import st_fft, st_cross, saturated

# (label, run dir, log file) -- log supplies the configuration line
# EVERY ROW STATES BOTH DOMAIN PARAMETERS.  Labelling a run by the parameter
# that makes it different from its neighbour hides that the same run is the
# BASELINE for a different comparison: _cyl_H20_N6dt0.1ac is simultaneously the
# Xu = 10 reference for the upstream study and the H = 20 reference for the
# lateral one, and a label naming only one of them invites the reader to go
# looking for a case that is already in the table.
RUNS = [
    ('N8  dt0.200 +AC  Xu10 H20', '_cyl_dt0.2',          'logs_cyl_dt0.2.log'),
    ('N8  dt0.100 noAC Xu10 H20', '_cyl_re100',          'logs_cyl_re100.log'),
    ('N8  dt0.100 +AC  Xu10 H20', '_cyl_dt0.1ac',        'logs_cyl_dt0.1ac.log'),
    ('N8  dt0.050 +AC  Xu10 H20', '_cyl_dt0.05ac',       'logs_cyl_dt0.05ac.log'),
    ('N8  dt0.025 +AC  Xu10 H20', '_cyl_dt0.025ac',      'logs_cyl_dt0.025ac.log'),
    ('N8  dt0.0125+AC  Xu10 H20', '_cyl_dt0.0125ac',     'logs_cyl_dt0.0125ac.log'),
    ('N6  dt0.050 +AC  Xu10 H20', '_cyl_N6dt0.05ac',     'logs_cyl_N6dt0.05ac.log'),
    ('N10 dt0.050 +AC  Xu10 H20', '_cyl_N10dt0.05ac',    'logs_cyl_N10dt0.05ac.log'),
    ('N6  dt0.100 +AC  Xu10 H20', '_cyl_H20_N6dt0.1ac',  'logs_cyl_H20_N6dt0.1.log'),
    ('N6  dt0.100 +AC  Xu10 H40', '_cyl_H40_N6dt0.1ac',  'logs_cyl_H40_N6dt0.1.log'),
    ('N6  dt0.100 +AC  Xu20 H20', '_cyl_Xu20_N6dt0.1ac', 'logs_cyl_Xu20_N6dt0.1.log'),
    ('N8  dt0.100 noAC Xu10 H20 n=5', '_cyl_noac_nw5_tol6',
     'logs_cyl_noac_nw5_tol6.log'),
]



def reduce_run(d):
    for name in ('final.npz', 'chk_latest.npz'):
        f = os.path.join('scratch', d, name)
        if os.path.exists(f):
            break
    else:
        return None
    h = np.asarray(np.load(f, allow_pickle=True)['hist'], float)
    if len(h) < 128:
        return None
    t, cd, cl = h[:, 0], h[:, 1], h[:, 2]
    i0 = saturated(t, cl)
    t, cd, cl = t[i0:], cd[i0:], cl[i0:]
    if len(t) < 128:
        return None
    sf, _, _ = st_fft(t, cl)
    sc, sd, n = st_cross(t, cl)
    fd, _, _ = st_fft(t, cd)
    npd = int((t[-1] - t[0])*sf)
    keep = t <= t[0] + npd/sf + 1e-12
    return dict(t0=t[0], t1=t[-1], n=n, sf=sf, sc=sc, sd=sd, h=fd/sf,
                cd=cd[keep].mean(), cdamp=0.5*(cd[keep].max() - cd[keep].min()),
                clrms=cl[keep].std(), clamp=np.abs(cl[keep]).max())


def main():
    print(f'{"configuration":30s} {"window":>13s} {"cyc":>4s} {"St(FFT)":>8s} '
          f'{"St(cross)":>17s} {"2f/f":>6s} {"mean C_D":>9s} {"C_D amp":>8s} '
          f'{"C_L rms":>8s} {"C_L amp":>8s}')
    print('-'*128)
    for lab, d, _ in RUNS:
        r = reduce_run(d)
        if r is None:
            print(f'{lab:30s} {"(too short / absent)":>13s}')
            continue
        flag = '' if abs(r['h'] - 2.0) < 0.02 else '   <-- NOT a limit cycle'
        print(f'{lab:30s} {r["t0"]:6.0f}-{r["t1"]:6.0f} {r["n"]:4d} {r["sf"]:8.4f} '
              f'{r["sc"]:9.4f} +-{r["sd"]:.4f} {r["h"]:6.3f} {r["cd"]:9.4f} '
              f'{r["cdamp"]:8.4f} {r["clrms"]:8.4f} {r["clamp"]:8.4f}{flag}')
    print('-'*128)
    print('literature (Qu et al. 2013 Table 1, n=8, large domain):'
          '      St 0.1646 +- 0.0007   C_D 1.3238 +- 0.0063   C_L rms ~0.228')
    print('Behr et al. (1995) M320, two formulations:'
          '                  St 0.1624 / 0.1661     C_D 1.370 / 1.389   '
          'C_L amp 0.371 / 0.366')


if __name__ == '__main__':
    main()
