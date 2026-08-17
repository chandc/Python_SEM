"""Summary table for the Fortran Poiseuille outflow study: errors + convergence rate.

    uv run --quiet python scratch/pois_f90_summary.py

Convergence rate is measured, not assumed: the driver prints the per-step
residual, so log10(res) is fitted against PHYSICAL time over the decade before
the run's floor.  Reporting decades per unit time (rather than per step) makes
the rate comparable across dt -- a rate per step would just restate dt.

Rate is only meaningful where the run actually converged; NaN/blow-up rows show
the time at which the residual last exceeded 1.0 instead.
"""
import os, re, sys
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
from fsol import load_solution
from lssem2d.lgl import diff_matrix, lgl_weights

FD = '/Users/danielchan/Dropbox/F90_SEM/pmg_clean/poiseuille_dt'
RE, L, NU = 100.0, 4.0, 0.01
DP_EXACT = 12.0*L/RE
XCUT = 3.0


def history(logpath):
    """(t, res) from the driver log; res may be NaN or huge."""
    if not os.path.exists(logpath):
        return np.array([]), np.array([])
    txt = open(logpath, errors='ignore').read()
    ts, rs = [], []
    blocks = re.findall(r'At time=\s*([0-9.E+-]+)\s*\n\s*\d+\s+([0-9.EN+aInf-]+)', txt)
    for t, r in blocks:
        try:
            ts.append(float(t))
            rs.append(float(r.replace('NaN', 'nan')))
        except ValueError:
            ts.append(float(t)); rs.append(np.nan)
    return np.array(ts), np.array(rs)


def t_to(ts, rs, level):
    """physical time at which the residual first drops below `level` and stays
    there.  A fitted decay slope was tried first and is useless: the residual
    reaches its floor early and then sits flat, so the fit returns ~0 regardless
    of how fast it got there.  Time-to-level is what actually distinguishes the
    runs."""
    ok = np.isfinite(rs)
    if ok.sum() < 5:
        return None
    t, r = ts[ok], rs[ok]
    below = r < level
    if not below.any() or not below[-1]:
        return None
    # last index where it was still above -> it settles just after
    above = np.where(~below)[0]
    i = above[-1]+1 if above.size else 0
    return t[i] if i < len(t) else None


def errors(path, xcut=None):
    d = load_solution(path)
    U, xn, yn, N = d['U'], d['xnod'], d['ynod'], d['N']
    n = N+1; D = diff_matrix(N); w = lgl_weights(N)
    hx = xn[:, -1]-xn[:, 0]; hy = yn[:, -1]-yn[:, 0]
    eu = eom = div2 = area = 0.0; dvmax = 0.0
    for e in range(U.shape[0]):
        if xcut is not None and xn[e, 0] >= xcut-1e-9:
            continue
        u = U[e, :, :, 0]; v = U[e, :, :, 1]; om = U[e, :, :, 3]
        dvg = (D @ u)*(2.0/hx[e]) + (v @ D.T)*(2.0/hy[e])
        wq = np.outer(w, w)*0.25*hx[e]*hy[e]
        div2 += np.sum(dvg**2*wq); area += wq.sum()
        ye = yn[e][None, :]*np.ones((n, 1))
        eu += np.sum((u-6.0*ye*(1.0-ye))**2*wq)
        eom += np.sum((om-(12.0*ye-6.0))**2*wq)
        dvmax = max(dvmax, np.abs(v).max())

    def pm(xt, i):
        num = den = 0.0
        for e in range(U.shape[0]):
            if abs(xn[e, i]-xt) < 1e-9:
                num += np.sum(U[e, i, :, 2]*w)*(0.5*hy[e]); den += hy[e]
        return num/den
    xref = xn.max() if xcut is None else xcut
    iref = n-1 if xcut is None else 0
    dp = pm(xn.min(), 0)-pm(xref, iref)
    dpx = 12.0*(xref-xn.min())/RE
    return dict(maxu=np.abs(U[..., 0]).max(), maxv=dvmax,
                rms_div=np.sqrt(div2/area), l2_u=np.sqrt(eu/area),
                l2_om=np.sqrt(eom/area), dp=dp, dp_err=(dp/dpx-1)*100)


# p = 0 on the outlet plane only.  The free-outflow families are dropped: no free
# run reproduced the exact solution at any dt or tolerance, so they contribute
# nothing to an accuracy table -- see FORTRAN_POISEUILLE_OUTFLOW.md sec 3.
# The main sweep plus the dt-band probe, all p = 0.
CASES = [('1e-12', 0.01, 'logP_dt0.01', 'solP_dt0.01'),
         ('1e-12', 0.05, 'logP_dt0.05', 'solP_dt0.05'),
         ('1e-12', 0.10, 'logP_dt0.1',  'solP_dt0.1'),
         ('1e-12', 0.50, 'logP_dt0.5',  'solP_dt0.5'),
         ('1e-12', 1.00, 'logP_dt1',    'solP_dt1'),
         ('1e-12', 1.50, 'logQ_dt1.5',  'solQ_dt1.5'),
         ('1e-12', 2.00, 'logP_dt2',    'solP_dt2'),
         ('1e-10', 2.00, 'logQ_dt2t10', 'solQ_dt2t10'),
         ('1e-6',  2.00, 'logQ_dt2t6',  'solQ_dt2t6'),
         ('1e-12', 2.50, 'logQ_dt2.5',  'solQ_dt2.5'),
         ('1e-12', 3.00, 'logQ_dt3',    'solQ_dt3'),
         ('1e-12', 5.00, 'logP_dt5',    'solP_dt5')]

print('Fortran LSSEM, plane Poiseuille, Re = 100, channel [0,4]x[0,1], 4x2 elem, N = 10.')
print(f'Legacy weighting (a_mass = 1.5 fixed, a_flux = dt).  ntime*dt = 200 for every dt.')
print(f'Exact: u = 6y(1-y), v = 0, om = 12y-6, dp = 12L/Re = {DP_EXACT:.5f}\n')
hdr = (f"{'tol':>8}{'dt':>6}{'a_flux':>7}{'outcome':>11}{'res':>10}"
       f"{'t<1e-6':>8}{'t<1e-10':>8}{'max|u|':>9}{'max|v|':>10}{'rms div':>10}"
       f"{'L2 u':>10}{'L2 om':>10}{'dp err':>9}")
print('OUTLET: p = 0 on the whole outlet plane (SEM_2D_BFS_POUT).\n')
print(hdr); print('-'*len(hdr))
if True:
    for label, dtv, lp, sp in CASES:
        dt = f'{dtv:g}'
        ts, rs = history(f'{FD}/{lp}.txt')
        solf = f'{FD}/{sp}.dat'
        if ts.size == 0:
            print(f'{label:>8}{dt:>6}{dtv:>7g}{"not run":>11}'); continue
        # 'diverged' must be judged on the FINAL residual, not the transient max:
        # a healthy run can spike early and still land on the exact solution.
        final = rs[-1] if rs.size else np.nan
        bad = (not np.isfinite(final)) or final > 1.0
        t6 = t_to(ts, rs, 1e-6); t10 = t_to(ts, rs, 1e-10)
        if os.path.exists(solf):
            try:
                e = errors(solf)
            except Exception:
                e = None
        else:
            e = None
        if e is None or not np.isfinite(e['maxu']):
            out = 'NaN' if (~np.isfinite(rs)).any() else 'blew up'
            print(f'{label:>8}{dt:>6}{dtv:>7g}{out:>11}'
                  f'{"--":>10}{"--":>8}{"--":>8}{"--":>9}{"--":>10}{"--":>10}'
                  f'{"--":>10}{"--":>10}{"--":>9}')
            continue
        wrong = abs(e['dp_err']) > 5.0
        out = 'diverged' if bad else ('BAD STATE' if wrong else 'exact')
        print(f"{label:>8}{dt:>6}{dtv:>7g}{out:>11}{final:>10.1e}"
              f"{(f'{t6:.1f}' if t6 is not None else '--'):>8}"
              f"{(f'{t10:.1f}' if t10 is not None else '--'):>8}"
              f"{e['maxu']:>9.4f}{e['maxv']:>10.2e}"
              f"{e['rms_div']:>10.2e}{e['l2_u']:>10.2e}{e['l2_om']:>10.2e}"
              f"{e['dp_err']:>8.2f}%")
print(f"{'exact':>8}{'':>6}{'':>7}{'':>11}{'':>10}{'':>8}{'':>8}{1.5:>9.4f}"
      f"{0.0:>10.2e}{0.0:>10.2e}{0.0:>10.2e}{0.0:>10.2e}{0.0:>8.2f}%")
print('\nt<1e-6 / t<1e-10 = PHYSICAL time at which the per-step residual settles'
      '\n   below that level -- the convergence measure.  (A fitted decay slope was'
      '\n   tried and is useless: the residual reaches its floor early then sits'
      '\n   flat, so the fit returns ~0 however fast it got there.)'
      '\n"BAD STATE" = converged, but to the WRONG steady state.'
      '\nErrors are whole-domain, outflow plane included.')
