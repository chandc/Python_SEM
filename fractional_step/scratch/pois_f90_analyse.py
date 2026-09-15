"""Fortran LSSEM Poiseuille dt sweep: stability, divergence, profile, pressure drop.

    uv run --quiet python scratch/pois_f90_analyse.py

Reads the solution files written by SEM_2D_BFS_FREEOUT on the plane channel
[0,4] x [0,1] (grid from scratch/mesh_poiseuille_f90.py), free outflow, Re = 100.

WHY THIS IS AN IMPLEMENTATION CHECK.  The inlet imposes fully-developed plane
Poiseuille, so the exact solution holds throughout the domain and is representable
exactly in the discrete space for any order >= 2:

    u = 6y(1-y)      v = 0      om = dv/dx - du/dy = 12y - 6
    dp/dx = nu * u'' = -12/Re    =>    dp = 12 L / Re  across the domain

A correct implementation must return all four to round-off, for every dt.  The
legacy weighting makes the momentum row  fac1*u + dt*N(u)  against constraint
rows of weight 1, i.e. a_mass = fac1 = 1.5 fixed and a_flux = dt, so dt is the
momentum weight -- see PRECONDITIONER_AND_DT_STUDY.md sec 5.  Any dt-dependence
found here is that weighting, not temporal error: the exact solution is a
steady state, so a converged run has no temporal error left to measure.
"""
import os, sys, glob
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
from fsol import load_solution
from lssem2d.lgl import diff_matrix, lgl_weights

FDIR = '/Users/danielchan/Dropbox/F90_SEM/pmg_clean/poiseuille_dt'
RE = 100.0
NU = 1.0/RE
L = 4.0
DTS = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]


def exact_u(y):
    return 6.0*y*(1.0-y)


def exact_om(y):
    return 12.0*y - 6.0


def analyse(path, xcut=None):
    """xcut: if given, exclude elements whose x range starts at or beyond it.
    The free outflow corrupts the solution in a boundary layer at x = L; cutting
    the last element column separates IMPLEMENTATION accuracy from that."""
    d = load_solution(path)
    U, xn, yn, N = d['U'], d['xnod'], d['ynod'], d['N']
    n = N+1
    D = diff_matrix(N); w = lgl_weights(N)
    nelem = U.shape[0]
    # element sizes from the node coordinates
    hx = xn[:, -1]-xn[:, 0]
    hy = yn[:, -1]-yn[:, 0]

    du = dv = dp = 0.0
    dvmax = 0.0
    eu = eom = 0.0
    div2 = 0.0; divmax = 0.0; area = 0.0
    for e in range(nelem):
        if xcut is not None and xn[e, 0] >= xcut - 1e-9:
            continue
        u = U[e, :, :, 0]; v = U[e, :, :, 1]; om = U[e, :, :, 3]
        ux = (D @ u)*(2.0/hx[e]); vy = (u*0 + (v @ D.T))*(2.0/hy[e])
        dvg = ux + vy
        jac = 0.25*hx[e]*hy[e]
        wq = np.outer(w, w)*jac
        div2 += np.sum(dvg**2 * wq); divmax = max(divmax, np.abs(dvg).max())
        area += wq.sum()
        ue = exact_u(yn[e][None, :]) * np.ones((n, 1))
        ome = exact_om(yn[e][None, :]) * np.ones((n, 1))
        eu += np.sum((u-ue)**2 * wq)
        eom += np.sum((om-ome)**2 * wq)
        dvmax = max(dvmax, np.abs(v).max())
    # pressure drop: area-weighted mean p on the inlet and outlet planes
    xmin, xmax = xn.min(), xn.max()

    def plane_mean(xtarget, i):
        num = den = 0.0
        for e in range(nelem):
            if abs(xn[e, i]-xtarget) < 1e-9:
                num += np.sum(U[e, i, :, 2]*w)*(0.5*hy[e])
                den += hy[e]
        return num/den

    xref = xmax if xcut is None else xcut
    iref = n-1 if xcut is None else 0
    p_in = plane_mean(xmin, 0); p_out = plane_mean(xref, iref)
    Lref = xref - xmin
    return dict(maxu=np.abs(U[..., 0]).max(), maxv=dvmax,
                rms_div=np.sqrt(div2/area), max_div=divmax,
                l2_u=np.sqrt(eu/area), l2_om=np.sqrt(eom/area),
                dp=p_in-p_out, Lref=Lref, time=d['time'], re=d['re'])


XCUT = 3.0          # exclude the last element column (x in [3,4])
DP_EXACT = 12.0*L/RE
DP_EXACT_CUT = 12.0*XCUT/RE
print(f'Fortran LSSEM, legacy weighting, FREE outflow.  Channel [0,{L:g}]x[0,1], '
      f'Re = {RE:g} (nu = {NU:g}), 4x2 elements, N = 10.')
print(f'Exact: u = 6y(1-y), v = 0, om = 12y-6, dp = 12L/Re = {DP_EXACT:.6f}\n')
import sys as _s
PREFIX = _s.argv[1] if len(_s.argv) > 1 else 'sol'
BCNAME = {'sol': 'FREE outflow (0 conditions), tol=1e-12',
          'sol6': 'FREE outflow (0 conditions), tol=1e-6',
          'solP': 'p = 0 on the outlet plane (1 condition), tol=1e-12'}.get(PREFIX, PREFIX)
print(f'>>> {BCNAME}\n')
for title, xcut, dpx in (('WHOLE DOMAIN  (includes the outflow plane)', None, DP_EXACT),
                         (f'INTERIOR ONLY  (x < {XCUT:g}: last element column excluded)',
                          XCUT, DP_EXACT_CUT)):
    print(f'--- {title} ---')
    print(f"{'dt':>6}{'a_flux':>8}{'max|u|':>10}{'max|v|':>11}{'rms div':>11}"
          f"{'max|div|':>11}{'L2 err u':>11}{'L2 err om':>11}{'dp':>10}{'dp err':>10}")
    for dt in DTS:
        f = f'{FDIR}/{PREFIX}_dt{dt:g}.dat'
        if not os.path.exists(f):
            print(f'{dt:>6g}   (no solution written -- diverged or still running)')
            continue
        try:
            r = analyse(f, xcut)
        except Exception as ex:
            print(f'{dt:>6g}   FAILED: {type(ex).__name__}'); continue
        if not np.isfinite(r['maxu']):
            print(f'{dt:>6g}{dt:>8g}      NaN  -- diverged'); continue
        print(f"{dt:>6g}{dt:>8g}{r['maxu']:>10.4f}{r['maxv']:>11.3e}"
              f"{r['rms_div']:>11.3e}{r['max_div']:>11.3e}{r['l2_u']:>11.3e}"
              f"{r['l2_om']:>11.3e}{r['dp']:>10.5f}"
              f"{(r['dp']/dpx-1)*100:>9.2f}%")
    print(f"{'exact':>6}{'':>8}{1.5:>10.4f}{0.0:>11.3e}{0.0:>11.3e}{0.0:>11.3e}"
          f"{0.0:>11.3e}{0.0:>11.3e}{dpx:>10.5f}{0.0:>9.2f}%")
    print()
