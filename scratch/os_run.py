"""Chan (1996) Fig. 2: Orr-Sommerfeld growth rate at three polynomial orders.

Base flow verified first (os_base.py): with the body force f_x = 2*nu the
parabola is held to max|u - U0| = 0 for 200 steps, so the mesh, the periodicity
on a SINGLE streamwise element, and the forcing weight are all right.

Perturbation: the least-stable Orr-Sommerfeld mode at Re = 7500, alpha = 1,
computed by Chebyshev collocation + scipy.linalg.eig (orr_sommerfeld.py) and
verified against Chan/Streett to 8.9e-10 in phase speed and 1.0e-05 in growth
rate.  phi is interpolated from the Chebyshev grid onto the SEM nodes by
BARYCENTRIC Lagrange -- the stable form; the naive product formula loses digits
badly at N ~ 200 Chebyshev points.

    u' = Re[phi'(y)  e^{i a x}]
    v' = Re[-i a phi e^{i a x}]
    om' = v'_x - u'_y = Re[(a^2 phi - phi'') e^{i a x}]

Total field = base + amp * perturbation, amp = 1e-4 (Chan's value).

Perturbation energy E' = 1/2 int (u-U)^2 + v^2 grows as exp(2 sigma t) with
sigma = alpha*Im(c) = 0.00223497, so ln(E'/E'0) should rise to 0.447 at t = 100
-- which is what Fig. 2's y-axis (0 to 0.50) shows.
"""
import os, sys, time
import numpy as np
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
sys.path.insert(0, SC)
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState, ls_coeffs
import lssem2d.solver as S
from os_base import build, base_state, RE, NU, LX, DT
from orr_sommerfeld import solve as os_solve, cheb

ALPHA = 1.0
AMP = 1.0e-4
TEND = 100.0
NCHEB = 160
SIGMA_REF = 0.00223497          # Chan / Streett
C_REF = 0.24989154


def bary_weights(x):
    """Barycentric weights for an arbitrary node set."""
    n = len(x)
    w = np.ones(n)
    for j in range(n):
        d = x[j] - np.delete(x, j)
        w[j] = 1.0/np.prod(d)
    return w


def bary_interp(xn, fn, wn, xq):
    """Barycentric Lagrange interpolation, the numerically stable form."""
    out = np.empty(len(xq), dtype=fn.dtype)
    for k, x in enumerate(xq):
        d = x - xn
        hit = np.where(np.abs(d) < 1e-14)[0]
        if hit.size:
            out[k] = fn[hit[0]]
        else:
            t = wn/d
            out[k] = np.dot(t, fn)/t.sum()
    return out


def perturbation(m, N, amp=AMP):
    """OS eigenmode on the SEM grid, scaled so max|u'| = amp."""
    c, phi, ych = os_solve(NCHEB, RE, ALPHA)
    D, _ = cheb(NCHEB)
    dphi = D @ phi
    d2phi = D @ dphi
    w = bary_weights(ych)

    n = N+1
    ymid = 0.5*(m.ynod.min() + m.ynod.max())
    P = np.zeros((m.nelem, n, n, 4))
    for e in range(m.nelem):
        yq = m.ynod[e, :] - ymid
        f0 = bary_interp(ych, phi, w, yq)
        f1 = bary_interp(ych, dphi, w, yq)
        f2 = bary_interp(ych, d2phi, w, yq)
        for i in range(n):
            x = m.xnod[e, i]
            ex = np.exp(1j*ALPHA*x)
            P[e, i, :, 0] = np.real(f1*ex)
            P[e, i, :, 1] = np.real(-1j*ALPHA*f0*ex)
            P[e, i, :, 3] = np.real((ALPHA**2*f0 - f2)*ex)
    P *= amp/np.abs(P[..., 0]).max()
    return P, c


def run(N, nsteps=None, verbose=True):
    m = build(N)
    n = N+1
    st = SolverState(m, diff_matrix(N), nu=NU, dt=DT, fac1=1.0)
    _, a_flux, _ = ls_coeffs(st)

    U0 = base_state(m, N)
    P, c = perturbation(m, N)
    U = U0 + P
    f = np.zeros_like(U0)
    f[..., 0] = a_flux*2.0*NU

    def epert(Uf):
        du = Uf[..., 0] - U0[..., 0]
        dv = Uf[..., 1] - U0[..., 1]
        return 0.5*float(np.sum((du*du + dv*dv)*m.wq))

    hist = [U]
    pin = (0, n//2, n//2)
    nsteps = nsteps or int(round(TEND/DT))
    ts, Es = [0.0], [epert(U)]
    t0 = time.perf_counter()
    for s in range(nsteps):
        U = S.step_bdf(st, hist, time=s*DT, max_newton=2,
                       newton_tol=0.0, newton_factor=0.0, f_known=f,
                       pin_p=pin, cgsfac=0.01, cg_tol=1e-14,
                       cg_max_iter=2000, line_search=False)
        if not np.all(np.isfinite(U)):
            return dict(N=N, status='NaN')
        ts.append((s+1)*DT); Es.append(epert(U))
        if verbose and (s+1) % 200 == 0:
            print(f"    N={N}  step {s+1:5d}  t={(s+1)*DT:7.1f}  "
                  f"ln(E/E0) = {np.log(Es[-1]/Es[0]):+.5f}  "
                  f"{time.perf_counter()-t0:6.0f}s", flush=True)
    ts, Es = np.array(ts), np.array(Es)
    k = len(ts)//4                       # skip the initial transient
    sig = 0.5*np.polyfit(ts[k:], np.log(Es[k:]/Es[0]), 1)[0]
    return dict(N=N, status='ok', sigma=sig, ts=ts, Es=Es,
                err=abs(sig-SIGMA_REF)/SIGMA_REF,
                wall=time.perf_counter()-t0, c=c)


if __name__ == '__main__':
    c, _, _ = os_solve(NCHEB, RE, ALPHA)
    print(f"Chan (1996) Fig. 2 -- Orr-Sommerfeld, Re = {RE}, alpha = {ALPHA}, dt = {DT}")
    print(f"  mesh 1 x 3 elements (y: 0.6/0.8/0.6), periodic in x, amp = {AMP:g}")
    print(f"  eigensolver: c = {c.real:.9f} + {c.imag:.9f}i")
    print(f"  Chan/Streett: growth {SIGMA_REF}, phase speed {C_REF}")
    print(f"  expected ln(E/E0) at t = {TEND:g}:  {2*SIGMA_REF*TEND:.4f}\n")
    print(f"{'N':>4}{'status':>9}{'sigma':>14}{'err vs Chan':>13}"
          f"{'ln(E/E0) at T':>15}{'wall':>8}")
    out = {}
    for N in (8, 10, 14):
        r = run(N)
        out[N] = r
        if r['status'] != 'ok':
            print(f"{N:>4}{r['status']:>9}")
            continue
        print(f"{N:>4}{'ok':>9}{r['sigma']:>14.8f}{r['err']:>12.3%}"
              f"{np.log(r['Es'][-1]/r['Es'][0]):>15.5f}{r['wall']:>7.0f}s")
        sys.stdout.flush()
    np.savez(f'{SC}/os_traces.npz',
             **{f'N{N}_{a}': v for N, r in out.items() if r['status'] == 'ok'
                for a, v in (('t', r['ts']), ('e', r['Es']/r['Es'][0]),
                             ('s', np.array([r['sigma']])))})
    print('\nsaved os_traces.npz')
