"""Instrument the Newton sub-iteration: does it contract WITHIN a time step?

Two conjectures for why nsub=5 destabilises where nsub=1 does not have now been
refuted (explicit-drift; the CG absolute tolerance floor).  Stop guessing and
measure: print |dU| for every sub-iteration of every time step, plus the true
least-squares residual J of the implicit system being solved at that level.

If |dU| contracts within a step, the implicit solve is healthy and the fault is
in the time integration.  If it grows, the sub-iteration itself diverges and the
bug is in the Newton step.
"""
import os, sys, time
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
sys.path.insert(0, SC)
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
from fgrid import load
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState, apply_L
import lssem2d.solver as S
from lssem2d import precond as P

RE, DT, WMOM, WMASS = 389.0, 0.1, 1.0, 1.0
NSTEP = int(os.environ.get('NSTEP', '8'))
NSUB = int(os.environ.get('NSUB', '5'))
_p = S.pcg_solve
_ns = S.newton_step


def build():
    m, _, _ = load('/Users/danielchan/Dropbox/F90_SEM/pmg_clean/cnos_short_grid.dat')
    for e in range(m.nelem):
        if m.bc[e, 1] == 4:
            m.bc[e, 1] = 0
    return m


def devc_ic(m):
    n = m.N+1
    U0 = np.zeros((m.nelem, n, n, 4)); Lb = 1.0
    u_dev = lambda y: 3.0*y*(1.0-y)
    du_dev = lambda y: 3.0 - 6.0*y

    def u_step(y):
        if y <= 0.5:
            return 0.0
        eta = 2.0*y - 1.0
        return 6.0*eta*(1.0-eta)

    def du_step(y):
        if y <= 0.5:
            return 0.0
        eta = 2.0*y - 1.0
        return 12.0*(1.0 - 2.0*eta)

    def G(y):
        I = 1.5*y*y - y**3
        if y > 0.5:
            eta = 2.0*y - 1.0
            I -= 0.5*(3.0*eta*eta - 2.0*eta**3)
        return I

    for e in range(m.nelem):
        for i in range(n):
            xx = m.xnod[e, i]
            if xx <= 0.0:
                t, sp, spp = 0.0, 0.0, 0.0
            else:
                t = min(xx/Lb, 1.0)
                sp = (6.0*t - 6.0*t*t)/Lb
                spp = (6.0 - 12.0*t)/Lb**2
                if xx >= Lb:
                    sp = spp = 0.0
            sv = 3.0*t*t - 2.0*t**3
            for j in range(n):
                yy = m.ynod[e, j]
                if xx < 0.0:
                    eta = (yy-0.5)/0.5
                    U0[e, i, j, 0] = 6.0*eta*(1.0-eta)
                    U0[e, i, j, 3] = -12.0*(1.0-2.0*eta)
                else:
                    U0[e, i, j, 0] = (1.0-sv)*u_step(yy) + sv*u_dev(yy)
                    U0[e, i, j, 1] = -sp*G(yy)
                    U0[e, i, j, 3] = (-spp*G(yy)
                                      - ((1.0-sv)*du_step(yy) + sv*du_dev(yy)))
    return U0


m = build(); N = m.N
st = SolverState(m, diff_matrix(N), nu=1.0/RE, dt=DT, fac1=1.0,
                 w_mom=WMOM, w_mass=WMASS)
inlet = lambda x, y, t: 6.0*((y-0.5)/0.5)*(1.0-(y-0.5)/0.5)

log = []


def implicit_J(state, U, su_history):
    """||L(U) - su_history||^2 -- the residual of THIS time level's system."""
    f = np.ascontiguousarray(U[..., 0]/2.0); g = np.ascontiguousarray(U[..., 1]/2.0)
    state.update_linearisation(f, g)
    r = apply_L(state, U, f, g) - su_history
    return float(np.sum(r*r/state.mesh.wq[..., None]))


def probe_newton(state, U, su_history, M_inv, mw, **kw):
    Jb = implicit_J(state, U, su_history)
    Un, dU, it = _ns(state, U, su_history, M_inv, mw, **kw)
    Ja = implicit_J(state, Un, su_history)
    log.append((float(np.max(np.abs(dU))), Jb, Ja,
                float(np.abs(Un[..., 0]).max()), it))
    return Un, dU, it


def pcg(state, b, fu, fv, M, mw, pin_p=False, max_iter=5000, tol=None,
        cgsfac=None, precond=None, **kwargs):
    pre = P.make('pmg2', state, fu, fv, M, pin_p,
                 pc=max(2, N//2), deg=4, coarse_deg=10)
    return _p(state, b, fu, fv, M, mw, pin_p=pin_p, max_iter=300000,
              tol=1e-10, cgsfac=1e-3, precond=pre)


S.pcg_solve = pcg
S.newton_step = probe_newton

print(f"nsub = {NSUB},  dt = {DT},  w_mom = w_mass = 1.0,  cold developed IC")
print("Within each time step the implicit residual J MUST fall if Newton is healthy.\n")
print(f"{'step':>5}{'sub':>5}{'|dU|':>12}{'J before':>13}{'J after':>13}"
      f"{'ratio':>9}{'max|u|':>9}{'CG':>8}")

U = devc_ic(m); hist = [U]
try:
    for s in range(NSTEP):
        log.clear()
        U = S.step_bdf(st, hist, time=s*DT, max_newton=NSUB, newton_tol=1e-12,
                       newton_factor=0.0, custom_inlet=inlet, pin_p=None,
                       cgsfac=1e-3, cg_max_iter=300000, verbose=False)
        for k, (d, jb, ja, um, it) in enumerate(log):
            ratio = ja/jb if jb > 0 else float('nan')
            flag = '' if ratio < 1.0 else '   <-- GREW'
            print(f"{s:>5}{k:>5}{d:>12.3e}{jb:>13.4e}{ja:>13.4e}"
                  f"{ratio:>9.3f}{um:>9.3f}{it:>8}{flag}")
        sys.stdout.flush()
        if not np.all(np.isfinite(U)) or np.abs(U[..., 0]).max() > 50:
            print("  ** blew up **"); break
finally:
    S.pcg_solve = _p
    S.newton_step = _ns
