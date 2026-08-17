"""Admissible BC set 4 -- (p, omega) -- at the outlet, with the bc=4 bug fixed.

A least-squares method for this first-order system is well posed only if the
functional is norm-equivalent, and the ADN complementing condition wants TWO
scalar conditions per boundary point.  Free outflow supplies 0.  Of the four
classical admissible pairs for 2D velocity-vorticity-pressure --

    (u.n, u.t)   (u.n, omega)   (u.t, p)   (p, omega)

-- only the last is natural at an outflow, and it is the one never tested.

FIRST, A BUG.  bc = 4 does NOT enforce p = 0.  The two BC implementations
disagree: `bc.apply_mask` has a bc==4 branch that freezes the pressure DOF,
`SolverState.get_global_mask` does not -- and it is get_global_mask that builds
`b = -c_gs * mask_global` in newton_step, i.e. the one that decides which DOFs
the update may touch.  So apply_bc writes p = 0 at the top of each Newton
iteration and the update moves it straight off.  Measured: max|p| on the outlet
plane = 4.87e-01 after 60 steps, against the 0 the BC claims.

    get_global_mask at the outlet, p component : 1.0  (update NOT frozen)
    bc.apply_mask   at the outlet, p component : 0.0  (frozen)

Consequence: variant B in OUTFLOW_BC_STUDY.md never imposed anything, so
"it is the velocity, not the pressure" is unsupported -- B matched free outflow
because for pressure it WAS free outflow.  No other script is affected: every
one that builds bc_E = 4 overrides it to 0 first, so the bug is latent.

Fixed here by editing the cached mask, the same way the omega constraint is
applied.  Four variants, dt from 1 down to 0.05:

    free    nothing at the outlet (baseline), p pinned at the inlet
    P       p = 0 across the outlet plane, ENFORCED, inlet pin removed
    Z       d(omega)/dx = 0, p free (pinned at inlet)
    P+Z     both -- admissible set 4

The question set 4 is meant to answer: Z alone supplies roughly one condition
and fails below dt = 0.25 (|dU| = 359 at dt = 0.1).  If that floor is the
missing SECOND condition, P+Z should remove it.
"""
import os, sys, time
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
import lssem2d
lssem2d.set_backend('numpy')
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix, lgl_weights
from lssem2d.lssem import SolverState
import lssem2d.solver as S
import lssem2d.bc as BC

N, EX, EY = 8, 10, 2
NU = 0.01
ue = lambda y: 6.0*y*(1.0-y)
D = diff_matrix(N)
wq = lgl_weights(N)
OB = BC.apply_bc


def run(dt, pressure, omega, wall=900.0, tmax=600.0):
    m = build_channel(10., 1., EX, EY, N, bcs=(3, 4, 1, 1))
    n = N+1
    inlet_pin = next((e, 0, 0) for e in range(m.nelem)
                     if m.bc[e, 0] == 3 and m.bc[e, 2] == 1)
    for e in range(m.nelem):
        if m.bc[e, 1] == 4 and not pressure:
            m.bc[e, 1] = 0                       # free outflow for p
    xn = m.xnod; xmax = xn.max(); xmin = xn.min()
    out = [e for e in range(m.nelem) if abs(xn[e, -1]-xmax) < 1e-9]
    pin = False if pressure else inlet_pin       # p set at outlet => no inlet pin

    def bc2(mesh, U, **kw):
        U = OB(mesh, U, **kw)                    # writes p=0 at outlet when bc=4
        if omega:
            for e in out:
                U[e, -1, :, 3] = -(D[-1, :-1] @ U[e, :-1, :, 3])/D[-1, -1]
        return U

    st = SolverState(m, D, nu=NU, dt=dt, fac1=1.0, w_mom=1.0, w_mass=1.0)
    st.get_global_mask(pin_p=pin)                # populate cache, then correct it
    for e in out:
        if pressure:
            st._global_mask[e, -1, :, 2] = 0.0   # THE BUG FIX: freeze p at bc=4
        if omega:
            st._global_mask[e, -1, :, 3] = 0.0
    S.apply_bc = bc2
    t0 = time.perf_counter(); status = 'TCAP'; d = np.nan
    try:
        U = np.zeros((m.nelem, n, n, 4)); h = [U.copy()]
        for s in range(int(tmax/dt)):
            prev = h[0].copy()
            U = S.step_bdf(st, h, time=s*dt, max_newton=1, newton_tol=1e-12,
                           newton_factor=0.0, custom_inlet=lambda x, y, t: ue(y),
                           pin_p=pin, cgsfac=1e-8, cg_tol=1e-10,
                           cg_max_iter=300000)
            if not np.all(np.isfinite(U)):
                return None
            d = float(np.abs(U-prev).max())
            if d < 1e-13:
                status = 'conv'; break
            if time.perf_counter()-t0 > wall:
                status = 'WALLCAP'; break
    finally:
        S.apply_bc = OB

    def pbar(edge):
        tot = a = 0.0
        for e in range(m.nelem):
            xe = xn[e, 0] if edge == 'in' else xn[e, -1]
            ref = xmin if edge == 'in' else xmax
            if abs(xe-ref) < 1e-9:
                i = 0 if edge == 'in' else -1
                tot += np.sum(wq*U[e, i, :, 2])*(m.hy[e]/2); a += m.hy[e]
        return tot/a
    pout = max(np.abs(U[e, -1, :, 2]).max() for e in out)
    prof = float(np.sqrt(np.mean((U[..., 0]-ue(m.ynod)[:, None, :])**2))/1.5)
    return (status, s+1, d, float(pbar('in')-pbar('out')), prof, float(pout),
            time.perf_counter()-t0)


VARIANTS = [('free', False, False), ('P', True, False),
            ('Z', False, True), ('P+Z', True, True)]
HDR = (f"{'dt':>6}{'outlet BC':>11}{'status':>9}{'steps':>7}{'|dU|':>11}"
       f"{'dp':>11}{'dp err':>10}{'prof err':>11}{'max|p_out|':>12}{'wall s':>8}")


def report(dt, lab, r):
    if r is None:
        print(f"{dt:>6g}{lab:>11}   DIVERGED", flush=True); return
    print(f"{dt:>6g}{lab:>11}{r[0]:>9}{r[1]:>7}{r[2]:>11.3e}{r[3]:>11.5f}"
          f"{abs(r[3]-1.2)/1.2:>10.2e}{r[4]:>11.3e}{r[5]:>12.3e}{r[6]:>8.0f}",
          flush=True)


def main(dts=(1.0, 0.5, 0.25, 0.1, 0.05), variants=VARIANTS, wall=900.0):
    print("Admissible set 4 = (p, omega) at the outlet.  bc=4 mask bug fixed.")
    print("max|p_out| verifies the pressure condition is actually enforced.\n")
    print(HDR)
    for dt in dts:
        for lab, p, w in variants:
            report(dt, lab, run(dt, p, w, wall=wall))
        print(flush=True)


# NOTE: guard added 2026-08-12.  Without it, `import pois_option4` re-ran the
# whole sweep -- which is exactly what happened when this module was imported to
# reuse run() for a targeted small-dt job.
if __name__ == '__main__':
    main()
