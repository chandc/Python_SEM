"""Was the slaved-omega failure the IDEA, or my lagged implementation of it?

The first test (pois_omega_bc.log) imposed  om := v_x - u_y  at the outlet once
per step and reported dp = 1.87, 58% high.  That verdict does not stand up:

    variant      |dU| final      slip = |om - (v_x - u_y)|
    free          0.000e+00      2.565e-08
    Z zero-grad   0.000e+00      2.428e-08
    S slaved      1.658e-04      1.636e-04     <-- slip ~= |dU|

Slaving should drive slip to zero BY CONSTRUCTION.  Instead it is the worst of
the three, and essentially equal to |dU| -- the signature of a LAG, not of a
constraint: apply_bc computes om from the field at the top of newton_step, the
solve then moves u and v, and the final om no longer matches the final velocity.
|dU| stalling at 1e-04 also means that run never converged, so dp = 1.87 is just
where a non-convergent iteration happened to be at step 600.

Proper slaving means ELIMINATING the omega DOF -- substituting om = v_x - u_y
into the equations so the Jacobian knows that perturbing u or v moves om at the
boundary.  That is static condensation and needs apply_L/apply_LT changed.  This
script tests the cheap proxy instead: re-apply the constraint inside every
Newton sub-iteration, so the lag shrinks as the sub-iterations converge.  If S
then converges and dp -> 1.2, the idea is sound and only the implementation was
at fault.  If it stalls at 1e-04 regardless of nsub, the lag is not the cause.

Also re-checks the claim that a definitionally-true constraint "carries no
information".  In a least-squares system the vorticity row is enforced only
WEAKLY, so imposing it STRONGLY at the boundary does change the discrete system
-- it removes the slack that lets omega drift.  It is not redundant, and that
sentence in OUTFLOW_BC_STUDY.md sec 7a needs softening either way.
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
from lssem2d.operators import dUdx, dUdy
import lssem2d.solver as S
import lssem2d.bc as BC

N, EX, EY = 8, 10, 2
NU = 0.01
ue = lambda y: 6.0*y*(1.0-y)
D = diff_matrix(N)
wq = lgl_weights(N)
OB = BC.apply_bc


def run(dt, mode, nsub, nst=600, wall=900.0):
    m = build_channel(10., 1., EX, EY, N, bcs=(3, 4, 1, 1))
    n = N+1
    pin = next((e, 0, 0) for e in range(m.nelem)
               if m.bc[e, 0] == 3 and m.bc[e, 2] == 1)
    for e in range(m.nelem):
        if m.bc[e, 1] == 4:
            m.bc[e, 1] = 0
    xn = m.xnod; xmax = xn.max(); xmin = xn.min()
    out = [e for e in range(m.nelem) if abs(xn[e, -1]-xmax) < 1e-9]

    def bc2(mesh, U, **kw):
        # called once per NEWTON iteration inside newton_step, so raising nsub
        # shrinks the lag between the imposed omega and the current velocity
        U = OB(mesh, U, **kw)
        if mode == 'slaved':
            vx = dUdx(np.ascontiguousarray(U[..., 1]), D, mesh.facx)
            uy = dUdy(np.ascontiguousarray(U[..., 0]), D, mesh.facy)
            for e in out:
                U[e, -1, :, 3] = vx[e, -1, :] - uy[e, -1, :]
        elif mode == 'zerograd':
            for e in out:
                U[e, -1, :, 3] = -(D[-1, :-1] @ U[e, :-1, :, 3])/D[-1, -1]
        return U

    st = SolverState(m, D, nu=NU, dt=dt, fac1=1.0, w_mom=1.0, w_mass=1.0)
    st.get_global_mask(pin_p=pin)
    for e in out:
        st._global_mask[e, -1, :, 3] = 0.0
    S.apply_bc = bc2
    t0 = time.perf_counter(); status = 'TCAP'; d = np.nan
    try:
        U = np.zeros((m.nelem, n, n, 4)); h = [U.copy()]
        for s in range(nst):
            prev = h[0].copy()
            U = S.step_bdf(st, h, time=s*dt, max_newton=nsub, newton_tol=1e-13,
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
    vx = dUdx(np.ascontiguousarray(U[..., 1]), D, m.facx)
    uy = dUdy(np.ascontiguousarray(U[..., 0]), D, m.facy)
    slip = max(np.abs(U[e, -1, :, 3]-(vx[e, -1, :]-uy[e, -1, :])).max() for e in out)
    prof = float(np.sqrt(np.mean((U[..., 0]-ue(m.ynod)[:, None, :])**2))/1.5)
    return (status, s+1, d, float(pbar('in')-pbar('out')), prof, float(slip),
            time.perf_counter()-t0)


print("Does sub-iterating remove the lag that broke slaved omega?")
print("nsub re-applies the constraint inside each Newton iteration.\n")
hdr = (f"{'dt':>6}{'mode':>11}{'nsub':>6}{'status':>9}{'steps':>7}{'|dU|':>11}"
       f"{'dp':>11}{'dp err':>10}{'prof err':>11}{'slip':>11}{'wall s':>8}")
print(hdr)
for dt in (1.0, 0.5):
    for mode, nsub in (('slaved', 1), ('slaved', 5), ('slaved', 20),
                       ('zerograd', 1)):
        r = run(dt, mode, nsub)
        if r is None:
            print(f"{dt:>6g}{mode:>11}{nsub:>6}   DIVERGED", flush=True); continue
        print(f"{dt:>6g}{mode:>11}{nsub:>6}{r[0]:>9}{r[1]:>7}{r[2]:>11.3e}"
              f"{r[3]:>11.5f}{abs(r[3]-1.2)/1.2:>10.2e}{r[4]:>11.3e}"
              f"{r[5]:>11.3e}{r[6]:>8.0f}", flush=True)
