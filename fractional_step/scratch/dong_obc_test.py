"""Dong OBC (bc = 6) test ladder -- OUTFLOW_DONG_OBC_PLAN.md stages 0 and 2+.

    uv run --quiet python scratch/dong_obc_test.py stage0
    uv run --quiet python scratch/dong_obc_test.py ladder
    uv run --quiet python scratch/dong_obc_test.py uniform
    uv run --quiet python scratch/dong_obc_test.py theta

stage0   Plane Poiseuille, [0,12]x[0,1], 12x2 elements N=10, Re=100, dt=0.5,
         D0 = 0, switch off -> traction-free  -p + nu*du/dx = 0.
         Criterion (plan sec 5): dp = 1.44000 exact, rms div <= 1e-8.

ladder   The OUTFLOW_BC_STUDY.md sec 7b configuration: [0,10]x[0,1], 10x2
         elements, N = 8, Re = 100, parabolic inlet, COLD START, tight solve,
         dt in {1, 0.5, 0.25, 0.1, 0.05}.  Published for comparison:
             free : conv at dt=1 only (orbit 9.2 at 0.5, worse below)
             P or Z: conv down to 0.25, fails at 0.1
             P+Z  : conv down to 0.1 (393 steps), fails at 0.05
         Dong supplies TWO scalar conditions, so the ADN count says it should
         behave like P+Z.  Run at D0 = 0 and D0 = 1 (= 1/U_c, U_c = 1).

uniform  Same mesh, UNIFORM inlet -- genuinely developing at the exit, the
         exact solution is NOT representable.  Published: free/Z give
         dp = 1.60273/1.60272 at dt=1; free fails at dt=0.5, Z converges.

theta    Poiseuille dt=0.5 with the backflow switch ARMED (delta=0.05, U0=1).
         No backflow exists here, so Theta0 ~ 0 and the answer must not move:
         an inertness check of the switch machinery, not of its physics.

Fields are saved to scratch/dong_<mode>_dt<dt>_D0<d0>.npz -- always, per the
always-save-simulation-output rule.
"""
import os, sys, time
SC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SC)
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import numpy as np
import lssem2d
lssem2d.set_backend('numpy')
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix, lgl_weights
from lssem2d.lssem import SolverState
import lssem2d.solver as S

RE = 100.0
NU = 1.0 / RE


def inlet(kind):
    if kind == 'uniform':
        return lambda x, y, t: np.ones_like(np.asarray(y, dtype=float))
    return lambda x, y, t: 6.0 * np.asarray(y, dtype=float) * (1.0 - np.asarray(y, dtype=float))


def diagnostics(U, m, D, w):
    n = m.N + 1
    eu = div2 = area = 0.0
    for e in range(m.nelem):
        u = U[e, :, :, 0]; v = U[e, :, :, 1]
        ux = (D @ u) * (2.0 / m.hx[e]); vy = (v @ D.T) * (2.0 / m.hy[e])
        wq = np.outer(w, w) * 0.25 * m.hx[e] * m.hy[e]
        div2 += np.sum((ux + vy)**2 * wq); area += wq.sum()
        ye = m.ynod[e][None, :] * np.ones((n, 1))
        eu += np.sum((u - 6.0 * ye * (1.0 - ye))**2 * wq)

    def pmean(i, xt):
        num = den = 0.0
        for e in range(m.nelem):
            if abs(m.xnod[e, i] - xt) < 1e-9:
                num += np.sum(U[e, i, :, 2] * w) * (0.5 * m.hy[e]); den += m.hy[e]
        return num / den
    dp = pmean(0, m.xnod.min()) - pmean(n - 1, m.xnod.max())
    out = [e for e in range(m.nelem) if m.bc[e, 1] == 6]
    p_out = max(np.abs(U[e, -1, :, 2]).max() for e in out)
    om_err = max(np.abs(U[e, -1, :, 3] - (12.0 * m.ynod[e] - 6.0)).max() for e in out)
    return dict(dp=dp, maxu=np.abs(U[..., 0]).max(), maxv=np.abs(U[..., 1]).max(),
                rms_div=np.sqrt(div2 / area), l2_u=np.sqrt(eu / area),
                p_out=p_out, om_out_err=om_err)


def run(L, Ex, Ey, N, dt, D0, kind='parabolic', delta=None, cap=1500,
        nsub=1, tag=''):
    m = build_channel(L_x=L, L_y=1.0, E_x=Ex, E_y=Ey, N=N, bcs=(3, 6, 1, 1))
    D = diff_matrix(N); w = lgl_weights(N); n = N + 1
    st = SolverState(m, D, nu=NU, dt=dt, fac1=1.0, w_mom=1.0, w_mass=1.0)
    st.obc_D0 = D0
    st.obc_delta = delta
    U = np.zeros((m.nelem, n, n, 4)); h = [U.copy()]
    inl = inlet(kind)
    t0 = time.perf_counter(); status = 'cap'; d = np.nan
    for s in range(cap):
        prev = h[0].copy()
        U = S.step_bdf(st, h, time=(s + 1) * dt, max_newton=nsub,
                       newton_tol=1e-13, newton_factor=1e-6, custom_inlet=inl,
                       pin_p=False, cgsfac=1e-8, cg_tol=1e-10,
                       cg_max_iter=200000, line_search=(nsub > 1))
        if not np.all(np.isfinite(U)):
            status = f'NaN@{(s + 1) * dt:.2f}'; break
        if np.abs(U[..., 0]).max() > 20.0:
            status = f'BLEWUP@{(s + 1) * dt:.2f}'; break
        d = float(np.abs(U - prev).max())
        if d < 1e-12:
            status = 'conv'; break
    ok = np.all(np.isfinite(U)) and 'BLEWUP' not in status
    g = diagnostics(U, m, D, w) if ok else {}
    np.savez(f'{SC}/dong_{tag}_dt{dt:g}_D0{D0:g}.npz', U=U, xnod=m.xnod,
             ynod=m.ynod, N=N, dt=dt, D0=D0, kind=kind,
             delta=(np.nan if delta is None else delta), status=status)
    return dict(dt=dt, D0=D0, status=status, steps=s + 1, dU=d,
                wall=time.perf_counter() - t0, **g)


def show(rows, dp_exact=None):
    hdr = (f"{'dt':>6}{'D0':>5}{'status':>13}{'steps':>7}{'|dU|':>10}{'dp':>10}"
           f"{'max|u|':>9}{'max|v|':>10}{'rms div':>10}{'L2 u err':>10}"
           f"{'max|p|out':>11}{'om out err':>11}{'wall s':>8}")
    print(hdr); print('-' * len(hdr))
    for r in rows:
        if 'dp' not in r:
            print(f"{r['dt']:>6g}{r['D0']:>5g}{r['status']:>13}{r['steps']:>7}"
                  f"{r['dU']:>10.2e}")
            continue
        print(f"{r['dt']:>6g}{r['D0']:>5g}{r['status']:>13}{r['steps']:>7}"
              f"{r['dU']:>10.2e}{r['dp']:>10.5f}{r['maxu']:>9.4f}"
              f"{r['maxv']:>10.2e}{r['rms_div']:>10.2e}{r['l2_u']:>10.2e}"
              f"{r['p_out']:>11.2e}{r['om_out_err']:>11.2e}{r['wall']:>8.1f}",
              flush=True)
    if dp_exact is not None:
        print(f"{'exact':>6}{'':>5}{'':>13}{'':>7}{'':>10}{dp_exact:>10.5f}"
              f"{1.5:>9.4f}{0.0:>10.2e}{0.0:>10.2e}{0.0:>10.2e}")


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'stage0'

    if mode == 'stage0':
        print('STAGE 0: Poiseuille [0,12], 12x2 N=10, dt=0.5, traction-free '
              '(D0=0, switch off).  Criterion: dp = 1.44000, rms div <= 1e-8.\n')
        r = run(12.0, 12, 2, 10, 0.5, 0.0, cap=800, nsub=5, tag='stage0')
        show([r], dp_exact=1.44)
        ok = ('conv' in r['status'] and abs(r.get('dp', 9) - 1.44) < 1e-4
              and r.get('rms_div', 9) <= 1e-8)
        print('\nSTAGE 0:', 'PASS' if ok else 'FAIL')

    elif mode == 'ladder':
        print('LADDER: OUTFLOW_BC_STUDY sec 7b config -- [0,10], 10x2 N=8, '
              'Re=100, parabolic, cold start, nsub=1, tight solve.')
        print('Published: free conv @ dt=1 only; P or Z down to 0.25; '
              'P+Z down to 0.1 (393 steps); nothing at 0.05.\n')
        rows = []
        for D0 in (0.0, 1.0):
            for dt in (1.0, 0.5, 0.25, 0.1, 0.05):
                rows.append(run(10.0, 10, 2, 8, dt, D0, cap=1500, tag='ladder'))
                show(rows[-1:]) if len(rows) > 1 else show(rows)
        print('\nFull table:')
        show(rows, dp_exact=1.2)

    elif mode == 'uniform':
        print('UNIFORM INLET: developing flow, not representable. Published: '
              'dp = 1.60273 (free, dt=1) / 1.60272 (Z, dt=1); free FAILS at '
              'dt=0.5, Z converges (1.60370).\n')
        rows = [run(10.0, 10, 2, 8, dt, D0, kind='uniform', cap=1500,
                    tag='uniform')
                for D0 in (0.0, 1.0) for dt in (1.0, 0.5)]
        show(rows)

    elif mode == 'theta':
        print('THETA INERTNESS: Poiseuille dt=0.5, switch ARMED '
              '(delta=0.05, U0=1) vs off.  No backflow -> answers must agree.\n')
        r_off = run(10.0, 10, 2, 8, 0.5, 1.0, cap=1500, tag='thetaoff')
        r_on = run(10.0, 10, 2, 8, 0.5, 1.0, delta=0.05, cap=1500, tag='thetaon')
        show([r_off, r_on], dp_exact=1.2)
        if 'dp' in r_off and 'dp' in r_on:
            print(f"\n|dp_on - dp_off| = {abs(r_on['dp'] - r_off['dp']):.2e}")
    else:
        raise SystemExit(f'unknown mode {mode!r}')
