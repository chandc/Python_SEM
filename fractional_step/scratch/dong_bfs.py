"""Dong OBC on the SHORT-domain BFS -- the genuine-backflow test (Stage 4).

The short domain ends at x = 2.5 while the recirculation reattaches at
x_r ~ 4.1, so the reversed flow CROSSES the outlet plane -- the boundary sits
in inflow.  OUTFLOW_BC_STUDY.md sec 7c measured, at Re = 389, dt = 1:

    free outflow : BLOWS UP ON THE FIRST STEP from all three ICs
                   (max|u| 3603 / 2890 / 398 against a physical 1.5)
    P+Z          : converges from all three to the same state (~1e-08 apart)

Dong's switch term Theta0 exists precisely for this backflow: it removes the
uncontrolled energy influx 1/2|u|^2(n.u) where n.u < 0.  Here we run the
Dong outlet (bc = 6, D0 = 1/U_c = 2, switch ARMED, delta = 0.05) from the
same three ICs, plus the switch DISARMED for contrast, and a P+Z run for a
field-level comparison away from the boundary (truncation error is local --
sec 7c found short-vs-long agreement of 0.02-0.5% within 2 step heights).

    uv run --quiet python scratch/dong_bfs.py
"""
import os, sys, time
SC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SC)
sys.path.insert(0, ROOT)
sys.path.insert(0, SC)
os.chdir(ROOT)
import numpy as np
import lssem2d
lssem2d.set_backend('numpy')
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState
import lssem2d.solver as S
import lssem2d.bc as BC
from bfs_outflow_ic import build, ic_para, ic_devc, merit, reattach, RE

OB = BC.apply_bc
DT = 1.0


def run_dong(ic, D0=2.0, delta=0.05, cap=400, wallcap=900.0, picard=False,
             nsub=1):
    m, n, _ = build()
    D = diff_matrix(m.N)
    for e in range(m.nelem):
        if m.bc[e, 1] == 4:
            m.bc[e, 1] = 6                      # Dong outlet
    st = SolverState(m, D, nu=1.0 / RE, dt=DT, fac1=1.0, w_mom=1.0, w_mass=1.0)
    st.obc_D0 = D0
    st.obc_delta = delta                        # None disarms the switch
    st.obc_picard = picard                      # E at the CURRENT iterate
    inlet = lambda x, y, t: 6.0 * ((np.asarray(y) - 0.5) / 0.5) * (1.0 - (np.asarray(y) - 0.5) / 0.5)
    U = {'cold': lambda: np.zeros((m.nelem, n, n, 4)),
         'para': lambda: ic_para(m, n),
         'devc': lambda: ic_devc(m, n)}[ic]()
    h = [U.copy()]
    t0 = time.perf_counter(); status = 'CAP'; d = np.nan
    for s in range(cap):
        prev = h[0].copy()
        U = S.step_bdf(st, h, time=s * DT, max_newton=nsub, newton_tol=1e-12,
                       newton_factor=(1e-6 if nsub > 1 else 0.0),
                       custom_inlet=inlet, pin_p=False,
                       cgsfac=1e-8, cg_tol=1e-10, cg_max_iter=300000,
                       line_search=(nsub > 1))
        if not np.all(np.isfinite(U)):
            status = 'NaN'; break
        if np.abs(U[..., 0]).max() > 20.0:
            status = 'BLEWUP'; break
        d = float(np.abs(U - prev).max())
        if d < 1e-12:
            status = 'conv'; break
        if time.perf_counter() - t0 > wallcap:
            status = 'WALLCAP'; break
    ok = np.all(np.isfinite(U)) and status != 'BLEWUP'
    out = [e for e in range(m.nelem) if m.bc[e, 1] == 6]
    umin_out = min(U[e, -1, :, 0].min() for e in out) if ok else np.nan
    tag = (f'dong_bfs_{ic}_D0{D0:g}_' + ('on' if delta else 'off')
           + ('_picard' if picard else ''))
    np.savez(f'{SC}/{tag}.npz', U=U, xnod=m.xnod, ynod=m.ynod, N=m.N,
             status=status)
    return dict(ic=ic, sw=('on' if delta else 'off'), status=status,
                steps=s + 1, dU=d, J=(merit(st, U) if ok else np.nan),
                maxu=(float(np.abs(U[..., 0]).max()) if ok else np.nan),
                umin_out=umin_out,
                xr=(reattach(U, m, D) if ok else np.nan),
                wall=time.perf_counter() - t0, U=U, mesh=m)


def run_pz(cap=400, wallcap=900.0):
    """P+Z reference on the same grid (mask edit + omega wrapper, post-fix)."""
    m, n, _ = build()
    D = diff_matrix(m.N)
    xmax = m.xnod.max()
    out = [e for e in range(m.nelem) if abs(m.xnod[e, -1] - xmax) < 1e-9]

    def bc2(mesh, U, **kw):
        U = OB(mesh, U, **kw)
        for e in out:
            U[e, -1, :, 3] = -(D[-1, :-1] @ U[e, :-1, :, 3]) / D[-1, -1]
        return U

    st = SolverState(m, D, nu=1.0 / RE, dt=DT, fac1=1.0, w_mom=1.0, w_mass=1.0)
    st.get_global_mask(pin_p=False)
    for e in out:
        st._global_mask[e, -1, :, 2] = 0.0
        st._global_mask[e, -1, :, 3] = 0.0
    S.apply_bc = bc2
    inlet = lambda x, y, t: 6.0 * ((np.asarray(y) - 0.5) / 0.5) * (1.0 - (np.asarray(y) - 0.5) / 0.5)
    U = np.zeros((m.nelem, n, n, 4)); h = [U.copy()]
    t0 = time.perf_counter(); status = 'CAP'; d = np.nan
    try:
        for s in range(cap):
            prev = h[0].copy()
            U = S.step_bdf(st, h, time=s * DT, max_newton=1, newton_tol=1e-12,
                           newton_factor=0.0, custom_inlet=inlet, pin_p=False,
                           cgsfac=1e-8, cg_tol=1e-10, cg_max_iter=300000)
            if not np.all(np.isfinite(U)):
                status = 'NaN'; break
            if np.abs(U[..., 0]).max() > 20.0:
                status = 'BLEWUP'; break
            d = float(np.abs(U - prev).max())
            if d < 1e-12:
                status = 'conv'; break
            if time.perf_counter() - t0 > wallcap:
                status = 'WALLCAP'; break
    finally:
        S.apply_bc = OB
    np.savez(f'{SC}/dong_bfs_pzref.npz', U=U, xnod=m.xnod, ynod=m.ynod, N=m.N,
             status=status)
    return dict(ic='cold', sw='P+Z', status=status, steps=s + 1, dU=d,
                J=merit(st, U), maxu=float(np.abs(U[..., 0]).max()),
                umin_out=min(U[e, -1, :, 0].min() for e in out),
                xr=reattach(U, m, D), wall=time.perf_counter() - t0,
                U=U, mesh=m)


def upstream_diff(rA, rB, xcut=1.5):
    """max |U_A - U_B| over u,v restricted to elements ending before xcut."""
    m = rA['mesh']
    es = [e for e in range(m.nelem) if m.xnod[e, -1] <= xcut + 1e-9]
    return max(np.abs(rA['U'][e, :, :, :2] - rB['U'][e, :, :, :2]).max()
               for e in es)


if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'picard':
    # Switch ARMED but SEMI-IMPLICIT (Picard, nsub = 5): the lagged-explicit
    # form blew up by step 11 from every IC -- the remedy-E lesson.  If these
    # converge, the lag was the instability, not the condition.
    print(f'SHORT-domain BFS, Re = {RE:g}, dt = {DT:g}, switch ARMED, '
          f'PICARD (nsub = 5).\n')
    hdr = (f"{'IC':>6}{'outlet':>8}{'status':>9}{'steps':>7}{'|dU|':>11}"
           f"{'J end':>11}{'max|u|':>9}{'min u out':>11}{'x_r':>7}{'wall s':>8}")
    print(hdr); print('-' * len(hdr))
    rows = []
    for ic in ('cold', 'para', 'devc'):
        r = run_dong(ic, delta=0.05, picard=True, nsub=5, wallcap=1800.0)
        rows.append(r)
        print(f"{r['ic']:>6}{'DongPi':>8}{r['status']:>9}{r['steps']:>7}"
              f"{r['dU']:>11.3e}{r['J']:>11.3e}{r['maxu']:>9.3f}"
              f"{r['umin_out']:>11.3e}{r['xr']:>7.3f}{r['wall']:>8.0f}", flush=True)
    ref = np.load(f'{SC}/dong_bfs_pzref.npz')
    good = [r for r in rows if np.isfinite(r['maxu'])]
    if good:
        m = good[0]['mesh']
        es = [e for e in range(m.nelem) if m.xnod[e, -1] <= 1.5 + 1e-9]
        d_up = max(np.abs(good[0]['U'][e, :, :, :2] - ref['U'][e, :, :, :2]).max()
                   for e in es)
        print(f"\nDongPicard(cold) vs P+Z upstream of x = 1.5 (u,v): {d_up:.2e}")
        if len(good) >= 2:
            print(f"Same-state check across ICs: "
                  f"{max(upstream_diff(good[0], g, xcut=99) for g in good[1:]):.2e}")
    sys.exit(0)

if __name__ == '__main__':
    print(f'SHORT-domain BFS, Re = {RE:g}, dt = {DT:g}, nsub = 1, cold/para/devc ICs.')
    print('Published: free outflow BLOWS UP ON STEP 1 from all three ICs; '
          'P+Z converges from all three.\n')
    hdr = (f"{'IC':>6}{'outlet':>8}{'status':>9}{'steps':>7}{'|dU|':>11}"
           f"{'J end':>11}{'max|u|':>9}{'min u out':>11}{'x_r':>7}{'wall s':>8}")
    print(hdr); print('-' * len(hdr))
    rows = []
    for ic in ('cold', 'para', 'devc'):
        r = run_dong(ic, delta=0.05)
        rows.append(r)
        print(f"{r['ic']:>6}{'Dong':>8}{r['status']:>9}{r['steps']:>7}"
              f"{r['dU']:>11.3e}{r['J']:>11.3e}{r['maxu']:>9.3f}"
              f"{r['umin_out']:>11.3e}{r['xr']:>7.3f}{r['wall']:>8.0f}", flush=True)
    r_off = run_dong('cold', delta=None)
    print(f"{r_off['ic']:>6}{'D-off':>8}{r_off['status']:>9}{r_off['steps']:>7}"
          f"{r_off['dU']:>11.3e}{r_off['J']:>11.3e}{r_off['maxu']:>9.3f}"
          f"{r_off['umin_out']:>11.3e}{r_off['xr']:>7.3f}{r_off['wall']:>8.0f}",
          flush=True)
    r_pz = run_pz()
    print(f"{r_pz['ic']:>6}{'P+Z':>8}{r_pz['status']:>9}{r_pz['steps']:>7}"
          f"{r_pz['dU']:>11.3e}{r_pz['J']:>11.3e}{r_pz['maxu']:>9.3f}"
          f"{r_pz['umin_out']:>11.3e}{r_pz['xr']:>7.3f}{r_pz['wall']:>8.0f}",
          flush=True)

    good = [r for r in rows if r['status'] in ('conv', 'CAP', 'WALLCAP')
            and np.isfinite(r['maxu'])]
    if len(good) >= 2:
        print(f"\nSame-state check, Dong ICs (max|dU,dv| whole field): "
              f"{max(upstream_diff(good[0], g, xcut=99) for g in good[1:]):.2e}")
    if good and np.isfinite(r_pz['maxu']):
        print(f"Dong(cold) vs P+Z, upstream of x = 1.5 (u,v): "
              f"{upstream_diff(good[0], r_pz):.2e}")
        print(f"Dong(cold) vs P+Z, whole field (u,v):         "
              f"{upstream_diff(good[0], r_pz, xcut=99):.2e}")
