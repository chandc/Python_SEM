"""Pressure across the outlet plane for Poiseuille.

The exact solution is p = p0 - Gx, independent of y, so p(y) at the outlet must
be FLAT.  Any y-variation is pure discretisation/weighting error, and because
pressure enters only the momentum rows (weight dt) this is the most direct
probe of the under-weighting: it should be badly non-flat at small dt and flat
at dt = 1.

Outlet BC is FREE (bc 4 -> 0, nothing imposed); pressure is pinned only at the
inlet lower-left corner, so the outlet pressure is entirely a prediction.
"""
import os, sys, time
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix, lgl_weights
from lssem2d.lssem import SolverState
import lssem2d.solver as S

SC = os.path.dirname(os.path.abspath(__file__))
LX, LY, RE = 10.0, 1.0, 100.0
NU = 1.0*LY/RE
N, EX, EY = 8, 10, 2
RATE_TOL, T_MIN, MAXSTEP = 1.0e-9, 300.0, 20000
DTS = [0.05, 0.1, 0.5, 1.0, 2.0]
u_exact = lambda y: 6.0*y*(1.0-y)


def solve(dt, w_mom=None):
    mesh = build_channel(LX, LY, EX, EY, N, bcs=(3, 4, 1, 1))
    n = N+1
    pin = next((e, 0, 0) for e in range(mesh.nelem)
               if mesh.bc[e, 0] == 3 and mesh.bc[e, 2] == 1)
    for e in range(mesh.nelem):
        if mesh.bc[e, 1] == 4:
            mesh.bc[e, 1] = 0                 # FREE outflow
    st = SolverState(mesh, diff_matrix(N), nu=NU, dt=dt, fac1=1.0, w_mom=w_mom)
    inlet = lambda x, y, t: u_exact(y)
    U = np.zeros((mesh.nelem, n, n, 4)); hist = [U]
    for s in range(MAXSTEP):
        Up = hist[0].copy()
        U = S.step_bdf(st, hist, time=s*dt, max_newton=1, newton_tol=1e-12,
                       newton_factor=0.0, custom_inlet=inlet, pin_p=pin,
                       cgsfac=1e-3, cg_max_iter=40000, verbose=False)
        if (s+1)*dt >= T_MIN and np.max(np.abs(U-Up))/dt < RATE_TOL:
            break
    return mesh, U


def plane(mesh, U, where):
    """p(y) and u(y) on the inlet or outlet plane"""
    xn, yn = mesh.xnod, mesh.ynod
    ref = xn.min() if where == 'in' else xn.max()
    i = 0 if where == 'in' else -1
    ys, ps, us = [], [], []
    for e in range(mesh.nelem):
        if abs((xn[e, 0] if where == 'in' else xn[e, -1]) - ref) < 1e-9:
            for j in range(yn.shape[1]):
                ys.append(yn[e, j]); ps.append(U[e, i, j, 2]); us.append(U[e, i, j, 0])
    o = np.argsort(ys)
    ys, ps, us = np.array(ys)[o], np.array(ps)[o], np.array(us)[o]
    k = np.concatenate(([True], np.diff(ys) > 1e-12))
    return ys[k], ps[k], us[k]


fig, axs = plt.subplots(1, 3, figsize=(15.5, 5.0))
cols = plt.cm.viridis(np.linspace(0.05, 0.85, len(DTS)))
print(f"{'dt':>6}{'p_out spread':>15}{'/ |dp|':>10}{'p_out mean':>13}{'u err':>11}")
res = []
for c, dt in zip(cols, DTS):
    mesh, U = solve(dt)
    y, p, u = plane(mesh, U, 'out')
    spread = p.max()-p.min()
    res.append((dt, spread, p.mean()))
    print(f"{dt:>6}{spread:>15.4e}{spread/1.2:>10.2e}{p.mean():>13.5f}"
          f"{np.sqrt(np.mean((u-u_exact(y))**2))/1.5:>11.2e}")
    np.savez(f'{SC}/pout_dt{dt:g}.npz', y=y, p=p, u=u)
    axs[0].plot(p, y, '-o', color=c, ms=3.5, lw=1.7, label=f'dt = {dt}')
    axs[1].plot(p-p.mean(), y, '-o', color=c, ms=3.5, lw=1.7, label=f'dt = {dt}')

# decoupled weight, for contrast
pass  # w_mom contrast lives in poiseuille_wmom.py

axs[0].axvline(0.0, color='k', lw=1.4)
axs[0].annotate('exact: $p_{out}=0$, flat', (0.0, 0.5), rotation=90,
                textcoords='offset points', xytext=(-16, -40), fontsize=8.5)
axs[0].set_xlabel('$p$ at the outlet'); axs[0].set_ylabel('y')
axs[0].set_title('outlet pressure (absolute)', fontsize=10)
axs[0].legend(fontsize=8, loc='center right'); axs[0].grid(alpha=.3)

axs[1].axvline(0.0, color='k', lw=1.4)
axs[1].set_xlabel('$p - \\overline{p}$ at the outlet')
axs[1].set_title('shape only — exact is a vertical line', fontsize=10)
axs[1].legend(fontsize=8, loc='center right'); axs[1].grid(alpha=.3)
axs[1].set_xscale('symlog', linthresh=1e-6)

d = np.array([r[0] for r in res]); sp = np.array([r[1] for r in res])
axs[2].loglog(d, sp, 'o-', color='tab:red', lw=2.2, ms=7)
axs[2].axvline(1.0, color='k', ls='--', lw=1.4)
axs[2].annotate('equal weight', (1.0, sp.max()/3), textcoords='offset points',
                xytext=(7, 0), fontsize=8.5)
axs[2].set_xlabel('dt'); axs[2].set_ylabel('outlet pressure spread  max-min')
axs[2].set_title('non-flatness vs dt', fontsize=10); axs[2].grid(alpha=.3, which='both')

fig.suptitle('Poiseuille Re=100, FREE outflow, pressure pinned at the inlet — the exact '
             'outlet pressure is constant across the channel', fontsize=11.5)
fig.tight_layout()
out = f'{SC}/poiseuille_pout.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f"\nsaved {out}")
