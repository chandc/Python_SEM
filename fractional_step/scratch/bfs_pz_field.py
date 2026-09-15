"""Short-domain BFS under the admissible outflow pair: the converged field.

The cold and parabolic ICs converge to the SAME state (max|cold - para| = 9e-09
in u, 1e-08 in v, 1e-09 in p, 4e-07 in omega), so one field is shown.

Free outflow cannot be shown at all: it reaches max|u| = 3603 on step 1.

Reattachment is absent from the picture and that is CORRECT -- x_r/h ~ 8.19-8.21
(WEIGHT_VS_TIMESTEP_STUDY.md) means x_r ~ 4.1 at h = 0.5, against a domain that
ends at x = 2.5.  The short domain truncates the recirculation, as
STEADY_FORM_STUDY.md sec 5 says.  Note what that implies for the BC: the reversed
flow reaches the outlet plane, so the outflow condition sits in INFLOW -- the
hardest case there is, and why free outflow fails here far more violently than
on Poiseuille.

Saves the state so the field can be re-plotted without a 5 minute re-solve.
"""
import os, sys, time
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
sys.path.insert(0, SC)
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, LinearTriInterpolator
import lssem2d
lssem2d.set_backend('numpy')
from fgrid import load
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState
import lssem2d.solver as S
import lssem2d.bc as BC
from bfs_outflow_ic import GRID, RE

H = 0.5
OB = BC.apply_bc
NPZ = f'{SC}/bfs_pz_state.npz'


def solve():
    m, _, _ = load(GRID); n = m.N+1; N = m.N
    D = diff_matrix(N)
    xmax = m.xnod.max()
    out = [e for e in range(m.nelem) if abs(m.xnod[e, -1]-xmax) < 1e-9]

    def bc2(mesh, U, **kw):
        U = OB(mesh, U, **kw)
        for e in out:
            U[e, -1, :, 3] = -(D[-1, :-1] @ U[e, :-1, :, 3])/D[-1, -1]
        return U

    st = SolverState(m, D, nu=1.0/RE, dt=1.0, fac1=1.0, w_mom=1.0, w_mass=1.0)
    st.get_global_mask(pin_p=False)
    for e in out:
        st._global_mask[e, -1, :, 2] = 0.0
        st._global_mask[e, -1, :, 3] = 0.0
    S.apply_bc = bc2
    inlet = lambda x, y, t: 6.0*((y-0.5)/0.5)*(1.0-(y-0.5)/0.5)
    U = np.zeros((m.nelem, n, n, 4)); h = [U.copy()]
    try:
        for s in range(400):
            prev = h[0].copy()
            U = S.step_bdf(st, h, time=s, max_newton=1, newton_tol=1e-12,
                           newton_factor=0.0, custom_inlet=inlet, pin_p=False,
                           cgsfac=1e-8, cg_tol=1e-10, cg_max_iter=300000)
            d = float(np.abs(U-prev).max())
            if d < 1e-12:
                break
    finally:
        S.apply_bc = OB
    print(f"  converged: {s+1} steps, |dU| = {d:.3e}", flush=True)
    np.savez(NPZ, U=U, xnod=m.xnod, ynod=m.ynod, hy=m.hy, N=m.N)
    return U, m


if os.path.exists(NPZ):
    d = np.load(NPZ)
    U, xn, yn, hy, N = d['U'], d['xnod'], d['ynod'], d['hy'], int(d['N'])
    print("loaded saved state")
else:
    U, m = solve()
    xn, yn, hy, N = m.xnod, m.ynod, m.hy, m.N
n = N+1
D = diff_matrix(N)

px, py, pu, pv, pp = [], [], [], [], []
for e in range(U.shape[0]):
    for i in range(n):
        for j in range(n):
            px.append(xn[e, i]); py.append(yn[e, j])
            pu.append(U[e, i, j, 0]); pv.append(U[e, i, j, 1]); pp.append(U[e, i, j, 2])
px, py, pu, pv, pp = map(np.array, (px, py, pu, pv, pp))
tri = Triangulation(px, py)
cx = px[tri.triangles].mean(axis=1); cy = py[tri.triangles].mean(axis=1)
tri.set_mask((cx < 0) & (cy < 0.5))
fu = LinearTriInterpolator(tri, pu); fv = LinearTriInterpolator(tri, pv)
fp = LinearTriInterpolator(tri, pp)
gx = np.linspace(px.min(), px.max(), 900); gy = np.linspace(0, 1, 240)
GX, GY = np.meshgrid(gx, gy)
ui = np.array(fu(GX, GY).filled(np.nan))
vi = np.array(fv(GX, GY).filled(np.nan))
pi = np.array(fp(GX, GY).filled(np.nan))

fig, axs = plt.subplots(3, 1, figsize=(13.4, 11.0),
                        gridspec_kw={'height_ratios': [1.15, 1.15, 1.0], 'hspace': .40})

# 1 -- streamlines over speed
ax = axs[0]
cf = ax.contourf(GX, GY, np.ma.masked_invalid(np.hypot(ui, vi)), levels=40, cmap='viridis')
rev = np.ma.masked_where(np.nan_to_num(ui) >= 0, np.ones_like(GX))
ax.contourf(GX, GY, rev, levels=[.5, 1.5], colors=['red'], alpha=.25)
ax.streamplot(gx, gy, np.nan_to_num(ui), np.nan_to_num(vi), density=2.6,
              color='w', linewidth=.7, arrowsize=.7)
ax.add_patch(plt.Rectangle((px.min(), 0), -px.min(), .5, fc='0.85', ec='k', lw=1.2, zorder=5))
ax.axvline(xn.max(), color='yellow', lw=3.5, zorder=6)
ax.set_title('Streamlines over |u|  —  red = reversed flow.  The recirculation '
             'reaches the OUTLET (yellow),\nso the outflow BC sits in inflow.  '
             'x_r/h ≈ 8.2 ⇒ x_r ≈ 4.1, beyond this domain (ends at 2.5).', fontsize=10)
ax.set_ylabel('y'); ax.set_xlim(px.min(), px.max()); ax.set_ylim(0, 1)
plt.colorbar(cf, ax=ax, pad=.01, fraction=.026, label='|u|')

# 2 -- pressure field
ax = axs[1]
cf = ax.contourf(GX, GY, np.ma.masked_invalid(pi), levels=40, cmap='coolwarm')
ax.contour(GX, GY, np.ma.masked_invalid(pi), levels=18, colors='k', linewidths=.45, alpha=.55)
ax.add_patch(plt.Rectangle((px.min(), 0), -px.min(), .5, fc='0.85', ec='k', lw=1.2, zorder=5))
ax.axvline(xn.max(), color='yellow', lw=3.5, zorder=6)
ax.set_title('Pressure field p(x, y).  p = 0 imposed on the outlet plane; '
             'the inlet level is PREDICTED.', fontsize=10)
ax.set_ylabel('y'); ax.set_xlim(px.min(), px.max()); ax.set_ylim(0, 1)
plt.colorbar(cf, ax=ax, pad=.01, fraction=.026, label='p')

# 3 -- pressure profiles
ax = axs[2]
for lab, sel, jj, col in (('bottom wall  y = 0', lambda e: yn[e, 0] < 0.01 and xn[e, 0] > -1e-9, 0, 'tab:blue'),
                          ('top wall  y = 1', lambda e: yn[e, -1] > 0.99, -1, 'tab:red')):
    xs, ps = [], []
    for e in range(U.shape[0]):
        if not sel(e):
            continue
        for i in range(n):
            xs.append(xn[e, i]); ps.append(U[e, i, jj, 2])
    o = np.argsort(xs)
    ax.plot(np.array(xs)[o], np.array(ps)[o], lw=2.0, color=col, label=lab)
# centreline of the expanded channel
xs, ps = [], []
for e in range(U.shape[0]):
    if xn[e, 0] < -1e-9:
        continue
    jm = int(np.argmin(np.abs(yn[e, :]-0.5)))
    if abs(yn[e, jm]-0.5) > 0.02:
        continue
    for i in range(n):
        xs.append(xn[e, i]); ps.append(U[e, i, jm, 2])
if xs:
    o = np.argsort(xs)
    ax.plot(np.array(xs)[o], np.array(ps)[o], lw=1.6, color='tab:green',
            ls='--', label='y = 0.5 (step height)')
ax.axvline(xn.max(), color='goldenrod', lw=2.5, label='outlet (p = 0 imposed)')
ax.axvline(0.0, color='k', lw=1.0, ls=':', label='step')
ax.set_xlabel('x'); ax.set_ylabel('p'); ax.grid(alpha=.3)
ax.legend(fontsize=8.5, loc='upper left')
ax.set_title('Pressure profiles along the domain', fontsize=10)

fig.suptitle('Short-domain BFS, Re = 389, dt = 1, w_mom = w_mass = 1,  outlet: '
             'p = 0 and ∂ω/∂x = 0\nIdentical from a cold start and from a '
             'parabolic IC (max difference 1e-08).', fontsize=12)
fig.savefig(f'{SC}/../figs/bfs_pz_field.png', dpi=125, bbox_inches='tight')
print('figs/bfs_pz_field.png')
