"""Is the small cross-stream pressure variation at x = 2.5 physically right?

Objection under test: the plane cuts a recirculation, so p cannot be nearly
constant across the channel.

Two independent checks.

A. THE FORTRAN SOLUTION.  pmg_clean's validated long-domain result is a
   completely separate code.  If it also gives a small cross-stream spread at
   x = 2.5, the Python field is not the thing being questioned.

B. THE Y-MOMENTUM BALANCE.  In this VVP formulation the steady y-momentum row is

       u v_x + v v_y + p_y - nu*om_x = 0     =>   p_y = -(u v_x + v v_y) + nu*om_x

   Evaluate both sides along x = 2.5.  If they agree, the pressure gradient the
   field carries is the one the momentum equation demands, and its magnitude is
   then a physical statement, not a numerical one.  Integrating p_y across the
   channel must also reproduce the observed spread.
"""
import os, sys
import numpy as np
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
sys.path.insert(0, SC)
from lssem2d.lgl import lgl_nodes, diff_matrix
from upper_wall import read_fortran

RE = 389.0
NU = 1.0/RE
XCUT = 2.5


def bary_w(z):
    n = len(z); w = np.ones(n)
    for j in range(n):
        for k in range(n):
            if k != j:
                w[j] /= (z[j]-z[k])
    return w


def make_col(U, xn, yn):
    """Return f(xq, comp) -> (y, value, dvalue/dy) along the line x = xq."""
    n = U.shape[1]
    z = lgl_nodes(n-1); wb = bary_w(z); D = diff_matrix(n-1)

    def lag(xq):
        d = xq - z
        hit = np.where(np.abs(d) < 1e-13)[0]
        if hit.size:
            r = np.zeros(n); r[hit[0]] = 1.0; return r
        r = wb/d
        return r/r.sum()

    def dlag(xq):
        """d/dx of the Lagrange basis at xq, via the nodal derivative matrix."""
        l = lag(xq)
        return l @ D                      # row vector of d(phi_i)/dxi at xq

    def col(xq, comp):
        ys, vs, dys, dxs = [], [], [], []
        for e in range(U.shape[0]):
            x0, x1 = xn[e, 0], xn[e, -1]
            if not (x0-1e-9 <= xq <= x1+1e-9):
                continue
            xi = 2.0*(xq-x0)/(x1-x0) - 1.0
            lx, dlx = lag(xi), dlag(xi)*(2.0/(x1-x0))
            hy = yn[e, -1]-yn[e, 0]
            for j in range(n):
                ys.append(yn[e, j])
                vs.append(float(lx @ U[e, :, j, comp]))
                dxs.append(float(dlx @ U[e, :, j, comp]))
            # d/dy at this station
            vals = np.array([float(lx @ U[e, :, jj, comp]) for jj in range(n)])
            dys.extend((D @ vals)*(2.0/hy))
        ys = np.array(ys); o = np.argsort(ys)
        ys, vs, dys, dxs = ys[o], np.array(vs)[o], np.array(dys)[o], np.array(dxs)[o]
        k = np.concatenate(([True], np.diff(ys) > 1e-12))
        return ys[k], vs[k], dys[k], dxs[k]
    return col


print("=" * 76)
print("A.  THE FORTRAN SOLUTION -- an entirely separate code")
print("=" * 76)
_, XPf, YPf, Uf = read_fortran(
    '/Users/danielchan/Dropbox/F90_SEM/pmg_clean/run_chan389_long/chan389_long.dat')
colF = make_col(Uf, XPf, YPf)
d = np.load(f'{SC}/bfswm_0.1.npz')
colP = make_col(d['U'], d['xnod'], d['ynod'])

print(f"  {'x':>6}{'x/h':>6} | {'FORTRAN p spread':>18}{'u min':>9}{'% rev':>7}"
      f" | {'PYTHON p spread':>17}{'u min':>9}{'% rev':>7}")
for xq in (1.0, 2.0, 2.5, 3.0, 4.0):
    yF, pF, _, _ = colF(xq, 2); _, uF, _, _ = colF(xq, 0)
    yP, pP, _, _ = colP(xq, 2); _, uP, _, _ = colP(xq, 0)
    print(f"  {xq:>6.2f}{xq/0.5:>6.1f} | {pF.max()-pF.min():>18.5f}{uF.min():>9.4f}"
          f"{100*np.mean(uF < 0):>6.0f}% | {pP.max()-pP.min():>17.5f}{uP.min():>9.4f}"
          f"{100*np.mean(uP < 0):>6.0f}%")

print()
print("=" * 76)
print(f"B.  Y-MOMENTUM BALANCE at x = {XCUT}  (Python field)")
print("=" * 76)
y, p, p_y, p_x = colP(XCUT, 2)
_, u, u_y, u_x = colP(XCUT, 0)
_, v, v_y, v_x = colP(XCUT, 1)
_, om, om_y, om_x = colP(XCUT, 3)

lhs = p_y
rhs = -(u*v_x + v*v_y) + NU*om_x
print(f"  max |p_y|                     = {np.abs(lhs).max():.5f}")
print(f"  max |-(u v_x + v v_y) + nu om_x| = {np.abs(rhs).max():.5f}")
print(f"  max |residual of y-momentum|  = {np.abs(lhs-rhs).max():.3e}")
print(f"  rms |residual|                = {np.sqrt(np.mean((lhs-rhs)**2)):.3e}")
print(f"\n  term sizes along this line:")
print(f"    max |u v_x|   = {np.abs(u*v_x).max():.5f}")
print(f"    max |v v_y|   = {np.abs(v*v_y).max():.5f}")
print(f"    max |nu om_x| = {np.abs(NU*om_x).max():.5f}")
print(f"    max |v|       = {np.abs(v).max():.5f}   (transverse velocity)")
print(f"    max |u|       = {np.abs(u).max():.5f}   (streamwise)")

# integrate p_y across the channel and compare with the observed spread

integ = np.trapezoid(p_y, y)
print(f"\n  integral of p_y dy across the channel = {integ:+.5f}")
print(f"  p(top) - p(bottom) from the field     = {p[-1]-p[0]:+.5f}")
print(f"  full spread max(p) - min(p)           = {p.max()-p.min():.5f}")
print(f"\n  quasi-parallel estimate: for v << u the leading balance gives")
print(f"  dp/dy ~ O(v^2 / L) -- here max|v| = {np.abs(v).max():.4f}, "
      f"v^2 = {np.abs(v).max()**2:.5f}")
print(f"  reverse-flow dynamic head u_rev^2/2 = {u.min()**2/2:.5f}")
print(f"  observed cross-stream spread        = {p.max()-p.min():.5f}")
