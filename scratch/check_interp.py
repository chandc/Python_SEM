"""Independent check of the interpolated exit-plane pressure.

Claim under test: the LONG-domain solution has a cross-stream pressure spread of
only 0.024 at x = 2.5, a plane that cuts through the recirculation.

Three checks:
  1. Re-evaluate the long-domain polynomial at x = 2.5 with a SEPARATE code path
     (per-element Lagrange along x only, at the element's own y nodes), and
     compare with what the interpolation produced.
  2. Sanity-check the interpolation where it must be EXACT -- the inlet channel,
     where short and long grids share nodes.
  3. Profile the cross-stream spread of p as a function of x along the whole
     long domain, to see whether 0.024 is typical or anomalous, and compare it
     with the streamwise pressure drop.
"""
import os, sys
import numpy as np
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
from lssem2d.lgl import lgl_nodes

XCUT = 2.5
d = np.load(f'{SC}/bfswm_0.1.npz')
UL, xl, yl = d['U'], d['xnod'], d['ynod']
nl = UL.shape[1]
z = lgl_nodes(nl-1)


def bary_w(zz):
    n = len(zz); w = np.ones(n)
    for j in range(n):
        for k in range(n):
            if k != j:
                w[j] /= (zz[j]-zz[k])
    return w


wb = bary_w(z)


def lag(xq):
    dd = xq - z
    hit = np.where(np.abs(dd) < 1e-13)[0]
    if hit.size:
        r = np.zeros(len(z)); r[hit[0]] = 1.0; return r
    r = wb/dd
    return r/r.sum()


def column(xq, comp=2):
    """(y, value) along the vertical line x = xq, from the long solution."""
    ys, vs = [], []
    for e in range(UL.shape[0]):
        x0, x1 = xl[e, 0], xl[e, -1]
        if not (x0-1e-9 <= xq <= x1+1e-9):
            continue
        xi = 2.0*(xq-x0)/(x1-x0) - 1.0
        lx = lag(xi)
        for j in range(nl):
            ys.append(yl[e, j])
            vs.append(float(lx @ UL[e, :, j, comp]))
    ys, vs = np.array(ys), np.array(vs)
    o = np.argsort(ys); ys, vs = ys[o], vs[o]
    k = np.concatenate(([True], np.diff(ys) > 1e-12))
    return ys[k], vs[k]


print("=" * 78)
print("CHECK 1 -- independent evaluation of the long solution at x = 2.5")
print("=" * 78)
y_ref, p_ref = column(XCUT, 2)
_, u_ref = column(XCUT, 0)

ic = np.load(f'{SC}/bfsint2_IC.npz')
Ui, xi_, yi_ = ic['U'], ic['xnod'], ic['ynod']
ni = Ui.shape[1]
ys, ps, us = [], [], []
for e in range(Ui.shape[0]):
    if abs(xi_[e, -1]-xi_.max()) < 1e-9:
        for j in range(ni):
            ys.append(yi_[e, j]); ps.append(Ui[e, -1, j, 2]); us.append(Ui[e, -1, j, 0])
o = np.argsort(ys); ys, ps, us = np.array(ys)[o], np.array(ps)[o], np.array(us)[o]
k = np.concatenate(([True], np.diff(ys) > 1e-12))
ys, ps, us = ys[k], ps[k], us[k]

print(f"  independent long-domain column at x=2.5 : spread {p_ref.max()-p_ref.min():.6f}"
      f"   min {p_ref.min():+.5f}  max {p_ref.max():+.5f}")
print(f"  interpolated IC exit column             : spread {ps.max()-ps.min():.6f}"
      f"   min {ps.min():+.5f}  max {ps.max():+.5f}")
print(f"  max |difference| in p : {np.abs(np.interp(ys, y_ref, p_ref)-ps).max():.3e}")
print(f"  max |difference| in u : {np.abs(np.interp(ys, y_ref, u_ref)-us).max():.3e}")
print(f"  u at x=2.5 ranges {us.min():+.4f} .. {us.max():+.4f}"
      f"  ({100*np.mean(us<0):.0f}% reversed) -- confirms the plane cuts the bubble")

print()
print("=" * 78)
print("CHECK 2 -- where the grids SHARE nodes the interpolation must be exact")
print("=" * 78)
for xq in (-1.0, -0.5, -0.19245009, 0.0):
    yq, pq = column(xq, 2)
    # same station from the interpolated field
    got = None
    for e in range(Ui.shape[0]):
        for i in range(ni):
            if abs(xi_[e, i]-xq) < 1e-9:
                got = np.array([Ui[e, i, j, 2] for j in range(ni)])
                yy = np.array([yi_[e, j] for j in range(ni)])
                break
        if got is not None:
            break
    if got is None:
        print(f"  x = {xq:+.5f}: not a node of the short grid")
        continue
    ref = np.interp(yy, yq, pq)
    print(f"  x = {xq:+.5f}: max|p_interp - p_long| = {np.abs(got-ref).max():.3e}")

print()
print("=" * 78)
print("CHECK 3 -- cross-stream pressure spread along the LONG domain")
print("=" * 78)
print(f"  {'x':>8}{'x/h':>7}{'p spread across y':>20}{'mean p':>12}{'u min':>9}{'% rev':>8}")
for xq in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 4.17, 5.0, 6.0, 7.0, 8.0, 8.5):
    if xq > xl.max():
        continue
    yq, pq = column(xq, 2)
    _, uq = column(xq, 0)
    print(f"  {xq:>8.2f}{xq/0.5:>7.2f}{pq.max()-pq.min():>20.5f}{pq.mean():>12.5f}"
          f"{uq.min():>9.4f}{100*np.mean(uq<0):>7.0f}%")

y0, p0 = column(0.0, 2)
y8, p8 = column(xl.max(), 2)
print(f"\n  streamwise drop, mean p at x=0 minus x={xl.max():.2f}: "
      f"{p0.mean()-p8.mean():.5f}")
print(f"  cross-stream spread at x=2.5 as a fraction of that drop: "
      f"{(p_ref.max()-p_ref.min())/abs(p0.mean()-p8.mean()):.4%}")
print(f"\n  for contrast, the SHORT domain's own state at its exit plane:")
so = np.load(f'{SC}/bfsnp2_off_nopin.npz')
Us, xs_, ys_ = so['U'], so['xnod'], so['ynod']
pe = np.array([Us[e, -1, j, 2] for e in range(Us.shape[0])
               if abs(xs_[e, -1]-xs_.max()) < 1e-9 for j in range(Us.shape[1])])
print(f"     spread {pe.max()-pe.min():.5f}  -- "
      f"{(pe.max()-pe.min())/abs(p0.mean()-p8.mean()):.1%} of the whole streamwise drop")
