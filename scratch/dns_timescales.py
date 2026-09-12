"""Is the DNS time step set by physics or by numerical stability?  Measure both.

    uv run python scratch/dns_timescales.py [checkpoint.npz ...] [--dt 8e-4]

For a direct numerical simulation the time step has to resolve the fastest
motion in the flow, which is the near-wall end of the dissipation range; it must
*also* stay inside whatever stability limit the scheme has, but a DNS that is
limited by stability rather than by physics is one whose scheme is in the way.
This script computes the two limits from a real field and prints the ratio, so
the claim is a measurement rather than a preference.

WHAT IS COMPUTED, all from the checkpointed velocity field:

  * the local dissipation from velocity gradients,
    eps = nu * sum_ij (du_i/dx_j)^2, plane-averaged and pointwise-maximum;
  * the Kolmogorov time tau_eta = sqrt(nu/eps), whose MINIMUM over the channel
    is the fastest time scale the simulation must resolve;
  * the convective CFL on the true smallest GLL spacing, and the RKW3 stability
    limit of sqrt(3), which together give the largest stable step for this field;
  * dt+ = dt*u_tau^2/nu, the step in wall units, which is how the channel-flow
    literature quotes it.

Viscous terms are implicit (each RKW3 stage is a Stokes solve), so there is no
diffusive stability limit to report -- the explicit convection is the only one.
"""
import argparse
import glob
import os
import sys

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '4')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R)
sys.path.insert(0, os.path.join(_R, 'scratch'))
os.chdir(_R)

import numpy as np

CFL_LIMIT = np.sqrt(3.0)              # RKW3 (Spalart-Moser-Rogers) convective limit


def timescales(s, U, nz, dt):
    from lssem3d import operator as OP, fourier as FR, deriv as DV
    import minchan as MC
    from minchan_stats import PlaneStats

    m, D, kz, nu = s['m'], s['D'], s['kz'], s['nu']
    Uc = OP.to_complex(U)
    P = FR.to_physical(Uc[..., (OP.U_, OP.V_, OP.W_), :], nz)

    # eps(x) = nu * sum_ij (du_i/dx_j)^2, from the velocity gradients rather
    # than the stored vorticity: omega is an independent VVP unknown that
    # satisfies omega = curl u only to the least-squares residual, and it is
    # left free on the walls, which is where the dissipation peaks.
    g2 = np.zeros(P.shape[:-2] + (nz,))
    for f in (OP.U_, OP.V_, OP.W_):
        c = Uc[..., f, :]
        for q in (DV.ddx(c, D, m.facx), DV.ddy(c, D, m.facy), 1j*kz*c):
            g2 += FR.to_physical(q, nz)**2
    eps = nu*g2

    st = PlaneStats(s, nz)
    flat = eps.reshape(-1, nz)[st.order]
    epsy = np.array([float((flat[g]*st.gw[j][:, None]).sum()/(st.gwsum[j]*nz))
                     for j, g in enumerate(st.groups)])
    y = st.y
    utau = MC.u_tau(s, U)
    tau = np.sqrt(nu/np.maximum(epsy, 1e-30))
    jmin = int(np.argmin(tau))
    tmin_pt = float(np.sqrt(nu/max(eps.max(), 1e-30)))

    cfl = MC.cfl(s, U, dt)
    ywall = np.minimum(y, 2.0 - y)          # distance to the NEARER wall
    return dict(y=y, yp=ywall*utau/nu, eps=epsy, tau=tau, utau=utau, nu=nu,
                tau_min=float(tau[jmin]), y_tau_min=float(y[jmin]),
                yp_tau_min=float(ywall[jmin]*utau/nu), tau_min_point=tmin_pt,
                cfl=cfl, dt_stable=dt*CFL_LIMIT/max(cfl, 1e-30), dt=dt,
                dtp=dt*utau**2/nu, eps_max=float(epsy.max()))


def report(r):
    dt = r['dt']
    print(f'  u_tau = {r["utau"]:.4f}   nu = {r["nu"]:.5f}   '
          f'dt = {dt:.2e}  =  dt+ {r["dtp"]:.3f} wall units')
    print(f'  PHYSICS   min plane-averaged Kolmogorov time tau_eta = {r["tau_min"]:.2e} '
          f'at y+ = {r["yp_tau_min"]:.1f}   (pointwise min {r["tau_min_point"]:.2e})')
    print(f'            tau_eta / dt = {r["tau_min"]/dt:6.1f}      '
          f'largest step resolving tau_eta/10 = {r["tau_min"]/10:.2e}')
    print(f'  STABILITY CFL = {r["cfl"]:.3f} of the RKW3 limit {CFL_LIMIT:.3f}  '
          f'-> largest stable step = {r["dt_stable"]:.2e}')
    lim = 'PHYSICS' if r['tau_min']/10 < r['dt_stable'] else 'STABILITY'
    print(f'  BINDING   {lim}: physics allows {r["tau_min"]/10:.2e}, '
          f'stability allows {r["dt_stable"]:.2e}  '
          f'(ratio {r["tau_min"]/10/r["dt_stable"]:.2f})')
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='*')
    ap.add_argument('--dt', type=float, default=8e-4)
    ap.add_argument('--nz', type=int, default=32)
    a = ap.parse_args()
    files = a.files or sorted(glob.glob('scratch/run01_ck/checkpoint_*.npz'))[-1:]
    if not files:
        raise SystemExit('give a checkpoint npz')

    import lssem3d
    lssem3d.set_backend('numpy')
    import minchan as MC
    s = MC.setup(nz=a.nz)

    print(f'minimal channel Re_tau = {MC.RE_TAU:g}, {s["m"].nelem} elements N = {s["N"]}, '
          f'nz = {a.nz},  dt = {a.dt:g}\n')
    out = []
    for f in files:
        with np.load(f) as z:
            U, t = z['U'], float(z['t']) if 't' in z.files else np.nan
        print(f'{os.path.basename(f)}   t = {t:.3f}')
        out.append(report(timescales(s, U, a.nz, a.dt)))
        print()
    r = out[-1]
    print('reading: the step is set by the smaller of the two limits.  Resolving the '
          'Kolmogorov\n         time to a tenth needs dt <= '
          f'{r["tau_min"]/10:.2e}; RKW3 stability on this field allows\n         '
          f'dt <= {r["dt_stable"]:.2e}.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
