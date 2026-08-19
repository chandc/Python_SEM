"""Does artificial compressibility damage an UNSTEADY solution? Measure div u.

    uv run --quiet python scratch/ac_unsteady_divergence.py

THE CONCERN, stated as a prediction so the measurement can refute it.  In 2D,
AC's accuracy argument rests on sub-iterating until p^k -> p^{k-1}, at which
point the AC term vanishes identically.  The RKW3/CN driver has NO
sub-iterations by construction: each stage is ONE linear solve carrying
kappa_p*p_prev on the right-hand side.  So the continuity row solves

    kappa_p*p^k + div u^k = kappa_p*p^{k-1}   =>   div u^k = -kappa_p*(p^k - p^{k-1})

At a steady state p^k = p^{k-1} and this is exact -- which is why the M2 cavity
gate could not see it.  Unsteady, the scaling is alarming:

    kappa_p = 1/(beta_k*dt) ~ 1/dt      and     (p^k - p^{k-1}) ~ dp/dt * dt

so their product is **O(1) in dt**: refining the time step would NOT reduce the
divergence error.  If that is what happens, AC is a steady-state method being
used on an unsteady DNS, and M7's statistics inherit a Mach-like error.

WHY IT MIGHT NOT.  This is a LEAST-SQUARES formulation: continuity is one of
eight weighted rows, not a hard constraint, so the minimiser trades the AC row
off against the momentum and vorticity rows rather than satisfying it exactly.
The prediction above is therefore an upper bound on the concern, not a
derivation of it.  Hence: measure.

THE TEST.  Same physical time, same everything, dt refined 4x, AC on vs off.
  * div u falls with dt        -> AC is not corrupting the unsteady solution
  * div u flat in dt with AC on, and small with AC off -> the concern is real
"""
import os, sys, time
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
sys.path.insert(0, SC)
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
import channel3d as C
from lssem3d import operator as OP, timestep as T, deriv as DV

GRID = dict(N=6, ex=3, ey=3, nz=16, re=180.0)
AMP = 0.05
TEND = 0.08


def divergence(s, U):
    """rms and max of |div u| = |u_x + v_y + i k w|, over the whole field."""
    m, D, kz = s['m'], s['D'], s['kz']
    Uc = OP.to_complex(U)
    d = (DV.ddx(Uc[..., OP.U_, :], D, m.facx)
         + DV.ddy(Uc[..., OP.V_, :], D, m.facy)
         + 1j*kz*Uc[..., OP.W_, :])
    return float(np.sqrt(np.mean(np.abs(d)**2))), float(np.abs(d).max())


def velocity_scale(s, U):
    Uc = OP.to_complex(U)
    return float(max(np.abs(Uc[..., f, :]).max() for f in (OP.U_, OP.V_, OP.W_)))


def run(dt, ac, tend=TEND):
    s = C.setup(**GRID)
    U = C.initial_state(s, amp=AMP)
    a = T.a_mass_worst(dt)
    kap = a if ac else 0.0
    Minv = C.make_precond(s, dt, kap)
    Nprev = np.zeros(OP.to_complex(U).shape[:-2] + (3, s['nk']), dtype=complex)
    nstep = int(round(tend/dt))
    for _ in range(nstep):
        U, Nprev, _ = C.step(s, U, Nprev, dt, kap, Minv=Minv, tol=1e-10,
                             max_iter=20000)
    rms, mx = divergence(s, U)
    return dict(dt=dt, ac=ac, a_mass=a, nstep=nstep, rms=rms, max=mx,
                scale=velocity_scale(s, U))


if __name__ == '__main__':
    print(f'Re={GRID["re"]:g}  perturbed channel, amp={AMP}, '
          f'same physical time t={TEND} in every run\n')
    print(f"{'AC':>5}{'dt':>9}{'a_mass':>9}{'steps':>7}"
          f"{'rms|div u|':>13}{'max|div u|':>13}{'rel to |u|':>12}")
    out = []
    for ac in (True, False):
        for dt in (0.008, 0.004, 0.002, 0.001):
            t0 = time.perf_counter()
            r = run(dt, ac)
            r['wall'] = time.perf_counter()-t0
            out.append(r)
            print(f"{('on' if ac else 'off'):>5}{dt:>9g}{r['a_mass']:>9.0f}"
                  f"{r['nstep']:>7}{r['rms']:>13.4e}{r['max']:>13.4e}"
                  f"{r['rms']/max(r['scale'],1e-30):>12.3e}", flush=True)
    print('\n  If rms|div u| FALLS with dt for AC on, the unsteady concern is')
    print('  refuted.  If it is FLAT with AC on but falls with AC off, the AC')
    print('  term is injecting an O(1)-in-dt divergence error.')
    for ac in (True, False):
        r = [x for x in out if x['ac'] == ac]
        print(f"  AC {'on ' if ac else 'off'}: rms ratio finest/coarsest = "
              f"{r[-1]['rms']/r[0]['rms']:.3f}  (dt refined {r[0]['dt']/r[-1]['dt']:.0f}x)")
