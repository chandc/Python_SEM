"""M2 GATE: the 3D solver at k_z = 0 must reproduce the 2D Ghia result.

    uv run --quiet python scratch/cavity3d_kz0.py [dt] [nstep]

3D_DEVELOPMENT_PLAN.md Stage 1 calls this the anchor: RMS u = 1.568e-02 against
Ghia is already measured by the 2D code on this mesh (ARTIFICIAL_COMPRESSIBILITY.md
sec 5.1), so it is a number, not a judgement call.  test_stage1_vs_2d.py already
showed the OPERATOR matches to 1e-13; this exercises the whole stack --
operator + quadrature weights + gather-scatter + multiplicity + BC masking +
batched CG -- on an actual flow.

TIME INTEGRATION IS RKW3/CRANK-NICOLSON, and that is not optional.  The first
version used backward Euler with explicit convection, i.e. FORWARD EULER on the
convective term -- whose stability interval on the imaginary axis is exactly
ZERO (|1+iy| = sqrt(1+y^2) > 1 for any y).  It is unconditionally unstable for
advection and survived 36 steps only on viscous damping before going NaN.  A
smaller dt postpones that, it does not fix it.  RK3 is the lowest-order explicit
RK with a genuine interval, |lam|dt <= sqrt(3); AB2 and RK2 have none either.

ARTIFICIAL COMPRESSIBILITY IS ON.  At k_z = 0 the pressure reaches the operator
only through grad p -- the i*k_z*p path in the w-momentum row vanishes -- so the
missing a33 diagonal AC supplies is worst exactly here.  2D measured 27x fewer CG
iterations at a_mass = 30 (ARTIFICIAL_COMPRESSIBILITY.md sec 5.2), and this gate
runs at c = 1/(beta_k*dt), far above that.  The term is consistent at a steady
state (p = p_prev), which is what this gate measures.

Stage k solves   c*u^k + grad p + nu*curl(omega) = f,   c = 1/(beta_k*dt),
    f = c*[ u^{k-1} + dt*(gamma_k N^{k-1} + zeta_k N^{k-2} + alpha_k L^{k-1}) ]
with N = -(u.grad u) and L = -(grad p + nu*curl omega), the latter obtained by
evaluating the momentum rows with c = 0.
"""
import os, sys, time
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem3d import operator as OP, solver3d as S3, bc as BC, convect as CV
from lssem3d import timestep as T

RE, EX, N = 1000.0, 6, 10
NU = 1.0/RE
NZ, LZ = 1, 2.0*np.pi          # a single k_z = 0 mode
GH = np.load('cavity_re1000_data.npz')
# Ghia Table II: v(x) on the horizontal centreline y = 0.5, Re = 1000.
# NOTE lssem2d/tests/plot_verification.py carries a DIFFERENT ghia_v -- Re=100.
GHIA_XV = np.array([1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594,
                    0.8047, 0.5000, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781,
                    0.0703, 0.0625, 0.0000])
GHIA_V = np.array([0.0000, -0.21388, -0.27669, -0.33714, -0.39188, -0.51550,
                   -0.42665, -0.31966, 0.02526, 0.32235, 0.33075, 0.37095,
                   0.32627, 0.30353, 0.29012, 0.27485, 0.0000])


def lagrange(xn, xq):
    n = len(xn); w = np.ones(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                w[i] /= (xn[i]-xn[j])
    dd = xq-xn
    if np.any(np.abs(dd) < 1e-13):
        L = np.zeros(n); L[np.argmin(np.abs(dd))] = 1.0; return L
    num = w/dd
    return num/num.sum()


def centreline_u(mesh, U, n):
    """u(y) on x = 0.5, from the real part of the k_z = 0 mode."""
    ys, us = [], []
    for e in range(mesh.nelem):
        xs = mesh.xnod[e]
        if xs[0]-1e-9 <= 0.5 <= xs[-1]+1e-9:
            L = lagrange(xs, 0.5)
            for j in range(n):
                ys.append(mesh.ynod[e, j])
                us.append(np.dot(L, U[e, :, j, OP.U_, 0]))
    o = np.argsort(ys); ys, us = np.array(ys)[o], np.array(us)[o]
    k = np.concatenate(([True], np.diff(ys) > 1e-9))
    return ys[k], us[k]


def centreline_v(mesh, U, n):
    """v(x) on y = 0.5.  Tracking BOTH components is not optional: the 2D study
    found RMS u improving while RMS v did not move, and it was the v column that
    established AC is accuracy-NEUTRAL rather than better
    (ARTIFICIAL_COMPRESSIBILITY.md sec 5.1).  A gate on u alone can be passed by
    a solution that is wrong in v."""
    xs, vs = [], []
    for e in range(mesh.nelem):
        yr = mesh.ynod[e]
        if yr[0]-1e-9 <= 0.5 <= yr[-1]+1e-9:
            L = lagrange(yr, 0.5)
            for i in range(n):
                xs.append(mesh.xnod[e, i])
                vs.append(np.dot(L, U[e, i, :, OP.V_, 0]))
    o = np.argsort(xs); xs, vs = np.array(xs)[o], np.array(vs)[o]
    k = np.concatenate(([True], np.diff(xs) > 1e-9))
    return xs[k], vs[k]


def rms_both(mesh, U, n):
    ys, us = centreline_u(mesh, U, n)
    xs, vs = centreline_v(mesh, U, n)
    ru = float(np.sqrt(np.mean((np.interp(GH['ghia_y'], ys, us)-GH['ghia_u'])**2)))
    rv = float(np.sqrt(np.mean((np.interp(GHIA_XV[::-1], xs, vs)[::-1]-GHIA_V)**2)))
    return ru, rv


def run(cfl_target=0.8, tmax=25.0, kap_frac=1.0, nstep_cap=40000, tol=1e-9):
    n = N+1
    mesh = build_channel(1.0, 1.0, EX, EX, N, bcs=(1, 1, 1, 2))
    D = diff_matrix(N)
    kz = np.zeros(NZ)
    shape = (mesh.nelem, n, n, OP.NVAR_R, NZ)

    Uprobe = np.zeros((mesh.nelem, n, n, OP.NVAR, NZ))
    Uprobe[..., OP.U_, :] = 1.0
    dt = CV.max_dt_for_cfl(Uprobe, D, mesh.facx, mesh.facy, LZ, NZ, cfl_target)
    cs = [1.0/(T.BETA[k]*dt) for k in range(T.NSTAGE)]
    kap = kap_frac*max(cs)
    print(f'  dt = {dt:.3e} (CFL {cfl_target:g}; RK3 limit sqrt3 = {T.cfl_limit():.3f})')
    print(f'  c per stage = {[f"{c:.0f}" for c in cs]}   kappa_p = {kap:.0f}')

    mask = BC.build_mask(mesh, NZ, pin_p=True)
    U = np.zeros(shape)
    BC.apply_values(mesh, U, NZ, lid_speed=1.0, pin_p=True)

    # Row weights: lssem2d's legacy least-squares scaling, without which the
    # momentum rows outweigh the constraints by c^2 (3D_STATUS.md sec 7A.2).
    RWS = [OP.momentum_row_weights(c) if ROWWEIGHT else None for c in cs]
    _fw = lambda rw, n2: (1.0 if rw is None
                          else np.concatenate([rw, rw]).reshape((1, 1, 1, n2, 1)))

    t0 = time.perf_counter(); Minv = []
    for k, c in enumerate(cs):
        d = S3.jacobi_diagonal_analytic(shape, D, mesh.facx, mesh.facy, kz, NU, c,
                               mesh, mask, mesh.wq, kap, rw=RWS[k])
        Minv.append(S3.jacobi_inverse(d, mask))
    print(f'  jacobi probed (3 stages) in {time.perf_counter()-t0:.0f}s', flush=True)

    wqR = mesh.wq[..., None, None]
    Nprev = np.zeros((mesh.nelem, n, n, 3, NZ), dtype=complex)
    nstep = min(nstep_cap, int(round(tmax/dt)))
    t0 = time.perf_counter(); status = 'CAP'; cg_tot = 0
    for s_ in range(nstep):
        Uold = U.copy()
        for k in range(T.NSTAGE):
            Uc = OP.to_complex(U)
            Nk = -CV.convective(Uc, D, mesh.facx, mesh.facy, kz, NZ)
            R0 = OP.apply_L0_complex(Uc, D, mesh.facx, mesh.facy, kz, NU, 0.0, kap)
            Lk = -R0[..., 4:7, :]
            c = cs[k]
            fc = np.zeros((mesh.nelem, n, n, OP.NROW, NZ), dtype=complex)
            for row, fld in ((4, OP.U_), (5, OP.V_), (6, OP.W_)):
                i = row-4
                fc[..., row, :] = c*(Uc[..., fld, :] + dt*(
                    T.GAMMA[k]*Nk[..., i, :] + T.ZETA[k]*Nprev[..., i, :]
                    + T.ALPHA[k]*Lk[..., i, :]))
            fc[..., 0, :] = kap*Uc[..., OP.P_, :]
            f = np.concatenate([fc.real, fc.imag], axis=-2)

            r = OP.apply_LT(OP.apply_L(U, D, mesh.facx, mesh.facy, kz, NU, c,
                                       mesh.wq, kap, RWS[k])
                            - f*wqR*_fw(RWS[k], f.shape[-2]),
                            D, mesh.facx, mesh.facy, kz, NU, c, kap)
            b = -S3.gs(mesh, r)*mask
            dU, it, _ = S3.pcg(b, D, mesh.facx, mesh.facy, kz, NU, c, mesh=mesh,
                               mask=mask, M_inv=Minv[k], tol=1e-8,
                               max_iter=20000, wq=mesh.wq, kap=kap, rw=RWS[k])
            U = U + dU
            cg_tot += it
            Nprev = Nk

        if not np.all(np.isfinite(U)):
            status = f'NaN@{s_+1}'; break
        dUs = float(np.abs(U-Uold).max())
        if (s_+1) % 200 == 0 or s_ < 2:
            ru, rv = rms_both(mesh, U, n)
            print(f'  step {s_+1:5d} t={dt*(s_+1):6.2f}  |dU| {dUs:.2e}  '
                  f'cg/step {cg_tot/(s_+1):6.0f}  RMS u {ru:.4e}  RMS v {rv:.4e}'
                  f'  {time.perf_counter()-t0:6.0f}s', flush=True)
        if s_ > 3 and dUs < tol:
            status = 'conv'; break

    ru, rv = rms_both(mesh, U, n)
    np.savez(f'{SC}/cavity3d_kz0_rkw3{TAG}.npz', U=U, xnod=mesh.xnod, ynod=mesh.ynod,
             dt=dt, rms=ru, rms_v=rv, status=status, steps=s_+1, kappa_p=kap)
    return (ru, rv), status, s_+1, time.perf_counter()-t0


ROWWEIGHT = 'rw' in sys.argv
TAG = ('_rw' if ROWWEIGHT else '') + ('_noac' if '0.0' in sys.argv[3:4] else '')

if __name__ == '__main__':
    cflt = float(sys.argv[1]) if len(sys.argv) > 1 else 0.8
    tmax = float(sys.argv[2]) if len(sys.argv) > 2 else 25.0
    kapf = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    print(f'3D solver at k_z = 0, cavity Re={RE:g}, {EX}x{EX} N={N}, RKW3/CN')
    print(f'GATE: RMS u vs Ghia should approach 1.568e-02 '
          f'(the measured 2D value on this mesh)\n')
    (ru, rv), status, steps, wall = run(cflt, tmax, kapf)
    print(f'\nRMS u = {ru:.4e}   target 1.568e-02   ratio {ru/1.5682e-02:.2f}')
    print(f'RMS v = {rv:.4e}   target 2.079e-02   ratio {rv/2.0790e-02:.2f}')
    print(f'[{status}, {steps} steps, {wall:.0f}s]')
