"""PHASE 1: does the projection splitting preserve second-order accuracy?

    python scratch/fs_phase1.py

Two tests, deliberately in this order.

  1a PERIODIC STOKES.  Triply periodic, no convection, solenoidal initial
     field u = (sin x cos y, -cos x sin y, 0).  For this mode grad^2 u = -2u,
     so the exact solution decays as exp(-2 nu t) and the pressure is
     identically zero.  A projection scheme must reproduce the decay rate
     EXACTLY and must produce phi ~ 0: if it does not, the machinery is wrong
     before any question of splitting error arises.

  1b GATE 1, the project's own test.  Stokes decay in a CHANNEL against the
     analytic sigma = 9.3137399, which the least-squares path passes at order
     2.00 (sigma 9.3153041 / 9.3141300 / 9.3138373 at dt = 0.01 / 0.005 /
     0.0025).  This exercises the WALL treatment as well as the splitting,
     which is why 1a runs first -- a failure here alone points at walls.
"""
import sys
sys.path.insert(0, '.'); sys.path.insert(0, 'scratch')
import numpy as np
import lssem3d; lssem3d.set_backend('numpy')
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem3d import project as PJ, helmholtz as HH, fourier as FR
from lssem3d import solver3d as S3, timestep as T

L = 2*np.pi


def make(N, ex, ey, nz, nu, wall=False, tol=1e-11):
    m = build_channel(L, L, ex, ey, N, bcs=(0, 0, 0, 0))
    m.periodic_x = L
    if not wall:
        m.periodic_y = L
    m.compute_global_indices()
    nk = nz//2 + 1
    n = N + 1
    X = np.empty((m.nelem, n, n)); Y = np.empty_like(X)
    for e in range(m.nelem):
        X[e] = m.xnod[e][:, None]; Y[e] = m.ynod[e][None, :]
    mask_u = PJ.build_masks(m, nk, nz, 3, wall=wall)
    mask_p = PJ.build_masks(m, nk, nz, 1, wall=False)
    ind = np.zeros(mask_p.shape); ind[0, 0, 0, 0, 0] = 1.0
    mask_p[..., 0, 0] *= (S3.gs(m, ind)[..., 0, 0] < 0.5)   # pin p at kz = 0
    kz = FR.wavenumbers(nz, L)
    s = dict(m=m, D=diff_matrix(N), N=N, nz=nz, nk=nk, nu=nu, kz=kz,
             X=X, Y=Y, mask_u=mask_u, mask_p=mask_p, tol=tol,
             wq3=m.wq[..., None, None], wq1=m.wq[..., None, None])
    s['Mu'] = HH.fdm_preconditioner(m, N, 0.0, nu, mask_u, 6, nk)  # rebuilt per dt
    s['Mp'] = HH.fdm_preconditioner(m, N, kz**2, 1.0, mask_p, 2, nk)
    return s


def rebuild_u_precond(s, dt, k):
    lam = T.implicit_coeff(dt, k) + s['nu']*(s['kz']**2)
    s['Mu'] = HH.fdm_preconditioner(s['m'], s['N'], lam, s['nu'],
                                    s['mask_u'], 6, s['nk'])


def energy(s, Uc):
    wz = L/s['nz']
    return float(0.5*wz*np.sum(np.abs(Uc)**2 * s['m'].wq[..., None, None]))


print("1a  PERIODIC STOKES: exact decay exp(-2 nu t), pressure identically 0")
nu = 0.05
print("\n  (i) fixed dt, raise N -- if the error is SPATIAL it falls spectrally")
for N in (6, 8, 10, 12):
    s_ = make(N, 2, 2, 4, nu)
    nk = s_["nk"]
    Uc = np.zeros((s_["m"].nelem, N+1, N+1, 3, nk), dtype=complex)
    Uc[..., 0, 0] = np.sin(s_["X"])*np.cos(s_["Y"])*s_["nz"]
    Uc[..., 1, 0] = -np.cos(s_["X"])*np.sin(s_["Y"])*s_["nz"]
    pc = np.zeros((s_["m"].nelem, N+1, N+1, 1, nk), dtype=complex)
    Z = np.zeros_like(Uc); E0 = energy(s_, Uc); dt = 0.02
    nstep = int(round(0.4/dt))
    for i_ in range(nstep):
        for k in range(T.NSTAGE):
            rebuild_u_precond(s_, dt, k)
            Uc, pc, _ = PJ.substage(s_, Uc, pc, Z, Z, k, dt)
    sig = -0.5*np.log(energy(s_, Uc)/E0)/(nstep*dt)
    print(f"   N={N:>3}  sigma = {sig:.10f}  rel err {abs(sig-2*nu)/(2*nu):.3e}"
          f"   max|p| = {np.abs(pc).max():.2e}")

print("\n  (ii) N=12 (spatially converged), sweep dt -- temporal order")
errs = []
for dt in (0.05, 0.025, 0.0125):
    s_ = make(12, 2, 2, 4, nu)
    nk = s_["nk"]
    Uc = np.zeros((s_["m"].nelem, 13, 13, 3, nk), dtype=complex)
    Uc[..., 0, 0] = np.sin(s_["X"])*np.cos(s_["Y"])*s_["nz"]
    Uc[..., 1, 0] = -np.cos(s_["X"])*np.sin(s_["Y"])*s_["nz"]
    pc = np.zeros((s_["m"].nelem, 13, 13, 1, nk), dtype=complex)
    Z = np.zeros_like(Uc); E0 = energy(s_, Uc)
    nstep = int(round(0.4/dt))
    for i_ in range(nstep):
        for k in range(T.NSTAGE):
            rebuild_u_precond(s_, dt, k)
            Uc, pc, _ = PJ.substage(s_, Uc, pc, Z, Z, k, dt)
    sig = -0.5*np.log(energy(s_, Uc)/E0)/(nstep*dt)
    e = abs(sig-2*nu)/(2*nu); errs.append(e)
    print(f"   dt={dt:<8g} sigma = {sig:.10f}   rel err {e:.3e}")
if len(errs) > 2 and errs[-1] > 0:
    print(f"   observed order in dt = {np.log2(errs[0]/errs[-1])/2:.2f}")
