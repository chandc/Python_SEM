"""The mechanism of the least-squares time-step pathology, on a 1D model problem
whose fixed point can be written down in closed form.

THE POINT.  A least-squares (FOSLS) time step minimises, over the state U,

    J(U) = || a_mass*Pi_u U + a_flux*(A U - f) - hist*Pi_u U^n ||_W^2  +  || C U ||_W^2
             \\________________ momentum row ________________/          \\_ constraints _/

with Pi_u the velocity part, A the spatial momentum operator (grad p + nu curl w),
C the constraint rows (continuity, vorticity definition).  Every individual step
is a symmetric positive-definite problem.  The FIXED POINT of the step map is not.

With BDF1 (fac1 = 1) the history scaling equals the mass coefficient, hist = a_mass
=: m, so at a fixed point U* = U^n the momentum residual collapses to a_flux*(A U*-f)
and stationarity gives, after dividing by a_flux =: a,

    [ m Pi_u^H W A  +  a A^H W A  +  (1/a) C^H W C ] U*  =  m Pi_u^H W f + a A^H W f
      \\___________/    \\________/    \\___________/
       NON-symmetric     symmetric      symmetric
       cross term        Gauss-Newton   constraints

which is the equation this file solves directly (no time marching: the model is
linear, so the fixed point is one linear solve).  Three facts follow, and all
three are verified numerically below.

 1. THE FIXED POINT IS NOT THE MINIMISER OF ANY STEADY LEAST-SQUARES FUNCTIONAL.
    The cross term m*Pi_u^H W A is not symmetric; the steady functional
    a^2||AU-f||^2 + ||CU||^2 would give [a^2 A^H W A + C^H W C] U = a^2 A^H W f.

 2. THE WEIGHTING DECIDES WHICH TERM SURVIVES AS dt -> 0.  With c := m/a the
    mass coefficient,

      legacy   (w_mom = w_mass = dt):    m = 1,        a = dt
                cross : momentum : constraints  =  1  :  dt  :  1/dt
                -> momentum enters ONLY through the non-symmetric term, and is
                   dt^2 smaller than the constraints.  The fixed point is any
                   constraint-satisfying field, selected at O(dt^2).  The
                   selection is what appears as a mesh-scale mode.

      balanced (w_mom = w_mass = sqrt(dt)):  m = 1/sqrt(dt), a = sqrt(dt)
                cross : momentum : constraints  =  1/sqrt(dt) : sqrt(dt) : 1/sqrt(dt)
                -> cross term and constraints stay at the SAME order for every dt.
                   Multiplying through by sqrt(dt) gives a dt-independent limit
                   [Pi_u^H W A + C^H W C] U = Pi_u^H W f.

 3. THE ERROR IS PROPORTIONAL TO THE IRREDUCIBLE SPATIAL RESIDUAL.  If the
    discrete space can satisfy every row exactly (control case below), both
    weightings return that solution for every dt and there is no pathology.  The
    defect appears only when the rows cannot be satisfied simultaneously -- which
    is the C_1*dt*||R_h|| term of the error model.

THE MODEL.  One Fourier mode (wavenumber k in y) of the 2D velocity-vorticity-
pressure Stokes system, reduced to a two-point boundary-value problem in x, which
is exactly the per-mode structure of the 3D channel code:

    continuity    u_x + i k v
    vorticity     w - v_x + i k u
    momentum x    c u + p_x + nu (i k w)        - f1
    momentum y    c v + i k p - nu w_x          - f2

on x in [0,1] with u = v = 0 at both walls, discretised by C0 spectral elements
exactly as lssem3d does: residuals collocated at the GLL nodes of each element,
summed with the GLL quadrature weights, continuity imposed by assembly.

    uv run python scratch/model1d_fixedpoint.py
"""
import os
import sys

import numpy as np

_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R)
from lssem2d.lgl import lgl_nodes, lgl_weights, diff_matrix

NF = 4                      # u, v, p, omega
U_, V_, P_, W_ = range(NF)


# ----------------------------------------------------------------- discretisation

def mesh1d(E, N, L=1.0):
    """E equal C0 spectral elements of order N on [0, L].  Returns nodal
    coordinates, the element->global map, the element size and the GLL data."""
    xi = np.asarray(lgl_nodes(N), dtype=float)
    wq = np.asarray(lgl_weights(N), dtype=float)
    D = np.asarray(diff_matrix(N), dtype=float)
    h = L/E
    x = np.empty((E, N + 1))
    gid = np.empty((E, N + 1), dtype=int)
    for e in range(E):
        x[e] = e*h + 0.5*h*(xi + 1.0)
        gid[e] = np.arange(e*N, e*N + N + 1)          # last node of e == first of e+1
    return x, gid, h, wq, D


def blocks(E, N, k, nu, L=1.0):
    """Dense complex A (momentum spatial part), C (constraints), Pi_u (velocity
    selection) and the quadrature weight vectors, all mapping the GLOBAL dof
    vector to element-local collocated residuals."""
    x, gid, h, wq, D = mesh1d(E, N, L)
    n = N + 1
    ng = E*N + 1
    ndof = ng*NF
    nres = E*n                                        # one collocation point per node per row
    Dx = (2.0/h)*D

    A = np.zeros((2*nres, ndof), dtype=complex)       # momentum x, momentum y
    C = np.zeros((2*nres, ndof), dtype=complex)       # continuity, vorticity
    Pu = np.zeros((2*nres, ndof), dtype=complex)      # (u, v) at the same points
    wr = np.zeros(nres)

    g = lambda e, j, f: gid[e, j]*NF + f
    for e in range(E):
        for i in range(n):
            r = e*n + i
            wr[r] = wq[i]*h/2.0
            # continuity: u_x + i k v
            for j in range(n):
                C[r, g(e, j, U_)] += Dx[i, j]
            C[r, g(e, i, V_)] += 1j*k
            # vorticity: w - v_x + i k u
            C[nres + r, g(e, i, W_)] += 1.0
            for j in range(n):
                C[nres + r, g(e, j, V_)] -= Dx[i, j]
            C[nres + r, g(e, i, U_)] += 1j*k
            # momentum x spatial part: p_x + nu i k w
            for j in range(n):
                A[r, g(e, j, P_)] += Dx[i, j]
            A[r, g(e, i, W_)] += nu*1j*k
            # momentum y spatial part: i k p - nu w_x
            A[nres + r, g(e, i, P_)] += 1j*k
            for j in range(n):
                A[nres + r, g(e, j, W_)] -= nu*Dx[i, j]
            # velocity selection at the same collocation points
            Pu[r, g(e, i, U_)] = 1.0
            Pu[nres + r, g(e, i, V_)] = 1.0
    W2 = np.concatenate([wr, wr])                      # weights for a 2-row block
    return dict(x=x, gid=gid, ng=ng, ndof=ndof, nres=nres, A=A, C=C, Pu=Pu, W=W2, h=h)


def free_dofs(B, k, pin_p=None):
    """u = v = 0 at both walls; pressure pinned at one node when k = 0 (there the
    i k p path into the y-momentum row vanishes and p is defined up to a constant)."""
    ng, ndof = B['ng'], B['ndof']
    free = np.ones(ndof, dtype=bool)
    for nd in (0, ng - 1):
        free[nd*NF + U_] = False
        free[nd*NF + V_] = False
    if k == 0.0:
        free[(pin_p if pin_p is not None else ng//2)*NF + P_] = False
    return free


# ------------------------------------------------------------------ the two solves

def fixed_point(B, free, m, a, f):
    """Solve [ m Pu^H W A + a A^H W A + (1/a) C^H W C ] U = m Pu^H W f + a A^H W f,
    the fixed point of the least-squares step map (derivation in the docstring)."""
    A, C, Pu, W = B['A'], B['C'], B['Pu'], B['W']
    WA = W[:, None]*A
    K = m*(Pu.conj().T @ WA) + a*(A.conj().T @ WA) + (1.0/a)*(C.conj().T @ (W[:, None]*C))
    rhs = m*(Pu.conj().T @ (W*f)) + a*(A.conj().T @ (W*f))
    U = np.zeros(B['ndof'], dtype=complex)
    U[free] = np.linalg.solve(K[np.ix_(free, free)], rhs[free])
    return U, K[np.ix_(free, free)]


def steady_ls(B, free, a, f):
    """Minimiser of the STEADY functional a^2||AU-f||^2 + ||CU||^2 -- what the
    fixed point would be if the cross term were absent."""
    A, C, W = B['A'], B['C'], B['W']
    K = (a*a)*(A.conj().T @ (W[:, None]*A)) + (C.conj().T @ (W[:, None]*C))
    rhs = (a*a)*(A.conj().T @ (W*f))
    U = np.zeros(B['ndof'], dtype=complex)
    U[free] = np.linalg.solve(K[np.ix_(free, free)], rhs[free])
    return U


# --------------------------------------------------------------- manufactured data

def manufactured(B, k, nu, kind):
    """Nodal exact state and the forcing.  'poly': psi and p are polynomials of
    low degree, so every discrete row can be satisfied exactly (the control).
    'trig': psi = sin^2(pi x), not in the space -- an irreducible residual."""
    x, gid, ng = B['x'], B['gid'], B['ng']
    xg = np.zeros(ng)
    for e in range(x.shape[0]):
        xg[gid[e]] = x[e]
    if kind == 'poly':
        psi = xg**2*(1 - xg)**2
        dpsi = 2*xg*(1 - xg)**2 - 2*xg**2*(1 - xg)
        d2psi = 2*(1 - xg)**2 - 8*xg*(1 - xg) + 2*xg**2
        d3psi = -12*(1 - xg) + 12*xg
        p = xg.copy()
        dp = np.ones_like(xg)
    else:
        pi = np.pi
        psi = np.sin(pi*xg)**2
        dpsi = pi*np.sin(2*pi*xg)
        d2psi = 2*pi*pi*np.cos(2*pi*xg)
        d3psi = -4*pi**3*np.sin(2*pi*xg)
        p = np.cos(pi*xg)
        dp = -pi*np.sin(pi*xg)
    u = 1j*k*psi                                   # from the streamfunction: div u = 0
    v = -dpsi
    om = -d2psi + k*k*psi                          # omega = v_x - u_y
    dom = -d3psi + k*k*dpsi
    U = np.zeros(B['ndof'], dtype=complex)
    U[np.arange(ng)*NF + U_] = u
    U[np.arange(ng)*NF + V_] = v
    U[np.arange(ng)*NF + P_] = p
    U[np.arange(ng)*NF + W_] = om
    # forcing that makes U the steady solution: f = A U (analytically)
    f1 = dp + nu*1j*k*om
    f2 = 1j*k*p - nu*dom
    nres = B['nres']
    f = np.zeros(2*nres, dtype=complex)
    for e in range(B['x'].shape[0]):
        for i in range(B['x'].shape[1]):
            r = e*(B['x'].shape[1]) + i
            nd = gid[e, i]
            f[r] = f1[nd]
            f[nres + r] = f2[nd]
    return U, f


def zigzag(B, U, field=V_):
    """Node-to-node sign alternations of the increment along x -- the discrete
    signature of a mesh-scale mode."""
    ng = B['ng']
    q = U[np.arange(ng)*NF + field].real
    d = np.diff(q)
    if d.size < 2:
        return 0.0, 0
    s = np.sign(d)
    return float(np.mean(np.abs(d))), int(np.sum(s[1:]*s[:-1] < 0))


# ------------------------------------------------------------------------- driver

def run(E=4, N=4, k=2.0, nu=1.0/180.0, kind='trig', dts=(1e-1, 1e-2, 1e-3, 1e-4, 1e-5)):
    B = blocks(E, N, k, nu)
    free = free_dofs(B, k)
    Ue, f = manufactured(B, k, nu, kind)
    A, C, W = B['A'], B['C'], B['W']
    r_mom = np.sqrt(np.sum(W*np.abs(A @ Ue - f)**2))
    r_con = np.sqrt(np.sum(W*np.abs(C @ Ue)**2))
    nrm = np.sqrt(np.sum(np.abs(Ue)**2))
    print(f'--- {kind}: E={E} N={N} k={k} nu={nu:g}   irreducible residual at the exact state: '
          f'momentum {r_mom:.2e}, constraints {r_con:.2e}')
    print(f'{"dt":>8} | {"legacy err":>11} {"sym defect":>11} {"cond":>9} {"|dv| zz":>9} {"sign":>5} | '
          f'{"balanced err":>12} {"sym defect":>11} {"cond":>9} {"|dv| zz":>9} {"sign":>5} | {"steady-LS err":>13}')
    out = {}
    for dt in dts:
        row = [f'{dt:8.0e} |']
        for name, (m, a) in (('legacy', (1.0, dt)), ('balanced', (1.0/np.sqrt(dt), np.sqrt(dt)))):
            U, K = fixed_point(B, free, m, a, f)
            err = np.sqrt(np.sum(np.abs(U - Ue)**2))/nrm
            sym = np.linalg.norm(K - K.conj().T)/np.linalg.norm(K)
            cond = np.linalg.cond(K)
            zz, sg = zigzag(B, U - Ue)
            out[(name, dt)] = (err, sym, cond, zz, sg)
            row.append(f'{err:11.3e} {sym:11.2e} {cond:9.1e} {zz:9.2e} {sg:5d} |')
        Us = steady_ls(B, free, np.sqrt(dt), f)
        row.append(f'{np.sqrt(np.sum(np.abs(Us - Ue)**2))/nrm:13.3e}')
        print(' '.join(row), flush=True)
    return B, out


if __name__ == '__main__':
    print(__doc__.split('THE MODEL')[0].strip()[:0] or '', end='')
    print('LEAST-SQUARES TIME-STEP FIXED POINT, 1D Fourier-mode Stokes model\n')
    run(kind='poly', E=2, N=6)
    print()
    run(kind='trig', E=4, N=4)
    print()
    run(kind='trig', E=8, N=6)
