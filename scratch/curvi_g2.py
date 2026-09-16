"""Gate G2 — spectral convergence of a manufactured solution on a deformed mesh.

    uv run python scratch/curvi_g2.py

CURVILINEAR_2D_PLAN.md G2.  G0 and G1 test the geometry and the operator with
answers known exactly; neither of them solves anything.  This is the first gate
that inverts the system, and it asks the one question that separates a correct
curvilinear discretisation from a plausible one:

    does the error fall EXPONENTIALLY in N on a curved mesh?

Algebraic decay is the signature of inconsistent metrics.  A metric error acts
like a low-order quadrature or geometry error: it is invisible at coarse
resolution, where the solution error dominates, and it FLOORS the convergence
once the solution error drops beneath it.  So the gate is the slope at high N,
not the error at any single N -- which is why the affine mesh is run alongside
as the control, with the identical solution, forcing and solver.

THREE DECISIONS THAT MAKE THE MEASUREMENT MEAN WHAT IT SAYS.

**The solution is steady**, so BDF2 contributes exactly zero.  A time-dependent
manufactured solution would mix a temporal error of fixed order into a spatial
error falling exponentially, and the mixture would floor -- looking exactly like
the metric bug this gate exists to find.  One step at dt = 1e6 from exactly
seeded history is a steady Newton solve: the mass term is 1e-6 of the operator
and perturbs the answer by 1e-6 OF THE ERROR ITSELF.  The run repeats at
dt = 1e4 to show that number is not doing anything.

**The linear solves are exact**, by QR of the least-squares operator.  A CG
tolerance that is generous at N = 4 silently becomes the error at N = 14 -- again
a floor, again mistakable for the bug.  Assembling the NORMAL equations and
inverting them would be no better: A = L^T W L squares the condition number, and
measures 1.2e16 already at N = 4.  So the rectangular operator is assembled
instead, from `apply_L` itself, and its triangular QR factor solves the Newton
system -- the same operator the iterative path applies, assembled, with the
condition number of L rather than of L^T L.  The patch preconditioner on
curvilinear meshes is gate G5's business, deliberately not entangled here.

**THE ROTATED MESH, NOT THE DEFORMED ONE, IS WHAT TESTS `bc.py`.**  Worth
stating because the figure makes it look otherwise.  `curvi.deform`'s
perturbation is a product of sines that vanish on the domain boundary, so the
DOMAIN is still exactly the unit square and only the interior element interfaces
bend -- boundary nodes have not moved, and their tensor-product coordinates are
still correct.  The rotated mesh is the one whose edges are no longer axis
aligned: x and y both vary along every edge, the old tensor-product form
(`xnod[e,0]` as a single x for the whole west edge) is simply wrong there, and
the rotated column would not converge at all if `bc.py` were still using it.
Gate G3's annulus then tests it on a boundary that is genuinely curved.
"""
import os
import sys
import time as _time

_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R)
os.chdir(_R)

import numpy as np

import lssem2d
from lssem2d import curvi
from lssem2d.lgl import diff_matrix
from lssem2d.lssem import SolverState, ls_coeffs
from lssem2d.mesh import build_channel
from lssem2d.solver import apply_A
import lssem2d.solver as S

OUT = os.path.join(_R, 'figs', 'curvi_g2.png')
NU = 0.05
DT = 1.0e6


# ------------------------------------------------------------ manufactured data

def _fields():
    """Steady (u, v, p, omega) and the forcing that makes them exact.

    Same stream function as scratch/mms2d_temporal.py with the cos(t) removed,
    so the two studies are directly comparable.  Momentum rows carry
    grad p + nu curl omega, matching lssem2d.lssem's a13/a14 and a23/a24.
    """
    import sympy as sp
    x, y, nu = sp.symbols('x y nu', real=True)
    psi = sp.sin(sp.pi*x)**2*sp.sin(sp.pi*y)**2
    u = sp.diff(psi, y)
    v = -sp.diff(psi, x)
    p = sp.cos(sp.pi*x)*sp.cos(sp.pi*y)
    om = sp.diff(v, x) - sp.diff(u, y)
    f1 = u*sp.diff(u, x) + v*sp.diff(u, y) + sp.diff(p, x) + nu*sp.diff(om, y)
    f2 = u*sp.diff(v, x) + v*sp.diff(v, y) + sp.diff(p, y) - nu*sp.diff(om, x)
    L = lambda e: sp.lambdify((x, y, nu), sp.simplify(e), 'numpy')
    return dict(u=L(u), v=L(v), p=L(p), om=L(om), f1=L(f1), f2=L(f2))


_F = None


def coords(m):
    """Physical node coordinates, curvilinear or not."""
    if getattr(m, 'curvilinear', False):
        return m.X, m.Y
    X = m.xnod[:, :, None] + 0.0*m.ynod[:, None, :]
    Y = m.ynod[:, None, :] + 0.0*m.xnod[:, :, None]
    return X, Y


def exact(m, t=0.0):
    global _F
    if _F is None:
        _F = _fields()
    X, Y = coords(m)
    U = np.empty(X.shape + (4,))
    for i, k in enumerate(('u', 'v', 'p', 'om')):
        U[..., i] = _F[k](X, Y, NU)
    return U


def forcing(m, t=0.0):
    global _F
    if _F is None:
        _F = _fields()
    X, Y = coords(m)
    f = np.zeros(X.shape + (4,))
    f[..., 0] = _F['f1'](X, Y, NU)
    f[..., 1] = _F['f2'](X, Y, NU)
    return f


def errors(m, U, Ue):
    """Weighted L2 relative errors; pressure with its mean removed (it is
    determined only up to a constant, and the solver pins a node the exact
    field knows nothing about)."""
    w = m.wq
    tot = w.sum()
    out = {}
    for i, k in enumerate(('u', 'v', 'p', 'om')):
        a, b = U[..., i], Ue[..., i]
        if k == 'p':
            a = a - (a*w).sum()/tot
            b = b - (b*w).sum()/tot
        out[k] = float(np.sqrt((w*(a - b)**2).sum())/max(np.sqrt((w*b**2).sum()), 1e-300))
    num = np.sqrt((w*((U[..., 0] - Ue[..., 0])**2 + (U[..., 1] - Ue[..., 1])**2)).sum())
    den = np.sqrt((w*(Ue[..., 0]**2 + Ue[..., 1]**2)).sum())
    out['uv'] = float(num/den)
    return out


# --------------------------------------------------------------- exact solver --

def exact_factory(state, pin_p=True):
    """A `precond_factory` that solves the Newton system EXACTLY, by QR.

    NOT by inverting the normal equations.  A = L^T W L has the condition number
    of L squared -- 1.2e16 already at N = 4 with the legacy row weighting -- so
    assembling A and inverting it throws away more digits than the gate is
    trying to measure, and the convergence curve would floor on the solver
    rather than on the discretisation.  Instead the RECTANGULAR least-squares
    operator is assembled,

        Lh = sqrt(wq) * L_plain * mask * expand ,     A = Lh^T Lh  exactly,

    and `A g = b` is solved through the triangular factor of its QR: R^T y = b,
    R g = y.  R carries the condition number of L, not of A, so the formation
    loss is gone and only the (unavoidable) backsolve remains.

    `apply_L` returns rows already multiplied by wq, so the square root is
    recovered by dividing by sqrt(wq) -- this is the same operator the iterative
    path applies, assembled, not a second derivation of it.
    """
    m = state.mesh
    gidx = m.gidx
    ng = int(gidx.max()) + 1
    nf = 4
    rep = np.zeros((ng, 3), int)
    for e in range(m.nelem):
        for i in range(m.nterm):
            for j in range(m.nterm):
                rep[gidx[e, i, j]] = (e, i, j)
    re_, ri, rj = rep[:, 0], rep[:, 1], rep[:, 2]
    sw = np.sqrt(m.wq)[..., None]
    diag = {}

    def factory(st, fu, fv, M_inv, pp):
        from lssem2d.lssem import apply_L
        mask = st.get_global_mask(pin_p=pp)
        free = mask[re_, ri, rj, :].ravel() > 0.5
        nfree = int(free.sum())
        cols = np.zeros((m.nelem*m.nterm*m.nterm*nf, nfree))
        dU = np.zeros((m.nelem, m.nterm, m.nterm, nf))
        c = 0
        for g in range(ng):
            sel = gidx == g
            for f in range(nf):
                if not free[g*nf + f]:
                    continue
                dU[...] = 0.0
                dU[..., f][sel] = 1.0
                cols[:, c] = (apply_L(st, dU*mask, fu, fv)/sw).ravel()
                c += 1
        Q, R = np.linalg.qr(cols)
        diag['cond'] = float(np.linalg.cond(R))

        def solve(r):
            bg = r[re_, ri, rj, :].ravel()[free]
            y = np.linalg.solve(R.T, bg)
            g_ = np.linalg.solve(R, y)
            z = np.zeros(ng*nf)
            z[free] = g_
            return z.reshape(ng, nf)[gidx]
        return solve
    factory.diag = diag
    return factory


# ------------------------------------------------------------------------ run --

def make_mesh(N, E, kind, amp=0.10):
    m = build_channel(1.0, 1.0, E, E, N, bcs=(1, 1, 1, 1))
    m.compute_global_indices()
    if kind == 'deformed':
        curvi.deform(m, amp=amp, kx=2, ky=1)
    elif kind == 'rotated':
        curvi.rotate(m, np.pi/6)
    elif kind == 'curvi-affine':
        curvi.attach(m, *curvi.affine_coords(m))
    return m


def solve(N, E, kind, dt=DT, newton=8, amp=0.10):
    m = make_mesh(N, E, kind, amp)
    # w_mom = 1 puts the momentum row at weight 1 and the mass term at 1/dt.
    # The legacy default instead scales the whole momentum row by dt, which at
    # dt = 1e6 is a factor 1e6 of row imbalance and 1e12 of condition number --
    # measured at 1.2e16 for the assembled normal equations.  Same steady limit,
    # no manufactured ill-conditioning.
    st = SolverState(m, diff_matrix(N), nu=NU, dt=dt, fac1=1.0, w_mom=1.0)
    _, a_flux, _ = ls_coeffs(st)
    fac = exact_factory(st)
    st.precond_factory = fac
    # CG IS BYPASSED, NOT TUNED.  `pcg_solve` returns a ZERO increment whenever
    # p^T A p falls below a hardcoded 1e-20 -- which happens as soon as the
    # residual reaches ~1e-10, i.e. exactly the regime this gate lives in at
    # high N.  At N = 14 that silently reported an error of exactly 0.  The
    # preconditioner above is already the exact solve, so the Krylov wrapper has
    # nothing to add here and its floors have everything to lose.
    _orig_pcg = S.pcg_solve

    def _direct(state, b, fu_, fv_, M_inv, mw, pin_p=False, max_iter=5000,
                tol=1e-6, cgsfac=0.0, precond=None):
        return precond(b), 1
    S.pcg_solve = _direct
    Ue = exact(m)
    h = [Ue.copy(), Ue.copy()]
    f = a_flux*forcing(m)
    t0 = _time.perf_counter()
    S.step_bdf(st, h, time=0.0, max_newton=newton, newton_tol=1e-14,
               newton_factor=0.0, f_known=f, pin_p=True,
               exact_solution=lambda x, y, t: tuple(
                   _F[k](x, y, NU) for k in ('u', 'v', 'p', 'om')),
               cgsfac=0.0, cg_tol=1e-30, cg_max_iter=4, line_search=False)
    S.pcg_solve = _orig_pcg
    e = errors(m, h[0], Ue)
    e['wall'] = _time.perf_counter() - t0
    e['ndof'] = int(m.gidx.max() + 1)*4
    e['cond'] = fac.diag.get('cond', float('nan'))
    return e


def main():
    lssem2d.set_backend('numpy')
    global _F
    _F = _fields()
    E = 2
    Ns = [4, 6, 8, 10, 12, 14]
    kinds = ['affine', 'curvi-affine', 'rotated', 'deformed']
    res = {k: [] for k in kinds}
    print(f'{"N":>3s} ' + ' '.join(f'{k:>14s}' for k in kinds) + '   (|u| rel L2)')
    for N in Ns:
        row = []
        for k in kinds:
            e = solve(N, E, k)
            res[k].append(e)
            row.append(f'{e["uv"]:14.3e}')
        print(f'{N:3d} ' + ' '.join(row))

    # dt insensitivity: the steady limit is a limit, not a tuning knob
    print('\ndt insensitivity on the deformed mesh (N = 10):')
    for dt in (1e4, 1e6, 1e8):
        print(f'   dt = {dt:8.0e}   |u| err = {solve(10, E, "deformed", dt=dt)["uv"]:.6e}')

    # EXPONENTIAL OR ALGEBRAIC?  That, not the size of the error, is the gate.
    # A metric bug does not make the error large at coarse resolution -- the
    # solution error hides it -- it changes the SHAPE of the curve, flooring an
    # exponential into an algebraic tail.  So both models are fitted over the
    # same points and compared by how well they explain the data:
    #     exponential   log e = a - b N        (straight on a log-linear axis)
    #     algebraic     log e = a - q log N    (straight on a log-log axis)
    def fits(Ns_, es):
        x, y = np.asarray(Ns_, float), np.log(np.asarray(es))
        out = {}
        for tag, xx in (('exp', x), ('alg', np.log(x))):
            c = np.polyfit(xx, y, 1)
            r = y - np.polyval(c, xx)
            out[tag] = (-c[0], 1.0 - r.var()/y.var())
        return out

    print('\nconvergence model over N = 6..14   '
          '(exponential rate b, algebraic order q, R^2 of each fit):')
    rates, quality = {}, {}
    for k in kinds:
        f = fits(Ns[1:], [r['uv'] for r in res[k][1:]])
        rates[k], quality[k] = f['exp'][0], f
        print(f'   {k:14s} b = {f["exp"][0]:5.2f} (R2 {f["exp"][1]:.4f})   '
              f'q = {f["alg"][0]:5.1f} (R2 {f["alg"][1]:.4f})   '
              f'final error {res[k][-1]["uv"]:.2e}')

    reg = abs(res['affine'][-1]['uv'] - res['curvi-affine'][-1]['uv'])
    ok = all(quality[k]['exp'][1] > 0.99 and quality[k]['exp'][1] > quality[k]['alg'][1]
             and rates[k] > 0.8 for k in ('rotated', 'deformed')) and reg < 1e-14
    print(f'\ngate: on BOTH curved meshes the exponential model fits better than the '
          f'algebraic one\n      with R^2 > 0.99 and rate > 0.8, and the curvilinear path '
          f'reproduces the affine\n      one (delta {reg:.1e}) -> {"PASS" if ok else "FAIL"}')
    figure(Ns, res, rates, E)
    return 0 if ok else 1


def figure(Ns, res, rates, E):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(13, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.25, 1])

    ax = fig.add_subplot(gs[0, 0])
    m = make_mesh(8, E, 'deformed')
    n = m.nterm
    for e in range(m.nelem):
        for idx in (0, n-1):
            ax.plot(m.X[e, idx, :], m.Y[e, idx, :], 'k-', lw=0.9)
            ax.plot(m.X[e, :, idx], m.Y[e, :, idx], 'k-', lw=0.9)
    ax.plot(m.X.ravel(), m.Y.ravel(), '.', ms=1.5, color='C0', alpha=0.6)
    ma = make_mesh(8, E, 'affine')
    for e in range(ma.nelem):
        for idx in (0, n-1):
            ax.plot(np.repeat(ma.xnod[e, idx], n), ma.ynod[e, :], color='C7', ls='--', lw=0.8)
            ax.plot(ma.xnod[e, :], np.repeat(ma.ynod[e, idx], n), color='C7', ls='--', lw=0.8)
    ax.set_aspect('equal'); ax.tick_params(labelsize=8)
    ax.set_title('the deformed mesh (10 %), affine dashed', fontsize=10)

    ax = fig.add_subplot(gs[0, 1])
    style = {'affine': ('C7', 'o', '--'), 'curvi-affine': ('C0', 's', '-'),
             'rotated': ('C2', '^', '-'), 'deformed': ('C3', 'D', '-')}
    for k, v in res.items():
        c, mk, ls = style[k]
        ax.semilogy(Ns, [r['uv'] for r in v], marker=mk, ls=ls, color=c,
                    label=f'{k}  (b = {rates[k]:.2f})', ms=5)
    ax.set_xlabel('polynomial order $N$'); ax.set_ylabel(r'relative $L^2$ error in $\mathbf{u}$')
    ax.set_title('G2: exponential convergence survives the deformation', fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which='both')

    ax = fig.add_subplot(gs[0, 2])
    for f, c in zip(('u', 'p', 'om'), ('C0', 'C1', 'C4')):
        ax.semilogy(Ns, [r[f] for r in res['deformed']], 'o-', color=c, ms=4, label=f)
    ax.set_xlabel('polynomial order $N$'); ax.set_ylabel('relative $L^2$ error')
    ax.set_title('deformed mesh, by field', fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which='both')

    fig.suptitle('Gate G2 — steady manufactured solution, exact linear solves: '
                 'the slope at high $N$ is the test', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=140)
    print(f'wrote {OUT}')


if __name__ == '__main__':
    sys.exit(main())
