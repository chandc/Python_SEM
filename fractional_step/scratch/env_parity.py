"""Do the Anaconda 3.9 and the repo's uv .venv 3.12 give the SAME solver answer?

    python           scratch/env_parity.py anaconda     # bare `python` -> Anaconda 3.9
    uv run --quiet python scratch/env_parity.py venv     # repo .venv 3.12
    uv run --quiet python scratch/env_parity.py compare  # diff the two

The BFS runs in this study were launched with bare `python`, which resolves to
Anaconda 3.9, while requirements.txt documents a uv-managed .venv on 3.12.  Those
carry different numpy/scipy/BLAS, so this checks whether that changed any result.

20 BDF steps on the ER-1.94 short grid, P+Z outlet, identical settings to
armaly_run.py.  Writes envparity_<tag>.npz -- never touches armaly_*.npz.
"""
import os, sys, time, hashlib
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
sys.path.insert(0, SC)
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np

STEPS = 20


def env():
    import scipy
    return dict(python=sys.version.split()[0], exe=sys.executable,
                numpy=np.__version__, scipy=scipy.__version__)


def run(tag):
    import lssem2d
    lssem2d.set_backend('numpy')
    from fgrid import load
    from lssem2d.lgl import diff_matrix
    from lssem2d.lssem import SolverState
    import lssem2d.solver as S
    import lssem2d.bc as BC
    from armaly_run import GRIDS, NU

    OB = BC.apply_bc
    m, _, _ = load(GRIDS['short']); N = m.N; n = N+1
    D = diff_matrix(N)
    xmax = m.xnod.max()
    out = [e for e in range(m.nelem) if abs(m.xnod[e, -1]-xmax) < 1e-9]

    def bc2(mesh, U, **kw):
        U = OB(mesh, U, **kw)
        for e in out:
            U[e, -1, :, 3] = -(D[-1, :-1] @ U[e, :-1, :, 3])/D[-1, -1]
        return U

    from armaly_run import inlet_profile
    st = SolverState(m, D, nu=NU, dt=1.0, fac1=1.0, w_mom=1.0, w_mass=1.0)
    st.get_global_mask(pin_p=False)
    for e in out:
        st._global_mask[e, -1, :, 3] = 0.0
    S.apply_bc = bc2
    inl = lambda x, y, t: inlet_profile(y)
    U = np.zeros((m.nelem, n, n, 4)); h = [U.copy()]
    t0 = time.perf_counter()
    try:
        for s in range(STEPS):
            U = S.step_bdf(st, h, time=s*1.0, max_newton=1, newton_tol=1e-12,
                           newton_factor=0.0, custom_inlet=inl, pin_p=False,
                           cgsfac=1e-3, cg_tol=1e-6, cg_max_iter=300000)
    finally:
        S.apply_bc = OB
    e = env()
    np.savez(f'{SC}/envparity_{tag}.npz', U=U, **{k: str(v) for k, v in e.items()})
    d = hashlib.sha256(np.ascontiguousarray(U).tobytes()).hexdigest()[:16]
    print(f"[{tag}] python {e['python']}  numpy {e['numpy']}  scipy {e['scipy']}")
    print(f"        {e['exe']}")
    print(f"        {STEPS} steps in {time.perf_counter()-t0:.0f}s")
    print(f"        max|u| = {np.abs(U[...,0]).max():.12f}   sha256[:16] = {d}")


def compare():
    a = np.load(f'{SC}/envparity_anaconda.npz')
    b = np.load(f'{SC}/envparity_venv.npz')
    Ua, Ub = a['U'], b['U']
    print(f"anaconda: python {a['python']} numpy {a['numpy']} scipy {a['scipy']}")
    print(f"venv    : python {b['python']} numpy {b['numpy']} scipy {b['scipy']}")
    d = np.abs(Ua-Ub)
    print(f"\nmax|dU|            = {d.max():.6e}")
    print(f"max relative       = {(d.max()/max(np.abs(Ua).max(),1e-300)):.6e}")
    print(f"bit-identical      = {np.array_equal(Ua, Ub)}")
    for k, nm in enumerate(('u', 'v', 'p', 'omega')):
        print(f"  {nm:>5}: max|d| = {d[...,k].max():.4e}   max|val| = {np.abs(Ua[...,k]).max():.6f}")


def converge(tag, domain='short'):
    """Run to steady state and report the two numbers the study quotes.

    Anaconda 3.9 reference values, from armaly_run.py / armaly_short_pz_tol.py:
        short : x_r/S = 5.174, du/dy_outlet = +1.41057, max|u| = 1.5000
        long  : x_r/S = 8.145  (the Armaly validation number, 1.2% vs measured 8.05)
    """
    import lssem2d
    lssem2d.set_backend('numpy')
    from fgrid import load
    from lssem2d.lgl import diff_matrix
    from lssem2d.lssem import SolverState
    import lssem2d.solver as S
    import lssem2d.bc as BC
    from armaly_run import GRIDS, NU, S_STEP, inlet_profile, reattach

    OB = BC.apply_bc
    m, _, _ = load(GRIDS[domain]); N = m.N; n = N+1
    D = diff_matrix(N)
    xmax = m.xnod.max()
    out = [e for e in range(m.nelem) if abs(m.xnod[e, -1]-xmax) < 1e-9]

    def bc2(mesh, U, **kw):
        U = OB(mesh, U, **kw)
        for e in out:
            U[e, -1, :, 3] = -(D[-1, :-1] @ U[e, :-1, :, 3])/D[-1, -1]
        return U

    st = SolverState(m, D, nu=NU, dt=1.0, fac1=1.0, w_mom=1.0, w_mass=1.0)
    st.get_global_mask(pin_p=False)
    for e in out:
        st._global_mask[e, -1, :, 3] = 0.0
    S.apply_bc = bc2
    inl = lambda x, y, t: inlet_profile(y)
    U = np.zeros((m.nelem, n, n, 4)); h = [U.copy()]
    t0 = time.perf_counter(); status = 'CAP'; d = np.nan
    try:
        for s in range(1200):
            prev = h[0].copy()
            U = S.step_bdf(st, h, time=s*1.0, max_newton=1, newton_tol=1e-12,
                           newton_factor=0.0, custom_inlet=inl, pin_p=False,
                           cgsfac=1e-3, cg_tol=1e-6, cg_max_iter=300000)
            if not np.all(np.isfinite(U)):
                status = 'NaN'; break
            d = float(np.abs(U-prev).max())
            if np.abs(U[..., 0]).max() > 20.0:
                status = 'BLEWUP'; break
            if d < 1e-11:
                status = 'conv'; break
    finally:
        S.apply_bc = OB
    e = env()
    np.savez(f'{SC}/envparity_conv_{domain}_{tag}.npz', U=U, xnod=m.xnod,
             ynod=m.ynod, hy=m.hy, N=N, status=status, steps=s+1, dU=d,
             **{k: str(v) for k, v in e.items()})
    xr = reattach(U, m.xnod, m.ynod, m.hy, N)
    tw = np.nan
    for el in out:
        if m.ynod[el, 0] < 0.01:
            tw = float(np.dot(D[0, :], U[el, -1, :, 0])*(2.0/m.hy[el]))
    print(f"[{tag}/{domain}] python {e['python']} numpy {e['numpy']} scipy {e['scipy']}")
    print(f"   {status}  {s+1} steps  |dU| = {d:.3e}  max|u| = {np.abs(U[...,0]).max():.6f}")
    print(f"   du/dy_outlet = {tw:+.5f}   x_r/S = {xr/S_STEP:.4f}   "
          f"{time.perf_counter()-t0:.0f}s")


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'venv'
    if what == 'compare':
        compare()
    elif what.startswith('conv:'):
        _, tag, dom = what.split(':')
        converge(tag, dom)
    else:
        run(what)
