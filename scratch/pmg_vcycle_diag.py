"""Is the V-cycle reducing error at all on the CHANNEL operator?

The direct diagnostic, not a timing.  Run PMG as a STATIONARY iteration --
x <- x + M(b - Ax) -- and watch ||r||.  A working V-cycle gives a reduction
factor of 0.05-0.3 per cycle.  sec 7J saw "exactly 1.0000" at w7=1; if that is
what comes back at w7=1e-4 too, PMG is stalling and no amount of tuning the
smoother degree will help.

Also prints the c actually in force: c = 1/(beta_k*dt) = 5405 at dt=8e-4, TEN
TIMES the c=525 that sec 7Q studied.  sec 7Q found PMG behaviour is a REGIME
effect of c, so this is the regime that matters, not the one measured.
"""
import os, sys, time
for _v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '8')
_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R,'scratch')); os.chdir(_R)
os.environ.setdefault('LSSEM3D_BACKEND', 'numba')
import numpy as np

def main():
    import lssem3d; lssem3d.set_backend(os.environ['LSSEM3D_BACKEND'])
    from lssem3d import operator as OP, precond as P3, solver3d as S3, timestep as T
    import channel3d as C, minchan as MC
    s = MC.setup(); dt = 8.0e-4
    cc = T.implicit_coeff(dt, 0)
    rw = OP.momentum_row_weights(cc)
    print(f'channel N=8, 108 elems, nk={s["nk"]}; dt={dt:g} -> c = 1/(beta_0*dt) = {cc:.1f}')
    print(f'(sec 7Q studied c=525; this is {cc/525:.1f}x that)\n')
    shape = (s['m'].nelem, s['N']+1, s['N']+1, OP.NVAR_R, s['nk'])
    A = lambda x: S3.normal_op(x, s['D'], s['m'].facx, s['m'].facy, s['kz'],
                               s['nu'], cc, s['m'], s['mask'], s['m'].wq, 0.0, rw)
    rng = np.random.default_rng(0)
    b = (rng.standard_normal(shape)*s['mask'])
    b /= np.linalg.norm(b)

    for tag, mk in (('Jacobi', lambda: C.make_precond(s, dt, 0.0, rowweight=True)[0]),
                    ('PMG (8,4,2) cheb-coarse',
                     lambda: P3.PMG(s['m'], s['nk'], s['nz'], s['nu'], cc, s['kz'],
                                    kap=0.0, rw=rw, orders=(8,4,2), deg=6,
                                    pin_p=True, direct_coarse=False, mask=s['mask'])),
                    ('PMG (8,4) cheb-coarse',
                     lambda: P3.PMG(s['m'], s['nk'], s['nz'], s['nu'], cc, s['kz'],
                                    kap=0.0, rw=rw, orders=(8,4), deg=6,
                                    pin_p=True, direct_coarse=False, mask=s['mask']))):
        t0 = time.perf_counter(); M = mk(); tb = time.perf_counter()-t0
        ap = M if callable(M) else (lambda r: r*M)
        x = np.zeros(shape); r = b.copy(); n0 = np.linalg.norm(r)
        hist, t0 = [], time.perf_counter()
        for it in range(8):
            x = x + ap(r); r = b - A(x)
            hist.append(np.linalg.norm(r)/n0)
        tc = (time.perf_counter()-t0)/8
        fac = (hist[-1]/hist[-4])**(1/3) if hist[-4] > 0 else float('nan')
        print(f'{tag:26s} build {tb:6.1f}s  {tc:6.2f}s/cycle')
        print('   ||r||/||r0||: ' + ' '.join(f'{h:.3e}' for h in hist))
        print(f'   asymptotic reduction factor per cycle = {fac:.4f}'
              f'{"   <-- STALLED" if fac > 0.9 else ""}\n')

if __name__ == '__main__':
    main()
