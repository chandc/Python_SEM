"""What the pointwise divergence error actually costs, measured on both DNS runs.

    uv run python scratch/divergence_consequence.py

The velocity statistics of the two formulations agree to ~2 % (see the matched
window comparison), so mean profiles and Reynolds stresses do not distinguish
them.  This asks where the 2000x difference in pointwise divergence DOES show up.

CHOOSING THE RIGHT QUANTITY IS THE WHOLE PROBLEM, and two obvious candidates are
traps:

  * THE INTEGRATED ENERGY LEAK, -1/2 int |u|^2 (div u) dV, is not a
    discriminator.  div u is largely uncorrelated with |u|^2, so the integral
    cancels and comes out comparable for both.  Quoting it would support the
    conclusion that divergence does not matter.
  * MASS FLUX through constant-x planes is conserved to 3e-4 % by BOTH.  The
    projection enforces the WEAK divergence G^T u = 0 to machine precision, so
    every integrated conservation statement is excellent.  The failure is purely
    pointwise.

What does discriminate is the vorticity equation:

    Domega/Dt = omega . grad u  -  omega (div u)  +  nu lap omega
                ^^^^^^^^^^^^^^     ^^^^^^^^^^^^^
                stretching:        IDENTICALLY ZERO in incompressible flow
                the cascade

Whatever a scheme leaves in div u appears in the second term MULTIPLIED BY THE
VORTICITY, so the error is largest exactly where the vorticity is -- in the
near-wall vortices that sustain the turbulence.  Vortex stretching is the
mechanism of the cascade, so a spurious fraction of it is a dynamical statement
rather than a diagnostic one.
"""
import glob
import os
import sys

_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R)
os.chdir(_R)

import numpy as np

from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix, lgl_weights
from lssem3d import operator as OP

LX, LZ, N, EX, EY, NZ, NU = np.pi, 0.34*np.pi, 8, 6, 18, 32, 1/180.
M = build_channel(LX, 2.0, EX, EY, N, bcs=(0, 0, 1, 1))
M.compute_global_indices()
D = diff_matrix(N)
W1 = lgl_weights(N)
KZ = 2*np.pi*np.fft.rfftfreq(NZ, d=LZ/NZ)
WQ = ((M.hx*M.hy/4)[:, None, None]*W1[None, :, None]*W1[None, None, :])[..., None]*(LZ/NZ)

ddx = lambda f: np.einsum('ij,ejkm->eikm', D, f)*(2.0/M.hx)[:, None, None, None]
ddy = lambda f: np.einsum('ij,ekjm->ekim', D, f)*(2.0/M.hy)[:, None, None, None]
ddz = lambda f: 1j*KZ[None, None, None, :]*f
phys = lambda f: np.fft.irfft(f, NZ, axis=-1)
rms = lambda a: float(np.sqrt((WQ*a**2).sum()/WQ.sum()))


def load(path, fosls):
    z = np.load(path, allow_pickle=True)['U']
    if fosls:
        return [z[..., k, :] + 1j*z[..., OP.NVAR + k, :]
                for k in (OP.U_, OP.V_, OP.W_)], float(np.load(path, allow_pickle=True)['t'])
    return [z[..., k, :] for k in (0, 1, 2)], float(np.load(path, allow_pickle=True)['t'])


def diagnose(uvw):
    u, v, w = uvw
    du = [ddx(u), ddy(u), ddz(u)]
    dv = [ddx(v), ddy(v), ddz(v)]
    dw = [ddx(w), ddy(w), ddz(w)]
    div = du[0] + dv[1] + dw[2]
    o = [phys(dw[1] - dv[2]), phys(du[2] - dw[0]), phys(dv[0] - du[1])]
    d = phys(div)
    grad = np.sqrt(sum(phys(g)**2 for g in du + dv + dw))
    omag = np.sqrt(sum(x**2 for x in o))
    stretch = np.sqrt(sum((o[0]*phys(du[i]) + o[1]*phys(dv[i]) + o[2]*phys(dw[i]))**2
                          for i in range(3)))
    spur = omag*np.abs(d)
    return dict(div=rms(d), rel=rms(d)/rms(grad),
                spur=rms(spur), stretch=rms(stretch),
                frac=100*rms(spur)/rms(stretch))


def yprofile(uvw):
    """Spurious omega*(div u) and true stretching as functions of y."""
    u, v, w = uvw
    du=[ddx(u),ddy(u),ddz(u)]; dv=[ddx(v),ddy(v),ddz(v)]; dw=[ddx(w),ddy(w),ddz(w)]
    d = phys(du[0]+dv[1]+dw[2])
    o = [phys(dw[1]-dv[2]), phys(du[2]-dw[0]), phys(dv[0]-du[1])]
    omag = np.sqrt(sum(x**2 for x in o))
    st = np.sqrt(sum((o[0]*phys(du[i])+o[1]*phys(dv[i])+o[2]*phys(dw[i]))**2
                     for i in range(3)))
    sp = omag*np.abs(d)
    ys, a, b = [], [], []
    for ey in range(EY):
        es = [ex*EY+ey for ex in range(EX)]
        for j in range(N+1):
            ys.append(M.ynod[es[0]][j])
            a.append(np.sqrt((sp[es,:,j,:]**2).mean()))
            b.append(np.sqrt((st[es,:,j,:]**2).mean()))
    ys=np.array(ys); o_=np.argsort(ys)
    return ys[o_], np.array(a)[o_], np.array(b)[o_]


def figure(runs):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for tag, files, c in (('FOSLS', runs['FOSLS'], 'C0'),
                          ('fractional step', runs['fractional step'], 'C3')):
        fr = []
        for f, fo in files:
            uvw, _ = load(f, fo)
            y, sp, st = yprofile(uvw)
            fr.append(100*sp/np.maximum(st, 1e-300))
        fr = np.mean(fr, axis=0)
        yp = np.minimum(y, 2-y)*180; k = np.argsort(yp)
        ax[0].loglog(yp[k], np.maximum(fr[k], 1e-4), color=c, lw=1.8, label=tag)
        ax[1].semilogx(np.maximum(fr, 1e-4), y, color=c, lw=1.8, label=tag)
    for a, xl, yl in ((ax[0], '$y^+$', 'spurious / stretching  [%]'),
                      (ax[1], 'spurious / stretching  [%]', '$y$')):
        a.set_xlabel(xl); a.set_ylabel(yl); a.grid(alpha=0.3, which='both')
        a.legend(fontsize=9, loc='center right')
    # EXPLICIT LIMITS.  The two runs differ by 2000x, so autoscaling clips one of
    # them off the axis entirely -- which the first version of this figure did.
    ax[0].set_xlim(0.5, 180); ax[0].set_ylim(1e-3, 1e2)
    ax[1].set_ylim(0, 2); ax[1].set_xlim(1e-3, 1e2)
    ax[0].set_title(r'$\omega\,(\nabla\!\cdot\mathbf{u})$ as a fraction of '
                    r'$\omega\cdot\nabla\mathbf{u}$' '\nwall units, folded',
                    fontsize=10)
    ax[1].set_title('across the channel', fontsize=10)
    fig.suptitle('The term incompressibility should kill, averaged over the '
                 'archived snapshots', fontsize=12)
    fig.tight_layout()
    fig.savefig('figs/divergence_consequence.png', dpi=140, bbox_inches='tight')
    print('wrote figs/divergence_consequence.png')


def main():
    runs = {'FOSLS': [(f, True) for f in sorted(glob.glob('scratch/_dns_drive/checkpoint_*.npz'))],
            'fractional step': [(f, False) for f in sorted(glob.glob('scratch/_fs_drive/field_t*.npz'))]}
    out = {}
    for tag, files in runs.items():
        print(f'\n{tag}')
        print(f'  {"t":>7s} {"rms div u":>11s} {"rel |grad u|":>13s} '
              f'{"spurious/stretching":>21s}')
        rec = []
        for f, fo in files:
            uvw, t = load(f, fo)
            r = diagnose(uvw)
            rec.append((t, r))
            print(f'  {t:7.3f} {r["div"]:11.3e} {r["rel"]:13.3e} {r["frac"]:20.3f} %')
        out[tag] = rec
        a = np.array([[r['div'], r['rel'], r['frac']] for _, r in rec])
        print(f'  {"mean":>7s} {a[:,0].mean():11.3e} {a[:,1].mean():13.3e} '
              f'{a[:,2].mean():20.3f} %')
        print(f'  {"spread":>7s} {a[:,0].std():11.3e} {a[:,1].std():13.3e} '
              f'{a[:,2].std():20.3f} %')
    A = np.array([[r['div'], r['rel'], r['frac']] for _, r in out['FOSLS']])
    B = np.array([[r['div'], r['rel'], r['frac']] for _, r in out['fractional step']])
    print(f'\nRATIO fractional step / FOSLS, over all snapshots:')
    for k, lab in ((0, 'rms div u'), (1, 'relative to |grad u|'),
                   (2, 'spurious fraction of vortex stretching')):
        print(f'  {lab:40s} {B[:,k].mean()/A[:,k].mean():8.0f} x')
    print('\nNOTE: three of the five FOSLS checkpoints sit at t = 29.6, 29.92, 30.0')
    print('and are nearly the same instant, so that run has ~3 independent')
    print('samples, not 5.  The spread above is correspondingly optimistic.')
    figure(runs)


if __name__ == '__main__':
    main()
