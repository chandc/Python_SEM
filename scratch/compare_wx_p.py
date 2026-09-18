"""Streamwise vorticity and pressure, FOSLS vs fractional step, at t = 26.

    uv run python scratch/compare_wx_p.py

THESE ARE DIFFERENT REALISATIONS AND WILL NOT MATCH POINTWISE.  Both runs were
seeded from the same field, but turbulence is chaotic: by t = 26 they are on
different trajectories and any pointwise comparison is meaningless.  What CAN be
compared is structure and magnitude -- the number and scale of the quasi-
streamwise vortices, how far they sit from the wall, the range of omega_x and of
p.  Read the panels as two samples from the same statistical state, not as a
difference plot.

WHY omega_x IS THE RIGHT FIELD.  It is the signature of the near-wall cycle: the
quasi-streamwise vortices that lift low-speed fluid and sustain the streaks.  It
is also where the two formulations differ structurally -- FOSLS carries omega as
a SOLVED VARIABLE, the fractional step must differentiate the velocity for it,
so this is a like-for-unlike comparison and the panels are labelled accordingly.
"""
import os
import sys

_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R)
os.chdir(_R)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem3d import operator as OP

OUT = os.path.join(_R, 'figs', 'fosls_vs_fs_wx_umean.png')
NZ_FINE = 256        # z refinement by FFT zero-padding: spectrally EXACT
NY_SUB = 10          # y points per element from the Lagrange interpolant
LX, LZ, DELTA = np.pi, 0.34*np.pi, 1.0
EX, EY, N, NZ = 6, 18, 8, 32
NU = 1/180.


def mesh():
    m = build_channel(LX, 2.0*DELTA, EX, EY, N, bcs=(0, 0, 1, 1))
    m.compute_global_indices()
    return m


def _irfft_fine(fh):
    """Inverse transform onto NZ_FINE points, preserving amplitude.

    Zero-padding in mode space samples the SAME truncated Fourier series more
    densely -- exact, not smoothing.  But numpy's irfft normalises by 1/n with n
    the OUTPUT length, so padding 32 -> 256 divides every amplitude by 8.  The
    structure looks perfect and the colourbar is wrong by that factor, which is
    precisely the kind of error a picture cannot show: caught here only because
    the rms fell from 17.2 to 2.13 against the unpadded transform.
    """
    fh = fh.copy()
    # THE NYQUIST MODE MUST BE HALVED.  At the original length NZ (even) the top
    # mode is a single real cosine counted once; in the longer transform it is an
    # ordinary +-pair.  Padding without halving it leaves a residual of ~25 % of
    # the rms on the shared sample points -- verified against the unpadded
    # transform, which is the only way to see it.
    if NZ % 2 == 0 and fh.shape[-1] == NZ//2 + 1:
        fh[..., -1] *= 0.5
    return np.fft.irfft(fh, NZ_FINE, axis=-1)*(NZ_FINE/NZ)


def fosls_fields(path, m):
    z = np.load(path, allow_pickle=True)
    U = z['U']                                   # (nelem, n, n, 14, nmode)
    wx = U[..., OP.OX_, :] + 1j*U[..., OP.NVAR + OP.OX_, :]
    return _irfft_fine(wx), float(z['t'])


def fs_fields(path, m):
    z = np.load(path, allow_pickle=True)
    U = z['U']                                   # (nelem, n, n, 3, nmode) complex
    p = z['p'][..., 0, :]
    # omega_x = dw/dy - dv/dz.  dv/dz is i*kz*v in mode space; dw/dy is the SEM
    # derivative along the second node axis, scaled by the element's 2/hy.
    kz = 2*np.pi*np.fft.rfftfreq(NZ, d=LZ/NZ)
    D = diff_matrix(N)
    w = U[..., 2, :]
    dwdy = np.einsum('ij,ejkm->eikm', D, w.transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3)
    dwdy = dwdy*(2.0/m.hy)[:, None, None, None]
    dvdz = 1j*kz[None, None, None, :]*U[..., 1, :]
    wx = dwdy - dvdz
    return _irfft_fine(wx), float(z['t'])


def _lag(nodes, r):
    n = len(nodes)
    L = np.ones((len(r), n))
    for i in range(n):
        for j in range(n):
            if i != j:
                L[:, i] *= (r - nodes[j])/(nodes[i] - nodes[j])
    return L


def plane(fld, m, ix_elem=EX//2, ix_node=N//2):
    """(y, z) cross-plane at one streamwise station.

    Element-local storage duplicates every shared interface node, so the
    concatenated y is NOT monotone and pcolormesh silently mis-draws the cells.
    Sort and drop the duplicates.
    """
    vals, ys = [], []
    for ey in range(EY):
        e = ix_elem*EY + ey
        vals.append(fld[e, ix_node, :, :])
        ys.append(m.ynod[e])
    # EVALUATE THE POLYNOMIAL, do not smooth the nodal values.  GLL nodes are
    # clustered at element edges, so plotting them as cells looks blocky and
    # under-represents the interior.  The element's interpolant is a degree-N
    # polynomial and can be sampled anywhere exactly, which is what a spectral
    # field deserves -- gouraud shading would merely blur the same blocks.
    from lssem2d.lgl import lgl_nodes
    xi = lgl_nodes(N)
    rf = np.linspace(-1, 1, NY_SUB)
    L = _lag(xi, rf)                                   # (NY_SUB, N+1)
    yy, vv = [], []
    for ey in range(EY):
        e = ix_elem*EY + ey
        y0, y1 = m.ynod[e][0], m.ynod[e][-1]
        yy.append(y0 + (y1 - y0)*(rf + 1)/2)
        vv.append(L @ fld[e, ix_node, :, :])
    y = np.concatenate(yy); v = np.concatenate(vv, axis=0)
    o = np.argsort(y); y, v = y[o], v[o]
    keep = np.concatenate(([True], np.diff(y) > 1e-12))
    return y[keep], v[keep]


def windowed_mean_U():
    """Mean streamwise profile from the MATCHED window of each campaign.

    FOSLS is one continuous accumulator, so its window is a straight difference.
    The fractional-step campaign lost a resume, so its snapshots are TWO
    cumulative series and must be bridged additively -- differencing across the
    break gives 2270 samples where the window holds 6204, which is arithmetic on
    invalid inputs rather than a window.
    """
    D, F = 'scratch/_dns_drive/', 'scratch/_fs_drive/'
    fa = np.load(D + 'stats_0006500.npz', allow_pickle=True)
    fb = np.load(D + 'stats_0037500.npz', allow_pickle=True)
    A = np.load(F + 'stats_t005.090.npz', allow_pickle=True)
    B = np.load(F + 'stats_t020.698.npz', allow_pickle=True)
    C = np.load(F + 'stats_t029.908.npz', allow_pickle=True)
    y = np.asarray(fa['y']); o = np.argsort(y); y = y[o]
    out = {}
    for tag, sums, n in (
            ('FOSLS', fb['sums'] - fa['sums'], int(fb['nsamp']) - int(fa['nsamp'])),
            ('fractional step', (B['sums'] - A['sums']) + C['sums'],
             (int(B['nsamp']) - int(A['nsamp'])) + int(C['nsamp']))):
        U = (sums/n)[0][o]
        dl = (U[1] - U[0])/(y[1] - y[0]); dh = (U[-1] - U[-2])/(y[-1] - y[-2])
        ut = np.sqrt(NU*0.5*(abs(dl) + abs(dh)))
        # fold the two halves: the channel is symmetric, so both walls sample the
        # same profile and folding halves the statistical error for free
        yp = np.minimum(y, 2.0 - y)*ut/NU
        k = np.argsort(yp)
        out[tag] = (yp[k], (U/ut)[k], ut, n)
    return out


def akm(path):
    """Abe-Antonia-Kawamura Re_tau = 180 reference mean profile."""
    yp, up, block = [], [], 0
    for ln in open(path):
        if 'u_mean+' in ln:
            block = 1; continue
        if block and '-uv+' in ln:
            break
        s = ln.split()
        # columns are:  j  y+  u_mean+  uu+  ww+   -- the first is a ROW INDEX,
        # not a coordinate.  Reading s[0] as y+ plots the index against y+ and
        # produces a curve that rises through U+ = 20 by y+ = 25, which is not a
        # channel profile at all.  It looked like data.
        if block and len(s) >= 3:
            try:
                yp.append(float(s[1])); up.append(float(s[2]))
            except ValueError:
                pass
    return np.array(yp), np.array(up)


def main():
    m = mesh()
    wf, tf = fosls_fields('scratch/_dns_drive/checkpoint_0032500.npz', m)
    ws, ts = fs_fields('scratch/_fs_drive/field_t026.0.npz', m)
    zc = np.linspace(0, LZ, NZ_FINE, endpoint=False)
    yF, WF = plane(wf, m)
    yS, WS = plane(ws, m)
    print(f'omega_x  FOSLS [{WF.min():+8.2f}, {WF.max():+8.2f}] rms {WF.std():7.3f}'
          f'   FS [{WS.min():+8.2f}, {WS.max():+8.2f}] rms {WS.std():7.3f}')
    print(f'rendered on {NZ_FINE} z points (FFT zero-pad, exact) x '
          f'{len(yF)} y points ({NY_SUB}/element from the interpolant)')

    U = windowed_mean_U()
    fig = plt.figure(figsize=(14, 9.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.34, wspace=0.16)
    lim = np.percentile(np.abs(np.concatenate([WF.ravel(), WS.ravel()])), 99.0)
    for c, (tag, Fv, note) in enumerate(
            ((f'FOSLS   $t = {tf:.1f}$', WF, 'SOLVED variable'),
             (f'fractional step   $t = {ts:.1f}$', WS,
              r'computed as $\partial w/\partial y - \partial v/\partial z$'))):
        a = fig.add_subplot(gs[0, c])
        im = a.pcolormesh(zc, yF, Fv, cmap='RdBu_r', vmin=-lim, vmax=lim,
                          shading='gouraud')
        fig.colorbar(im, ax=a, shrink=0.9)
        a.set_ylim(0, 2); a.set_xlabel('$z$'); a.set_ylabel('$y$')
        a.tick_params(labelsize=8)
        a.set_title(r'streamwise vorticity $\omega_x$ — ' + tag + f'\n({note})',
                    fontsize=10)

    a = fig.add_subplot(gs[1, :])
    ref = os.path.join(_R, 'reference/akm_chan180/ch180.dat')
    if os.path.exists(ref):
        ya, ua = akm(ref)
        a.semilogx(ya, ua, 'k-', lw=2.4, alpha=0.5, label='Abe–Antonia–Kawamura DNS')
    for (tag, (yp, up, ut, n)), st in zip(U.items(), (('C0', '-'), ('C3', '--'))):
        a.semilogx(yp[yp > 0.3], up[yp > 0.3], color=st[0], ls=st[1], lw=1.8,
                   label=f'{tag}  ($u_\\tau$ = {ut:.4f}, {n} samples)')
    yv = np.logspace(-0.5, 1, 40); a.semilogx(yv, yv, ':', color='0.5', lw=1.2,
                                              label=r'$U^+ = y^+$')
    yv = np.logspace(1.1, 2.3, 40)
    a.semilogx(yv, np.log(yv)/0.41 + 5.2, '-.', color='0.5', lw=1.2,
               label=r'$\frac{1}{0.41}\ln y^+ + 5.2$')
    a.set_xlim(0.3, 200); a.set_ylim(0, 20)
    a.set_xlabel('$y^+$'); a.set_ylabel('$U^+$')
    a.set_title('mean streamwise velocity, MATCHED window $t = 5.1 \\to 30$ '
                '(24.8 turnovers), both walls folded', fontsize=11)
    a.legend(fontsize=9, loc='upper left'); a.grid(alpha=0.3, which='both')

    fig.suptitle('Minimal channel $Re_\\tau = 180$: FOSLS vs fractional step\n'
                 'instantaneous $\\omega_x$ in a $(y,z)$ plane — DIFFERENT '
                 'REALISATIONS, compare structure — and the averaged profile, '
                 'which IS a like-for-like comparison', fontsize=12)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches='tight')
    for tag, (yp, up, ut, n) in U.items():
        print(f'  {tag:16s} u_tau {ut:.4f}  U+ at y+=30: '
              f'{np.interp(30.0, yp[yp<90], up[yp<90]):.3f}')
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
