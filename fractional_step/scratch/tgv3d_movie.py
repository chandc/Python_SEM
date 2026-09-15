"""Render the TGV movie and the diagnostics figure from saved frames.

    uv run --quiet python scratch/tgv3d_movie.py re100 [--movie-only|--diag-only]

Reads scratch/tgv_frames_<tag>/frame_####.npz (complex64 mode-space state) and
scratch/tgv_diag_<tag>.npz.  Produces:

    figs/tgv_<tag>_movie.mp4 (ffmpeg) or .gif (fallback)
    scratch/tgv_frames_<tag>/png/f####.png       the individual movie frames
    figs/tgv_<tag>_diagnostics.png               E, Omega, balance, CG history

Each movie frame: |omega| on the three mid-planes (z=pi, y=pi, x=pi) plus the
energy/enstrophy history with a time cursor.  A fixed colour scale across all
frames (scanned first) so growth and decay of the vortical structures is
visible rather than autoscaled away.
"""
import os, sys, glob, shutil, subprocess
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
sys.path.insert(0, SC)
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tgv3d as G
from lssem3d import fourier as FR, operator as OP

L = 2.0*np.pi


def load_case(tag):
    d = np.load(f'{SC}/tgv_diag_{tag}.npz')
    frames = sorted(glob.glob(f'{SC}/tgv_frames_{tag}/frame_*.npz'))
    cfg = G.CASES[tag]
    s = G.setup(N=cfg['N'], ex=cfg['ex'], ey=cfg['ey'], nz=cfg['nz'],
                nu=cfg['nu'])
    return d, frames, s


def omega_mag_phys(s, Uc):
    P = FR.to_physical(Uc.astype(np.complex128), s['nz'])
    return np.sqrt(sum(np.abs(P[..., f, :])**2
                       for f in (OP.OX_, OP.OY_, OP.OZ_)))


def plane_xy(s, W, ziq):
    """|omega| on the (x,y) plane at z-index ziq -> tricontour arrays."""
    m = s['m']; n = s['N'] + 1
    px, py, val = [], [], []
    for e in range(m.nelem):
        for i in range(n):
            for j in range(n):
                px.append(m.xnod[e, i]); py.append(m.ynod[e, j])
                val.append(W[e, i, j, ziq])
    return np.array(px), np.array(py), np.array(val)


def plane_xz(s, W, ycut=np.pi):
    """|omega| on the (x,z) plane at y = ycut (an element boundary)."""
    m = s['m']; n = s['N'] + 1
    xs, rows = [], []
    for e in range(m.nelem):
        for j in range(n):
            if abs(m.ynod[e, j] - ycut) < 1e-9:
                for i in range(n):
                    xs.append(m.xnod[e, i]); rows.append(W[e, i, j, :])
    xs = np.array(xs); rows = np.array(rows)
    o = np.argsort(xs); xs, rows = xs[o], rows[o]
    keep = np.concatenate([[True], np.diff(xs) > 1e-12])
    return xs[keep], s['zpl'], rows[keep]


def plane_yz(s, W, xcut=np.pi):
    m = s['m']; n = s['N'] + 1
    ys, rows = [], []
    for e in range(m.nelem):
        for i in range(n):
            if abs(m.xnod[e, i] - xcut) < 1e-9:
                for j in range(n):
                    ys.append(m.ynod[e, j]); rows.append(W[e, i, j, :])
    ys = np.array(ys); rows = np.array(rows)
    o = np.argsort(ys); ys, rows = ys[o], rows[o]
    keep = np.concatenate([[True], np.diff(ys) > 1e-12])
    return ys[keep], s['zpl'], rows[keep]


def render(tag, movie=True, diag=True):
    d, frames, s = load_case(tag)
    nu = float(d['nu'])

    if diag:
        t, E, Om = d['t'], d['E'], d['Om']
        eps = -np.gradient(E, t)
        fig, axs = plt.subplots(2, 2, figsize=(13, 8.5))
        ax = axs[0, 0]
        ax.plot(t, E/E[0], 'C0-', label='$E/E_0$')
        ax.plot(t, Om/Om[0], 'C3-', label='$\\Omega/\\Omega_0$')
        ax.set_xlabel('t'); ax.legend(); ax.grid(alpha=.3)
        ax.set_title('Energy and enstrophy (normalised)')
        ax = axs[0, 1]
        ax.plot(t, eps, 'C0-', label='$-dE/dt$')
        ax.plot(t, 2*nu*Om, 'k--', label='$2\\nu\\Omega$')
        ax.set_xlabel('t'); ax.legend(); ax.grid(alpha=.3)
        ax.set_title('Dissipation: kinetic-energy balance (must coincide)')
        ax = axs[1, 0]
        with np.errstate(divide='ignore', invalid='ignore'):
            bal = eps/(2*nu*Om)
        ax.plot(t[1:], bal[1:], 'C2-')
        ax.axhline(1.0, color='k', ls=':')
        ax.set_ylim(0.9, 1.1)
        ax.set_xlabel('t'); ax.grid(alpha=.3)
        ax.set_title('Balance ratio  $(-dE/dt)/(2\\nu\\Omega)$ — target 1')
        ax = axs[1, 1]
        ax.plot(d['t'][1:], d['cg'][1:], 'C4-')
        ax.set_xlabel('t'); ax.set_ylabel('CG iters / step'); ax.grid(alpha=.3)
        ncap = int(np.sum(d['capped']))
        ax.set_title(f'Solver cost (capped steps: {ncap})')
        fig.suptitle(f'TGV {tag}:  $\\nu$ = {nu:g}, dt = {float(d["dt"]):.4g}, '
                     f'row weights on, AC off', fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        out = f'figs/tgv_{tag}_diagnostics.png'
        fig.savefig(out, dpi=140); plt.close(fig)
        print('wrote', out)

    if not movie:
        return
    pngdir = f'{SC}/tgv_frames_{tag}/png'
    os.makedirs(pngdir, exist_ok=True)
    # first pass: global colour scale
    vmax = 0.0
    for f in frames:
        W = omega_mag_phys(s, np.load(f)['U'])
        vmax = max(vmax, float(W.max()))
    ziq = s['nz']//2
    t_all, E_all = d['t'], d['E']
    for fi, f in enumerate(frames):
        z = np.load(f)
        W = omega_mag_phys(s, z['U']); tf = float(z['t'])
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        px, py, val = plane_xy(s, W, ziq)
        tc = axs[0, 0].tricontourf(px, py, val, levels=40, cmap='inferno',
                                   vmin=0, vmax=vmax)
        axs[0, 0].set_title(f'$|\\omega|$, z = $\\pi$'); axs[0, 0].set_aspect(1)
        xs, zp, rows = plane_xz(s, W)
        axs[0, 1].contourf(xs, zp, rows.T, levels=40, cmap='inferno',
                           vmin=0, vmax=vmax)
        axs[0, 1].set_title('$|\\omega|$, y = $\\pi$'); axs[0, 1].set_aspect(1)
        ys, zp, rows = plane_yz(s, W)
        axs[1, 0].contourf(ys, zp, rows.T, levels=40, cmap='inferno',
                           vmin=0, vmax=vmax)
        axs[1, 0].set_title('$|\\omega|$, x = $\\pi$'); axs[1, 0].set_aspect(1)
        ax = axs[1, 1]
        ax.plot(t_all, E_all/E_all[0], 'C0-', label='$E/E_0$')
        ax.plot(t_all, d['Om']/d['Om'][0], 'C3-', label='$\\Omega/\\Omega_0$')
        ax.axvline(tf, color='k', lw=1)
        ax.set_xlabel('t'); ax.legend(fontsize=9); ax.grid(alpha=.3)
        ax.set_title(f't = {tf:.2f}')
        fig.colorbar(tc, ax=axs[0, 0], shrink=0.8)
        fig.suptitle(f'Taylor–Green vortex, {tag} ($\\nu$ = {nu:g}) — '
                     f'vortex stretching and mode interaction', fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(f'{pngdir}/f{fi:04d}.png', dpi=100)
        plt.close(fig)
    print(f'{len(frames)} PNG frames in {pngdir}')
    if shutil.which('ffmpeg'):
        out = f'figs/tgv_{tag}_movie.mp4'
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', '8',
                        '-i', f'{pngdir}/f%04d.png', '-pix_fmt', 'yuv420p',
                        '-vf', 'crop=trunc(iw/2)*2:trunc(ih/2)*2', out],
                       check=True)
        print('wrote', out)
    else:
        from PIL import Image
        imgs = [Image.open(p) for p in sorted(glob.glob(f'{pngdir}/f*.png'))]
        out = f'figs/tgv_{tag}_movie.gif'
        imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=125,
                     loop=0)
        print('wrote', out)


if __name__ == '__main__':
    tag = sys.argv[1] if len(sys.argv) > 1 else 're100'
    movie = '--diag-only' not in sys.argv
    diag = '--movie-only' not in sys.argv
    render(tag, movie=movie, diag=diag)
