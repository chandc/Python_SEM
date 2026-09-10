"""Plot ln(E'/E'0) vs t for the Orr-Sommerfeld runs: legacy vs balanced weighting."""
import glob, os, re
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
SC = os.path.dirname(os.path.abspath(__file__))
SIG = 0.00223497

if __name__ == '__main__':
    files = sorted(glob.glob(f'{SC}/os_bal_*.npz'))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for fn in files:
        mo = re.search(r'os_bal_(\w+?)_N(\d+)_dt([0-9p]+)(_cg([0-9p]+))?\.npz', fn)
        w, N, dt = mo.group(1), int(mo.group(2)), float(mo.group(3).replace('p', '.'))
        cg = float(mo.group(5).replace('p', '.')) if mo.group(5) else 0.01
        d = np.load(fn)
        if abs(dt-0.1) < 1e-9:
            panel = 0
        elif N == 10 and cg < 0.01:
            panel = 1
        else:
            continue
        ls = '-' if w == 'balanced' else '--'
        lab = f"{w} N={N} dt={dt:g}  σ={d['sigma'][0]:.5f}"
        ax[panel].plot(d['t'], np.log(d['e']), ls, lw=1.2, label=lab)
    for a, ttl in zip(ax, ('dt = 0.1, N = 8/10/14', 'N = 10, dt = 0.02 / 0.01, CG rel. tol 1e-4')):
        t = np.linspace(0, 100, 2)
        a.plot(t, 2*SIG*t, 'k:', lw=2, label=f'linear theory σ={SIG:.5f}')
        a.set_xlabel('t'); a.set_ylabel("ln(E'/E'0)"); a.set_title(ttl)
        a.set_ylim(-0.05, 0.6); a.grid(alpha=.3); a.legend(fontsize=7)
    fig.suptitle('Orr–Sommerfeld Re=7500, α=1: balanced (solid) vs legacy (dashed) weighting; mesh 0.3/1.4/0.3')
    fig.tight_layout()
    out = 'figs_fosls_vs_fs/os_balanced_vs_legacy.png'
    fig.savefig(out, dpi=130); print('wrote', out)
