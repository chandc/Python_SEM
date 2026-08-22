"""Compare our TGV runs against Gourianov et al. (2022) on THEIR metric.

    uv run --quiet python scratch/tgv_vs_gourianov.py

REFERENCE.  N. Gourianov, M. Lubasch, S. Dolgov, Q. Y. van den Berg, H. Babaee,
P. Givi, M. Kiffner, D. Jaksch, "A Quantum Inspired Approach to Exploit
Turbulence Structures", arXiv:2106.05782v3 / Nature Comp. Sci. 2 (2022).
reference/2106.05782v3.pdf.

WHY THIS PAPER.  Its 3-D TGV study uses the SAME diagnostic this project
adopted independently as its resolution meter: for incompressible periodic
flow the INSE imply

    zeta(t) = nu * int |curl V|^2 dr   ==   eps(t) = -1/2 d/dt int |V|^2 dr

and any gap is numerical diffusion.  Their Eq. (20) integrates that gap into a
single number

    e(s,c) = (1/E0) int_0^{2 T0} |zeta(t) - eps(t)| dt ,   T0 = Lbox/u0,

and Table 1 publishes it for DNS (256^3, 8th-order FD + RK2 + Chorin
projection), for their tensor-network MPS scheme, and for under-resolved DNS
on coarse grids -- all at Re = 800.  So our runs can be placed on a PUBLISHED
scale rather than compared by eye to a digitised curve.

TWO CAVEATS, stated because they bound the claim:

 1. THEIR Re IS 800; ours are 100 and 400.  Lower Re is easier to resolve, so
    e alone flatters us.  The resolution-adequacy column (k_max * eta, the
    standard DNS criterion, >~ 1 is resolved) is the like-for-like axis: it
    normalises for Reynolds number, and our Re = 400 run sits at 0.82 against
    their 1:25 URDNS at ~0.88 -- comparable adequacy, 9x better e.
 2. E0 AMBIGUITY.  The paper states "the corresponding energy at t = 0 is
    E0 = u0^2/2", but the TGV initial condition carries mean kinetic energy
    density u0^2/8 -- a factor of 4.  We normalise by the ACTUAL initial
    kinetic energy; if their E0 is the stated u0^2/2, every number of ours
    below should be divided by 4 (which improves our standing, so the
    conservative choice is the one made here).
"""
import numpy as np

V = (2*np.pi)**3
T0 = 2*np.pi                     # Lbox/u0 in our units (Lbox = 2 pi, u0 = 1)
TW = 2*T0                        # their integration window

# Gourianov et al., Table 1 (TGV rows), all at Re = 800, DNS on 256^3.
# URDNS grids inferred from the quoted compression ratios: 256/c^(1/3).
PAPER = [('DNS 256$^3$', 0.002, 256), ('MPS 1:25', 0.0385, None),
         ('MPS 1:49', 0.0844, None), ('MPS 1:78', 0.2618, None),
         ('URDNS ~88$^3$', 0.1599, 88), ('URDNS ~70$^3$', 0.2133, 70),
         ('URDNS ~60$^3$', 0.4563, 60)]
GRID = {'re100': 24, 're400': 48}


def analyse(tag):
    d = np.load(f'scratch/tgv_diag_{tag}.npz')
    t, E, Om, nu = d['t'], d['E'], d['Om'], float(d['nu'])
    zeta = 2*nu*Om                       # nu int |curl V|^2
    eps = -np.gradient(E, t)             # -1/2 d/dt int |V|^2
    m = t <= min(TW, t[-1])
    e = float(np.trapezoid(np.abs(zeta[m] - eps[m]), t[m])/E[0])
    epsv = zeta/V
    ip = int(np.argmax(epsv))
    eta = (nu**3/epsv[ip])**0.25         # Kolmogorov scale at peak dissipation
    return dict(tag=tag, nu=nu, Re=1.0/nu, e=e, t=t[m], gap=np.abs(zeta-eps)[m]/E[0],
                eta=eta, kmax_eta=GRID[tag]/2*eta, cover=t[m][-1]/TW,
                tpk=t[ip], epsmax=epsv[ip])


def main():
    rows = [analyse('re100'), analyse('re400')]
    print('OUR RUNS, on Gourianov et al. Eq. (20):\n')
    print(f"{'case':>7}{'Re':>6}{'grid':>8}{'coverage':>10}{'e':>10}"
          f"{'k_max*eta':>11}{'t_peak/T0':>11}")
    for r in rows:
        print(f"{r['tag']:>7}{r['Re']:>6.0f}{GRID[r['tag']]:>6}^3"
              f"{100*r['cover']:>9.0f}%{r['e']:>10.4f}{r['kmax_eta']:>11.2f}"
              f"{r['tpk']/T0:>11.2f}")
    print('\nPAPER, Table 1 (Re = 800):')
    for lab, e, _ in PAPER:
        print(f"{lab.replace('$','').replace('^',''):>16}   e = {e:.4f}")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.4))
    for r, col in zip(rows, ('C2', 'C0')):
        ax1.plot(r['t']/T0, r['gap'], col, lw=2,
                 label=f"ours, Re = {r['Re']:.0f}, {GRID[r['tag']]}$^3$ "
                       f"($k_{{max}}\\eta$ = {r['kmax_eta']:.2f}),  e = {r['e']:.4f}")
    ax1.set_xlabel('$t/T_0$'); ax1.set_ylabel(r'$|\zeta - \varepsilon|\,/\,E_0$')
    ax1.set_title('Instantaneous numerical diffusion\n'
                  r'(the integrand of Gourianov Eq. 20; DNS-quality $\approx$ 0)')
    ax1.legend(fontsize=8.5); ax1.grid(alpha=0.3)

    labs = [f"ours Re=100\n24$^3$", f"ours Re=400\n48$^3$"] + [p[0] for p in PAPER]
    vals = [rows[0]['e'], rows[1]['e']] + [p[1] for p in PAPER]
    cols = ['C2', 'C0'] + ['0.35'] + ['C1']*3 + ['C3']*3
    b = ax2.bar(range(len(vals)), vals, color=cols)
    ax2.set_xticks(range(len(labs)))
    ax2.set_xticklabels(labs, rotation=45, ha='right', fontsize=8)
    ax2.set_yscale('log'); ax2.set_ylabel('$e$  (lower is better)')
    ax2.set_title('Time-integrated numerical diffusion $e$\n'
                  'ours (Re 100/400) vs Gourianov et al. Table 1 (Re = 800)')
    for i, v in enumerate(vals):
        ax2.text(i, v*1.15, f'{v:.4f}'.rstrip('0'), ha='center', fontsize=8)
    ax2.grid(alpha=0.3, axis='y', which='both')
    fig.suptitle('Taylor–Green vortex: our LSSEM/VVP solver against a published '
                 'DNS / tensor-network / under-resolved-DNS scale', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig('figs/tgv_vs_gourianov.png', dpi=150)
    print('\nwrote figs/tgv_vs_gourianov.png')


if __name__ == '__main__':
    main()
