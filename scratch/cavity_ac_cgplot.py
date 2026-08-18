"""CG iterations with and without artificial compressibility -- cavity Re = 1000.

    uv run --quiet python scratch/cavity_ac_cgplot.py

Reads the MEASURED table scratch/cavity_ac_cgiters.csv (written by
scratch/cavity_ac_cgiters.py) -- nothing here is hard-coded, so re-measuring and
re-plotting cannot drift apart.  Documented in ARTIFICIAL_COMPRESSIBILITY.md
sec 5.2.

    figs/cavity_ac_cg_iterations.png
"""
import os, sys, csv
SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
os.chdir('/Users/danielchan/Dropbox/Apple_MLX_CFD/sem_demo')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

CSV = f'{SC}/cavity_ac_cgiters.csv'
with open(CSV) as fh:
    ROWS = [{k: (v if k == 'tag' else float(v)) for k, v in r.items()}
            for r in csv.DictReader(fh)]

# Keep only a_mass values measured for ALL THREE settings.  cavity_ac_cgiters.py
# checkpoints by rewriting the whole csv after every case, so reading it while
# that sweep is still running catches a partial file -- which silently produced a
# plot missing its highest points until this guard was added.
_TAGS = {'off', 'half', 'match'}
_seen = {}
for r in ROWS:
    _seen.setdefault(r['a_mass'], set()).add(r['tag'])
AM = sorted(a for a in _seen if _seen[a] >= _TAGS)
_partial = sorted(set(_seen) - set(AM))
if _partial:
    print(f'WARNING: dropping incomplete a_mass {_partial} '
          f'(is cavity_ac_cgiters.py still running?)', file=sys.stderr)
if not AM:
    raise SystemExit(f'{CSV} has no a_mass measured for all of {sorted(_TAGS)}')
DT = {r['a_mass']: r['dt'] for r in ROWS}


def series(tag):
    """(a_mass, its/call, wall) for one AC setting, ordered by a_mass."""
    d = {r['a_mass']: r for r in ROWS if r['tag'] == tag}
    return [(a, d[a]['its_per_call'], d[a]['wall_s']) for a in AM if a in d]


SERIES = [('off', 'AC off', 'tab:red', 'o'),
          ('half', r'AC on, $\kappa_p=a_{mass}/2$', 'tab:blue', 's'),
          ('match', r'AC on, $\kappa_p=a_{mass}$', 'tab:green', '^')]

fig, axs = plt.subplots(1, 2, figsize=(14.0, 5.8))

# --- (a) iterations per CG solve vs a_mass ---
ax = axs[0]
for tag, lab, col, mk in SERIES:
    pts = series(tag)
    ax.plot([p[0] for p in pts], [p[1] for p in pts], mk+'-', color=col, ms=9,
            lw=2.0, label=lab)
    for a, y, _ in pts:
        ax.annotate(f'{y:.0f}', (a, y), textcoords='offset points',
                    xytext=(0, 10 if tag != 'match' else -15), ha='center',
                    fontsize=8.5, color=col)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(min(AM)/1.6, max(AM)*3.0)
ax.set_ylim(min(p[1] for p in series('match'))*0.62,
            max(p[1] for p in series('off'))*2.1)
ax.set_xticks(AM); ax.set_xticklabels([f'{a:g}' for a in AM])
ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xlabel(r'$a_{mass} = w_{mass}\,fac_1/dt$'
              '\n(dt = ' + ',  '.join(f'{DT[a]:g}' for a in AM) + ')')
ax.set_ylabel('CG iterations per solve')
ax.set_title('AC is cheaper at every $a_{mass}$', fontsize=12)
ax.grid(alpha=.3, which='both'); ax.legend(fontsize=9.5, loc='lower left')

# The AC-off cost is U-SHAPED in a_mass, not monotone -- it rises towards BOTH
# ends.  An earlier two-point version of this plot sampled only the right-hand
# branch and was captioned "AC reverses the slope", which is wrong: refining dt
# only makes the solve harder above the minimum.  Mark the minimum from the data
# rather than asserting a direction.
off_pts = series('off')
imin = int(np.argmin([p[1] for p in off_pts]))
a_min, y_min, _ = off_pts[imin]
interior = 0 < imin < len(off_pts)-1
if interior:
    ax.annotate(f'AC off is cheapest near $a_{{mass}}$ = {a_min:g};\n'
                'cost rises towards BOTH ends',
                xy=(a_min, y_min), xytext=(a_min*1.25, y_min*0.47), fontsize=9,
                color='tab:red', ha='left', va='center',
                arrowprops=dict(arrowstyle='->', color='tab:red', lw=1.2))
mt = series('match')
ax.annotate('with AC the cost falls\nmonotonically as dt is refined',
            xy=mt[-1][:2], xytext=(max(AM)*1.22, mt[-1][1]*1.5), fontsize=9,
            color='tab:green', ha='left', va='center',
            arrowprops=dict(arrowstyle='->', color='tab:green', lw=1.2))

# --- (b) reduction factor ---
ax = axs[1]
off = {a: y for a, y, _ in series('off')}
wall = {t: {a: w for a, _, w in series(t)} for t, _, _, _ in SERIES}
w = 0.36
x = np.arange(len(AM))
for k, (tag, lab, col, _) in enumerate([s for s in SERIES if s[0] != 'off']):
    d = {a: y for a, y, _ in series(tag)}
    h = [off[a]/d[a] for a in AM]
    bars = ax.bar(x + (k-0.5)*w, h, w, color=col, label=lab)
    for b in bars:
        ax.annotate(f'{b.get_height():.1f}x',
                    (b.get_x()+b.get_width()/2, b.get_height()),
                    textcoords='offset points', xytext=(0, 3), ha='center',
                    fontsize=9.5)
ax.axhline(1.0, color='k', lw=1.0, ls='--')
ax.set_xticks(x)
ax.set_xticklabels([f'{a:g}\n(dt = {DT[a]:g})\n'
                    f'{wall["off"][a]:.0f}s $\\rightarrow$ {wall["match"][a]:.0f}s'
                    for a in AM], fontsize=9)
ax.set_xlabel(r'$a_{mass}$,  with wall time  AC off $\rightarrow$ '
              r'$\kappa_p=a_{mass}$')
ax.set_ylim(0, max(off[a]/{aa: y for aa, y, _ in series('match')}[a]
                   for a in AM)*1.20)
ax.set_ylabel('CG-iteration reduction vs AC off')
ax.set_title('Benefit grows with $a_{mass}$', fontsize=12)
ax.grid(alpha=.3, axis='y'); ax.legend(fontsize=9.5, loc='upper left')

fig.suptitle('Artificial compressibility and linear-solver cost -- lid-driven '
             'cavity Re = 1000, 6x6 elements N = 10\n'
             '40 steps from rest, nsub = 5, cg_tol = 1e-8, Jacobi-preconditioned '
             'CG (200 solves per case, identical work)', fontsize=11.5)
fig.text(0.5, 0.005, 'Mechanism: pressure enters only the momentum rows, so '
         r'without AC the Jacobi preconditioner has no pressure diagonal '
         r'($a_{33}=0$).  AC supplies $a_{33}=\kappa_p P$ — exactly where the '
         'conditioning is worst.', ha='center', fontsize=10)
fig.tight_layout(rect=[0, 0.035, 1, 0.90])
fig.savefig('figs/cavity_ac_cg_iterations.png', dpi=125, bbox_inches='tight')
print('figs/cavity_ac_cg_iterations.png\n')
hdr = (f"{'dt':>7}{'a_mass':>8}{'kappa_p':>9}{'tag':>7}{'CG its':>9}"
       f"{'its/call':>10}{'reduction':>11}{'wall':>8}")
print(hdr); print('-'*len(hdr))
for r in ROWS:
    red = off[r['a_mass']]/r['its_per_call']
    print(f"{r['dt']:>7g}{r['a_mass']:>8g}{r['kappa_p']:>9g}{r['tag']:>7}"
          f"{int(r['cg_its']):>9d}{r['its_per_call']:>10.1f}"
          f"{('--' if r['tag'] == 'off' else f'{red:.1f}x'):>11}"
          f"{r['wall_s']:>7.1f}s")
