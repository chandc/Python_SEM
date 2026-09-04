# Minimal channel Re_tau = 180, K-path statistics run (COMPLETE)

Run: 2026-08-26/27, Mac M3 Max, RK3-CN substage + skew convection,
K-path (weak-Laplacian) projection, mode-pool + freezing + GEMM stack.
Grid 6x18 elements N=8, Nz=32 (Jimenez-Moin minimal box, Lx+=565, Lz+=192),
dt = 3.5e-4, statistics window t = 3..18 (15 units, ~6 bursting cycles).

## Files
- `stats_MERGED.npz` -- y, sums (5 x NY: U, <uu>, <vv>, <ww>, <uv>),
  nsamp = 8571, utau_series, nu.  Profiles = sums/nsamp; fold halves
  (u'v' antisymmetric); normalise by measured u_tau = 1.0148.
- `final_state.npz` -- U, p at t = 18 (restart-ready).
- `stats_seg1/2/final.npz` -- per-segment accumulators (additive).
  MERGE_NOTE.txt: seg3 was a duplicate of seg2, excluded.
- figures + stats_run.log.

## Headline numbers (vs KMM Re180 full-box DNS)
kappa 0.408 (0.40) | B 5.69 (5.5) | U+_c 18.38 (18.2)
u' 2.84@y+=14 (2.65@15, known minimal-box excess) | v' 0.87 (0.85)
w' 1.04 (1.05) | -<u'v'> 0.735 (0.73)
u_tau = 1.0148 +/- 0.02 (Re_tau 182.7); burst periods t+ ~ 139 / ~450.

## Caveats
Minimal box (single near-wall cycle; profile above y+~100 and u' carry
domain-size effects).  K-path projection: pointwise div ~0.15 (weak div
uncontrolled) -- the E-path twin (Spark, ~2026-08-29) is the operator A/B.
Provenance: DIVERGENCE_AND_CONSISTENCY.md, SCHEME_COMPARISON.md.
