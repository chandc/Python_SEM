# Minimal channel Re_tau = 180, E-path statistics run + the operator A/B

2026-08-27..29, Spark GB10 (CuPy).  RK3-CN substage + skew + CONSISTENT
P_N-P_N projection (E = G^T M^-1 G, E-multigrid deg 6, tol_p 1e-4), same
tripped initial state as the K-path twin (results/minchan_re180_K).
Window t = 3..15.95 (run stopped when E-solves entered a stagnation
plateau near the iteration cap at t~15.95 -- statistics were checkpointed
continuously, nothing lost; the stall is an E-multigrid improvement item).

## Files
stats_E.npz (same schema as the K archive), state_t15.95.npz,
AB_comparison.png (the verdict figure).

## The A/B verdict (matched windows, 7401 samples each)
u_tau -0.5% | U+c +1.4% | u' +0.03% | v' -2.7% | w' -1.1% | -u'v' -1.1%
-- all within the ~1-3% finite-window sampling error.  THE OPERATOR CHOICE
DOES NOT MEASURABLY AFFECT LOW-ORDER STATISTICS at this resolution.
K-path (2-5x cheaper) is the production-statistics scheme; E-path is the
correctness instrument (weak div 1e-6 vs uncontrolled; no spurious
pressure-work) for wall-pressure-sensitive / long-horizon cases.
