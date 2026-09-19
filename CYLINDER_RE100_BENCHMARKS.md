# Benchmark aerodynamic quantities for flow past a circular cylinder at Re = 100

**Provenance.** Sections 1-4 were supplied by the user on 2026-09-19 as a
reference compilation. They are archived here verbatim in substance. Section 5
is this project's own cross-check against them and is clearly separated; where
a number here disagrees with one I extracted independently earlier, section 5
says so rather than quietly reconciling it.

---

## Executive summary

At Re = 100, flow past an unconfined circular cylinder resides strictly within
the two-dimensional, laminar, periodic vortex-shedding regime (the laminar von
Karman vortex street). Because spanwise three-dimensional instabilities do not
emerge until the Mode A transition (Re ~ 188-190), two-dimensional simulations
resolve the exact physical dynamics. Consequently Re = 100 serves as the
foundational gold-standard benchmark for both code verification and
experimental validation.

## 1. Consensus benchmark ranges (unbounded flow)

Synthesising high-resolution simulations extrapolated to zero blockage
(B -> 0) with boundary-corrected experimental wind-tunnel and towing-tank data:

| quantity | consensus range | central value |
|---|---|---|
| Strouhal number St | 0.163 - 0.166 | **0.164** |
| time-averaged drag C_D | 1.31 - 1.35 | unconfined asymptote **1.31 - 1.33** |
| r.m.s. lift C_L,rms | 0.22 - 0.24 | peak amplitude sqrt(2)*C_L,rms = **0.32 - 0.34** |

## 2. Tabulated computational and experimental results

| study | type / method | domain or experimental detail | St | C_D | C_L,rms |
|---|---|---|---|---|---|
| Williamson (1989) | experiment, towing tank | end-plate angle control for parallel 2D shedding | 0.1643 | — | — |
| Roshko (1954) | experiment, wind tunnel | hot-wire wake, blockage corrected | 0.164-0.167 | — | — |
| Tritton (1959) | experiment, quartz fibre | direct force via fibre deflection | — | 1.26-1.32 | — |
| Wieselsberger (1921) | experiment, wind tunnel | classical baseline, confinement corrected | — | ~1.30-1.34 | — |
| **Posdziech & Grundmann (2007)** | **spectral element** | **H >= 140D, Richardson extrapolation to B -> 0** | **0.1633** | **1.310** | **0.226** |
| Henderson (1995) | spectral element | high-order expansion, B -> 0 | 0.1660 | 1.350 | — |
| Ding et al. (2007) | mesh-free least-squares / FDM | [-50D, 60D] x [-30D, 30D] | 0.164 | 1.325 | 0.228 |
| Qu et al. (2013) | high-order finite difference | multigrid domain-independence study | 0.165 | 1.326 | 0.231 |
| Mittal & Balachandar (1995) | spectral finite difference | outer radial boundary r_inf = 30D | 0.165 | 1.33 | — |
| Braza et al. (1986) | finite difference | moderate domain, H ~ 20D-30D | 0.160 | 1.36 | 0.25 |
| **Silva et al. (2003)** | **finite volume / immersed boundary** | **blockage B = 0.05 (H = 20D)** | **0.168** | **1.39** | — |
| He et al. (2000) | lattice Boltzmann | regularised 2D grid | 0.166 | 1.36 | 0.23 |
| Le et al. (2006) | immersed boundary | Cartesian uniform near-wall mesh | 0.160 | 1.37 | 0.22 |
| Liu et al. (1998) | finite element | multi-scale FEM | 0.164 | 1.35 | 0.235 |

## 3. Critical analysis of the discrepancies

### A. Drag inflation from lateral blockage (B = D/H)

Reported C_D >= 1.38 is almost universally associated with domains where the
lateral height H <= 20D (B >= 5 %). Under solid or slip boundary conditions,
lateral confinement prevents streamline expansion and accelerates flow around
the cylinder flanks (Venturi effect). This deepens base suction and inflates
C_D by 5-10 %. Expanded to H >= 140D, or extrapolated Richardson-style to
B -> 0, drag settles to the open-flow benchmark of 1.31-1.33.

### B. Resilience of the Strouhal number

Unlike drag, the shedding frequency is governed primarily by the local absolute
instability within the immediate recirculation zone (x <= 3D). Because the
formation region is shielded from the far field, St converges rapidly even in
moderately confined domains (H ~ 20D-30D), clustering tightly around 0.164.
Williamson's universal formula for parallel shedding,

    St = 0.2665 - 1.018/sqrt(Re)

gives St(100) = 0.1647, matching high-order simulation to better than 1 %.

### C. Lift reporting conventions

Shedding at Re = 100 produces a nearly monochromatic sinusoidal lift, so
C_L,max = sqrt(2) * C_L,rms. Values around 0.32-0.34 are PEAK amplitude; values
around 0.22-0.24 are r.m.s. Mixing the two is a common source of apparent
disagreement.

## 4. Primary references

- Williamson, C. H. K. (1989). *J. Fluid Mech.* **206**, 579-627.
- Roshko, A. (1954). NACA Report 1191.
- Tritton, D. J. (1959). *J. Fluid Mech.* **6**(4), 547-567.
- Wieselsberger, C. (1921). *Phys. Z.* **22**, 321-328.
- Posdziech, O. & Grundmann, R. (2007). *J. Fluids Struct.* **23**(3), 479-499.
- Henderson, R. D. (1995). *Phys. Fluids* **7**(9), 2102-2104.
- Ding, H., Shu, C., Yeo, K. S. & Xu, D. (2007). *Int. J. Numer. Meth. Fluids* **53**(5), 785-807.
- Qu, L., Norberg, C., Davidson, L., Peng, S. H. & Wang, F. (2013). *J. Fluids Struct.* **39**, 347-370.
- Mittal, R. & Balachandar, S. (1995). *Phys. Fluids* **7**(8), 1841-1865.
- Braza, M., Chassaing, P. & Minh, H. H. (1986). *J. Fluid Mech.* **165**, 79-130.
- Silva, A. L. F. L., Silveira-Neto, A. & Damasceno, J. J. R. (2003). *J. Braz. Soc. Mech. Sci. Eng.* **25**(2), 168-178.
- He, X., Zou, Q., Luo, L. S. & Dembo, M. (2000). *J. Stat. Phys.* **107**(1), 109-128.
- Le, D. V., Khoo, B. C. & Peraire, J. (2006). *J. Comput. Phys.* **220**(1), 109-138.
- Liu, C., Zheng, X. & Sung, C. H. (1998). *J. Comput. Phys.* **139**(1), 35-57.

---

## 5. What this means for our results (added by this project)

### 5.1 The single most useful row is Silva et al. (2003)

It is the only entry computed at OUR blockage ratio, B = 0.05 (H = 20D), and it
is the closest thing available to a like-for-like comparison:

| | St | C_D |
|---|---|---|
| Silva et al. (2003), B = 0.05 | 0.168 | 1.39 |
| **ours, N6 dt0.1 +AC Xu10 H20** | **0.1687** | **1.3952** |
| difference | **+0.4 %** | **+0.4 %** |

An independent code, a completely different method (finite volume / immersed
boundary against our least-squares spectral element), at the same blockage,
agreeing to 0.4 % in both quantities.  That is a far stronger statement about
our solver than the 4 % gap to the unbounded consensus, because it compares
like with like.

### 5.2 Section 3A describes our situation exactly

"Reported values of C_D >= 1.38 are almost universally associated with
computational domains where H <= 20D."  Ours is H = 20D and our C_D is 1.395.
We are a textbook instance of the effect, not an outlier from it.  This retires
the framing used earlier in this project, in which the gap to the large-domain
consensus was treated as an unexplained property of the solver.

### 5.3 Section 3B explains the St/C_D asymmetry we measured

St is set by the absolute instability within x <= 3D and is shielded from the
far field; drag is not.  That is precisely the asymmetry the domain study
found: widening the box moved C_D by 0.9 % of its value while moving St by only
0.24 %.  It also predicts the domain ladder will improve C_D substantially and
St barely -- which is what to expect from the runs in flight.

### 5.4 The extrapolation target

P&G's B -> 0 values, **St = 0.1633, C_D = 1.310, C_L,rms = 0.226**, are the
numbers the Posdziech-Grundmann ladder (Xu = 30, Xd = 50, N = 8, H = 20/40/80/
160) is aiming at.  NOTE these differ from the P&G row in Qu et al.'s Table 1
(St 0.1644, C_D 1.325, C_L' 0.228), which quotes them at H = 140 rather than
extrapolated -- the difference between the two IS the residual blockage at
H = 140, and it is not negligible: 0.0011 in St and 0.015 in C_D.

### 5.5 A caution on C_L

Our reduction reports both `C_L rms` and `C_L amp`.  Section 3C is the reason
both columns exist: our C_L rms of 0.2517 must be compared against 0.22-0.24,
NOT against 0.32-0.34, and our C_L amp of 0.3587 against the latter.  Earlier
in this project a 0.3378 amplitude was at one point set beside an rms-convention
literature value; that comparison was wrong by a factor sqrt(2).
