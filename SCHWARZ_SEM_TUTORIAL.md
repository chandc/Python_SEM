# Vertex-patch Schwarz preconditioning in a spectral-element code

A self-contained explanation of what the preconditioner is, why it works
for the FOSLS operator, and exactly how it is built and applied with
matrix-free spectral elements. Equations are kept to the minimum needed to
make each step unambiguous; every diagram corresponds to a piece of
`scratch/vertex_schwarz2d.py`.

---

## 1. The setting: spectral elements, nodes and the matrix-free operator

The domain is tiled by quadrilateral elements. Inside each element the
solution is a polynomial of degree $N$ in each direction, represented by its
values at the $(N+1)^2$ Gauss–Lobatto–Legendre (GLL) nodes. Neighbouring
elements share the nodes on their common edge, and the solution is
continuous because those shared nodes carry one value.

```
      element e1              element e2
   o---o---o---o---o       o---o---o---o---o
   |   |   |   |   |       |   |   |   |   |
   o---o---o---o---o       o---o---o---o---o        o : GLL node  (N = 4 here)
   |   |   |   |   ●  ==   ●   |   |   |   |        ● : shared edge node,
   o---o---o---o---o       o---o---o---o---o            ONE global unknown,
   |   |   |   |   |       |   |   |   |   |            stored TWICE locally
   o---o---o---o---o       o---o---o---o---o
```

Two numberings coexist:

- **local storage** `U[e, i, j, f]`: element $e$, node $(i,j)$, field $f$. A
  shared node is stored once per element that owns it (redundantly).
- **global dofs**: one index per physical unknown, `g[e,i,j,f] =
  gidx[e,i,j]·F + f`, where `gidx` maps each local node to its global node
  and $F$ is the number of fields (4 in 2D: $u, v, \omega, p$; 14 real per
  Fourier mode in 3D).

The least-squares operator is never stored. Applying it to a field means:

$$
A\,x \;=\; Q^{\!\top}\!\Big(\sum_e L_e^{\top} W_e L_e\Big) Q\,x ,
$$

where $L_e$ is the first-order residual operator on element $e$ (derivatives
by tensor-product differentiation matrices), $W_e$ the diagonal of
quadrature weights times row weights, and $Q$ the scatter from global to
local storage (its transpose $Q^{\!\top}$, "gather-scatter", **sums** the
copies of every shared node). Each element block

$$
A_e \;=\; L_e^{\top} W_e L_e \qquad\big((N+1)^2 F \times (N+1)^2 F\big)
$$

is dense within the element and couples nothing outside it. The global
matrix is the sum of the element blocks placed at their global positions:

$$
A \;=\; \sum_e Q_e^{\!\top} A_e Q_e .
$$

CG needs only $x \mapsto Ax$, so $A$ itself is never assembled.

---

## 2. Why pointwise preconditioning fails here

CG converges in a number of iterations governed by how far apart the
eigenvalues of $M^{-1}A$ are. With Jacobi, $M = \mathrm{diag}(A)$. For the
FOSLS operator at the channel's time step there is a large family of modes,
divergence-free velocity swirls with their matched vorticity, on which $A$
acts almost like the identity (eigenvalue $\sim 1$) while the diagonal of
$A$ at those nodes is $\sim N^4/h^2$. Jacobi therefore mis-scales them by
$10^4$–$10^5$, and no node-by-node rule can do better because a single node
cannot distinguish a swirl from a gradient (ADN_FOSLS.md §3–5).

```
      a divergence-free swirl centred on a vertex
                 ↑
            ←    ●    →          A(swirl)  ≈ 1 · swirl
                 ↓               diag(A)   ≈ N^4/h^2   at every node of it
```

A correction that "sees" the whole swirl needs all four elements around the
vertex at once. That is the vertex patch.

---

## 3. The Schwarz idea in three equations

Choose overlapping sets of unknowns $V_1,\dots,V_P$ (the patches) and let
$R_i$ be the restriction that picks out the unknowns of patch $i$
($R_i$ is a $|V_i| \times n$ selection matrix). The **patch matrix** is the
piece of $A$ that lives on the patch:

$$
A_i \;=\; R_i\,A\,R_i^{\!\top} .
$$

The **additive Schwarz preconditioner** solves every patch exactly, with
everything outside the patch frozen, and adds the corrections:

$$
M^{-1} r \;=\; \sum_{i=1}^{P} R_i^{\!\top} A_i^{-1} R_i\, r
\;\;+\;\; P\,A_c^{-1} P^{\!\top} r .
$$

The last term is a **coarse correction** on a low-order space (here the
same elements at degree 2): $P$ interpolates coarse to fine, $A_c$ is the
operator rediscretised at degree 2 and solved directly. Patches fix
everything local; the coarse level fixes what every patch sees as a constant
(a slow drift across the whole mesh, e.g. of pressure).

Because $M^{-1}$ is a sum of symmetric positive semi-definite terms and at
least one of them is definite on every unknown, $M^{-1}$ is symmetric
positive definite and CG applies unchanged.

---

## 4. What a vertex patch is

For each mesh vertex $v$, the patch is every unknown on every element that
has $v$ as a corner: four elements in the interior, two on a wall, one in a
corner. The patch covers a $(2N+1)\times(2N+1)$ block of nodes.

```
          patch of vertex v  (4 elements, 2N+1 = 9 nodes across for N = 4)

    o o o o o o o o o        elements:  e1 | e2
    o o o o o o o o o                   ----+----
    o o o o o o o o o                    e3 | e4
    o o o o o o o o o
    o o o o ● o o o o   ← v
    o o o o o o o o o
    o o o o o o o o o        dofs in patch  n_p = (2N+1)^2 · F
    o o o o o o o o o        N = 8, F = 4  :  1156
    o o o o o o o o o        N = 8, F = 14 :  4046   (3D, per Fourier mode)
```

Patches overlap heavily: every element belongs to the patches of its four
corners, so an interior unknown is corrected by up to four patches per
sweep. The overlap is not optional. Measured with the same operator
(ADN_FOSLS.md §8.5): element blocks (no overlap) 655 iterations; element +
one node layer 330; + two layers 231; **full vertex patch 42**. Only the
element-sized overlap contains the local swirls, which is Pavarino's
condition for $p$-independence on spectral elements.

```
   element block       element + 1 layer        vertex patch
   ┌────────┐          ┌──┬────────┬──┐         ┌────────┬────────┐
   │        │          │  │        │  │         │        │        │
   │   e    │          │  │   e    │  │         │   e1   │   e2   │
   │        │          │  │        │  │         │        │        │
   └────────┘          └──┴────────┴──┘         ├────────●────────┤
                                                │   e3   │   e4   │
      655 it              330 it                │        │        │
                                                └────────┴────────┘
                                                      42 it
```

---

## 5. Building the patch matrices without a global matrix

### 5.1 Element blocks by probing

Column $j$ of any matrix is that matrix applied to the $j$-th unit vector.
Place a unit value at local node $(i,j)$ and field $f$ **in every element
at the same time**, apply the element operator (no gather-scatter), and the
result holds column $((i\cdot n)+j)F+f$ of every element's $A_e$
simultaneously:

$$
A_e[:, c] \;=\; \big(L_e^{\top} W_e L_e\big)\, \mathbf e_c ,\qquad
c = 0,\dots,(N+1)^2F-1 .
$$

That is $(N+1)^2F$ operator applications for the whole mesh, independent of
the number of elements (and, in 3D, of the number of Fourier modes, since
the batched apply evaluates all modes at once). No hand-derived matrix, so
the blocks are the operator CG applies, to round-off.

```
   probe c = (i,j,f)             every element answers at once

   e1: [0 .. 1 .. 0]  ─┐        A_e1[:, c]
   e2: [0 .. 1 .. 0]  ─┼─ L^T W L ─▶  A_e2[:, c]
   e3: [0 .. 1 .. 0]  ─┘        A_e3[:, c]
```

### 5.2 Patch topology

```
   corners[v]    = elements having v as a corner              → the patch
   node_elems[n] = elements containing global node n
   ring(v)       = ∪_{n ∈ nodes of the patch} node_elems[n]   → patch + neighbours
```

The patch dof list is the sorted set of global dofs of the patch elements,
minus the fixed ones (wall values, lid, pinned pressure).

### 5.3 Assembling $A_i = R_i A R_i^{\top}$ — the ring matters

Because $A = \sum_e Q_e^{\top} A_e Q_e$, the restriction to the patch is

$$
A_i \;=\; \sum_{e\,\in\,\mathrm{ring}(v)} \big(R_i Q_e^{\!\top}\big)\, A_e\, \big(Q_e R_i^{\!\top}\big),
$$

i.e. every element that touches **any** patch dof contributes its block,
restricted to the patch's dofs. The elements outside the patch but adjacent
to its boundary (the ring) contribute to the patch-boundary nodes. Leaving
them out under-stiffens those nodes and made 16 of 25 patch matrices
singular in the first attempt.

```
              ring(v): patch elements  █  and their neighbours ░

        ░  ░  ░  ░                 a boundary node ○ of the patch
        ░  █  █  ░                 receives stiffness from █ AND ░ ;
        ░  █  █  ░                 the ░ interiors are frozen (Dirichlet),
        ░  ░  ░  ░                 only their contribution to ○ is kept
```

### 5.4 Equilibrate, factor, store

Pressure rows carry a weight $a_{\text{flux}}^2 \sim dt^2 \approx 10^{-8}$
against $O(1)$ mass rows, so a raw Cholesky loses those pivots. Scale
symmetrically first:

$$
s = \mathrm{diag}(A_i)^{-1/2},\qquad \tilde A_i = s A_i s = \tilde L \tilde L^{\top},\qquad
A_i^{-1} r = s\,\tilde L^{-\top}\tilde L^{-1}(s\,r).
$$

Store $\tilde L$ and $s$ per patch (per Fourier mode in 3D).

---

## 6. Applying $M^{-1}$ inside CG

CG hands the preconditioner a residual in **redundant local storage**: after
gather-scatter every copy of a shared node already holds the assembled
value. The apply is:

```
   r  (local, nelem × n × n × F)          copies of a shared node are equal
     │
     ▼  rg = bincount(g, r · mw)          mw = 1/multiplicity, so summing the
     │                                    copies returns the value once
   rg (global)
     │
     ▼  for each patch i:   z_g[dofs_i] += s ⊙ chol_solve( s ⊙ rg[dofs_i] )
     │                      (all patches independent → batched / parallel)
   z_g (global)
     │
     ▼  z = z_g[g] · mask                 expand back to local copies,
     │                                    zero the prescribed dofs
     ▼  z += P A_c^{-1} P^T r             coarse term (PMG transfers)
   z  (local)
```

The multiplicity weight `mw` is what makes the gather the true adjoint of
the scatter; with it the whole preconditioner is symmetric in the
multiplicity-weighted inner product CG uses.

---

## 7. The coarse correction

The operator is rediscretised on the same elements at degree $p_c = 2$
(one 3×3 node block per element) and factored directly. Transfers are
nodal interpolation in each direction,

$$
P = I_{p_c\to N}\otimes I_{p_c\to N}\quad\text{per element},\qquad
\text{restriction} = \text{multiplicity-weighted } P^{\!\top},
$$

so the coarse term $P A_c^{-1} P^{\top}$ is symmetric. It removes the
$h$-dependence that patches alone leave: on the 2D tests the one-level
method went 53 → 90 iterations from 4×4 to 8×8 elements, the two-level
42 → 53 (ADN_FOSLS.md §8.4).

---

## 8. Why it works, in one line each

- **Inside a patch the solve is exact**, so every mode that fits in a patch
  is corrected with the right eigenvalue — swirls included.
- **Overlap** lets neighbouring patches agree on modes straddling their
  boundaries; element-sized overlap is what contains the local swirls.
- **The coarse level** handles what every patch sees as constant.
- **Symmetry** is kept by additive combination and multiplicity-weighted
  gather/scatter, so plain CG applies.

Measured result (2D, production `pcg_solve`, c = 5405): 31 → 26 → 25 CG
iterations for N = 8 → 12 → 16 (Jacobi 1286 → 3497); the same at c = 1;
at N = 30 in the cavity 19 iterations against 8365 for Jacobi.

---

## 9. What it costs, and the exact way to cut it

A dense patch factor holds $\tfrac12 n_p^2$ entries with
$n_p=(2N+1)^2F$: storage and apply $\propto (2N+1)^4$, factorisation
$\propto (2N+1)^6$; 32 GB for the 3D channel at $N=8$, 44 GB for the 2D
cavity at $N=30$ (ADN_FOSLS.md §13).

**Static condensation** removes most of it without changing the result.
Interior nodes of an element ($1\le i,j\le N-1$) couple only to that
element's own dofs, so inside a patch

$$
A_i = \begin{pmatrix} K_{II} & K_{IB}\\ K_{BI} & K_{BB}\end{pmatrix},\qquad
K_{II} = \mathrm{blockdiag}_e\,(A_e)_{II},
$$

with $I$ = interior dofs of the patch's elements and $B$ = dofs on element
edges. Factor each element's interior block **once** (it is the same for all
four patches containing the element) and each patch's dense Schur complement
on its edge dofs:

$$
S_i = K_{BB} - K_{BI} K_{II}^{-1} K_{IB}\quad(\text{size} \approx 12N F),
$$

$$
z_B = S_i^{-1}\big(r_B - K_{BI}K_{II}^{-1} r_I\big),\qquad
z_I = K_{II}^{-1}\big(r_I - K_{IB} z_B\big).
$$

```
      interior dofs  ·  (per element, factored once, shared)
      edge dofs      ○  (patch Schur complement, dense but small)

      ○ ○ ○ ○ ○ ○ ○ ○ ○
      ○ · · · ○ · · · ○
      ○ · · · ○ · · · ○        stored:  Σ_e ((N-1)^2 F)^2 / 2   interiors
      ○ · · · ○ · · · ○               + Σ_patches (12 N F)^2 / 2  edges
      ○ ○ ○ ○ ● ○ ○ ○ ○
      ○ · · · ○ · · · ○        measured (2D, c = 5405): identical iterations,
      ○ · · · ○ · · · ○        memory 7.7× (N=8) → 13× (N=24) smaller,
      ○ · · · ○ · · · ○        apply 2–4× faster
      ○ ○ ○ ○ ○ ○ ○ ○ ○
```

(`scratch/adn_schwarz_condensed.py`; LOW_MEMORY_PATCH_SOLVERS.md §3.)

---

## 10. The 3D channel: one patch per vertex per Fourier mode

The 3D code is a 2D element mesh in $(x,y)$ with Fourier modes in $z$. The
implicit operator has constant coefficients in $z$ (convection is explicit),
so it is block-diagonal over modes, with $\partial_z \to i k_z$ inside each
block. Consequently:

- a patch is still the four elements around a vertex of the 2D mesh, with
  $F = 14$ real fields (7 complex: $u,v,w,\omega_x,\omega_y,\omega_z,p$),
  one dense block **per patch per mode**; modes never couple;
- the operator does not change during the run (it depends on the mesh,
  $\nu$, $k_z$ and $c = 1/(\beta_k\,dt)$ only), so the probes and factors
  are built **once**, for the three RKW3 values of $c$, and reused for the
  whole DNS;
- the mask must be built for the full set of modes and sliced ($k_z=0$ has
  its imaginary half masked; rebuilding on a subset masks the wrong modes);
- the pressure is pinned at $k_z = 0$ only.

Sizes at $N=8$ on the 6×18 production mesh: 114 patches × 17 modes ×
2023 complex dofs; 32 GB of dense factors, ≈ 4 GB condensed, per value of
$c$ (VERTEX_SCHWARZ_IMPLEMENTATION.md §3b).

---

## 11. Pseudocode

```
BUILD(state, mask):
    blocks   = probe_element_blocks(state)              # (nelem, n_loc, n_loc)  §5.1
    g        = gidx*F + f                               # local -> global dof
    free     = mask > 0.5 at any copy
    mw       = 1 / multiplicity(g)
    for each vertex v:
        elems = corners[v]; ring = elements touching any node of elems
        dofs  = unique(g[elems]) ∩ free
        K     = 0
        for e in ring:  K[loc(g[e]), loc(g[e])] += blocks[e]  restricted to dofs
        s = 1/sqrt(diag K);  Lf = cholesky(s K s)
        patches.append((dofs, s, Lf))
    coarse = PMG(orders=(N, 2), direct)                  # P, A_c^-1, P^T

APPLY(r):                                                # r in local storage
    rg = bincount(g, r*mw)
    zg = 0
    for (dofs, s, Lf) in patches:  zg[dofs] += s * solve(Lf, s * rg[dofs])
    z  = zg[g] * mask
    return z + coarse(r)
```

With condensation, `K` is never formed as one dense block: the element
interior factors and the edge Schur complements replace `(dofs, s, Lf)`
(§9).

---

## 12. Pitfalls that cost a run each

| symptom | cause | fix |
|---|---|---|
| CG stalls at residual ≈ 1, some patches fall back to LU | patch built from its own elements only | include the ring (§5.3) |
| Cholesky fails (`potrf info > 0`) | pressure pivots $\sim dt^2$ | symmetric equilibration (§5.4) |
| divergence or wrong answers in 3D | mask rebuilt on a mode subset | build mask for full $n_k$, slice |
| preconditioner works on the assembled harness but not in `pcg_solve` | residual copies not gathered with `mw` | §6 |
| iterations grow with element count | no coarse level | §7 |
| iterations grow with $N$ despite patches | thin overlap, or patches only on coarse multigrid levels | element-sized overlap on the fine level (§4, ADN_FOSLS §12) |
| multiplicative/undamped use inside a V-cycle makes CG produce NaN | indefinite smoother | damp by $1/\lambda_{\max}(M^{-1}A)$, or use it additively |

---

## 13. Where to look

- Code: `scratch/vertex_schwarz2d.py` (production-path class),
  `scratch/adn_schwarz_condensed.py` (condensed variant),
  `scratch/pmgv2d.py` (as a multigrid smoother, for the record).
- Measurements: ADN_FOSLS.md §8, §10–13; LOW_MEMORY_PATCH_SOLVERS.md §3.
- 3D port plan: VERTEX_SCHWARZ_IMPLEMENTATION.md §4.
- Theory: Toselli & Widlund, *Domain Decomposition Methods* (2005), ch. 2–3
  (abstract Schwarz theory); Pavarino, Numer. Math. 66 (1994) (spectral
  elements, overlap); Arnold, Falk & Winther, Numer. Math. 85 (2000)
  (H(div) kernels and patch smoothers).
