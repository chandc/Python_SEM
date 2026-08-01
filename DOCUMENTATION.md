# `pmg_dtdiv` — VVP Least-Squares Spectral Element Solver

### Divided-form time discretisation, matrix-free tensor-product operator

This document describes the solver in `pmg_dtdiv/`: the governing equations, the
spectral-element discretisation, the least-squares formulation, and — in detail — how
the matrix-free operator and **its transpose** are constructed without ever forming a
matrix.

`pmg_dtdiv` is a sibling of `../pmg_clean`. The two differ in exactly one respect: where
$\Delta t$ sits in the momentum equations (§3.2). They are mathematically equivalent and
verified to agree (§10).

---

## 1. Governing equations

The incompressible Navier–Stokes equations are recast as a **first-order** system in four
unknowns $\mathbf{U} = (u, v, p, \omega)^{T}$ by promoting vorticity to a primary variable:

$$
\begin{aligned}
\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} + v\frac{\partial u}{\partial y}
  + \frac{\partial p}{\partial x} + \nu\frac{\partial \omega}{\partial y} &= 0
  &&\text{($x$-momentum)}\\[4pt]
\frac{\partial v}{\partial t} + u\frac{\partial v}{\partial x} + v\frac{\partial v}{\partial y}
  + \frac{\partial p}{\partial y} - \nu\frac{\partial \omega}{\partial x} &= 0
  &&\text{($y$-momentum)}\\[4pt]
\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} &= 0
  &&\text{(continuity)}\\[4pt]
\omega + \frac{\partial u}{\partial y} - \frac{\partial v}{\partial x} &= 0
  &&\text{(vorticity definition)}
\end{aligned}
$$

with $\nu = 1/\mathrm{Re}$ (code: `pr = 1./re`) and the sign convention
$\omega = \partial v/\partial x - \partial u/\partial y$.

### 1.1 Why the viscous terms take this form

For a divergence-free field,

$$
\nabla^{2} u = -\frac{\partial \omega}{\partial y},
\qquad
\nabla^{2} v = +\frac{\partial \omega}{\partial x}
$$

obtained by differentiating $\omega = v_x - u_y$ and eliminating $u_{xx}$ with
$u_x = -v_y$. Hence $-\nu\nabla^{2}u \mapsto +\nu\,\omega_y$ and
$-\nu\nabla^{2}v \mapsto -\nu\,\omega_x$: the second-order viscous operator is replaced by
first derivatives of $\omega$.

### 1.2 Consequences

| | |
|---|---|
| **Only first derivatives appear** | $C^{0}$ Lagrange elements suffice; no mixed spaces, no inf–sup (LBB) condition — $u,v,p,\omega$ all use the identical basis |
| **The LS operator is SPD** | $\mathcal{L}^{T}\mathcal{L}$ is symmetric positive semi-definite for any $\mathrm{Re}$; no upwinding or stabilisation is needed |
| **Cost** | four unknowns per node instead of three |
| **Pressure enters only via $\nabla p$** | the discrete operator carries a constant-pressure null mode (§8.3) |

---

## 2. Notation

| symbol | meaning |
|---|---|
| $N$ | polynomial order; `nterm` $= N+1$ points per direction |
| $\xi_i$ | Legendre–Gauss–Lobatto (LGL) nodes on $[-1,1]$ |
| $w_i$ | LGL quadrature weights |
| $D_{in} = \ell_n'(\xi_i)$ | 1-D differentiation matrix (`d(i,n)`) |
| $h_x, h_y$ | element width / height (`wid`, `wht`) |
| $\mathcal{L}$ | the $4\times4$ first-order operator |
| $\mathbf{r} = \mathcal{L}\mathbf{U} - \mathbf{f}$ | residual |
| $\mathcal{N}(u)$ | all spatial terms: convection $+$ pressure gradient $+$ viscous |

---

## 3. Time discretisation

### 3.1 BDF family

$$
\frac{\texttt{fac1}\,u^{n+1} + \texttt{fac2}\,u^{n} + \texttt{fac3}\,u^{n-1}}{\Delta t}
\;+\; \mathcal{N}(u^{n+1}) \;=\; 0
$$

| scheme | `fac1` | `fac2` | `fac3` | used |
|---|---|---|---|---|
| BDF1 (implicit Euler) | $1$ | $-1$ | $0$ | first step / cold start |
| BDF2 | $3/2$ | $-2$ | $1/2$ | thereafter |

Continuity and the vorticity definition carry no time derivative — they are **algebraic
constraints** imposed at level $n+1$.

### 3.2 The defining choice: divided form

`pmg_dtdiv` implements the momentum equations exactly as written above — **divided by
$\Delta t$** — so that $\Delta t$ appears *only* on the transient term:

$$
\boxed{\;
\underbrace{\frac{\texttt{fac1}\,u^{n+1} + \texttt{fac2}\,u^{n} + \texttt{fac3}\,u^{n-1}}{\Delta t}}_{\text{only place }\Delta t\text{ occurs}}
\;+\; \underbrace{u\,u_x + v\,u_y + p_x + \nu\,\omega_y}_{\mathcal{N}(u),\ \text{no }\Delta t}
\;=\;0 \;}
$$

`../pmg_clean` instead stores the same equation multiplied through by $\Delta t$:

$$
\texttt{fac1}\,u^{n+1} + \texttt{fac2}\,u^{n} + \texttt{fac3}\,u^{n-1} + \Delta t\,\mathcal{N}(u) = 0
$$

**Why the divided form is useful.** With $\Delta t$ confined to the transient term,
$\Delta t \to \infty$ annihilates it and leaves the *steady* equations $\mathcal{N}(u)=0$
directly. $\Delta t$ becomes a pure continuation parameter that never touches the spatial
operator — the basis of pseudo-transient continuation.

**Caveat.** The $\texttt{fac1}/\Delta t$ term is also what keeps the Jacobian diagonally
dominant. As $\Delta t \to \infty$ that stabilisation vanishes with the transient, so
Newton needs a good initial guess: ramp $\Delta t$ up from a converged state rather than
cold-starting at a large value.

---

## 4. Spectral element discretisation

### 4.1 Mapping

Elements are axis-aligned rectangles mapped to $[-1,1]^2$, so the metric is **diagonal and
constant per element**:

$$
J = \frac{h_x h_y}{4}\ (\texttt{ajac}), \qquad
\frac{\partial \xi}{\partial x} = \frac{2}{h_x}\ (\texttt{facx}), \qquad
\frac{\partial \eta}{\partial y} = \frac{2}{h_y}\ (\texttt{facy})
$$

No Jacobian matrix, no chain-rule cross terms.

### 4.2 Tensor-product basis

Within an element the solution is a tensor-product Lagrange interpolant on LGL nodes:

$$
u(\xi,\eta) \;=\; \sum_{i=1}^{N+1}\sum_{j=1}^{N+1} u_{ij}\,\ell_i(\xi)\,\ell_j(\eta)
$$

Nodes come from `jacobl`, weights from `quad`, and $D$ from `derv` (`lgl_baseline.f90`).

### 4.3 Node and DOF indexing

Node $(i,j)$ — $i$ the $x$-index, $j$ the $y$-index — maps to the linear index

$$
\texttt{ij} = (i-1)(N+1) + j \qquad\text{(code: \texttt{li(i) + j}, \texttt{li(n)=(n-1)*nterm})}
$$

With `ndep` $=4$ fields interleaved ($\texttt{iu}=1,\ \texttt{iv}=2,\ \texttt{ip}=3,\ \texttt{iom}=4$):

$$
\texttt{dof} = \bigl[(i-1)(N+1) + (j-1)\bigr]\cdot 4 + \texttt{component}
$$

Interleaving keeps all four unknowns at a node contiguous — what makes nodal-block
preconditioning cheap.

---

## 5. Tensor-product derivatives — the matrix-free core

### 5.1 Forward application

The two derivative operators are tensor products with the identity:

$$
\mathbf{D}_x \;=\; D \otimes I, \qquad \mathbf{D}_y \;=\; I \otimes D
$$

so each acts on **one index only**:

$$
\bigl(\mathbf{D}_x u\bigr)_{ij}
= \frac{2}{h_x}\sum_{n=1}^{N+1} D_{in}\,u_{nj},
\qquad
\bigl(\mathbf{D}_y u\bigr)_{ij}
= \frac{2}{h_y}\sum_{n=1}^{N+1} D_{jn}\,u_{in}
$$

In code (`rhs`, lines 182–200):

```fortran
! x-derivative:  contract over the FIRST index
ij = li(i) + j          ! target node (i,j)
kk = li(n) + j          ! source node (n,j)   <- x-index varies
dx = d(i,n)*facx
dudx(ij) = dudx(ij) + dx*u(kk,ne)

! y-derivative:  contract over the SECOND index
ij = li(i) + j          ! target node (i,j)
kk = li(i) + n          ! source node (i,n)   <- y-index varies
dy = d(j,n)*facy
dudy(ij) = dudy(ij) + dy*u(kk,ne)
```

**Cost.** A full $(N+1)^2 \times (N+1)^2$ matrix–vector product would be
$\mathcal{O}(N^4)$ per element. The tensor-product contraction is
$\mathcal{O}(N^3)$ — the standard sum-factorisation saving, and the reason no matrix is
stored.

### 5.2 The adjoint of a tensor-product derivative

This is the key identity for §7. For any vectors $u, s$,

$$
\langle \mathbf{D}_x u,\ s\rangle
= \sum_{ij} s_{ij}\sum_n D_{in} u_{nj}
= \sum_{nj} u_{nj} \sum_i D_{in} s_{ij}
= \langle u,\ \mathbf{D}_x^{T} s\rangle
$$

hence

$$
\boxed{\ \bigl(\mathbf{D}_x^{T} s\bigr)_{ij} = \frac{2}{h_x}\sum_{n} D_{ni}\,s_{nj}
\qquad
\bigl(\mathbf{D}_y^{T} s\bigr)_{ij} = \frac{2}{h_y}\sum_{n} D_{nj}\,s_{in}\ }
$$

**The transpose is the same contraction with the index pair of $D$ swapped**:
`d(i,n)` $\to$ `d(n,i)`. Nothing is transposed in memory; the loop simply reads $D$ the
other way round. This is what makes $\mathcal{L}^{T}$ as cheap as $\mathcal{L}$.

---

## 6. Least-squares formulation

Write the time-discrete system as $\mathcal{L}\mathbf{U} = \mathbf{f}$. LSSEM minimises the
residual in a discrete $L^2$ norm:

$$
J(\mathbf{U}) \;=\; \tfrac{1}{2}\sum_{e}\ \sum_{i,j}
w_{ij}^{e}\,\bigl|\mathcal{L}\mathbf{U} - \mathbf{f}\bigr|^{2}_{ij},
\qquad
w_{ij}^{e} = J^{e}\,w_i\,w_j \ \ (\texttt{facem})
$$

Stationarity gives the **normal equations**

$$
\mathcal{L}^{T} W \mathcal{L}\,\mathbf{U} \;=\; \mathcal{L}^{T} W \mathbf{f},
\qquad W = \mathrm{diag}(w_{ij})
$$

$\mathcal{L}^{T}W\mathcal{L}$ is symmetric positive semi-definite by construction for any
Reynolds number.

### 6.1 Newton linearisation of convection

Convection is linearised about the current iterate $(f_u, f_v)$, refreshed each
sub-iteration:

$$
u\frac{\partial u}{\partial x}
\;\approx\;
\underbrace{f_u\frac{\partial u}{\partial x} \;+\; u\frac{\partial f_u}{\partial x}}_{\text{linear in the unknown}}
\;-\;
\underbrace{f_u\frac{\partial f_u}{\partial x}}_{\text{known, moved to RHS}}
$$

Both the frozen-coefficient and frozen-gradient terms are retained — this is **true
Newton**, not Picard.

---

## 7. The matrix-free operator and its transpose

Neither $\mathcal{L}$ nor $\mathcal{L}^{T}W\mathcal{L}$ is ever assembled. Both `rhs`
(residual) and `lhs` (matrix–vector product) use the identical two-pass structure.

### 7.1 Pass 1 — apply $\mathcal{L}$

Evaluate the four equation residuals at every LGL node, pre-multiplied by $w_{ij}$:

$$
\begin{aligned}
su_1 &= \Bigl[\tfrac{\texttt{fac1}}{\Delta t}u
        + f_u u_x + f_v u_y + u\,\partial_x f_u + v\,\partial_y f_u
        + p_x + \nu\,\omega_y\Bigr] w_{ij} \\[2pt]
su_2 &= \Bigl[\tfrac{\texttt{fac1}}{\Delta t}v
        + f_u v_x + f_v v_y + u\,\partial_x f_v + v\,\partial_y f_v
        + p_y - \nu\,\omega_x\Bigr] w_{ij} \\[2pt]
su_3 &= \bigl[u_x + v_y\bigr] w_{ij} \\[2pt]
su_4 &= \bigl[\omega + u_y - v_x\bigr] w_{ij}
\end{aligned}
$$

```fortran
su(ij,1) = fac1*u(ij,ne)*facem/dt + ( fu*dudx + fv*dudy + u*dfudx + v*dfudy &
                                      + dpdx + pr*domdy )*facem
su(ij,3) = ( dudx(ij) + dvdy(ij) )*facem
su(ij,4) = ( om(ij,ne) + dudy(ij) - dvdx(ij) )*facem
```

In `rhs` a second block adds the known terms — the BDF history (divided by $\Delta t$) and
the deferred Newton product (not divided) — and negates:

$$
su_1 \;\leftarrow\; \Bigl[\tfrac{-\texttt{fac2}u^{n} - \texttt{fac3}u^{n-1}}{\Delta t}
+ \bigl(f_u \partial_x f_u + f_v \partial_y f_u\bigr)\Bigr] w_{ij} \;-\; su_1
$$

so $su_i = -r_i\,w_{ij}$, the weighted residual.

### 7.2 Pass 2 — construct and apply $\mathcal{L}^{T}$

We need $c_k = \partial J/\partial U_k$, i.e.

$$
c_k \;=\; \sum_{i=1}^{4}\Bigl(\frac{\partial R_i}{\partial U_k}\Bigr)^{T} su_i
$$

Each $\partial R_i/\partial U_k$ is one of only **three shapes**, and each has a simple
adjoint:

| shape in $\mathcal{L}$ | example | adjoint | implementation |
|---|---|---|---|
| **scalar multiple of identity** | $\texttt{fac1}/\Delta t$, $\ \omega$ in $R_4$ | itself | multiply $su$ at the **target** node |
| **nodal coefficient (diagonal)** | $\partial_x f_u$ in $R_1$ | itself | multiply $su$ at the **target** node |
| **coefficient $\times$ derivative** | $f_u\,\mathbf{D}_x$ | $\mathbf{D}_x^{T}\,\mathrm{diag}(f_u)$ | multiply $su$ by $f_u$ at the **source** node, then contract with $D_{ni}$ |

The third row is the one that is easy to get wrong: for $A = \mathrm{diag}(f_u)\mathbf{D}_x$
the adjoint is $A^{T} = \mathbf{D}_x^{T}\mathrm{diag}(f_u)$, so the coefficient must be
evaluated at the **source** index `kk`, not the target `ij`. The code does exactly this:

```fortran
kk = li(n) + j                     ! source node
dx = d(n,i)*facx                   ! <-- index pair SWAPPED  => D_x^T
c1(ij,ne) = dx*su(kk,3) + fu(kk,ne)*dx*su(kk,1) + c1(ij,ne)
!                          ^^^^^^^^ coefficient at the SOURCE node
```

Assembling all four rows:

$$
\begin{aligned}
c_1 \;=\;& \underbrace{\tfrac{\texttt{fac1}}{\Delta t} su_1 + (\partial_x f_u)su_1 + (\partial_x f_v)su_2}_{\text{diagonal}}
 + \underbrace{\mathbf{D}_x^{T}su_3 + \mathbf{D}_x^{T}(f_u su_1)}_{x\text{-contraction}}
 + \underbrace{\mathbf{D}_y^{T}su_4 + \mathbf{D}_y^{T}(f_v su_1)}_{y\text{-contraction}}\\[4pt]
c_2 \;=\;& \tfrac{\texttt{fac1}}{\Delta t} su_2 + (\partial_y f_u)su_1 + (\partial_y f_v)su_2
 - \mathbf{D}_x^{T}su_4 + \mathbf{D}_x^{T}(f_u su_2)
 + \mathbf{D}_y^{T}su_3 + \mathbf{D}_y^{T}(f_v su_2)\\[4pt]
c_3 \;=\;& \phantom{0} \mathbf{D}_x^{T}su_1 + \mathbf{D}_y^{T}su_2 \\[4pt]
c_4 \;=\;& su_4 \;-\; \nu\,\mathbf{D}_x^{T}su_2 \;+\; \nu\,\mathbf{D}_y^{T}su_1
\end{aligned}
$$

Two structural facts worth noting:

- **$c_3$ has no diagonal term.** Pressure appears in the PDE only as $\nabla p$, so
  $\partial R_i/\partial p$ is always a derivative. Adding a constant to $p$ leaves $J$
  unchanged $\Rightarrow$ exact constant-pressure null mode (§8.3). In code, `c3` is
  initialised to `0.0` and receives only contraction terms.
- **$c_4$ has a diagonal term $su_4$** from the bare $\omega$ in the vorticity definition,
  which is why the $\omega$ block is well-conditioned.

### 7.3 Consistency requirement

Because the Jacobian is never assembled, **pass 2 must be the exact transpose of pass 1**.
There is no mechanism that enforces this — it is maintained by hand. Any edit to the
operator must therefore change both passes in lockstep, in **all** routines that
implement it:

| routine | file | role |
|---|---|---|
| `rhs` | `lssem_baseline.f90` | nonlinear residual $\mathcal{L}^{T}W\mathbf{r}$ |
| `lhs` | `lssem_baseline.f90` | matrix–vector product $\mathcal{L}^{T}W\mathcal{L}\,\delta\mathbf{U}$ |
| `lhs_fast` | `solver_pmg2.f90` | same, with cached $\nabla f_u,\nabla f_v$ |
| `dge` | `solver_pmg2.f90` | diagonal of $\mathcal{L}^{T}W\mathcal{L}$ for preconditioning |

`dge` is the one place a Jacobian block *is* formed explicitly — the $4\times4$ array `aa`
with $\texttt{aa}(i,k) = \partial R_i/\partial U_k$ at a node — but only to extract
$\mathrm{diag} = \sum_i \texttt{aa}(i,k)^2$ (column norms), never to solve with.

---

## 8. Inter-element assembly and boundary conditions

### 8.1 Direct stiffness

Elements couple only through `collect`, which for each shared face adds the two elements'
contributions and writes the **sum back into both copies** — the $QQ^{T}$ gather–scatter
enforcing $C^{0}$ continuity:

```fortran
resu = res(ijs+iu,ne) + res(ijn+iu,isouth(ne))
res(ijs+iu,ne) = resu ;  res(ijn+iu,isouth(ne)) = resu
```

Only south and west neighbour lists are traversed, so each interface is visited once.
Applied to both the residual and the preconditioner diagonal.

### 8.2 Dirichlet conditions — the mask

With $M = \mathrm{diag}(\texttt{mask})$ a 0/1 projector, a Dirichlet condition is **three
coordinated actions**:

1. the value is written into $\mathbf{f}$ each sub-iteration;
2. the residual is projected, $\texttt{res} \leftarrow M\,\texttt{res}$;
3. the BiCGSTAB update is projected, $\mathbf{f} \leftarrow \mathbf{f} + M\,\delta$.

so the system actually solved is

$$
\bigl(M\,\mathcal{L}^{T}W\mathcal{L}\,M\bigr)\,\delta\mathbf{U}
= M\,\mathcal{L}^{T}W\bigl(\mathbf{f} - \mathcal{L}\mathbf{U}\bigr)
$$

This is **symmetric elimination** — rows *and* columns projected out — not "zero the row
and put 1 on the diagonal", which would destroy symmetry. Prescribed values still reach
the interior because $\mathbf{f}$ already holds them when `rhs` evaluates
$\mathcal{L}\mathbf{U}$.

Edge codes: `0` interior, `1` no-slip, `2` lid, `3` inlet, `4` outlet ($p=0$), `5`
symmetry ($v=0,\ \omega=0$). In the free-outflow driver the **east** code-4 branch is
deliberately empty: $u,v,p,\omega$ are all unknowns at the outlet.

### 8.3 Pressure pin

Because $c_3$ receives only differentiated contributions, $\mathcal{L}^{T}W\mathcal{L}$ has
a constant-pressure null mode when no Dirichlet pressure exists anywhere. Controls:

| `npin_e` | behaviour |
|---|---|
| $0$ | auto-detect the SE outlet corner |
| $>0$ | use that element (`npin_j` selects the east-edge node; $-1$ = top) |
| $<0$ | **no pin at all** — the operator keeps its null mode |

A Krylov method started from $\delta = 0$ on a *consistent* singular system stays in the
range space and never excites the null component, so `npin_e = -1` is well behaved in
practice (verified: 10 000 steps, zero capped solves, no NaN).

---

## 9. Solution algorithm

```
read mesh, build LGL basis, set BC masks
for it = 1 .. ntime
    t <- t + dt
    for im = 1 .. nsub                          ! Newton sub-iteration
        impose Dirichlet values into f
        dge                 -> preconditioner diagonal
        unpack f -> u,v,p,om ; freeze fu,fv <- u,v
        rhs                 -> weighted residual, then L^T
        res <- M * res      ; collect(res), collect(diag)
        res0 <- ||res||_2   ; exit if res0 <= tol
        cgstol <- max(cgsfac*res0, tol)          ! inexact Newton
        bicgstab(...)       -> update f in place
    fnn <- fn ; fn <- f ; switch BDF1 -> BDF2
```

**Inexact Newton (`cgsfac`).** The linear solve is asked only to reduce the residual by a
factor `cgsfac` relative to the current nonlinear residual, not to `tol`. Early
sub-iterations, where the Newton iterate is far from the solution, are therefore solved
loosely and cheaply. `cgsfac = 0.1` converts a cold start that diverges at step 4 under
tight solves into 10 000 clean steps with zero capped solves.

**Convergence.** A bit-identical residual is **not** proof of a fixed point — the
iteration can park at a constant residual while the flow still evolves. Require both a
long residual lock **and** a field-change check ($\max|\Delta u| < 1\%$ of range over
several hundred further time units).

---

## 10. Verification

**Equivalence with `pmg_clean` (long domain)** — identical settings
($\Delta t = 0.1$, `nsub=3`, `nitcgs=500`, `cgsfac=0.1`, 10 000 steps, cold, no pin):

| | reattachment $x_r$ | rms $\Delta u$ | rms $\Delta v$ | rms $\Delta p$ |
|---|---|---|---|---|
| `pmg_clean` | $8.11\,h$ | — | — | — |
| `pmg_dtdiv` | $8.09\,h$ | $0.035$ (2 % of range) | $0.034$ | $0.020$ |

Streamlines and profiles at $1$–$7.5\,h$ are indistinguishable. At $\Delta t = 1.0$ the two
trees agree to round-off, as they must (the division is then a no-op).

> **Do not use $\max|\Delta|/\text{range}$ to compare solutions.** The re-entrant step
> corner carries a genuine vorticity singularity whose discrete value never converges, so
> pointwise maxima are dominated by an unresolvable feature. On the comparison above
> $\max|\Delta v|/\text{range}$ reads 425 % while the rms is 2 %. **Report rms**, plus a
> physical scalar such as reattachment length.

**Do not use the truncated (5 inlet-height) case as a discriminating test.** It admits
several converged states; pin location alone changes $\max|\Delta u|$ by 81 %.

---

## 11. Build and usage

```bash
cd pmg_dtdiv
make                      # -> SEM_2D_BFS_FREEOUT
```

`FFLAGS = -O2 -g -Wall -Wextra -fcheck=bounds -fbacktrace -fdefault-real-8
-fdefault-double-8 -ffree-form`; links Accelerate for BLAS/LAPACK. The Makefile carries
explicit module dependencies (a fresh tree has no `.mod` files, so build order matters).

Representative namelist:

```fortran
&input
  fin='cnos_short_grid.dat', fout='o.dat', re=389., dt=0.1, ntime=10000, nsub=3,
  tol=1.0e-6, nitcgs=500, istart=0, frun='out.dat', iform=1,
  cgsfac=0.1, nsave=500, ystep=0.5, hinlet=0.5, npin_e=-1, npin_j=1,
  pmg_on=.false.
/
```

| parameter | role |
|---|---|
| `re` | Reynolds number on the **mesh length unit** ($\nu = 1/\texttt{re}$) |
| `dt` | time step; in this tree, only on the transient term |
| `nsub` | Newton sub-iterations per step |
| `cgsfac` | inexact-Newton forcing factor ($0$ = always solve to `tol`) |
| `nitcgs` | BiCGSTAB iteration cap |
| `npin_e`, `npin_j` | pressure pin ($-1$ disables) |
| `pmg_on` | p-multigrid preconditioner vs diagonal |

---

## 12. Differences from `pmg_clean`

1. **Momentum equations in divided form** — `fac1`, `fac2`, `fac3` divided by $\Delta t$;
   every convective, pressure-gradient, viscous, continuity and vorticity term
   $\Delta t$-free. Applied consistently in `rhs`, `lhs`, `lhs_fast`, `dge`.
2. **Dead code removed** — `lhs_NT` and `form_coefficient_matrix` (462 lines) carried the
   same $\Delta t$ pattern but were never called; they are dropped here rather than left
   half-transformed.
3. **Makefile with explicit module dependencies**, so the tree builds from clean.

Everything else — basis, mapping, assembly, boundary conditions, solver, preconditioners —
is byte-identical to `pmg_clean`.
