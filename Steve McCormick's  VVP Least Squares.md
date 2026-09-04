Steve McCormick's First-Order System Least Squares (FOSLS) framework reformulates elliptic partial differential equations into a first-order system, minimizing the $L^2$ norm of the residuals for the governing equations to yield robust error bounds and elliptic regularity. Boundary conditions, including essential or natural treatments for wall and outflow boundaries, are incorporated directly into the least-squares functional to ensure coercive bilinear forms. Algebraic Multigrid (AMG) methods are then naturally applied to the resulting symmetric, positive-definite FOSLS stiffness matrices, providing optimal $h$-independent convergence rates.
Below is the complete LaTeX markdown document detailing the FOSLS framework, boundary conditions, and AMG integration:

\documentclass{article}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{geometry}
\geometry{margin=1in}

\title{Steve McCormick's First-Order System Least Squares (FOSLS) Framework}
\author{Documentation Synthesis}
\date{\today}

\begin{document}

\maketitle

\section{Introduction to FOSLS}
The First-Order System Least Squares (FOSLS) methodology, pioneered by Steve McCormick and collaborators, is a general framework for solving elliptic partial differential equations (PDEs). Instead of dealing with higher-order operators or indefinite saddle-point systems directly, FOSLS rewrites a given second-order (or higher) scalar PDE or system as a coupled first-order system of differential equations. 

Let a general elliptic boundary value problem be represented abstractly as:
\begin{equation}
    L u = f \quad \text{in } \Omega,
\end{equation}
where $\Omega \subset \mathbb{R}^d$ is a bounded domain. By introducing auxiliary flux or gradient variables $\mathbf{U} = \mathcal{A}u$, the problem is transformed into a first-order system:
\begin{equation}
    \mathcal{D} \mathbf{Q} = \mathbf{F} \quad \text{in } \Omega,
\end{equation}
where $\mathbf{Q} = (u, \mathbf{U})^T$ and $\mathcal{D}$ is a first-order differential operator.

\section{The Least-Squares Functional}
The FOSLS functional $\mathcal{F}(\mathbf{Q}; \mathbf{F})$ is defined as the sum of squared $L^2$-norms of the first-order residuals over the domain $\Omega$ and appropriate norms on the boundary $\partial\Omega$:
\begin{equation}
    \mathcal{F}(\mathbf{Q}; \mathbf{F}) = \|\mathcal{D}\mathbf{Q} - \mathbf{F}\|_{0,\Omega}^2 + \|\mathcal{B}\mathbf{Q} - \mathbf{g}\|_{0,\partial\Omega}^2,
\end{equation}
where $\mathcal{B}$ represents the boundary operator corresponding to the constraints, and $\mathbf{g}$ denotes the boundary data. 

The minimization problem reads: Find $\mathbf{Q} \in V$ such that
\begin{equation}
    \mathcal{F}(\mathbf{Q}; \mathbf{F}) = \min_{\mathbf{V} \in V} \mathcal{F}(\mathbf{V}; \mathbf{F}),
\end{equation}
which is equivalent to the symmetric, positive-definite variational problem: Find $\mathbf{Q} \in V$ such that $a(\mathbf{Q}, \mathbf{V}) = (\mathbf{F}, \mathcal{D}\mathbf{V})_\Omega + (\mathbf{g}, \mathcal{B}\mathbf{V})_{\partial\Omega}$ for all $\mathbf{V} \in V$.

\section{Wall and Outflow Boundary Conditions}
In fluid-dynamic or transport-dominated applications, the boundary $\partial\Omega$ is partitioned into distinct segments such as solid walls $\Gamma_w$ and outflow boundaries $\Gamma_{out}$ (along with inflow $\Gamma_{in}$).

\subsection{Wall Boundary Conditions ($\Gamma_w$)}
On a solid wall, physical constraints typically dictate no-slip or no-penetration conditions for velocity components, or Dirichlet/Neumann conditions for scalar potentials. In FOSLS:
\begin{itemize}
    \item Essential boundary conditions (e.g., $u = g_D$ on $\Gamma_w$) are enforced strongly by restricting the trial space $V$, or weakly by incorporating penalty terms into the boundary functional $\|\mathcal{B}\mathbf{Q} - \mathbf{g}\|_{0,\Gamma_w}^2$.
    \item Normal and tangential flux components are coupled into the first-order system to maintain ellipticity and equivalence to the $H^1$ or $H(\text{div})$ norms.
\end{itemize}

\subsection{Outflow Boundary Conditions ($\Gamma_{out}$)}
Outflow boundaries require careful formulation to prevent spurious numerical reflections and ensure coercivity of the least-squares functional:
\begin{itemize}
    \item Natural boundary conditions are often accommodated by minimizing the residual of the traction or normal derivative directly in the boundary functional norm.
    \item Downstream convective or absorbing boundary conditions are integrated into $\mathcal{B}\mathbf{Q} = \mathbf{g}$ on $\Gamma_{out}$, penalizing departures from the desired asymptotic behavior without over-specifying internal characteristics.
\end{itemize}

\section{Algebraic Multigrid (AMG) Integration}
Because FOSLS naturally yields symmetric and positive-definite (SPD) algebraic systems $A \mathbf{x} = \mathbf{b}$ that are uniformly elliptic and well-conditioned in mesh-independent $H^1$-equivalent norms, they are exceptionally well-suited for Algebraic Multigrid (AMG) solvers.

\subsection{Key Properties for AMG}
\begin{enumerate}
    \item \textbf{Matrix Structure:} The discretization of $\mathcal{F}$ yields an SPD matrix $A$ whose graph directly reflects the couplings of the first-order system variables.
    \item \textbf{Smoothing Property:} Standard relaxation schemes (such as Gauss-Seidel or robust distributive smoothers for coupled systems) efficiently damp high-frequency error components because the principal part of $A$ corresponds to elliptic operators.
    \item \textbf{Coarse-Grid Correction:} AMG constructs algebraic coarse spaces and restriction/prolongation operators based on the strong connections in $A$, leading to convergence rates bounded strictly away from 1 independently of the mesh size $h$ or the number of refinement levels.
\end{enumerate}

\end{document}


