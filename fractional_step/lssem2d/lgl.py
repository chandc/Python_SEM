import numpy as np

def lgl_nodes(N):
    """
    Compute the Legendre-Gauss-Lobatto (LGL) nodes on [-1, 1].
    N: polynomial degree (N >= 1)
    Returns: numpy array of size N+1
    """
    if N == 0:
        return np.array([0.0])
    
    # Initial guess for the interior nodes: roots of Chebyshev polynomials
    x = -np.cos(np.pi * np.arange(1, N) / N)
    
    def legendre_and_deriv(n, x):
        P0 = np.ones_like(x)
        P1 = x.copy()
        if n == 0:
            return P0, np.zeros_like(x)
        if n == 1:
            return P1, np.ones_like(x)
        
        dP0 = np.zeros_like(x)
        dP1 = np.ones_like(x)
        
        for k in range(1, n):
            P2 = ((2 * k + 1) * x * P1 - k * P0) / (k + 1)
            dP2 = ((2 * k + 1) * P1 + (2 * k + 1) * x * dP1 - k * dP0) / (k + 1)
            P0, P1 = P1, P2
            dP0, dP1 = dP1, dP2
            
        return P1, dP1

    # Newton iteration
    for _ in range(100):
        P_N, dP_N = legendre_and_deriv(N, x)
        d2P_N = (2 * x * dP_N - N * (N + 1) * P_N) / (1 - x**2)
        
        dx = -dP_N / d2P_N
        x = x + dx
        if np.max(np.abs(dx)) < 1e-15:
            break
            
    nodes = np.zeros(N + 1)
    nodes[0] = -1.0
    nodes[-1] = 1.0
    if N > 1:
        nodes[1:-1] = x
        
    return nodes

def lgl_weights(N):
    """
    Compute LGL quadrature weights.
    w_i = 2 / (N(N+1) [P_N(xi_i)]^2)
    """
    if N == 0:
        return np.array([2.0])
    
    xi = lgl_nodes(N)
    
    from numpy.polynomial.legendre import Legendre
    L_N = Legendre([0] * N + [1])
    P_N_xi = L_N(xi)
    
    w = 2.0 / (N * (N + 1) * P_N_xi**2)
    return w

def diff_matrix(N):
    """
    Compute the 1-D differentiation matrix D[i,n] = l_n'(xi_i)
    D[i,n] = P_N(xi_i) / (P_N(xi_n) (xi_i - xi_n))     i != n
    D[i,i] = 0 for interior;  D[0,0] = -N(N+1)/4;  D[N,N] = +N(N+1)/4
    """
    if N == 0:
        return np.zeros((1, 1))
        
    xi = lgl_nodes(N)
    from numpy.polynomial.legendre import Legendre
    L_N = Legendre([0] * N + [1])
    P_N_xi = L_N(xi)
    
    D = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for n in range(N + 1):
            if i != n:
                D[i, n] = (P_N_xi[i] / P_N_xi[n]) / (xi[i] - xi[n])
            else:
                if i == 0:
                    D[i, n] = -N * (N + 1) / 4.0
                elif i == N:
                    D[i, n] = N * (N + 1) / 4.0
                else:
                    D[i, n] = 0.0
                    
    return D
