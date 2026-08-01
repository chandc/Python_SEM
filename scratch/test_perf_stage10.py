import numpy as np
from lssem2d.mesh import build_channel
from lssem2d.lgl import diff_matrix
from lssem2d.operators import dUdx, dUdy, DxT, DyT

def run_test():
    N = 4
    mesh = build_channel(1.0, 1.0, 2, 2, N)
    D = diff_matrix(N)
    
    U = np.random.rand(mesh.nelem, N+1, N+1)
    facx = mesh.facx
    facy = mesh.facy
    
    res1 = dUdx(U, D, facx)
    res2 = facx[:, None, None] * np.matmul(D, U)
    print("dUdx match:", np.allclose(res1, res2))
    
    res3 = dUdy(U, D, facy)
    res4 = facy[:, None, None] * np.matmul(U, D.T)
    print("dUdy match:", np.allclose(res3, res4))
    
    res5 = DxT(U, D, facx)
    res6 = facx[:, None, None] * np.matmul(D.T, U)
    print("DxT match:", np.allclose(res5, res6))
    
    res7 = DyT(U, D, facy)
    res8 = facy[:, None, None] * np.matmul(U, D)
    print("DyT match:", np.allclose(res7, res8))
    
if __name__ == "__main__":
    run_test()
