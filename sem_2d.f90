!=======================================================================
! 2D Spectral Element Method (SEM) Solver
! Matrix-Free Tensor Contraction & Direct Stiffness Summation (DSS)
!
! This program solves the 2D Poisson equation:
!     -nabla^2 u = f   on Omega = [-1, 1] x [-1, 1]
!     u = 0            on dOmega (Dirichlet boundaries)
!
! The solution is approximated using Gauss-Lobatto-Legendre (GLL) 
! spectral elements. To achieve maximum performance, global sparse 
! matrix assembly is completely bypassed.
! 
! Key Equations & Methods:
! 1. 1D Matrices:
!    M_{ii} = w_i * (dx / 2)
!    K = D^T * M * D * (2/dx)^2
! 2. Local Matrix-Free Evaluation:
!    The stiffness operator A*u is applied locally on each element:
!    v_local = M_1dy * u_local * K_1dx^T + K_1dy * u_local * M_1dx^T
! 3. Direct Stiffness Summation (DSS):
!    C^0 continuity is enforced by exchanging and summing residual 
!    contributions across shared element interfaces (edges/corners).
!=======================================================================

program sem_2d
    implicit none

    integer :: p, E_x, E_y, num_elements
    integer :: N_x_global, N_y_global, num_global_nodes, N_local
    integer :: i, j, k, l, ex, ey, local_idx, global_idx
    integer :: iter, max_iters
    character(len=256) :: filename
    character(len=256) :: arg1, arg2, arg3, arg4
    
    real(8), allocatable :: x_gll(:), w_gll(:), D_matrix(:,:)
    real(8), allocatable :: M_1dx(:,:), K_1dx(:,:), M_1dy(:,:), K_1dy(:,:)
    real(8), allocatable :: X_local(:,:,:,:), Y_local(:,:,:,:)
    real(8), allocatable :: F_local(:,:,:,:)
    
    real(8) :: L_x, L_y, dx, dy, x_L, y_L, x_phys, y_phys, f_val
    real(8) :: pi
    
    ! CG Variables
    real(8), allocatable :: X(:,:,:,:), R(:,:,:,:), Z(:,:,:,:), P_vec(:,:,:,:), AP(:,:,:,:)
    real(8), allocatable :: inv_D(:,:,:,:)
    real(8), allocatable :: W_local(:,:,:,:)
    real(8) :: alpha, beta, rsold, rsnew, pAp
    
    integer(8) :: count1, count2, count_rate
    real(8) :: time_taken, max_err, ex_val
    
    pi = 4.0d0 * atan(1.0d0)
    
    call get_command_argument(1, arg1)
    if (len_trim(arg1) > 0) then
        filename = trim(arg1)
    else
        filename = 'matrices_p8.txt'
    end if

    call get_command_argument(2, arg2)
    if (len_trim(arg2) > 0) then
        read(arg2, *) E_x
    else
        E_x = 10
    end if

    call get_command_argument(3, arg3)
    if (len_trim(arg3) > 0) then
        read(arg3, *) E_y
    else
        E_y = 10
    end if

    call get_command_argument(4, arg4)
    if (len_trim(arg4) > 0) then
        read(arg4, *) max_iters
    else
        max_iters = 400
    end if

    open(unit=10, file=trim(filename), status='old', action='read')
    
    read(10, *) p
    
    allocate(x_gll(0:p), w_gll(0:p), D_matrix(0:p, 0:p))
    
    read(10, *) ! "x_gll"
    do i = 0, p
        read(10, *) x_gll(i)
    end do
    
    read(10, *) ! "w_gll"
    do i = 0, p
        read(10, *) w_gll(i)
    end do
    
    read(10, *) ! "D_matrix"
    do i = 0, p
        read(10, *) D_matrix(i, 0:p)
    end do
    close(10)
    
    L_x = 2.0d0
    L_y = 2.0d0
    dx = L_x / dble(E_x)
    dy = L_y / dble(E_y)
    
    ! 1D Matrices
    allocate(M_1dx(0:p, 0:p), K_1dx(0:p, 0:p))
    allocate(M_1dy(0:p, 0:p), K_1dy(0:p, 0:p))
    M_1dx = 0.0d0; K_1dx = 0.0d0
    M_1dy = 0.0d0; K_1dy = 0.0d0
    
    do i = 0, p
        M_1dx(i, i) = w_gll(i) * (dx / 2.0d0)
        M_1dy(i, i) = w_gll(i) * (dy / 2.0d0)
    end do
    
    K_1dx = matmul(transpose(D_matrix), matmul(M_1dx, D_matrix)) * (2.0d0 / dx)**2
    K_1dy = matmul(transpose(D_matrix), matmul(M_1dy, D_matrix)) * (2.0d0 / dy)**2
    
    allocate(X_local(0:p, 0:p, 0:E_x-1, 0:E_y-1))
    allocate(Y_local(0:p, 0:p, 0:E_x-1, 0:E_y-1))
    allocate(F_local(0:p, 0:p, 0:E_x-1, 0:E_y-1))
    F_local = 0.0d0
    
    do ey = 0, E_y - 1
        do ex = 0, E_x - 1
            x_L = -1.0d0 + ex * dx
            y_L = -1.0d0 + ey * dy
            
            do j = 0, p
                y_phys = y_L + (x_gll(j) + 1.0d0) * (dy / 2.0d0)
                do i = 0, p
                    x_phys = x_L + (x_gll(i) + 1.0d0) * (dx / 2.0d0)
                    
                    X_local(i, j, ex, ey) = x_phys
                    Y_local(i, j, ex, ey) = y_phys
                    
                    f_val = 32.0d0 * pi**2 * sin(4.0d0 * pi * x_phys) * sin(4.0d0 * pi * y_phys)
                    F_local(i, j, ex, ey) = M_1dx(i,i)*M_1dy(j,j) * f_val
                end do
            end do
        end do
    end do
    
    !===================================================================
    ! Conjugate Gradient (CG) Solver Setup
    !
    ! Solves A * X = F.
    ! Since variables are stored in localized element arrays, global 
    ! inner products (r^T * r) must use the multiplicity weight matrix W 
    ! to prevent over-counting nodes shared by multiple elements.
    !===================================================================
    allocate(X(0:p, 0:p, 0:E_x-1, 0:E_y-1))
    allocate(R(0:p, 0:p, 0:E_x-1, 0:E_y-1))
    allocate(Z(0:p, 0:p, 0:E_x-1, 0:E_y-1))
    allocate(P_vec(0:p, 0:p, 0:E_x-1, 0:E_y-1))
    allocate(AP(0:p, 0:p, 0:E_x-1, 0:E_y-1))
    allocate(inv_D(0:p, 0:p, 0:E_x-1, 0:E_y-1))
    allocate(W_local(0:p, 0:p, 0:E_x-1, 0:E_y-1))
    
    W_local = 1.0d0
    do ey = 0, E_y-1
        do ex = 0, E_x-2
            W_local(p, :, ex, ey) = W_local(p, :, ex, ey) / 2.0d0
            W_local(0, :, ex+1, ey) = W_local(0, :, ex+1, ey) / 2.0d0
        end do
    end do
    do ey = 0, E_y-2
        do ex = 0, E_x-1
            W_local(:, p, ex, ey) = W_local(:, p, ex, ey) / 2.0d0
            W_local(:, 0, ex, ey+1) = W_local(:, 0, ex, ey+1) / 2.0d0
        end do
    end do
    
    do ey = 0, E_y-1
        do ex = 0, E_x-1
            do j = 0, p
                do i = 0, p
                    inv_D(i, j, ex, ey) = M_1dy(j, j) * K_1dx(i, i) + K_1dy(j, j) * M_1dx(i, i)
                end do
            end do
        end do
    end do
    call apply_dss(inv_D)
    
    do ey = 0, E_y-1
        do ex = 0, E_x-1
            do j = 0, p
                do i = 0, p
                    if (inv_D(i, j, ex, ey) > 1d-14) then
                        inv_D(i, j, ex, ey) = 1.0d0 / inv_D(i, j, ex, ey)
                    else
                        inv_D(i, j, ex, ey) = 0.0d0
                    end if
                end do
            end do
        end do
    end do
    
    inv_D(0, :, 0, :) = 0.0d0
    inv_D(p, :, E_x-1, :) = 0.0d0
    inv_D(:, 0, :, 0) = 0.0d0
    inv_D(:, p, :, E_y-1) = 0.0d0
    
    call apply_dss(F_local)
    
    X = 0.0d0
    R = F_local
    Z = R * inv_D
    P_vec = Z
    
    print *, "Starting Fortran Matrix-Free PCG Solver (E=", E_x, ", p=", p, ")"
    
    call system_clock(count1, count_rate)
    
    rsold = sum(R * Z * W_local)
    
    do iter = 1, max_iters
        do ey = 0, E_y-1
            do ex = 0, E_x-1
                AP(:,:,ex,ey) = matmul(matmul(K_1dx, P_vec(:,:,ex,ey)), M_1dy) + &
                                matmul(matmul(M_1dx, P_vec(:,:,ex,ey)), K_1dy)
            end do
        end do
        
        call apply_dss(AP)
        
        pAp = sum(P_vec * AP * W_local)
        if (pAp < 1d-25) then
            alpha = 0.0d0
        else
            alpha = rsold / pAp
        end if
        
        X = X + alpha * P_vec
        R = R - alpha * AP
        
        Z = R * inv_D
        rsnew = sum(R * Z * W_local)
        
        if (sqrt(rsnew) < 1d-11) exit
        
        if (rsold < 1d-25) then
            beta = 0.0d0
        else
            beta = rsnew / rsold
        end if
        
        P_vec = Z + beta * P_vec
        rsold = rsnew
    end do
    
    call system_clock(count2)
    time_taken = real(count2 - count1, kind=8) / real(count_rate, kind=8)
    
    ! Error Calculation (Local)
    max_err = 0.0d0
    do ey = 0, E_y-1
        do ex = 0, E_x-1
            do j = 0, p
                do i = 0, p
                    ex_val = sin(4.0d0 * pi * X_local(i,j,ex,ey)) * sin(4.0d0 * pi * Y_local(i,j,ex,ey))
                    if (abs(X(i,j,ex,ey) - ex_val) > max_err) then
                        max_err = abs(X(i,j,ex,ey) - ex_val)
                    end if
                end do
            end do
        end do
    end do
    
    print *, "Fortran Solve Time: ", time_taken, " s"
    print *, "L_inf Error:      ", max_err
    
    contains

    !===================================================================
    ! subroutine apply_dss
    ! Direct Stiffness Summation (DSS)
    !
    ! Enforces C^0 continuity across all elements by summing the 
    ! interface residuals. This replaces the global topological 
    ! gather/scatter mapping (Q).
    !
    ! The summation is dimensionally split:
    ! 1. X-exchange: Adds right edge of element ex to left edge of ex+1
    ! 2. Y-exchange: Adds top edge of element ey to bottom edge of ey+1
    !===================================================================
    subroutine apply_dss(v)
        real(8), intent(inout) :: v(0:p, 0:p, 0:E_x-1, 0:E_y-1)
        integer :: ex_i, ey_i
        real(8) :: sum_val(0:p)
        
        ! X-exchange
        do ey_i = 0, E_y-1
            do ex_i = 0, E_x-2
                sum_val = v(p, :, ex_i, ey_i) + v(0, :, ex_i+1, ey_i)
                v(p, :, ex_i, ey_i) = sum_val
                v(0, :, ex_i+1, ey_i) = sum_val
            end do
        end do
        
        ! Y-exchange
        do ey_i = 0, E_y-2
            do ex_i = 0, E_x-1
                sum_val = v(:, p, ex_i, ey_i) + v(:, 0, ex_i, ey_i+1)
                v(:, p, ex_i, ey_i) = sum_val
                v(:, 0, ex_i, ey_i+1) = sum_val
            end do
        end do
        
        ! Global Boundary Conditions
        v(0, :, 0, :) = 0.0d0
        v(p, :, E_x-1, :) = 0.0d0
        v(:, 0, :, 0) = 0.0d0
        v(:, p, :, E_y-1) = 0.0d0
    end subroutine apply_dss

end program sem_2d
