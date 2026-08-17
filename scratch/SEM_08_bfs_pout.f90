!
!  least squares  discretization
!  using tensor product
!  dual time stepping
!
      use sem_data
      use pmg2_ctrl
      implicit none

! --- Parameters ---
      integer, parameter :: norder = 21, nem = 100, ndepp = 4
      integer, parameter :: ntdof = norder*norder*ndepp, npm = norder-1
      integer, parameter :: ndim = norder*norder

! --- Main Arrays ---
      real :: xp(ntdof,nem), yp(ntdof,nem)
      real :: f(ntdof,nem), fn(ntdof,nem), fnn(ntdof,nem)
!     real :: rms(8), temp(ntdof,nem)              ! Reserved for future use
      real :: wg(norder), d(norder,norder), zpts(norder)
      real :: wht(nem), wid(nem)
      integer :: inorth(nem), isouth(nem), iwest(nem), ieast(nem)
      integer :: ibcw(nem), ibce(nem), ibcs(nem), ibcn(nem)
      integer :: mask(ntdof,nem)  ! ✅ MASK declared as INTEGER
      real :: res(ntdof,nem)
      real :: p(ntdof,nem), rin(ntdof,nem)
!     real :: q(ntdof,nem), apn(ntdof,nem)          ! Reserved for future use
      real :: diag(ntdof,nem)
!     real :: u_rel(ntdof,nem), u_img(ntdof,nem),    ! Reserved for future use
!             v_rel(ntdof,nem), v_img(ntdof,nem)      ! Reserved for future use
      real :: u(ndim,nem), v(ndim,nem), pp(ndim,nem), &
              om(ndim,nem), temp(ndim,nem), &
              un(ndim,nem), vn(ndim,nem), &
              unn(ndim,nem), vnn(ndim,nem), &
              fu(ndim,nem), fv(ndim,nem), &
              u_res(ndim,nem), v_res(ndim,nem), &
              p_res(ndim,nem), om_res(ndim,nem)

      real :: r_hat(ntdof,nem), v_b(ntdof,nem)
      real :: s(ntdof,nem), t(ntdof,nem)
      real :: phat(ntdof,nem), shat(ntdof,nem)

      real :: c1(ndim,nem), c2(ndim,nem), &
              c3(ndim,nem), c4(ndim,nem)

!      real :: A_element(ntdof, ntdof, nem) 


      double precision :: small

! --- Character Variables ---
      character(len=80) :: fin, fout, frun

! --- Control Variables ---
      real :: re, dt, tol, cgsfac, fac1, fac2, fac3, pr
      integer :: ntime, nsub, iprt, nitcgs, istart, iform, nsave

      ! Loop indices and work variables
      integer :: i, j, k, ne, it, im, iter, ibg, iend, ii, ij, ipc, kk
      integer :: ippin_e, ippin_p    ! free-outflow: single pressure-pin (elem, dof)
      integer :: nelem, nterm, neig, nee, ndep, npoly
      real :: time, res0, qdrold, diffmax, eta
      real :: cgstol   ! inner (BiCGSTAB) tolerance for this sub-iteration
      real :: ystep = 0.5, hinlet = 0.5   ! inlet-channel bottom y and height
      integer :: npin_e = 0   ! pressure-pin element; 0 = auto (SE outlet corner)
      integer :: npin_j = 1   ! pin node on that element's east edge: 1 = bottom
                              ! (SE corner), -1 = top (NE corner), or 1..nterm
      integer :: jpin
      logical :: pin_on   ! .false. => no pressure pin at all (npin_e < 0)
      logical :: restart_needs_history_rebuild

      data small/1.0e-30/
!
      data re,dt,ntime,nsub,iprt,tol,nitcgs,istart,iform,cgsfac,nsave/ &
           7500.,0.01,10,2,0,1.0e-14,1000,0,0,0.01,50/
!
      namelist /input/fin,fout,re,dt,ntime,nsub,iprt, &
                      tol,nitcgs,istart,frun,iform,cgsfac,nsave, &
                      pmg_on,pmg_nu,pmg_omega,pmg_levels,pmg_pmin,pmg_nuc, &
                      pmg_galerkin,pmg_pmid,pmg_block, &
                      pmg_cheby,pmg_cheby_deg,pmg_cheby_lo,pmg_cheby4, &
                      pmg_cheby_opt,ystep,hinlet,npin_e,npin_j
!
      fac1 = 1.0
      fac2 =-1.0
      fac3 = 0.0
!
      read(5,nml=input)
      write(6,nml=input)
      pmg_verbose = (iprt .gt. 0)
      if (pmg_on) then
         write(6,'(A,I3,A,I2,A,F8.5)') ' PMG ENABLED (clean): levels=', &
              pmg_levels,'  nu=',pmg_nu,'  omega=',pmg_omega
      else
         write(6,'(A)') ' PMG disabled: diagonal preconditioner'
      endif
!
      open(9,file=fout,status='unknown')
!
      open(2,file=fin,form='formatted')
!
      read(2,*) nelem,nterm

      print *, nelem,nterm

      read(2,*) (wht(ne),ne=1,nelem)
      read(2,*) (wid(ne),ne=1,nelem)


      do ne=1,nelem
       read(2,*) (xp(k,ne),k=1,nterm)
       read(2,*) (yp(k,ne),k=1,nterm)
       read(2,*) iwest(ne),ieast(ne),isouth(ne),inorth(ne)
       read(2,*) ibcw(ne),ibce(ne),ibcs(ne),ibcn(ne)
      enddo

!134567
      do ne=1,nelem
       write(6,144) (xp(k,ne),k=1,nterm)
       write(6,144) (yp(k,ne),k=1,nterm)
       write(6,143) iwest(ne),ieast(ne),isouth(ne),inorth(ne)
       write(6,143) ibcw(ne),ibce(ne),ibcs(ne),ibcn(ne)
      enddo

!
      npoly = nterm - 1
!
      print *,'order of polynomials: ',npoly
      print *,'number of elements: ',nelem
!   
      if(nterm.gt.norder  .or. nelem.gt.nem) then
       print *,'Increase Dimension Size !!'
       print *,'Stop the program'
       stop
      endif
!
      call jacobl(npoly,0.,0.,zpts(1),npm)
      call quad(npoly,zpts(1),wg(1),npm)
      call derv(npoly,zpts(1),d(1,1),npm)

! 
      ndep = 4
      time = 0.0
!
      pr = 1./re

       neig = nterm*nterm
       nee = neig*ndep
!
! FREE-OUTFLOW pressure pin: SE corner of the main domain (element 20 = row1,col20
! for the 20x4 BFS meshes), east-edge bottom node (i=nterm, j=1). Pressure is left
! free everywhere else at the outlet; this single node removes the constant null mode.
! Auto-detect the SE corner of the outlet: east BC = 4 (outlet) and south BC = 1
! (bottom wall).  The previous hardcoded value 20 was that corner only for the
! original 20-column BFS meshes; on a mesh with a different column count it
! silently pins an interior node instead (e.g. (1.47,0.25) on the 15-column
! Armaly long mesh, (0.43,0.75) on the 6-column short mesh).
! npin_e > 0 in the namelist overrides the auto-detection (used to test that the
! pin is a pure gauge choice: only the additive pressure constant may change).
! npin_e < 0 disables the pin ENTIRELY: no pressure DOF is masked anywhere, so the
! discrete operator keeps its constant-pressure null mode.  This reproduces Chan &
! Mittal (1996), whose only outflow statement is that "points located on a 'free
! boundary' are treated as unknowns and solved directly" -- no pin is mentioned.
! A Krylov method started from a zero initial guess on a CONSISTENT singular system
! stays in the range space and never excites the null component, so this can be
! well behaved even though the operator is singular.
       pin_on  = ( npin_e .ge. 0 )
       ippin_e = 0
       if( .not. pin_on ) then
         write(6,*) 'free-outflow pressure pin: DISABLED (npin_e < 0)'
       else
       if( npin_e.gt.0 ) then
         ippin_e = npin_e
       else
       do ne = 1, nelem
         if( ibce(ne).eq.4 .and. ibcs(ne).eq.1 ) then
           ippin_e = ne
           exit
         endif
       enddo
       endif
       if( ippin_e.eq.0 ) then
         do ne = 1, nelem
           if( ibce(ne).eq.4 ) then
             ippin_e = ne
             exit
           endif
         enddo
       endif
       jpin = npin_j
       if( jpin.eq.-1 ) jpin = nterm
       if( jpin.lt.1 .or. jpin.gt.nterm ) jpin = 1
       ippin_p = ((nterm-1)*nterm + (jpin-1))*ndep + ip
       write(6,*) 'free-outflow pressure pin: element ', ippin_e, &
                  ' east-edge node j = ', jpin, ' of ', nterm
       endif
!
!
      if(istart.eq.1) then
       ! Provisional BDF2 factors; may be reset after sanity check
       fac1 =  1.5
       fac2 = -2.0
       fac3 =  0.5
       restart_needs_history_rebuild = .false.
       if(iform.eq.1) then
         open(1,file='rstart.dat',status='unknown')
         read(1,144)time,re
         read(1,143)nelem,neig,nterm,ndep,nee
         do ne=1,nelem
          read(1,144) (xp(k,ne),k=1,nterm)
          read(1,144) (yp(k,ne),k=1,nterm)
          read(1,144) (f(i,ne),i=1,nee)
          read(1,144) (fn(i,ne),i=1,nee)
          read(1,144) (fnn(i,ne),i=1,nee)
          read(1,143) (mask(i,ne),i=1,nee)
          read(1,143) iwest(ne),ieast(ne),isouth(ne),inorth(ne)
          read(1,143) ibcw(ne),ibce(ne),ibcs(ne),ibcn(ne)
         enddo

       else

        open(1,file='rstart.dat',form='unformatted')
         read(1)time,re
         read(1)nelem,neig,nterm,ndep,nee
         do ne=1,nelem
          read(1) (xp(k,ne),k=1,nterm),(yp(k,ne),k=1,nterm)
          read(1) (f(i,ne),i=1,nee),(fn(i,ne),i=1,nee),(fnn(i,ne),i=1,nee)
          read(1) (mask(i,ne),i=1,nee)
          read(1) iwest(ne),ieast(ne),isouth(ne),inorth(ne)
          read(1) ibcw(ne),ibce(ne),ibcs(ne),ibcn(ne)
         enddo

       endif
                   close(1)

! --- Post-restart sanity check: ensure current state f differs from history fn ---
                  diffmax = 0.0
                  do ne=1,nelem
                        do i=1,nee
                              diffmax = max(diffmax, abs(f(i,ne) - fn(i,ne)))
                        enddo
                  enddo
                              if(diffmax < 1.0e-12) then
                                    write(6,*) '*** WARNING: restart f and fn nearly identical; assuming old-style file (max diff=', diffmax, ')'
                                    write(6,*) '*** Using BDF1 for first step to rebuild history.'
                                    ! Fall back to BDF1 for one time step; switch to BDF2 after first step completes
                                    fac1 = 1.0
                                    fac2 = -1.0
                                    fac3 = 0.0
                                    restart_needs_history_rebuild = .true.
                              else
                                    write(6,*) 'Restart sanity: max |f-fn| =', diffmax
                                    restart_needs_history_rebuild = .false.
                              endif

                  print *,'restarted from rstart.dat at time= ', time

      else
        ! Initialize variables when not restarting
        iter = 0
!
! initialize all the variables
!
        do ne=1,nelem
         do i=1,nee
          f(i,ne) = 0.
          fn(i,ne) = 0.
          fnn(i,ne) = 0.
         enddo 
        enddo
      endif

143      format(10i5)
144      format(5e14.6)

! Initialize current solution from history only for fresh starts (avoid overwriting valid restart state).
                  if (istart == 0) then
                        do ne=1,nelem
                              do i=1,nee
                                    f(i,ne) = fn(i,ne)
                              enddo
                        enddo
                  endif

                  if (istart == 0) then
                        do ne=1,nelem
                              do j=1,nee
                                    mask(j,ne) = 1
                              enddo
                        enddo
                  endif   
!
! setting up connectivity and b.c. on different elements
!
!  fix pressure at one point
!  0 = interior
!  1 = wall
!  2 = moving lid
!  3 = inlet 
!  4 = outlet
!
! BFS: pressure is referenced by the outlet (P=0 on east edge); no interior pin.
      ij = (nterm-1)*nterm + nterm/2
      ipc = (ij-1)*ndep + ip
!      mask(ipc,11) = 0    ! (cavity-only pin removed for BFS)

! --- START OF NEW MASK SETUP ---
! Loop through each element and set masks based on BC codes.
      do ne=1,nelem

! West boundary
        if( ibcw(ne).eq.1 .or. ibcw(ne).eq.2 .or. ibcw(ne).eq.3 ) then 
!         Fix U and V for Wall (1), Lid (2), or Inlet (3)
          do j=1,nterm
            ij = (j-1)*ndep
            mask(ij+iu,ne) = 0 
            mask(ij+iv,ne) = 0 
          enddo
        else if( ibcw(ne).eq.4 ) then
!         Fix P for Outlet (4)
          do j=1,nterm
            ij = (j-1)*ndep
            mask(ij+ip,ne) = 0 
          enddo
        endif

! East boundary
        if( ibce(ne).eq.1 .or. ibce(ne).eq.2 .or. ibce(ne).eq.3 ) then 
          do j=1,nterm
            ii = (nterm-1)*nterm
            ij = (ii+j-1)*ndep
            mask(ij+iu,ne) = 0
            mask(ij+iv,ne) = 0
          enddo
        else if( ibce(ne).eq.4 ) then
!         P OUTLET: p = 0 imposed along the WHOLE outlet plane.  That is one of
!         the two scalar conditions ADN requires per boundary point in 2D
!         (free outflow supplies zero).  u, v and omega remain free.
          do j=1,nterm
            ii = (nterm-1)*nterm
            ij = (ii+j-1)*ndep
            mask(ij+ip,ne) = 0
          enddo
        endif

! South boundary
        if( ibcs(ne).eq.1 .or. ibcs(ne).eq.2 .or. ibcs(ne).eq.3 ) then 
          do i=1,nterm
            ii = (i-1)*nterm
            ij = ii*ndep
            mask(ij+iu,ne) = 0
            mask(ij+iv,ne) = 0
          enddo
        else if( ibcs(ne).eq.4 ) then
          do i=1,nterm
            ii = (i-1)*nterm
            ij = ii*ndep
            mask(ij+ip,ne) = 0
          enddo
        endif  

! North boundary
        if( ibcn(ne).eq.1 .or. ibcn(ne).eq.2 .or. ibcn(ne).eq.3 ) then 
          do i=1,nterm
            ii = (i-1)*nterm
            ij = (ii+nterm-1)*ndep
            mask(ij+iu,ne) = 0
            mask(ij+iv,ne) = 0
          enddo
        else if( ibcn(ne).eq.4 ) then
          do i=1,nterm
            ii = (i-1)*nterm
            ij = (ii+nterm-1)*ndep
            mask(ij+ip,ne) = 0
          enddo
        else if( ibcn(ne).eq.5 ) then
!         SYMMETRY plane (upper wall, UniFlo): stress-free + vanishing normal
!         velocity.  Normal = +y, so v = 0 (no through-flow) and the shear stress
!         is zero, which in VVP is omega = 0 (vorticity = dv/dx - du/dy = 0 when
!         du/dy = 0 and v = 0).  u and p remain free.
          do i=1,nterm
            ii = (i-1)*nterm
            ij = (ii+nterm-1)*ndep
            mask(ij+iv,ne)  = 0
            mask(ij+iom,ne) = 0
          enddo
        endif      

      enddo
! FREE OUTFLOW: pin pressure at a single outlet-corner node for uniqueness.
      if( pin_on ) mask(ippin_p, ippin_e) = 0
! --- END OF NEW MASK SETUP ---


      do it=1,ntime
!
      time = time + dt
      print *,'At time=',time
!
! sub-iteration to resolve linearization error
!
      do im=1,nsub

      do ne=1,nelem

!        print *,'ne= ',ne
!        print *, 'bcwes= ',ibcw(ne)
!        print *, 'bceast= ',ibce(ne)
!        print *, 'bcsouth= ',ibcs(ne)
!        print *, 'bcnorth= ',ibcn(ne)
        
! --- START OF INLET/OUTLET MODIFICATION BLOCK ---
!
!         f(ipc,11) = 0     ! (cavity-only pressure pin removed for BFS)


! Handle West Boundary
      if( ibcw(ne).eq.1 ) then  
!       Wall (U=0, V=0)
        do j=1,nterm
          ij = (j-1)*ndep
          f(ij+iu,ne)  =  0.0
          f(ij+iv,ne)  =  0.0
        enddo
      else if( ibcw(ne).eq.3 ) then
!       Inlet: Gartling BFS parabolic profile on y in [0.5,1]
!       u = 6*eta*(1-eta), eta=(y-0.5)/0.5  (u_ave=1, u_max=1.5)
        do j=1,nterm
          ij = (j-1)*ndep
          eta = (yp(j,ne) - ystep)/hinlet
          f(ij+iu,ne)  =  6.0*eta*(1.0-eta)
          f(ij+iv,ne)  =  0.0
        enddo
      else if( ibcw(ne).eq.4 ) then
!       Outlet (P=0, U/V float -> ~Neumann)
        do j=1,nterm
          ij = (j-1)*ndep
          f(ij+ip,ne)  =  0.0 ! Set P = 0.0
        enddo
      endif

! Handle East Boundary
      if( ibce(ne).eq.1 ) then  
!       Wall (U=0, V=0)
        do j=1,nterm
          ii = (nterm-1)*nterm
          ij = (ii+j-1)*ndep
          f(ij+iu,ne)  = 0.0
          f(ij+iv,ne)  = 0.0
        enddo
      else if( ibce(ne).eq.3 ) then
!       Inlet (U=1, V=0)
        do j=1,nterm
          ii = (nterm-1)*nterm
          ij = (ii+j-1)*ndep
          f(ij+iu,ne)  = 1.0 
          f(ij+iv,ne)  = 0.0 
        enddo
      else if( ibce(ne).eq.4 ) then
!       P OUTLET: p = 0 on the outlet plane; u, v, omega computed.
        do j=1,nterm
          ii = (nterm-1)*nterm
          ij = (ii+j-1)*ndep
          f(ij+ip,ne)  = 0.0
        enddo
      endif

! Handle South Boundary (Using ibg/iend for corners w/ walls)
      if( ibcs(ne).eq.1 ) then  
!       Wall (U=0, V=0)
        ibg = 1
        iend = nterm
        if(ibcw(ne).eq.1) ibg = 2
        if(ibce(ne).eq.1) iend = nterm - 1
        do i=ibg,iend
          ii = (i-1)*nterm
          ij = ii*ndep
          f(ij+iu,ne)  = 0.0
          f(ij+iv,ne)  = 0.0
        enddo
      else if( ibcs(ne).eq.3 ) then
!       Inlet (U=1, V=0)
        do i=1,nterm 
          ii = (i-1)*nterm
          ij = ii*ndep
          f(ij+iu,ne)  = 1.0 
          f(ij+iv,ne)  = 0.0 
        enddo
      else if( ibcs(ne).eq.4 ) then
!       Outlet (P=0, U/V float -> ~Neumann)
        do i=1,nterm 
          ii = (i-1)*nterm
          ij = ii*ndep
          f(ij+ip,ne)  = 0.0 ! Set P = 0.0
        enddo
      endif  
!
! Handle North Boundary (Using ibg/iend for corners w/ walls/lid)
      if( ibcn(ne).eq.1 ) then ! Added check for Wall on North
!       Wall (U=0, V=0)
        ibg = 1
        iend = nterm
        if(ibcw(ne).eq.1) ibg = 2
        if(ibce(ne).eq.1) iend = nterm - 1
        do i=ibg,iend
          ii = (i-1)*nterm
          ij = (ii+nterm-1)*ndep
          f(ij+iu,ne)  = 0.0
          f(ij+iv,ne)  = 0.0
        enddo
      else if( ibcn(ne).eq.2 ) then  
!       Moving Lid (U=1, V=0)
        ibg = 1
        iend = nterm
        if(ibcw(ne).eq.1) ibg = 2
        if(ibce(ne).eq.1) iend = nterm - 1
        do i=ibg,iend
          ii = (i-1)*nterm
          ij = (ii+nterm-1)*ndep
          f(ij+iu,ne)  = 1.0
          f(ij+iv,ne)  = 0.0
        enddo
      else if( ibcn(ne).eq.3 ) then
!       Inlet (U=1, V=0)
        do i=1,nterm 
          ii = (i-1)*nterm
          ij = (ii+nterm-1)*ndep
          f(ij+iu,ne)  = 1.0 
          f(ij+iv,ne)  = 0.0 
        enddo
      else if( ibcn(ne).eq.4 ) then
!       Outlet (P=0, U/V float -> ~Neumann)
        do i=1,nterm
          ii = (i-1)*nterm
          ij = (ii+nterm-1)*ndep
          f(ij+ip,ne)  = 0.0 ! Set P = 0.0
        enddo
      else if( ibcn(ne).eq.5 ) then
!       SYMMETRY plane: v = 0 (no through-flow), omega = 0 (zero shear stress)
        do i=1,nterm
          ii = (i-1)*nterm
          ij = (ii+nterm-1)*ndep
          f(ij+iv,ne)  = 0.0
          f(ij+iom,ne) = 0.0
        enddo

! --- END OF INLET/OUTLET MODIFICATION BLOCK ---
      endif   

      enddo  ! End of ne=1,nelem loop
!
! FREE OUTFLOW: hold the single pressure-pin node at 0 (the mask keeps it fixed).
      if( pin_on ) f(ippin_p, ippin_e) = 0.0
!
      call dge(nelem,nterm,ndep,ntdof,norder, &
               fac1,dt,pr, &
               diag,wid,wht,wg, &
               f,d)
!
      small = 1.0e-30
!
! calculate incomplete residuals
!
      do ne=1,nelem
      do ij=1,neig
       kk = (ij-1)*ndep
       u(ij,ne) = f(kk+1,ne)
       v(ij,ne) = f(kk+2,ne)
       pp(ij,ne) = f(kk+3,ne)
       om(ij,ne) = f(kk+4,ne)
       un(ij,ne) = fn(kk+1,ne)
       vn(ij,ne) = fn(kk+2,ne)
       unn(ij,ne) = fnn(kk+1,ne)
       vnn(ij,ne) = fnn(kk+2,ne)
       fu(ij,ne) = f(kk+1,ne)
       fv(ij,ne) = f(kk+2,ne)
      enddo
      enddo
!
      call rhs(u,v,pp,om,un,vn,unn,vnn, &
               fu,fv, &
               dt,pr,nelem,nterm,neig,norder, &
               fac1,fac2,fac3, &
               wid,wht,wg,d, &
               u_res,v_res,p_res,om_res,ndim)
!
      do ne=1,nelem
       do i=1,neig
        ij=(i-1)*ndep
        res(ij+1,ne) = u_res(i,ne)*mask(ij+1,ne)
        res(ij+2,ne) = v_res(i,ne)*mask(ij+2,ne)
        res(ij+3,ne) = p_res(i,ne)*mask(ij+3,ne)
        res(ij+4,ne) = om_res(i,ne)*mask(ij+4,ne)
       enddo
!
       do i=1,nee
        rin(i,ne) = res(i,ne)
       enddo
!
       enddo  
!
!  forming the complete residuals
!
      call collect(nelem,nterm,ndep,ntdof, &
                   isouth,inorth,iwest,ieast, &
                   res)
!
!  forming the complete diagonals
!
      call collect(nelem,nterm,ndep,ntdof, &
                   isouth,inorth,iwest,ieast, &
                   diag)


!
      qdrold = 1.0e30
!
! L2-Norm of residual
!
      res0 = 0.0
      do ne=1,nelem
       do i=1,nee
        res0 = res0 + res(i,ne)*res(i,ne)
       enddo
      enddo
!
      res0 = sqrt(res0)
!
      print *,im,res0
!
      if(res0 .le. tol) exit  ! Exit sub-iteration if converged
!
!  Inexact-Newton forcing term.  cgsfac is the residual-reduction factor demanded
!  of the linear solve within one sub-iteration: BiCGSTAB is asked to bring the
!  residual down to cgsfac*res0, not all the way to tol.  Early sub-iterations,
!  where the Newton iterate is still far from the solution, are therefore solved
!  loosely and cheaply; the tolerance tightens automatically as res0 falls.
!  Floored at tol so the inner solve is never asked for more accuracy than the
!  outer convergence test requires.
!    cgsfac = 0     -> cgstol = tol: the previous behaviour (always solve to tol).
!    cgsfac = 1e-3  -> demand a 1000x residual drop per sub-iteration.
!    cgsfac >= 1     -> degenerate: cgstol >= res0, so BiCGSTAB exits immediately
!                       and the sub-iteration makes no progress.  Keep cgsfac < 1.
      cgstol = max( cgsfac*res0, tol )
      if(iprt .gt. 0) print *,'   inner cgs tol =',cgstol,' (cgsfac=',cgsfac,')'
!
!  Preconditioned Conjugate Gradient Method
!
! --- INSERT THE BICGSTAB CALL HERE ---
      call bicgstab(f, res, diag, mask, &
                    p, v_b, s, t, r_hat, phat, shat, &
                    nelem, nterm, ndep, ntdof, nem, norder, ndim, &
                    pr, dt, fac1, nitcgs, cgstol, iprt, &
                    wid, wht, wg, d, small, &
                    u, v, pp, om, fu, fv, &
                    u_res, v_res, p_res, om_res, &
                    isouth, inorth, iwest, ieast)
! --- END OF BICGSTAB CALL ---

!
!      print *,'non-convergent in cgs ','residual= ',res1
!

      enddo  ! End of im=1,nsub loop

      ! Continue with time stepping regardless of sub-iteration convergence
!
                  ! periodic restart write BEFORE history shift so f (t^n), fn (t^{n-1}), fnn (t^{n-2}) stay distinct
                  if(mod(it,nsave).eq.0) then
                        if(iform.eq.1) then
                              open(1,file=frun,status='replace')
                              write(1,144)time,re
                              write(1,143)nelem,neig,nterm,ndep,nee
                              do ne=1,nelem
                                    write(1,144) (xp(k,ne),k=1,nterm)
                                    write(1,144) (yp(k,ne),k=1,nterm)
                                    write(1,144) (f(i,ne),i=1,nee)
                                    write(1,144) (fn(i,ne),i=1,nee)
                                    write(1,144) (fnn(i,ne),i=1,nee)
                                    write(1,143) (mask(i,ne),i=1,nee)
                                    write(1,143) iwest(ne),ieast(ne),isouth(ne),inorth(ne)
                                    write(1,143) ibcw(ne),ibce(ne),ibcs(ne),ibcn(ne)
                              enddo
                        else
                              open(1,file=frun,form='unformatted',status='replace')
                              write(1)time,re
                              write(1)nelem,neig,nterm,ndep,nee
                              do ne=1,nelem
                                    write(1) (xp(k,ne),k=1,nterm),(yp(k,ne),k=1,nterm)
                                    write(1) (f(i,ne),i=1,nee),(fn(i,ne),i=1,nee),(fnn(i,ne),i=1,nee)
                                    write(1) (mask(i,ne),i=1,nee)
                                    write(1) iwest(ne),ieast(ne),isouth(ne),inorth(ne)
                                    write(1) ibcw(ne),ibce(ne),ibcs(ne),ibcn(ne)
                              enddo
                        endif
                        close(1)
                        write(6,*) "*** finished writing data to ",frun
                  endif

                              ! If we used BDF1 to rebuild history on the first restarted step, switch to BDF2 now
                              if (restart_needs_history_rebuild .and. it == 1) then
                                    fac1 = 1.5
                                    fac2 = -2.0
                                    fac3 = 0.5
                                    restart_needs_history_rebuild = .false.
                                    write(6,*) '*** Switched to BDF2 after rebuilding history.'
                              endif

                              ! update in time (history shift) AFTER saving
                  if (it .lt. ntime) then
                        do ne=1,nelem
                              do i=1,nee
                                    fnn(i,ne) = fn(i,ne)
                                    fn(i,ne)  = f(i,ne)
                              enddo
                        enddo
                  endif

500   continue


!
      enddo  ! End of it=1,ntime loop
!
       print *,'finished at time= ',time
!
! Final restart write: MUST match restart read order (f, fn, fnn). Previously 'f' was omitted causing
! misalignment when reading (old fn loaded into f, old fnn into fn, and mask/BCs shifted). Fixed below.
                   if(iform.eq.1) then
                         open(1,file=frun,status='replace')
                         write(1,144)time,re
                         write(1,143)nelem,neig,nterm,ndep,nee
                         do ne=1,nelem
                              write(1,144) (xp(k,ne),k=1,nterm)
                              write(1,144) (yp(k,ne),k=1,nterm)
                              write(1,144) (f(i,ne), i=1,nee)   ! current state
                              write(1,144) (fn(i,ne),i=1,nee)   ! previous state (n-1)
                              write(1,144) (fnn(i,ne),i=1,nee)  ! previous state (n-2)
                              write(1,143) (mask(i,ne),i=1,nee)
                              write(1,143) iwest(ne),ieast(ne),isouth(ne),inorth(ne)
                              write(1,143) ibcw(ne),ibce(ne),ibcs(ne),ibcn(ne)
                         enddo
                   else
                        open(1,file=frun,form='unformatted',status='replace')
                         write(1)time,re
                         write(1)nelem,neig,nterm,ndep,nee
                         do ne=1,nelem
                              write(1) (xp(k,ne),k=1,nterm),(yp(k,ne),k=1,nterm)
                              write(1) (f(i,ne), i=1,nee),(fn(i,ne),i=1,nee),(fnn(i,ne),i=1,nee)
                              write(1) (mask(i,ne),i=1,nee)
                              write(1) iwest(ne),ieast(ne),isouth(ne),inorth(ne)
                              write(1) ibcw(ne),ibce(ne),ibcs(ne),ibcn(ne)
                         enddo
                   endif
!
      stop
      end program
