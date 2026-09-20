"""Low-order FOSLS by direct assembly: linear quadrilateral and triangular
finite elements for the velocity-vorticity-pressure first-order system.

Deliberately NOT a modification of `lssem2d`.  That code is matrix-free and
collocated, with tensor-product derivative matrices and GLL nodes doubling as
quadrature points -- machinery that exists to make HIGH order affordable and
that buys nothing at p = 1.  Here the element matrices are formed directly
(in closed form where the element map is affine) and the sparse normal
equations L^T L are assembled and solved.

Sharing no code path with `lssem2d` is the point: agreement between the two is
then real evidence rather than a shared bug.  The ONE thing imported from it is
`ls_coeffs`, because a_mass, a_flux and kappa_p are scalar row coefficients
with no element dependence, and re-deriving them would silently test a
different scheme from the one the paper argues for.
"""
