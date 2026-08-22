"""Fourier transforms in the periodic z direction, with the 3/2 dealias rule.

LAYOUT.  z is the LAST axis, always:

    U[e, i, j, var, k]      e   element
                            i,j GLL nodes
                            var field
                            k   z-plane (physical) or z-mode (spectral)

That is not arbitrary.  rfft/irfft on the last axis is then a stride-1
contiguous transform, and the 2D SEM contractions on (i,j) keep var*k as a fat
contiguous trailing block, which is what makes the batched matmul fast.  Both
requirements are satisfied at once only with z last.  See
3D_DEVELOPMENT_PLAN.md sec 1.1.

REAL-TO-COMPLEX.  Physical fields are real, so only Nz//2 + 1 modes are
independent.  Use rfft/irfft, never fft/ifft -- the latter doubles both memory
and the number of per-mode solves for no gain.
"""
import numpy as np

from . import device as DEV


def wavenumbers(nz, lz):
    """k_z for an rfft of nz real points on a period lz.  Shape (nz//2 + 1,)."""
    return 2.0*np.pi*np.fft.rfftfreq(nz, d=lz/nz)


def to_modes(u):
    """Physical -> spectral along the last axis."""
    return DEV.rfft(u)


def to_physical(uh, nz):
    """Spectral -> physical along the last axis.  nz is the physical count."""
    return DEV.irfft(uh, nz)


def ddz(uh, kz):
    """d/dz in mode space: multiply by i*k_z, broadcast over the trailing axis."""
    return 1j*kz*uh


def real_mode_indices(nz):
    """Modes whose coefficients are necessarily real for real input.

    k = 0 always, and the Nyquist mode when nz is even.  These can be solved as
    genuinely real systems (half the work).  A non-zero imaginary part at k = 0
    is the classic symptom of a botched transform -- see assert_hermitian_ok.
    """
    return (0,) if nz % 2 else (0, nz//2)


def assert_hermitian_ok(uh, nz, tol=1e-12):
    """The modes that must be real, are.  Cheap; call it in tests and debug runs."""
    for k in real_mode_indices(nz):
        bad = np.abs(uh[..., k].imag).max()
        assert bad < tol, f'mode k={k} should be real for real input, |Im| = {bad:.3e}'


# ---------------------------------------------------------------- dealiasing

def pad_factor():
    """3/2 rule."""
    return 1.5


def padded_size(nz):
    """Physical z-size for dealiased products.  Even, and >= 3*nz/2."""
    m = int(np.ceil(pad_factor()*nz))
    return m + (m % 2)


def dealias_forward(uh, nz):
    """Spectral (nz modes) -> PADDED physical, for forming quadratic products.

    Zero-pads the spectrum to padded_size(nz) before the inverse transform, so a
    product of two fields cannot alias back onto the resolved modes.  Convection
    is quadratic; without this a DNS piles up energy at high k_z.
    """
    m = padded_size(nz)
    out = DEV.zeros_complex(tuple(uh.shape[:-1]) + (m//2 + 1,), uh)
    out[..., :uh.shape[-1]] = uh
    # irfft normalises by the transform length, so rescale to preserve amplitude
    return DEV.irfft(out, m)*(m/nz)


def dealias_backward(u_pad, nz):
    """PADDED physical -> spectral (nz modes), truncating the padded tail."""
    m = u_pad.shape[-1]
    uh = DEV.rfft(u_pad)/(m/nz)
    return uh[..., :nz//2 + 1]
