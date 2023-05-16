import numpy as np
from astropy import units as u

from pywavefilters.optical_elements.filter.pinhole import Pinhole
from pywavefilters.optical_elements.general.lens import Lens
from pywavefilters.util.math import normalize_intensity
from pywavefilters.wavefronts.errors.zernike import get_zernike_error
from pywavefilters.wavefronts.wavefront import Wavefront

# Parameters
grid_size = 401
wavelength = 1e-5 * u.meter
beam_diameter = 0.003 * u.meter
zernike_modes = [(6, wavelength / 10)]

# Define wavefront
wavefront = Wavefront(wavelength,
                      beam_diameter,
                      grid_size)

# Add phase errors
phase_error_zernike = get_zernike_error(beam_diameter, zernike_modes, grid_size)
wavefront.add_phase(2 * np.pi * phase_error_zernike / wavelength)
wavefront.complex_amplitude = normalize_intensity(wavefront.complex_amplitude)

# Define optical elements
focal_length = 0.008 * u.meter
lens = Lens(focal_length)
pinhole = Pinhole(1.22, beam_diameter, grid_size)

# Plot wavefront in input plane
wavefront.plot_phase()
wavefront.plot_intensity_pupil_plane()

# Apply lens
wavefront.apply(lens)
wavefront.plot_intensity_focal_plane(dimensionless=True)

# Apply pinhole
wavefront.apply(pinhole)
wavefront.plot_intensity_focal_plane(dimensionless=True, title='Intensity Focal Plane After Pinhole')
