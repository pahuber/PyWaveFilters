import numpy as np
from astropy import units as u

from pywavefilters.optical_elements.filter.fiber import Fiber
from pywavefilters.optical_elements.general.lens import Lens
from pywavefilters.util.math import normalize_intensity
from pywavefilters.wavefronts.errors.power_spectral_density import get_power_spectral_density_error
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

phase_error_psd = get_power_spectral_density_error(wavelength, beam_diameter, 3.29e-18, 212.26, 7.8, grid_size)
wavefront.add_phase(2 * np.pi * phase_error_psd / wavelength.value)
wavefront.complex_amplitude = normalize_intensity(wavefront.complex_amplitude)

# Define optical elements
focal_length = 0.008 * u.meter
lens = Lens(focal_length)
fiber = Fiber(wavelength,
              beam_diameter,
              21e-6 * u.meter,
              125e-6 * u.meter,
              1.6,
              1.59,
              grid_size,
              lens)

# Plot wavefront in input plane
wavefront.plot_phase()
wavefront.plot_intensity_pupil_plane()

# Apply lens
wavefront.apply(lens)
wavefront.plot_intensity_focal_plane(dimensionless=True)

# Apply fiber
wavefront.apply(fiber)
wavefront.plot_intensity_focal_plane(dimensionless=True, title='Intensity Focal Plane After Fiber')
