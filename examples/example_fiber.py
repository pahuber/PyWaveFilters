from astropy import units as u

from pywavefilters.optical_elements.filter.fiber import Fiber
from pywavefilters.optical_elements.general.lens import Lens
from pywavefilters.wavefronts.errors.power_spectral_density import get_power_spectral_density_error
from pywavefilters.wavefronts.errors.zernike import get_zernike_error
from pywavefilters.wavefronts.wavefront import Wavefront

# Parameters
grid_size = 401
wavelength = 1e-5 * u.meter
beam_diameter = 0.003 * u.meter
rms = wavelength / 10
zernike_modes = [(6, rms)]

# Define wavefront
wavefront = Wavefront(wavelength, beam_diameter, grid_size)

# Add phase errors
phase_error_zernike = get_zernike_error(wavelength, beam_diameter, zernike_modes, grid_size)
wavefront.add_phase(phase_error_zernike)

phase_error_psd = get_power_spectral_density_error(wavelength, beam_diameter, rms, grid_size,
                                                   correlation_length=20000 / u.meter, plot_psd=True)
wavefront.add_phase(phase_error_psd)

# Define optical elements
focal_length = 0.074 * u.meter
lens = Lens(focal_length)
fiber = Fiber(wavelength,
              beam_diameter,
              2 * 12e-6 * u.meter,
              2 * 170e-6 * u.meter,
              2.7,
              2.6987,
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
