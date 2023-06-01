from astropy import units as u

from pywavefilters.optical_elements.filter.pinhole import Pinhole
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
phase_error_psd = get_power_spectral_density_error(wavelength, beam_diameter, rms, grid_size,
                                                   correlation_length=20000 / u.meter, plot_psd=True)
wavefront.add_phase(phase_error_zernike)
wavefront.add_phase(phase_error_psd)

# Define optical elements
focal_length = 0.008 * u.meter
lens = Lens(focal_length)
pinhole = Pinhole(1.22, beam_diameter, grid_size)

# Plot wavefront in input plane
wavefront.plot_phase()
wavefront.plot_intensity()

# Apply lens
wavefront.apply(lens)
wavefront.plot_intensity(dimensionless=True)

# Apply pinhole
wavefront.apply(pinhole)
wavefront.plot_intensity(dimensionless=True, title='Intensity Focal Plane After Pinhole')
