from astropy import units as u

from pywavefilters.optical_elements.filter.pinhole import Pinhole
from pywavefilters.optical_elements.lens import Lens
from pywavefilters.util.plot import plot_intensity_pupil_plane, plot_intensity_focal_plane, \
    plot_initial_wavefront_error
from pywavefilters.wavefronts.wavefront import Wavefront

# Define wavefront
wavelength = 1e-5 * u.meter
zernike_modes = [(6, wavelength / 10)]
beam_diameter = 0.003 * u.meter
number_of_pixels = 201

wavefront = Wavefront(wavelength,
                      zernike_modes,
                      beam_diameter,
                      number_of_pixels)

# Define optical elements
focal_length = 0.008 * u.meter

lens = Lens(focal_length)

pinhole = Pinhole(1.22, beam_diameter, number_of_pixels)

# Apply optical elements to wavefront and plot at each stage
plot_initial_wavefront_error(wavefront)

plot_intensity_pupil_plane(wavefront)

wavefront.apply(lens)

plot_intensity_focal_plane(wavefront, dimensionless=True)

wavefront.apply(pinhole)

plot_intensity_focal_plane(wavefront, dimensionless=True, title='Intensity Focal Plane After Pinhole')
