from astropy import units as u
from matplotlib import pyplot as plt

from wavefront_filtering.optical_elements.filter.pinhole import Pinhole
from wavefront_filtering.optical_elements.lens import Lens
from wavefront_filtering.wavefronts.wavefront import Wavefront

# Define wavefront
wavelength = 1e-5 * u.meter
initial_intensity = 1 * u.watt ** 0.5 / u.meter
zernike_modes = [(5, 0 * wavelength / 10)]
aperture_diameter = 0.003 * u.meter
array_dimension = 400

wavefront = Wavefront(wavelength,
                      initial_intensity,
                      zernike_modes,
                      aperture_diameter,
                      array_dimension)

# Define optical elements
lens_1 = Lens(100)
lens_2 = Lens(100)
pinhole = Pinhole(1.22, wavefront)

# Apply optical elements
wavefront.apply(lens_1)

plt.imshow(wavefront.intensity.value)
plt.show()

wavefront.apply(pinhole)

plt.imshow(wavefront.intensity.value)
plt.show()

wavefront.apply(lens_2)
