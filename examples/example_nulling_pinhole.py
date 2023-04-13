import numpy as np
from astropy import units as u

from pywavefilters.optical_elements.filter.pinhole import Pinhole
from pywavefilters.optical_elements.lens import Lens
from pywavefilters.wavefronts.wavefront import Wavefront

# Define wavefront
wavelength = 15e-6 * u.meter
zernike_modes_1 = [(5, 0 * wavelength / 100)]
zernike_modes_2 = [(6, wavelength / 100)]
beam_diameter = 0.003 * u.meter
number_of_pixels = 401

wavefront_1 = Wavefront(wavelength,
                        zernike_modes_1,
                        beam_diameter,
                        number_of_pixels)

wavefront_2 = Wavefront(wavelength,
                        zernike_modes_2,
                        beam_diameter,
                        number_of_pixels)

# Define optical elements
focal_length = 0.003 * u.meter
lens = Lens(focal_length)
pinhole = Pinhole(1.22, beam_diameter, number_of_pixels)

# Combine wavefronts and normalize
wavefront_const = wavefront_1 + wavefront_2
wavefront_dest = wavefront_1 - wavefront_2

intensity_unfiltered = wavefront_const.intensity

# Apply optical elements to wavefronts
wavefront_const.apply(lens)
wavefront_dest.apply(lens)

# plt.plot(wavefront_const.intensity.value[number_of_pixels // 2], label='co')
# plt.plot(wavefront_dest.intensity.value[number_of_pixels // 2], label='de')
# plt.legend()
# plt.show()

wavefront_const.apply(pinhole)
wavefront_dest.apply(pinhole)
#
# plt.plot(wavefront_const.intensity.value[number_of_pixels // 2], label='co')
# plt.plot(wavefront_dest.intensity.value[number_of_pixels // 2], label='de')
# plt.legend()
# plt.show()

wavefront_const.apply(lens)
wavefront_dest.apply(lens)

# Calculate null depth and throughput
null_depth = np.sum(wavefront_dest.intensity) / np.sum(wavefront_const.intensity)
intensity_filtered = wavefront_const.intensity
throughput = np.sum(intensity_filtered) / np.sum(intensity_unfiltered)
print('Null Depth: ', null_depth)
print('Throughput: ', throughput)
