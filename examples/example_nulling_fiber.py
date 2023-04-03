import numpy as np
from astropy import units as u

from pywavefilters.optical_elements.filter.fiber import Fiber
from pywavefilters.optical_elements.lens import Lens
from pywavefilters.wavefronts.wavefront import Wavefront

# Define wavefront
wavelength = 15e-6 * u.meter
zernike_modes_1 = [(5, wavelength / 100)]
zernike_modes_2 = [(6, wavelength / 100)]
beam_diameter = 0.003 * u.meter
number_of_pixels = 400

wavefront_1 = Wavefront(wavelength,
                        zernike_modes_1,
                        beam_diameter,
                        number_of_pixels)

wavefront_2 = Wavefront(wavelength,
                        zernike_modes_2,
                        beam_diameter,
                        number_of_pixels)

# Define optical elements
focal_length = 0.074 * u.meter
lens = Lens(focal_length)
fiber = Fiber(wavelength,
              beam_diameter,
              2 * 12e-6 * u.meter,
              2 * 170e-6 * u.meter,
              2.7,
              2.6987,
              number_of_pixels,
              lens)

# Combine wavefronts and normalize
wavefront_const = wavefront_1 + wavefront_2
wavefront_dest = wavefront_1 - wavefront_2

intensity_unfiltered = wavefront_const.intensity

# Apply optical elements to wavefronts
wavefront_const.apply(lens)
wavefront_dest.apply(lens)

# plt.plot(wavefront_const.intensity.value[number_of_pixels // 2], label='co')
# plt.plot(wavefront_dest.intensity.value[number_of_pixels // 2], label='de')
# plt.plot((abs(fiber.fundamental_fiber_mode) ** 2).value[number_of_pixels // 2], label='fiber')
# plt.legend()
# plt.show()

wavefront_const.apply(fiber)
wavefront_dest.apply(fiber)

# plt.plot(wavefront_const.intensity.value[number_of_pixels // 2], label='co')
# plt.plot(wavefront_dest.intensity.value[number_of_pixels // 2], label='de')
# plt.plot((abs(fiber.fundamental_fiber_mode) ** 2).value[number_of_pixels // 2], label='fiber')
# plt.legend()
# plt.show()

wavefront_1.apply(lens)
wavefront_2.apply(lens)

# Calculate null depth and throughput
null_depth = np.sum(wavefront_dest.intensity) / np.sum(wavefront_const.intensity)
intensity_filtered = wavefront_const.intensity
throughput = np.sum(intensity_filtered) / np.sum(intensity_unfiltered)
print('Null Depth: ', null_depth)
print('Throughput: ', throughput)
