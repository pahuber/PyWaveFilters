import numpy as np
from astropy import units as u

from wavefront_filtering.optical_elements.filter.fiber_optics.fiber import Fiber
from wavefront_filtering.optical_elements.lens import Lens
from wavefront_filtering.wavefronts.wavefront import Wavefront

# Define wavefront
wavelength = 1e-5 * u.meter
zernike_modes_1 = [(5, wavelength / 1000)]
zernike_modes_2 = [(6, wavelength / 1000)]
beam_diameter = 0.003 * u.meter
number_of_pixels = 100

wavefront_1 = Wavefront(wavelength,
                        zernike_modes_1,
                        beam_diameter,
                        number_of_pixels)

wavefront_2 = Wavefront(wavelength,
                        zernike_modes_2,
                        beam_diameter,
                        number_of_pixels)

# Define optical elements
focal_length = 0.008 * u.meter
lens = Lens(focal_length)
fiber_1 = Fiber(wavelength,
                beam_diameter,
                21e-6 * u.meter,
                125e-6 * u.meter,
                1.6,
                1.59,
                number_of_pixels,
                lens)
fiber_2 = Fiber(wavelength,
                beam_diameter,
                21e-6 * u.meter,
                125e-6 * u.meter,
                1.6,
                1.59,
                number_of_pixels,
                lens)

# Calculate null depth without filtering
wavefront_const_unfiltered = wavefront_1 + wavefront_2
wavefront_dest_unfiltered = wavefront_1 - wavefront_2
intensity_unfiltered = wavefront_const_unfiltered.intensity

null_depth_unfiltered = np.sum(wavefront_dest_unfiltered.intensity) / np.sum(wavefront_const_unfiltered.intensity)
print('Unfiltered Null Depth: ', null_depth_unfiltered)

# Apply optical elements to wavefronts
wavefront_1.apply(lens)
wavefront_2.apply(lens)

wavefront_1.apply(fiber_1)
wavefront_2.apply(fiber_2)

wavefront_1.apply(lens)
wavefront_2.apply(lens)

# Interfere wavefronts and calculate null depth and throughput
wavefront_const = wavefront_1 + wavefront_2
wavefront_dest = wavefront_1 - wavefront_2
intensity_filtered = wavefront_const.intensity

null_depth = np.sum(wavefront_dest.intensity) / np.sum(wavefront_const.intensity)
throughput = np.sum(intensity_filtered) / np.sum(intensity_unfiltered)  # TODO: check units and values
print('Filtered Null Depth: ', null_depth)
print('Throughput: ', throughput)
