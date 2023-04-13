import numpy as np
from astropy import units as u

from pywavefilters.optical_elements.filter.fiber import Fiber
from pywavefilters.optical_elements.lens import Lens
from pywavefilters.wavefronts.wavefront import Wavefront

# Define wavefront
wavelength = 15e-6 * u.meter
zernike_modes_1 = [(5, wavelength / 100)]
zernike_modes_2 = [(6, wavelength / 200)]
beam_diameter = 0.003 * u.meter
number_of_pixels = 200

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

intensity_unfiltered = (wavefront_1 + wavefront_2).intensity

# Transform to focal plane and apply fibers
wavefront_1.apply(lens)
wavefront_2.apply(lens)
wavefront_1.apply(fiber)
wavefront_2.apply(fiber)

# Remove piston difference with delay line
piston_diff = np.angle(wavefront_1.complex_amplitude) - np.angle(wavefront_2.complex_amplitude)
wavefront_1.complex_amplitude *= np.exp(-1j * piston_diff.value)

# Combine wavefronts and normalize
wavefront_const = wavefront_1 + wavefront_2
wavefront_dest = wavefront_1 - wavefront_2

# Transform to output plane
wavefront_const.apply(lens)
wavefront_dest.apply(lens)

# Calculate null depth and throughput
null_depth = np.sum(wavefront_dest.intensity) / np.sum(wavefront_const.intensity)
intensity_filtered = wavefront_const.intensity
throughput = np.sum(intensity_filtered) / np.sum(intensity_unfiltered)
print('Null Depth: ', null_depth)
print('Throughput: ', throughput)
