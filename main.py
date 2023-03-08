from numpy.fft import fft2, fftshift

from wavefront_filtering.optical_elements.lens import Lens
from wavefront_filtering.wavefronts.wavefront import Wavefront
import matplotlib.pyplot as plt
from astropy import units as u


if __name__ == '__main__':

    # Define wavefront
    wavelength = 1e-5 * u.meter
    initial_intensity = 1 * u.watt**0.5/u.meter
    zernike_modes = [(2, 1 * u.meter)]
    aperture_diameter = 0.01 * u.meter
    array_dimension = 100

    wavefront = Wavefront(wavelength,
                          initial_intensity,
                          zernike_modes,
                          aperture_diameter,
                          array_dimension)

    # Plot complex amplitude of wavefront in aperture plane
    plt.imshow(abs((wavefront.complex_amplitude.value)**2))
    plt.show()

    # Plot wavefront error phase map
    plt.imshow(abs(wavefront.wavefront_error.value)**2)
    plt.show()

    # Apply lens to transform to focal plane
    lens = Lens(100)
    wavefront.apply(lens)

    # Plot wavefront in focal plane
    plt.imshow(abs((wavefront.complex_amplitude.value)**2))
    plt.show()

    # Apply pinhole filter


    # Plot wavefront in focal plane after filter


    # Apply lens to transform to output plane


    # Plot wavefront in output plane