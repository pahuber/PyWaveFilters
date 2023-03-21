import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u

from wavefront_filtering.optical_elements.filter.pinhole import Pinhole
from wavefront_filtering.optical_elements.lens import Lens
from wavefront_filtering.wavefronts.wavefront import Wavefront

if __name__ == '__main__':
    # Define wavefront
    wavelength = 1e-5 * u.meter
    initial_intensity = 1 * u.watt ** 0.5 / u.meter
    zernike_modes = [(5, 0 * wavelength / 10)]
    aperture_diameter = 0.01 * u.meter
    array_dimension = 400

    wavefront = Wavefront(wavelength,
                          initial_intensity,
                          zernike_modes,
                          aperture_diameter,
                          array_dimension)

    # Plot complex amplitude of wavefront in aperture plane
    plt.imshow(abs((wavefront.complex_amplitude.value) ** 2),
               extent=[-wavefront.extent.value, wavefront.extent.value, -wavefront.extent.value,
                       wavefront.extent.value])
    plt.xlabel('$x$ (m)')
    plt.ylabel('$y$ (m)')
    plt.title('Wavefront in Aperture Plane')
    plt.colorbar()
    plt.show()
    print(np.sum(abs(wavefront.complex_amplitude.value) ** 2))

    # # Plot wavefront error phase map
    plt.imshow(abs(wavefront.initial_wavefront_error.value) ** 2)
    plt.title('Wavefront Phase Map')
    plt.colorbar()
    plt.show()

    # Apply lens to transform to focal plane
    lens = Lens(100)
    wavefront.apply(lens)

    # Plot wavefront in focal plane
    plt.imshow(abs((wavefront.complex_amplitude.value) ** 2))
    plt.title('Wavefront in Focal Plane')
    plt.colorbar()
    plt.show()

    # Apply pinhole filter
    pinhole = Pinhole(0.03 / u.meter, wavefront)
    wavefront.apply(pinhole)

    # Plot filtered wavefront in focal plane
    plt.imshow(abs((wavefront.complex_amplitude.value) ** 2))
    plt.title('Filtered Wavefront in Focal Plane')
    plt.colorbar()
    plt.show()

    # Apply lens to transform to output plane
    lens2 = Lens(100)
    wavefront.apply(lens2)

    # Plot wavefront in output plane
    plt.imshow(abs((wavefront.complex_amplitude.value) ** 2))
    plt.title('Wavefront in Output Aperture Plane')
    plt.colorbar()
    plt.show()
    print(np.sum(abs(wavefront.complex_amplitude.value) ** 2))
