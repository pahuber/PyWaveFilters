import matplotlib.pyplot as plt
from astropy import units as u

from wavefront_filtering.optical_elements.filter.pinhole import Pinhole
from wavefront_filtering.optical_elements.lens import Lens
from wavefront_filtering.wavefronts.wavefront import Wavefront

# Define wavefront
wavelength = 1e-5 * u.meter
initial_intensity = 1 * u.watt ** 0.5 / u.meter
zernike_modes = [(5, 0 * wavelength / 10)]
beam_diameter = 0.003 * u.meter
number_of_pixels = 1000

wavefront = Wavefront(wavelength,
                      initial_intensity,
                      zernike_modes,
                      beam_diameter,
                      number_of_pixels)

# Plot complex amplitude of wavefront in aperture plane
extent_pupil = wavefront.extent_pupil_plane_meters.value / 2
plt.imshow(abs((wavefront.complex_amplitude.value) ** 2),
           extent=[-extent_pupil, extent_pupil, -extent_pupil, extent_pupil])
plt.xlabel('$x$ (m)')
plt.ylabel('$y$ (m)')
plt.title('Wavefront in Aperture Plane')
plt.colorbar()
plt.show()

# Plot wavefront error phase map
plt.imshow(abs(wavefront.initial_wavefront_error.value) ** 2)
plt.title('Wavefront Phase Map')
plt.colorbar()
plt.show()

# Apply lens to transform to focal plane
lens = Lens(0.001 * u.meter)
wavefront.apply(lens)

# Plot wavefront in focal plane
extent_focal = wavefront.extent_focal_plane_meters.value / 2
plt.imshow(abs((wavefront.complex_amplitude.value) ** 2),
           extent=[-extent_focal, extent_focal, -extent_focal, extent_focal])
plt.title('Wavefront in Focal Plane')
plt.xlabel('$\lambda/D$')
plt.ylabel('$\lambda/D$')
plt.colorbar()
plt.show()

# Apply pinhole filter
pinhole = Pinhole(0.03 / u.meter, wavefront)
wavefront.apply(pinhole)

# Plot filtered wavefront in focal plane
plt.imshow(abs((wavefront.complex_amplitude.value) ** 2),
           extent=[-extent_focal, extent_focal, -extent_focal, extent_focal])
plt.title('Filtered Wavefront in Focal Plane')
plt.xlabel('$\lambda/D$')
plt.ylabel('$\lambda/D$')
plt.colorbar()
plt.show()

# Apply lens to transform to output plane
lens2 = Lens(100)
wavefront.apply(lens2)

# Plot wavefront in output plane
plt.imshow(abs((wavefront.complex_amplitude.value) ** 2),
           extent=[-extent_pupil, extent_pupil, -extent_pupil, extent_pupil])
plt.title('Wavefront in Output Aperture Plane')
plt.xlabel('$x$ (m)')
plt.ylabel('$y$ (m)')
plt.colorbar()
plt.show()
