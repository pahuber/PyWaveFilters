from astropy import units as u
from matplotlib import pyplot as plt

from wavefront_filtering.optical_elements.filter.fiber import Fiber
from wavefront_filtering.optical_elements.filter.pinhole import Pinhole
from wavefront_filtering.optical_elements.lens import Lens
from wavefront_filtering.wavefronts.wavefront import Wavefront

# Define wavefront
wavelength = 1e-5 * u.meter
initial_intensity = 10 * u.watt ** 0.5 / u.meter
zernike_modes = [(5, 0 * wavelength / 5)]
beam_diameter = 0.003 * u.meter
number_of_pixels = 100

wavefront = Wavefront(wavelength,
                      initial_intensity,
                      zernike_modes,
                      beam_diameter,
                      number_of_pixels)

# Define optical elements
focal_length = 0.008 * u.meter

lens = Lens(focal_length)

pinhole = Pinhole(1.22, wavefront)

fiber = Fiber(wavelength,
              21e-6 * u.meter,
              125e-6 * u.meter,
              1.6,
              1.59,
              wavefront,
              lens)

# Plot fiber mode
# e = wavefront.extent_focal_plane_dimensionless * focal_length.value / beam_diameter * wavelength / 2
# plt.imshow(abs(fiber.fundamental_fiber_mode) ** 2, extent=[-e, e, -e, e])
# plt.colorbar()
# plt.show()

e = 1
plt.imshow(wavefront.intensity.value, extent=[-e, e, -e, e])
plt.xticks([-2e-5, -1e-5, 0, 3.25e-5, 9.76e-5])
plt.colorbar()
plt.show()

# Apply optical elements
wavefront.apply(lens)

# e = wavefront.extent_focal_plane_meters.value / 2
plt.imshow(wavefront.intensity.value, extent=[-e, e, -e, e])
plt.xticks([-2e-5, -1e-5, 0, 3.25e-5, 9.76e-5])
plt.colorbar()
plt.title("after 1 lens")
plt.show()

# wavefront.apply(pinhole)


wavefront.apply(lens)

# e = wavefront.extent_focal_plane_meters.value / 2
plt.imshow(wavefront.intensity.value, extent=[-e, e, -e, e])
plt.xticks([-2e-5, -1e-5, 0, 3.25e-5, 9.76e-5])
plt.colorbar()
plt.title("after 2 lens")
plt.show()

wavefront.apply(lens)

wavefront.apply(fiber)
# e = wavefront.extent_focal_plane_meters.value / 2
plt.imshow(wavefront.intensity.value, extent=[-e, e, -e, e])
plt.xticks([-2e-5, -1e-5, 0, 3.25e-5, 9.76e-5])
plt.colorbar()
plt.title("after 3 lens")
plt.show()

wavefront.apply(lens)

# e = wavefront.extent_focal_plane_meters.value / 2
plt.imshow(wavefront.intensity.value, extent=[-e, e, -e, e])
plt.xticks([-2e-5, -1e-5, 0, 3.25e-5, 9.76e-5])
plt.colorbar()
plt.title("after 4 lens")
plt.show()

wavefront.apply(lens)

# e = wavefront.extent_focal_plane_meters.value / 2
plt.imshow(wavefront.intensity.value, extent=[-e, e, -e, e])
plt.xticks([-2e-5, -1e-5, 0, 3.25e-5, 9.76e-5])
plt.colorbar()
plt.title("after 5 lens")
plt.show()

# print(fiber.coupling_efficiency)

#
# e = 1
# plt.plot(np.angle(wavefront.complex_amplitude[number_of_pixels // 2]))
# plt.xticks([-2e-5, -1e-5, 0, 3.25e-5, 9.76e-5])
# # plt.colorbar()
# plt.show()
#
# # Apply optical elements
# wavefront.apply(lens)
#
# # e = wavefront.extent_focal_plane_meters.value / 2
# plt.plot(np.angle(wavefront.complex_amplitude[number_of_pixels // 2]))
# plt.title("after 1 lens")
# plt.show()
#
# # wavefront.apply(fiber)
# # wavefront.apply(pinhole)
#
# # wavefront.apply(fiber)
#
# wavefront.apply(lens)
#
# # e = wavefront.extent_focal_plane_meters.value / 2
# plt.plot(np.angle(wavefront.complex_amplitude[number_of_pixels // 2]))
# plt.title("after 2 lens")
# plt.show()
#
# wavefront.apply(lens)
#
# # e = wavefront.extent_focal_plane_meters.value / 2
# plt.plot(np.angle(wavefront.complex_amplitude[number_of_pixels // 2]))
# plt.title("after 3 lens")
# plt.show()
#
# wavefront.apply(lens)
#
# # e = wavefront.extent_focal_plane_meters.value / 2
# plt.plot(np.angle(wavefront.complex_amplitude[number_of_pixels // 2]))
# plt.title("after 4 lens")
# plt.show()
#
# wavefront.apply(lens)
#
# # e = wavefront.extent_focal_plane_meters.value / 2
# plt.plot(np.angle(wavefront.complex_amplitude[number_of_pixels // 2]))
# plt.title("after 5 lens")
# plt.show()
