import astropy
import numpy as np
from astropy import units as u

from wavefront_filtering.optical_elements.optical_element import OpticalElement
from wavefront_filtering.wavefronts.zernike import get_zernike_polynomial


class BaseWavefront:
    '''
    Base class to represent wavefronts.
    '''

    def __init__(self):
        '''
        Constructor for base wavefront object.
        '''
        self.complex_amplitude = None
        self.is_in_pupil_plane = None
        self.extent_pupil_plane_meters = None
        self.extent_focal_plane_dimensionless = None
        self.extent_focal_plane_meters = None  # Is reset to None after leaving the focal plane
        self.number_of_pixels = 1
        self.has_fiber_been_applied = None
        self._length_per_pixel = 300e-6 * u.meter
        self.beam_diameter = 1 * u.meter  # TODO: implement correctly
        self.wavelength = 1 * u.meter  # TODO: implement correctly

    def __add__(self, other_wavefront):
        '''
        Method to add two base wavefront together.

                Parameters:
                        other_wavefront: Base wavefront object to be added
                Returns:
                        Combined wavefront object
        '''
        if self.is_in_pupil_plane == other_wavefront.is_in_pupil_plane:
            return CombinedWavefront(self.complex_amplitude + other_wavefront.complex_amplitude,
                                     self.is_in_pupil_plane,
                                     self.extent_pupil_plane_meters,
                                     self.extent_focal_plane_dimensionless,
                                     self.extent_focal_plane_meters,
                                     self.number_of_pixels,
                                     self.has_fiber_been_applied)
        else:
            raise ValueError('Wavefronts must both be in pupil or in focal plane')

    def apply(self, optical_element: OpticalElement):
        '''
        Apply an optical element.
        '''
        optical_element.apply(self)


class Wavefront(BaseWavefront):
    '''
    Class representing a wavefront.
    '''

    def __init__(self,
                 wavelength: float,
                 initial_amplitude: float,
                 zernike_modes: list,
                 beam_diameter: float,
                 number_of_pixels: int):
        '''
        Constructor for wavefront object.

                Parameters:
                        wavelength: Wavelength of the wavefront in meters
                        initial_amplitude: Initial amplitude of the wavefront in sqrt(watts)/meter
                        zernike_modes: List containing the zernike mode indices and their coefficients in meters
                        beam_diameter: Beam diameter in the aperture plane in meters
                        number_of_pixels: Side length of the output array in pixels (1 pixel =^ 300 um)
        '''
        BaseWavefront.__init__(self)
        self.wavelength = wavelength
        self.initial_amplitude = initial_amplitude
        self.zernike_modes = zernike_modes
        self.beam_diameter = beam_diameter
        self.number_of_pixels = number_of_pixels

        self.extent_pupil_plane_meters = number_of_pixels * self._length_per_pixel
        self.extent_focal_plane_dimensionless = beam_diameter / self._length_per_pixel
        self.aperture_function = self.get_aperture_function()
        self.initial_wavefront_error = self.get_wavefront_error()
        self.complex_amplitude = self.get_initial_complex_amplitude()
        self.is_in_pupil_plane = True

    @property
    def wavelength(self) -> float:
        '''
        Return the wavelength.

                Returns:
                        Float corresponding to the wavelength
        '''
        return self._wavelength

    @wavelength.setter
    def wavelength(self, value):
        '''
        Setter method for the wavelength.
        '''
        if not (type(value) == astropy.units.quantity.Quantity and value.unit == u.meter):
            raise ValueError(f'Units of wavelength must be specified in meters.')
        self._wavelength = value

    @property
    def initial_amplitude(self) -> float:
        '''
        Return the initial amplitude.

                Returns:
                        Float corresponding to the initial amplitude
        '''
        return self._initial_amplitude

    @initial_amplitude.setter
    def initial_amplitude(self, value):
        '''
        Setter method for the initial amplitude.
        '''
        if not (type(value) == astropy.units.quantity.Quantity and value.unit ** 2 == u.watt / u.meter ** 2):
            raise ValueError(f'Units of the initial amplitude must be specified in sqrt(watt/meter^2).')
        self._initial_amplitude = value

    @property
    def beam_diameter(self) -> float:
        '''
        Return the beam diameter.

                Returns:
                        Float corresponding to beam diameter
        '''
        return self._beam_diameter

    @beam_diameter.setter
    def beam_diameter(self, value):
        '''
        Setter method for the beam diameter.
        '''
        if not (type(value) == astropy.units.quantity.Quantity and value.unit == u.meter):
            raise ValueError(f'Units of beam diameter must be specified in meters.')
        self._beam_diameter = value

    @property
    def number_of_pixels(self) -> int:
        '''
        Return the number of pixels.

                Returns:
                        Integer corresponding to the number of pixels
        '''
        return self._number_of_pixels

    @number_of_pixels.setter
    def number_of_pixels(self, value):
        '''
        Setter method for the number of pixels.
        '''
        if not (type(value) == int and value > 0):
            raise ValueError(f'Number of pixels must be a positive integer.')
        self._number_of_pixels = value

    @property
    def intensity(self) -> np.ndarray:
        '''
        Return the intensity as a function of the complex amplitude.

                Returns:
                        Array containing intensity
        '''
        return abs(self.complex_amplitude) ** 2

    def get_aperture_function(self) -> np.ndarray:
        '''
        Return an array containing a circular aperture.

                Returns:
                        Array containing circular aperture.
        '''
        extent = self.number_of_pixels / 2 * self._length_per_pixel
        extent_linear_space = np.linspace(-extent, extent, self.number_of_pixels)
        self._x_map, self._y_map = np.meshgrid(extent_linear_space, extent_linear_space)
        self._aperture_radius = self.beam_diameter / 2

        return self.initial_amplitude * (self._x_map ** 2 + self._y_map ** 2 < self._aperture_radius ** 2).astype(
            complex)

    def get_wavefront_error(self) -> np.ndarray:
        '''
        Return a wavefront error composed of a sum of several Zernike polynomial terms Z_j.

                Returns:
                        Array containing wavefront error
        '''
        if self.zernike_modes is None:
            return 0 * u.meter

        radial_map = np.sqrt(self._x_map ** 2 + self._y_map ** 2)
        angular_map = np.arctan2(self._y_map, self._x_map)

        wavefront_error = 0
        for element in self.zernike_modes:
            zernike_mode_index = element[0]
            mode_coefficient = element[1]
            wavefront_error += mode_coefficient * get_zernike_polynomial(zernike_mode_index, radial_map, angular_map,
                                                                         self._aperture_radius)
        return wavefront_error

    def get_initial_complex_amplitude(self) -> np.ndarray:
        '''
        Return an array containing the complex amplitude of the wavefront.

                Returns:
                        Array containing the complex amplitude.
        '''
        return self.aperture_function * np.exp(-2 * np.pi * 1j * self.initial_wavefront_error / self.wavelength)


class CombinedWavefront(BaseWavefront):
    '''
    Base class representing combined wavefronts.
    '''

    def __init__(self,
                 complex_amplitude: np.ndarray,
                 is_in_pupil_plane: bool,
                 extent_pupil_plane_meters: float,
                 extent_focal_plane_dimensionless: float,
                 extent_focal_plane_meters: float,
                 number_of_pixels: int,
                 is_fiber_applied: bool):
        '''
        Constructor for combined wavefront object.

                Parameters:
                        complex_amplitude: Complex amplitude of the combined wavefront
                        is_in_pupil_plane: Boolean specifying whether we are in the spatial domain or not
                        extent_pupil_plane_meters: Full array width in pupil plane in meters
                        extent_focal_plane_dimensionless: Full array width in focal plane dimensionless
                        extent_focal_plane_meters: Full array width in focal plane in meters
                        number_of_pixels: Number of pixels in array
                        is_fiber_applied: Boolean specifying whether a fiber has been applied
        '''
        self.complex_amplitude = complex_amplitude
        self.is_in_pupil_plane = is_in_pupil_plane
        self.extent_pupil_plane_meters = extent_pupil_plane_meters
        self.extent_focal_plane_dimensionless = extent_focal_plane_dimensionless
        self.extent_focal_plane_meters = extent_focal_plane_meters
        self.number_of_pixels = number_of_pixels
        self.is_fiber_applied = is_fiber_applied
