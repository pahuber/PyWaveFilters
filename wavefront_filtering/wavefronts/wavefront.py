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
        self.complex_amplitude = None

    def __add__(self, other_wavefront):
        return CombinedWavefront(self.complex_amplitude + other_wavefront.complex_amplitude)

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
                 aperture_diameter: float,
                 array_dimension: int):
        '''
        Constructor for wavefront object.

                Parameters:
                        wavelength: Wavelength of the wavefront in meters
                        initial_amplitude: Initial amplitude of the wavefront in sqrt(watts)/meter
                        zernike_modes: List containing the zernike mode indices and their coefficients in meters
                        aperture_diameter: Aperture diameter in the aperture plane in meters
                        array_dimension: Side length of the output array in pixels
        '''
        self.wavelength = wavelength
        self.initial_amplitude = initial_amplitude
        self.zernike_modes = zernike_modes
        self.aperture_diameter = aperture_diameter
        self.array_dimension = array_dimension

        self.aperture_function = self.get_aperture_function()
        self.initial_wavefront_error = self.get_wavefront_error()
        self.complex_amplitude = self.get_initial_complex_amplitude()

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
    def aperture_diameter(self) -> float:
        '''
        Return the aperture diameter.

                Returns:
                        Float corresponding to aperture diameter
        '''
        return self._aperture_diameter

    @aperture_diameter.setter
    def aperture_diameter(self, value):
        '''
        Setter method for the aperture diameter.
        '''
        if not (type(value) == astropy.units.quantity.Quantity and value.unit == u.meter):
            raise ValueError(f'Units of aperture diameter must be specified in meters.')
        self._aperture_diameter = value

    @property
    def array_dimension(self) -> int:
        '''
        Return the array dimension.

                Returns:
                        Integer corresponding to the array dimension
        '''
        return self._array_dimension

    @array_dimension.setter
    def array_dimension(self, value):
        '''
        Setter method for the array dimension.
        '''
        if not (type(value) == int and value > 0):
            raise ValueError(f'Array dimension must be a positive integer.')
        self._array_dimension = value

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
        self.extent = 15 * self.aperture_diameter
        extent_linear_space = np.linspace(-self.extent, self.extent, self.array_dimension)
        self._x_map, self._y_map = np.meshgrid(extent_linear_space, extent_linear_space)
        self._aperture_radius = self.aperture_diameter / 2

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
    def __init__(self, complex_amplitude: np.ndarray):
        self.complex_amplitude = complex_amplitude
