import astropy
import numpy as np
from astropy import units as u

from wavefront_filtering.optical_elements.optical_element import OpticalElement
from wavefront_filtering.wavefronts.zernike import get_zernike_polynomial


class Wavefront:
    def __init__(self,
                 wavelength: float,
                 initial_amplitude: float,
                 zernike_modes: list,
                 aperture_diameter: float,
                 array_dimension: int):
        self.wavelength = wavelength
        self.initial_amplitude = initial_amplitude
        self.zernike_modes = zernike_modes
        self.aperture_diameter = aperture_diameter
        self.array_dimension = array_dimension

        self.aperture_function = self.get_aperture_function()
        self.wavefront_error = self.get_wavefront_error()
        self.complex_amplitude = self.get_complex_amplitude()


    @property
    def wavelength(self):
        return self._wavelength


    @wavelength.setter
    def wavelength(self, value):
        if not (type(value) == astropy.units.quantity.Quantity and value.unit == u.meter):
            raise ValueError(f'Units of wavelength must be specified in meters.')
        self._wavelength = value


    @property
    def initial_amplitude(self):
        return self._initial_amplitude


    @initial_amplitude.setter
    def initial_amplitude(self, value):
        if not (type(value) == astropy.units.quantity.Quantity and value.unit ** 2 == u.watt / u.meter ** 2):
            raise ValueError(f'Units of the initial amplitude must be specified in sqrt(watt/meter^2).')
        self._initial_amplitude = value


    @property
    def aperture_diameter(self):
        return self._aperture_diameter


    @aperture_diameter.setter
    def aperture_diameter(self, value):
        if not (type(value) == astropy.units.quantity.Quantity and value.unit == u.meter):
            raise ValueError(f'Units of aperture diameter must be specified in meters.')
        self._aperture_diameter = value


    @property
    def array_dimension(self):
        return self._array_dimension


    @array_dimension.setter
    def array_dimension(self, value):
        if not (type(value) == int and value > 0):
            raise ValueError(f'Array dimension must be a positive integer.')
        self._array_dimension = value


    @property
    def intensity(self):
        return abs(self.complex_amplitude)**2


    def get_aperture_function(self):
        extent = np.linspace(-self.aperture_diameter * 2, self.aperture_diameter * 2, self.array_dimension)
        self._x_map, self._y_map = np.meshgrid(extent, extent)
        self._aperture_radius = self.aperture_diameter / 2

        return self.initial_amplitude * (self._x_map ** 2 + self._y_map ** 2 < self._aperture_radius ** 2).astype(
            complex)


    def get_wavefront_error(self) -> np.ndarray:
        '''
        Return a wavefronts error composed of a sum of several Zernike polynomial terms Z_j.

                Returns:
                        Wavefront error
        '''
        if self.zernike_modes is None:
            return 0 * u.meter

        radial_map = np.sqrt(self._x_map**2 + self._y_map**2)
        angular_map = np.arctan2(self._y_map, self._x_map)

        wavefront_error = 0
        for element in self.zernike_modes:
            zernike_mode_index = element[0]
            mode_coefficient = element[1]
            wavefront_error += mode_coefficient * get_zernike_polynomial(zernike_mode_index, radial_map, angular_map,
                                                                         self._aperture_radius)
        return wavefront_error


    def get_complex_amplitude(self):
        return self.aperture_function * np.exp(-2 * np.pi * 1j * self.wavefront_error / self.wavelength)


    def apply(self, optical_element: OpticalElement):
        optical_element.apply(self)