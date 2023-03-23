import astropy
import numpy as np
from astropy import units as u

from wavefront_filtering.optical_elements.optical_element import OpticalElement
from wavefront_filtering.wavefronts.wavefront import Wavefront, BaseWavefront


class Pinhole(OpticalElement):
    '''
    Class representing a pinhole, which can be used to filter a wavefront in the focal plane.
    '''

    def __init__(self, aperture_radius: float, wavefront: Wavefront):
        '''
        Constructor for pinhole object. Needs a aperture diameter.

                Parameters:
                        aperture_radius: Pinhole radius in wavelength/aperture diameter
                        wavefront: Wavefront object
        '''
        self.aperture_radius = aperture_radius
        self.wavefront = wavefront
        description = f'Pinhole with aperture diameter {self.aperture_radius}.'
        self.aperture_function = self.get_aperture_function()

    @property
    def aperture_radius(self) -> float:
        '''
        Return the aperture radius.

                Returns:
                        Float corresponding to aperture radius
        '''
        return self._aperture_radius

    @aperture_radius.setter
    def aperture_radius(self, value):
        '''
        Setter method for the aperture radius.
        '''
        if (type(value) == astropy.units.quantity.Quantity and value.unit == u.meter):
            raise ValueError(f'Units of pinhole aperture radius must be specified in dimensionless lambda/D.')
        self._aperture_radius = value

    def get_aperture_function(self) -> np.ndarray:
        '''
        Return an array containing a circular aperture.

                Returns:
                        Array containing circular aperture.
        '''
        extent = self.wavefront.array_width_focal_plane_dimensionless / 2
        extent_linear_space = np.linspace(-extent, extent,
                                          self.wavefront.array_dimension)
        x_map, y_map = np.meshgrid(extent_linear_space, extent_linear_space)
        return x_map ** 2 + y_map ** 2 < self.aperture_radius ** 2

    def apply(self, wavefront: BaseWavefront):
        '''
        Implementation of the apply method of the parent class. Used to apply the optical element to the wavefront.

                Parameters:
                        wavefront: Wavefront object
        '''
        if not (self.wavefront == wavefront):
            raise Exception('Pinhole must be applied to the same wavefront that was used to initialize it')
        else:
            if not wavefront.is_pupil_plane:
                wavefront.complex_amplitude *= self.aperture_function
            else:
                raise Exception('Pinholes can only be applied to wavefronts in the frequency domain')
