import astropy
import numpy as np
from astropy import units as u

from wavefront_filtering.optical_elements.optical_element import OpticalElement
from wavefront_filtering.wavefronts.wavefront import Wavefront, BaseWavefront


class Pinhole(OpticalElement):
    '''
    Class representing a pinhole, which can be used to filter a wavefront in the focal plane.
    '''

    def __init__(self, aperture_diameter: float, wavefront: Wavefront):
        '''
        Constructor for pinhole object. Needs a aperture diameter.

                Parameters:
                        aperture_diameter: Aperture diameter pinhole in 1/meters
                        wavefront: Wavefront object
        '''
        self.aperture_diameter = aperture_diameter
        self.wavefront = wavefront
        description = f'Pinhole with aperture diameter {self.aperture_diameter}.'
        self.aperture_function = self.get_aperture_function()

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
        if not (type(value) == astropy.units.quantity.Quantity and value.unit == 1 / u.meter):
            raise ValueError(f'Units of pinhole aperture diameter must be specified in 1/meters.')
        self._aperture_diameter = value

    def get_aperture_function(self) -> np.ndarray:
        '''
        Return an array containing a circular aperture.

                Returns:
                        Array containing circular aperture.
        '''
        extent = np.linspace(-self.wavefront.aperture_diameter, self.wavefront.aperture_diameter,
                             self.wavefront.array_dimension)
        x_map, y_map = np.meshgrid(extent, extent)
        aperture_radius = self.wavefront.wavelength / self.wavefront.aperture_diameter * 1.22 * u.meter
        return x_map ** 2 + y_map ** 2 < aperture_radius ** 2

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
