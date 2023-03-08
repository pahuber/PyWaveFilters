import astropy
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u

from wavefront_filtering.optical_elements.optical_element import OpticalElement
from wavefront_filtering.wavefronts.wavefront import Wavefront


class Pinhole(OpticalElement):
    '''
    Class representing a pinhole, which can be used to filter a wavefront in the focal plane.
    '''

    def __init__(self, aperture_diameter):
        '''
        Constructor for pinhole object. Needs a aperture diameter.

                Parameters:
                        aperture_diameter: Aperture diameter pinhole in meters
        '''
        self.aperture_diameter = aperture_diameter
        # description = f'Pinhole with aperture diameter {self.aperture_diameter}.'

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

    def get_aperture_function(self, wavefront: Wavefront) -> np.ndarray:
        '''
        Return an array containing a circular aperture.

                Returns:
                        Array containing circular aperture.
        '''
        extent = np.linspace(-wavefront.aperture_diameter * 2, wavefront.aperture_diameter * 2,
                             wavefront.array_dimension)
        x_map, y_map = np.meshgrid(extent, extent)
        aperture_radius = self.aperture_diameter / 2

        return x_map ** 2 + y_map ** 2 < aperture_radius ** 2

    def apply(self, wavefront: Wavefront):
        '''
        Implementation of the apply method of the parent class. Used to apply the optical element to the wavefront.

                Parameters:
                        wavefront: Wavefront object
        '''
        self.aperture_function = self.get_aperture_function(wavefront)
        plt.imshow(abs(wavefront.complex_amplitude.value * self.aperture_function) ** 2)
        plt.colorbar()
        plt.show()
        wavefront.complex_amplitude *= self.aperture_function
