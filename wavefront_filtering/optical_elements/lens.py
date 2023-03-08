import astropy
import numpy as np
from astropy import units as u
from numpy.dual import fft2
from numpy.fft import fftshift

from wavefront_filtering.optical_elements.optical_element import OpticalElement
from wavefront_filtering.wavefronts.wavefront import Wavefront


class Lens(OpticalElement):
    def __init__(self, focal_length):
        self.focal_length = focal_length
        description = f'Lens with focal length {self.focal_length}.'


    def apply(self, wavefront: Wavefront):
        wavefront.complex_amplitude = fftshift(fft2(wavefront.complex_amplitude))

# def get_aperture(aperture_diameter: float, pixels_per_dimension: int) -> np.ndarray:
#     '''
#         Returns an array of physical dimensions 0.1m x 0.1m containing a circular aperture.
#
#                 Parameters:
#                         aperture_diameter: Diameter of the aperture in meters
#                         pixels_per_dimension: Side length of the array, i.e. number of pixels per dimension
#
#                 Returns:
#                         Complex array containing a flat, circular aperture.
#     '''
#     aperture_plane_half_width = 0.05 * u.meter
#
#     if not (type(aperture_diameter) == astropy.units.quantity.Quantity and aperture_diameter.unit == u.meter):
#         raise ValueError(f'Units of aperture diameter must be specified in meters.')
#
#     if aperture_diameter > 2 * aperture_plane_half_width:
#         raise ValueError(f'Aperture diameter may not be larger than {2 * aperture_plane_half_width}')
#
#     linspace = np.linspace(-aperture_plane_half_width, aperture_plane_half_width, pixels_per_dimension)
#     x_map, y_map = np.meshgrid(linspace, linspace)
#     aperture_radius = aperture_diameter / 2
#     aperture_map = (x_map ** 2 + y_map ** 2 < aperture_radius ** 2).astype(complex)
#
#     return aperture_map