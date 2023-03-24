import astropy
from astropy import units as u
from numpy.fft import fft2, fftshift, ifft2

from wavefront_filtering.optical_elements.optical_element import OpticalElement
from wavefront_filtering.wavefronts.wavefront import BaseWavefront


class Lens(OpticalElement):
    '''
    Class representing a lens, which can be used to transform a wavefront from aperture plane to focal plane or
    vice-versa.
    '''

    def __init__(self, focal_length: float):
        '''
        Constructor for lens object. Needs a focal length.

                Parameters:
                        focal_length: Focal length of the lens in meters
        '''
        self.focal_length = focal_length
        self.description = f'Lens with focal length {self.focal_length}.'

    @property
    def focal_length(self) -> float:
        '''
        Return the focal length.

                Returns:
                        Float corresponding to the focal length
        '''
        return self._focal_length

    @focal_length.setter
    def focal_length(self, value):
        '''
        Setter method for the focal length.
        '''
        if not (type(value) == astropy.units.quantity.Quantity and value.unit == u.meter):
            raise ValueError(f'Units of focal length must be specified in meters.')
        self._focal_length = value

    def apply(self, wavefront: BaseWavefront):
        '''
        Implementation of the apply method of the parent class. Used to apply the optical element to the wavefront.

                Parameters:
                        wavefront: Wavefront object
        '''
        if wavefront.is_pupil_plane:
            wavefront.complex_amplitude = fftshift(fft2(wavefront.complex_amplitude))
            wavefront.is_pupil_plane = False
            wavefront.array_width_focal_plane_length = wavefront.array_width_focal_plane_dimensionless * \
                                                       self.focal_length
        else:
            wavefront.complex_amplitude = ifft2(wavefront.complex_amplitude)
            wavefront.is_pupil_plane = True
            wavefront.array_width_focal_plane_length = None
