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
                        wavefront: Base wavefront object
        '''
        if wavefront.is_pupil_plane:
            wavefront.complex_amplitude = fftshift(wavefront.complex_amplitude)  # TODO: double check this shift
            wavefront.complex_amplitude = fftshift(fft2(wavefront.complex_amplitude))
            wavefront.is_pupil_plane = False
            wavefront.extent_focal_plane_meters = wavefront.extent_focal_plane_dimensionless / \
                                                  wavefront.beam_diameter * self.focal_length * \
                                                  wavefront.wavelength
            # TODO: implement method to calculate focal plane extent in meters
        else:
            if wavefront.has_fiber_been_applied:
                wavefront.complex_amplitude = fftshift(ifft2(wavefront.complex_amplitude))
                wavefront.has_fiber_been_applied = None
            else:
                wavefront.complex_amplitude = ifft2(wavefront.complex_amplitude)

            wavefront.is_pupil_plane = True
            wavefront.extent_focal_plane_meters = None
