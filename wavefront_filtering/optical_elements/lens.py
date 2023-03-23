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
        description = f'Lens with focal length {self.focal_length}.'

    def apply(self, wavefront: BaseWavefront):
        '''
        Implementation of the apply method of the parent class. Used to apply the optical element to the wavefront.

                Parameters:
                        wavefront: Wavefront object
        '''
        if wavefront.is_pupil_plane:
            wavefront.complex_amplitude = fftshift(fft2(wavefront.complex_amplitude))
            wavefront.is_pupil_plane = False
        else:
            wavefront.complex_amplitude = ifft2(wavefront.complex_amplitude)
            wavefront.is_pupil_plane = True

    # TODO: add validation
