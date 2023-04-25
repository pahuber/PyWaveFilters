import astropy
from astropy import units as u

from pywavefilters.optical_elements.optical_element import BaseOpticalElement
from pywavefilters.util.math import get_2d_chirp_z_transform
from pywavefilters.wavefronts.wavefront import BaseWavefront


class Lens(BaseOpticalElement):
    """
    Class representing a lens, which can be used to transform a wavefront from aperture plane to focal plane or
    vice-versa.
    """

    def __init__(self, focal_length: float):
        """
        Constructor for lens object. Needs a focal length.

                Parameters:
                        focal_length: Focal length of the lens in meters
        """
        self.focal_length = focal_length
        self.description = f'Lens with focal length {self.focal_length}.'

    @property
    def focal_length(self) -> float:
        """
        Return the focal length.

                Returns:
                        Float corresponding to the focal length
        """
        return self._focal_length

    @focal_length.setter
    def focal_length(self, value):
        """
        Setter method for the focal length.
        """
        if not (type(value) == astropy.units.quantity.Quantity and value.unit == u.meter):
            raise ValueError(f'Units of focal length must be specified in meters.')
        self._focal_length = value

    def apply(self, wavefront: BaseWavefront):
        """
        Implementation of the apply method of the parent class. Used to apply the optical element to the wavefront.

                Parameters:
                        wavefront: Base wavefront object
        """
        # TODO: double check if phase behaviour is correct after lens transforms
        if wavefront.is_in_pupil_plane:
            wavefront.complex_amplitude = get_2d_chirp_z_transform(wavefront.complex_amplitude,
                                                                   wavefront.number_of_pixels,
                                                                   BaseWavefront._chirp_z_maximum_frequency) \
                                          * wavefront.complex_amplitude.unit

            # Normalize by number of pixels to make independent of amount of pixels
            wavefront.complex_amplitude /= wavefront.number_of_pixels

            wavefront.is_in_pupil_plane = False
            wavefront.extent_focal_plane_meters = wavefront.get_extent_focal_plane_meters(wavefront.wavelength,
                                                                                          wavefront.beam_diameter, self)
        else:
            # TODO: implement inverse CZT algorithm for transformations from the focal to the pupil plane
            raise Exception('Currently, lenses can only be applied to wavefronts in the pupil plane')
            # wavefront.complex_amplitude = fftshift(ifft2(ifftshift(wavefront.complex_amplitude)))
            #
            # # Normalize by number of pixels to make independent of amount of pixels
            # wavefront.complex_amplitude *= wavefront.number_of_pixels
            # if wavefront.has_fiber_been_applied:
            #     wavefront.has_fiber_been_applied = None
            #
            # wavefront.is_in_pupil_plane = True
            # wavefront.extent_focal_plane_meters = None
