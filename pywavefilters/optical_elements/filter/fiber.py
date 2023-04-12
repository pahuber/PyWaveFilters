import astropy
import numpy as np
from astropy import units as u

from pywavefilters.optical_elements.lens import Lens
from pywavefilters.optical_elements.optical_element import BaseOpticalElement
from pywavefilters.wavefronts.wavefront import BaseWavefront


class Fiber(BaseOpticalElement):
    """
    Class representing an optical fiber, which can be used to filter a wavefront in the focal plane.
    """

    def __init__(self,
                 intended_wavelength: float,
                 beam_diameter: float,
                 core_radius: float,
                 cladding_radius: float,
                 core_refractive_index: float,
                 cladding_refractive_index: float,
                 number_of_pixels: int,
                 lens: Lens):
        """
        Constructor for fiber object.

                Parameters:
                        intended_wavelength: Wavelength to calculate the fiber's V-number
                        beam_diameter: Beam diameter of the wavefronts
                        core_radius: Core radius of the fiber in meters
                        cladding_radius: Cladding radius of the fiber in meters
                        core_refractive_index: Refractive index of the core
                        cladding_refractive_index: Refractive index of the cladding
                        number_of_pixels: Number of pixels of one array dimension
                        lens: Lens used for coupling

        """
        self.lens = lens
        self.intended_wavelength = intended_wavelength
        self.beam_diameter = beam_diameter
        self.core_radius = core_radius
        self.cladding_radius = cladding_radius
        self.core_refractive_index = core_refractive_index
        self.cladding_refractive_index = cladding_refractive_index
        self.number_of_pixels = number_of_pixels
        self.description = f'Fiber with core radius {self.core_radius}, cladding radius {self.cladding_radius},' \
                           f'core refractive index {self.core_refractive_index}, cladding refractive index' \
                           f'{cladding_refractive_index} and intended wavelength {self.intended_wavelength}.'

        self.v_number = self.get_v_number()
        self.fundamental_fiber_mode = self.get_fundamental_fiber_mode()
        self.coupling_efficiency = None

    @property
    def core_radius(self) -> float:
        """
        Return the core radius.

                Returns:
                        Float corresponding to core radius
        """
        return self._core_radius

    @core_radius.setter
    def core_radius(self, value):
        """
        Setter method for the core radius.
        """
        if not (type(value) == astropy.units.quantity.Quantity and value.unit == u.meter):
            raise ValueError(f'Units of core radius must be specified in meters.')
        if BaseWavefront.get_extent_focal_plane_meters(self.intended_wavelength, self.beam_diameter,
                                                       self.lens) / 2 < value:
            raise ValueError(
                f'core diameter {2 * value} must be smaller than the wavefront array width in meters '
                f'{BaseWavefront.get_extent_focal_plane_meters(self.intended_wavelength, self.beam_diameter, self.lens)}')
        self._core_radius = value

    def get_v_number(self) -> float:
        """
        Return the V-number of a fiber with the given properties.

                Returns:
                        V-number of the fiber
        """
        v_number = 2 * np.pi / self.intended_wavelength * self.core_radius * \
                   np.sqrt(self.core_refractive_index ** 2 - self.cladding_refractive_index ** 2)
        if v_number.value >= 2.405:
            raise Exception(f'Fiber with V-number {v_number} is not single-mode')
        else:
            return v_number

    def get_fundamental_fiber_mode(self) -> np.ndarray:
        """
        Return an array representing the cross-section of the fundamental fiber mode, using the Gaussian approximation
        by Shaklan & Roddier 1988.

                Returns:
                        Array representing the fundamental fiber mode
        """
        gaussian_width = self.core_radius * (0.65 + 1.619 / self.v_number ** (3 / 2) + 2.879 / self.v_number ** 6)

        extent = BaseWavefront.get_extent_focal_plane_meters(self.intended_wavelength, self.beam_diameter,
                                                             self.lens) / 2
        extent_linear_space = np.linspace(-extent, extent, self.number_of_pixels)
        X, Y = np.meshgrid(extent_linear_space, extent_linear_space)

        angles = np.arctan2(Y, X)
        radii = np.sqrt(X ** 2 + Y ** 2)
        shape = (self.number_of_pixels, self.number_of_pixels)
        fundamental_fiber_mode = np.zeros(shape)

        for x in range(shape[0]):
            for y in range(shape[1]):
                fundamental_fiber_mode[x][y] = 1 / gaussian_width.value * np.exp(
                    -radii[x][y].value ** 2 / gaussian_width.value ** 2)

        normalization_constant = 1 / np.sqrt(np.sum(abs(fundamental_fiber_mode) ** 2))

        return normalization_constant * fundamental_fiber_mode * u.watt ** 0.5 / u.meter

    def get_coupling_efficiency(self, wavefront: BaseWavefront) -> float:
        """
        Return the coupling efficiency of the input field into the fiber.

                Returns:
                        A float corresponding to the coupling efficiency
        """
        coupling_efficiency = np.sum(self.fundamental_fiber_mode * wavefront.complex_amplitude) / (
                np.sqrt(np.sum(abs(self.fundamental_fiber_mode) ** 2)) * np.sqrt(np.sum(
            abs(wavefront.complex_amplitude) ** 2)))
        return coupling_efficiency

    def apply(self, wavefront: BaseWavefront):
        """
        Implementation of the apply method of the parent class. Used to apply the optical element to the wavefront. Sets
        the complex amplitude of the input wavefront to the (normalized) fiber mode, scaled by the coupling efficiency
        and the initial amplitude

                Parameters:
                        wavefront: Base wavefront object
        """
        if not (self.intended_wavelength == wavefront.wavelength and self.beam_diameter == wavefront.beam_diameter):
            raise Exception('Wavelength and beam diameter must match the ones from the wavefront')
        else:
            if not wavefront.is_in_pupil_plane:
                self.coupling_efficiency = self.get_coupling_efficiency(wavefront)
                wavefront.complex_amplitude = self.fundamental_fiber_mode * self.coupling_efficiency * np.sqrt(
                    np.sum(wavefront.intensity.value))
                # TODO: check correct output of complex amplitude
                wavefront.has_fiber_been_applied = True
            else:
                raise Exception('Fibers can only be applied to wavefronts in the focal plane')
