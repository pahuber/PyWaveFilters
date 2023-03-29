import warnings

import astropy
import numpy as np
from astropy import units as u
from scipy.optimize import fsolve

from wavefront_filtering.optical_elements.filter.bessel import get_system_of_equations, \
    get_mode_function
from wavefront_filtering.optical_elements.lens import Lens
from wavefront_filtering.optical_elements.optical_element import OpticalElement
from wavefront_filtering.wavefronts.wavefront import BaseWavefront


class Fiber(OpticalElement):
    '''
    Class representing an optical fiber, which can be used to filter a wavefront in the focal plane.
    '''

    def __init__(self,
                 intended_wavelength: float,
                 core_radius: float,
                 cladding_radius: float,
                 core_refractive_index: float,
                 cladding_refractive_index: float,
                 wavefront: BaseWavefront,
                 lens: Lens):
        '''
        Constructor for fiber object.

                Parameters:
                        intended_wavelength: Wavelength to calculate the fiber's V-number
                        core_radius: Core radius of the fiber in meters
                        cladding_radius: Cladding radius of the fiber in meters
                        core_refractive_index: Refractive index of the core
                        cladding_refractive_index: Refractive index of the cladding
                        wavefront: Base wavefront object
                        lens: Lens used for coupling

        '''
        self.wavefront = wavefront
        self.lens = lens
        self.intended_wavelength = intended_wavelength
        self.core_radius = core_radius
        self.cladding_radius = cladding_radius
        self.core_refractive_index = core_refractive_index
        self.cladding_refractive_index = cladding_refractive_index
        self.description = f'Fiber with core radius {self.core_radius}, cladding radius {self.cladding_radius},' \
                           f'core refractive index {self.core_refractive_index}, cladding refractive index' \
                           f'{cladding_refractive_index} and intended wavelength {self.intended_wavelength}.'

        self.v_number = self.get_v_number()
        self.fundamental_fiber_mode = self.get_fundamental_fiber_mode()
        self.coupling_efficiency = None

    @property
    def core_radius(self) -> float:
        '''
        Return the core radius.

                Returns:
                        Float corresponding to core radius
        '''
        return self._core_radius

    @core_radius.setter
    def core_radius(self, value):
        '''
        Setter method for the core radius.
        '''
        if not (type(value) == astropy.units.quantity.Quantity and value.unit == u.meter):
            raise ValueError(f'Units of core radius must be specified in meters.')
        if self.wavefront.extent_focal_plane_dimensionless * self.lens.focal_length / 2 < value:
            raise ValueError(
                f'core diameter {2 * value} must be smaller than the wavefront array width in meters '
                f'{self.wavefront.extent_focal_plane_dimensionless * self.lens.focal_length}')
        self._core_radius = value

    def get_v_number(self) -> float:
        '''
        Return the V-number of a fiber with the given properties.

                Returns:
                        V-number of the fiber
        '''
        v_number = 2 * np.pi / self.intended_wavelength * self.core_radius * \
                   np.sqrt(self.core_refractive_index ** 2 - self.cladding_refractive_index ** 2)
        if v_number.value >= 2.405:
            raise Exception(f'Fiber with V-number {v_number} is not single-mode')
        else:
            return v_number

    def get_fundamental_fiber_mode(self) -> np.ndarray:
        '''
        Return an array representing the cross-section of the fundamental fiber mode.

                Returns:
                        Array representing the fundamental fiber mode
        '''
        u_variable, w_variable = fsolve(get_system_of_equations,
                                        (np.sqrt(self.v_number).value, np.sqrt(self.v_number).value),
                                        self.v_number.value)

        extent = self.lens.get_extent_focal_plane_meters() / 2
        extent_linear_space = np.linspace(-extent, extent, self.wavefront.number_of_pixels)
        X, Y = np.meshgrid(extent_linear_space, extent_linear_space)

        angles = np.arctan2(Y, X)
        radii = np.sqrt(X ** 2 + Y ** 2)
        wavefront_shape = self.wavefront.complex_amplitude.shape
        fundamental_fiber_mode = np.zeros(wavefront_shape)

        for x in range(wavefront_shape[0]):
            for y in range(wavefront_shape[1]):
                fundamental_fiber_mode[x][y] = get_mode_function(radii[x][y], angles[x][y], u_variable, w_variable,
                                                                 self.core_radius)

        normalization_constant = 1 / np.sqrt(np.sum(abs(self.fundamental_fiber_mode) ** 2))

        return normalization_constant * fundamental_fiber_mode

    def get_coupling_efficiency(self) -> float:
        '''
        Return the coupling efficiency of the input field into the fiber.

                Returns:
                        A float corresponding to the coupling efficiency
        '''

        coupling_efficiency = abs(
            np.sum(self.fundamental_fiber_mode.conjugate() * self.wavefront.complex_amplitude)) ** 2 / (
                                      np.sum(abs(self.fundamental_fiber_mode) ** 2) * np.sum(
                                  abs(self.wavefront.complex_amplitude) ** 2))

        if coupling_efficiency <= 0.7:
            warnings.warn(f'Coupling efficiency is only {coupling_efficiency * 100} %')
        if coupling_efficiency > 0.85:
            warnings.warn(
                f'Coupling efficiency {coupling_efficiency * 100} % is larger than the theoretical limit.'
                f'Something must be wrong')

        return coupling_efficiency

    def apply(self, wavefront: BaseWavefront):
        '''
        Implementation of the apply method of the parent class. Used to apply the optical element to the wavefront.

                Parameters:
                        wavefront: Base wavefront object
        '''

        if not (self.wavefront == wavefront):
            raise Exception('Fiber must be applied to the same wavefront that was used to initialize it')
        else:
            if not wavefront.is_in_pupil_plane:
                self.coupling_efficiency = self.get_coupling_efficiency()
                wavefront.complex_amplitude = self.fundamental_fiber_mode * self.coupling_efficiency
                # TODO: check correct output of complex amplitude
                wavefront.has_fiber_been_applied = True
            else:
                raise Exception('Fibers can only be applied to wavefronts in the focal plane')
