import numpy as np
from scipy.optimize import fsolve

from wavefront_filtering.optical_elements.filter.bessel_equation import get_system_of_equations, \
    get_mode_function
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
                 wavefront: BaseWavefront):
        '''
        Constructor for fiber object.

                Parameters:
                        intended_wavelength: Wavelength to calculate the fiber's V-number
                        core_radius: Core radius of the fiber in meters
                        cladding_radius: Cladding radius of the fiber in meters
                        core_refractive_index: Refractive index of the core
                        cladding_refractive_index: Refractive index of the cladding
                        wavefront: Base wavefront object

        '''
        self.intended_wavelength = intended_wavelength
        self.core_radius = core_radius
        self.cladding_radius = cladding_radius
        self.core_refractive_index = core_refractive_index
        self.cladding_refractive_index = cladding_refractive_index
        self.wavefront = wavefront
        description = f'Fiber with core radius {self.core_radius}, cladding radius {self.cladding_radius},' \
                      f'core refractive index {self.core_refractive_index}, cladding refractive index' \
                      f'{cladding_refractive_index} and intended wavelength {self.intended_wavelength}.'

        self.v_number = self.get_v_number()
        self.fundamental_fiber_mode = self.get_fundamental_fiber_mode()

    def get_v_number(self):
        v_number = 2 * np.pi / self.intended_wavelength * self.core_radius * \
                   np.sqrt(self.core_refractive_index ** 2 - self.cladding_refractive_index ** 2)
        if v_number.value >= 2.405:
            raise Exception(f'Fiber with V-number {v_number} is not single-mode')
        else:
            return v_number

    def get_fundamental_fiber_mode(self):
        u_variable, w_variable = fsolve(get_system_of_equations,
                                        (np.sqrt(self.v_number).value, np.sqrt(self.v_number).value),
                                        self.v_number.value)
        # normalization_constant = fsolve(get_orthogonality_equations, 1, (u_variable, w_variable, self.core_radius,
        #
        #
        linspace = np.linspace(-self.cladding_radius, self.cladding_radius, self.wavefront.array_dimension)
        X, Y = np.meshgrid(linspace, linspace)

        angles = np.arctan2(Y, X)
        radii = np.sqrt(X ** 2 + Y ** 2)
        wavefront_shape = self.wavefront.complex_amplitude.shape
        fundamental_fiber_mode = np.zeros(wavefront_shape)

        for x in range(wavefront_shape[0]):
            for y in range(wavefront_shape[1]):
                fundamental_fiber_mode[x][y] = get_mode_function(radii[x][y], angles[x][y], u_variable, w_variable,
                                                                 self.core_radius)
                # TODO: crop map to cladding radius

        return fundamental_fiber_mode

    def apply(self):
        pass
