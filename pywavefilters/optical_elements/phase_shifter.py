from typing import Union

import astropy
import numpy as np
from astropy import units as u

from pywavefilters.optical_elements.optical_element import BaseOpticalElement
from pywavefilters.wavefronts.wavefront import BaseWavefront


class PhaseShifter(BaseOpticalElement):
    """
    Class representing a phase shifter, which can be used to add a certain phase to a wavefront's complex amplitude.
    """

    def __init__(self, phase_difference: Union[float, np.ndarray]):
        """
        Constructor for phase shifter object.

                Parameters:
                        phase_difference: Phase difference to be applied. Can be a single value for a constant phase
                                          shift or an array of phases for more complex pahse shifts
        """
        self.phase_difference = phase_difference
        self.description = f'Phase shifter.'

    @property
    def phase_difference(self) -> Union[float, np.ndarray]:
        """
        Return the phase difference.

                Returns:
                        Float or array corresponding to the phase difference
        """
        return self._phase_difference

    @phase_difference.setter
    def phase_difference(self, value):
        """
        Setter method for the phase_difference.
        """
        if not (type(value) == astropy.units.quantity.Quantity and value.unit == u.rad):
            raise ValueError(f'Units of phase shift must be specified in radians.')
        self._phase_difference = value

    def apply(self, wavefront: BaseWavefront):
        """
        Implementation of the apply method of the parent class. Used to apply the optical element to the wavefront.

                Parameters:
                        wavefront: Base wavefront object
        """
        wavefront.complex_amplitude *= np.exp(-1j * self.phase_difference.value)
