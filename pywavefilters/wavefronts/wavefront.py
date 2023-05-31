from typing import Union

import astropy
import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from pywavefilters.optical_elements.optical_element import BaseOpticalElement
from pywavefilters.util.math import get_normalized_intensity, get_x_y_grid, get_aperture_function


class BaseWavefront:
    """
    Base class to represent wavefronts.
    """

    _chirp_z_maximum_frequency = 100

    def __init__(self):
        """
        Constructor for base wavefront object.
        """
        self.wavelength = 0 * u.meter
        self.beam_diameter = 0 * u.meter
        self.complex_amplitude = None
        self.is_in_pupil_plane = None
        self.extent_pupil_plane_meters = None
        self.extent_focal_plane_dimensionless = None
        self.extent_focal_plane_meters = None  # Is reset to None after leaving the focal plane
        self.grid_size = 1
        self.has_fiber_been_applied = None

        self._x_map = None
        self._y_map = None

    def __add__(self, other_wavefront):
        """
        Method to add two base wavefront together.

                Parameters:
                        other_wavefront: Base wavefront object to be added
                Returns:
                        Combined wavefront object
        """
        if self.is_in_pupil_plane != other_wavefront.is_in_pupil_plane:
            raise Exception('Wavefronts must both be in pupil or in focal plane')
        elif self.beam_diameter != other_wavefront.beam_diameter:
            raise Exception('Wavefronts must have same beam diameter')
        elif self.wavelength != other_wavefront.wavelength:
            raise Exception('Wavefronts must have same wavelengths')
        else:
            return CombinedWavefront(self.wavelength,
                                     self.beam_diameter,
                                     self.complex_amplitude + other_wavefront.complex_amplitude,
                                     self.is_in_pupil_plane,
                                     self.extent_pupil_plane_meters,
                                     self.extent_focal_plane_dimensionless,
                                     self.extent_focal_plane_meters,
                                     self.grid_size,
                                     self.has_fiber_been_applied,
                                     self._x_map,
                                     self._y_map)

    def __sub__(self, other_wavefront):
        """
        Method to subtract one base wavefront from another.

                Parameters:
                        other_wavefront: Base wavefront object to be subtracted
                Returns:
                        Combined, i.e. subtracted, wavefront object
        """
        if self.is_in_pupil_plane != other_wavefront.is_in_pupil_plane:
            raise Exception('Wavefronts must both be in pupil or in focal plane')
        elif self.beam_diameter != other_wavefront.beam_diameter:
            raise Exception('Wavefronts must have same beam diameter')
        elif self.wavelength != other_wavefront.wavelength:
            raise Exception('Wavefronts must have same wavelengths')
        else:
            return CombinedWavefront(self.wavelength,
                                     self.beam_diameter,
                                     self.complex_amplitude - other_wavefront.complex_amplitude,
                                     self.is_in_pupil_plane,
                                     self.extent_pupil_plane_meters,
                                     self.extent_focal_plane_dimensionless,
                                     self.extent_focal_plane_meters,
                                     self.grid_size,
                                     self.has_fiber_been_applied,
                                     self._x_map,
                                     self._y_map)

    @property
    def aperture_radius(self) -> float:
        """
        Return the aperture radius.

                Returns:
                        Aperture radius
        """
        return self.beam_diameter / 2

    @property
    def amplitude(self) -> np.ndarray:
        """
        Return the amplitude of the complex amplitude.

                Returns:
                        Array containing amplitude
        """
        return abs(self.complex_amplitude)

    @property
    def phase(self) -> np.ndarray:
        """
        Return the phase of the complex amplitude.

                Returns:
                        Array containing phase
        """
        return np.angle(self.complex_amplitude)

    @property
    def intensity(self) -> np.ndarray:
        """
        Return the intensity of the complex amplitude.

                Returns:
                        Array containing intensity
        """
        return abs(self.complex_amplitude) ** 2

    @staticmethod
    def get_extent_pupil_plane_meters(beam_diameter: float):
        """
        Return a value corresponding to the full extent of the array in the pupil plane in units of meters.

                Returns:
                        Value corresponding to the full extent in meters
        """
        return beam_diameter

    @staticmethod
    def get_extent_focal_plane_dimensionless():
        """
        Return a value corresponding to the full extent of the array in the focal plane in units of wavelength over
        aperture diameter.

                Returns:
                        Value corresponding to the full extent in dimensionless units
        """
        return BaseWavefront._chirp_z_maximum_frequency / np.pi

    @staticmethod
    def get_extent_focal_plane_meters(wavelength: float, beam_diameter: float, lens: BaseOpticalElement) -> float:
        """
        Return a value corresponding to the full extent of the array in meters.

                Parameters:
                        wavelength: Wavelength of the wavefront
                        beam_diameter: Beam diameter of the wavefront
                        lens: Lens object

                Returns:
                        Value corresponding to the full extent in meters
        """
        return BaseWavefront.get_extent_focal_plane_dimensionless() / beam_diameter * lens.focal_length * wavelength

    def add_phase(self, phase: Union[float, np.ndarray], grid_size: Union[int, None] = None):
        """
        Add a constant offset (piston) or two-dimensional array to the phase of the complex amplitude of the wavefront.

                Parameters:
                        phase: Float corresponding to a piston offset or array containing the phase profile to add
                        grid_size: Grid size of the array; only needed if a constant piston offset is to be added
        """
        if isinstance(phase, float):
            if grid_size is not None:
                phase_array = np.ones((grid_size, grid_size)) * phase
                self.complex_amplitude *= np.exp(1j * phase_array)
            else:
                raise Exception('A grid size must be provided to add a constant phase offset')
        else:
            self.complex_amplitude *= np.exp(1j * phase)

    def apply(self, optical_element: BaseOpticalElement):
        """
        Apply an optical element.
        """
        optical_element.apply(self)

    def plot_phase(self, title=None):
        """
        Plot the wavefront phase.

                Parameters:
                        title: Optional title of the plot
        """
        plt.imshow(self.phase.value, cmap='bwr')
        plt.colorbar()

        if title is None:
            plt.title('Wavefront Phase')
        else:
            plt.title(title)

        plt.show()

    def plot_intensity_pupil_plane(self, title=None):
        """
        Plot the wavefront intensity in the pupil plane with correct axis scaling.

                Parameters:
                        title: Optional title of the plot
        """
        if not self.is_in_pupil_plane:
            raise Exception('Wavefront must be in pupil plane')

        half_extent = self.extent_pupil_plane_meters.value / 2
        plt.imshow(self.intensity.value, extent=[-half_extent, half_extent, -half_extent, half_extent],
                   norm=LogNorm(),
                   cmap='gist_heat')

        if title is None:
            plt.title('Intensity Pupil Plane')
        else:
            plt.title(title)

        colorbar = plt.colorbar()
        colorbar.set_label('Intensity (W/m$^2$)')

        plt.xlabel('Width (m)')
        plt.ylabel('Height (m)')

        plt.show()

    def plot_intensity_focal_plane(self, title=None, dimensionless=False):
        """
        Plot the wavefront intensity in the focal plane with correct axis scaling.

                Parameters:
                        title: Optional title of the plot
                        dimensionless: Boolean to specify whether to plot in units of meters or in dimensionless units
        """
        if self.is_in_pupil_plane:
            raise Exception('Wavefront must be in focal plane')

        if dimensionless:
            half_extent = self.extent_focal_plane_dimensionless / 2
        else:
            half_extent = self.extent_focal_plane_meters.value / 2

        plt.imshow(self.intensity.value, extent=[-half_extent, half_extent, -half_extent, half_extent],
                   norm=LogNorm(),
                   cmap='gist_heat')

        if title is None:
            plt.title('Intensity Focal Plane')
        else:
            plt.title(title)

        if dimensionless:
            plt.xlabel('Width ($\lambda/D$)')
            plt.ylabel('Height ($\lambda/D$)')
        else:
            plt.xlabel('Width (m)')
            plt.ylabel('Height (m)')

        colorbar = plt.colorbar()
        colorbar.set_label('Intensity (W/m$^2$)')  # TODO: check units

        plt.show()


class Wavefront(BaseWavefront):
    """
    Class representing a wavefront.
    """

    def __init__(self,
                 wavelength: float,
                 beam_diameter: float,
                 grid_size: int):
        """
        Constructor for wavefront object.

                Parameters:
                        wavelength: Wavelength of the wavefront in meters
                        zernike_modes: List containing the zernike mode indices and their coefficients in meters
                        beam_diameter: Beam diameter in the aperture plane in meters
                        grid_size: Side length of the output array in pixels (1 pixel =^ 300 um)
        """
        BaseWavefront.__init__(self)
        self.wavelength = wavelength
        self.beam_diameter = beam_diameter
        self.grid_size = grid_size

        self.extent_pupil_plane_meters = self.get_extent_pupil_plane_meters(self.beam_diameter)
        self.extent_focal_plane_dimensionless = self.get_extent_focal_plane_dimensionless()
        self.is_in_pupil_plane = True

        self._x_map, self._y_map = get_x_y_grid(self.grid_size, self.extent_pupil_plane_meters / 2)
        self.complex_amplitude = get_normalized_intensity(
            self.get_gaussian_beam_profile() * get_aperture_function(self._x_map, self._y_map,
                                                                     self.aperture_radius) * u.watt ** 0.5 / u.meter)

    @property
    def wavelength(self) -> float:
        """
        Return the wavelength.

                Returns:
                        Float corresponding to the wavelength
        """
        return self._wavelength

    @wavelength.setter
    def wavelength(self, value):
        """
        Setter method for the wavelength.
        """
        if not (type(value) == astropy.units.quantity.Quantity and value.unit == u.meter):
            raise ValueError(f'Units of wavelength must be specified in meters.')
        self._wavelength = value

    @property
    def beam_diameter(self) -> float:
        """
        Return the beam diameter.

                Returns:
                        Float corresponding to beam diameter
        """
        return self._beam_diameter

    @beam_diameter.setter
    def beam_diameter(self, value):
        """
        Setter method for the beam diameter.
        """
        if not (type(value) == astropy.units.quantity.Quantity and value.unit == u.meter):
            raise ValueError(f'Units of beam diameter must be specified in meters.')
        self._beam_diameter = value

    @property
    def grid_size(self) -> int:
        """
        Return the grid size.

                Returns:
                        Integer corresponding to the grid size
        """
        return self._grid_size

    @grid_size.setter
    def grid_size(self, value):
        """
        Setter method for the grid size.
        """
        if not (type(value) == int and value > 0 and value % 2 == 1):
            raise ValueError(f'Grid size must be an odd, positive integer.')
        self._grid_size = value

    def get_gaussian_beam_profile(self) -> np.ndarray:
        """
        Return an array containing a Gaussian beam profile.

                Returns:
                        Array containing the Gaussian beam profile.
        """
        return np.exp(-(self._x_map ** 2 + self._y_map ** 2) / (self.aperture_radius) ** 2)


class CombinedWavefront(BaseWavefront):
    """
    Base class representing combined wavefronts.
    """

    def __init__(self,
                 wavelength: float,
                 beam_diameter: float,
                 complex_amplitude: np.ndarray,
                 is_in_pupil_plane: bool,
                 extent_pupil_plane_meters: float,
                 extent_focal_plane_dimensionless: float,
                 extent_focal_plane_meters: float,
                 grid_size: int,
                 has_fiber_been_applied: bool,
                 _x_map: np.ndarray,
                 _y_map: np.ndarray):
        """
        Constructor for combined wavefront object.

                Parameters:
                        wavelength: Wavelength of each of the wavefronts
                        beam_diameter: Beam diameter of each of the wavefronts
                        complex_amplitude: Complex amplitude of the combined wavefront
                        is_in_pupil_plane: Boolean specifying whether we are in the spatial domain or not
                        extent_pupil_plane_meters: Full array width in pupil plane in meters
                        extent_focal_plane_dimensionless: Full array width in focal plane dimensionless
                        extent_focal_plane_meters: Full array width in focal plane in meters
                        grid_size: Grid size of array
                        has_fiber_been_applied: Boolean specifying whether a fiber has been applied
                        _x_map: X coordinate map of grid
                        _y_map: Y coordinate map of grid
        """
        self.wavelength = wavelength
        self.beam_diameter = beam_diameter
        self.complex_amplitude = complex_amplitude
        self.is_in_pupil_plane = is_in_pupil_plane
        self.extent_pupil_plane_meters = extent_pupil_plane_meters
        self.extent_focal_plane_dimensionless = extent_focal_plane_dimensionless
        self.extent_focal_plane_meters = extent_focal_plane_meters
        self.grid_size = grid_size
        self.has_fiber_been_applied = has_fiber_been_applied
        self._x_map = _x_map
        self._y_map = _y_map
