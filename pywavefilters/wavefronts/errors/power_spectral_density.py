from typing import Union

import cv2
import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt
from numpy.fft import fft2, fft

from pywavefilters.util.math import get_aperture_function, get_x_y_grid
from pywavefilters.wavefronts.wavefront import BaseWavefront


def prop_shift_center(image: np.ndarray) -> np.ndarray:
    """
    Shift a n by n image by (x,y)=(n/2,n/2), either shifting from the image
    origin to the center or vice-verse.
    Implemented as found in https://sourceforge.net/projects/proper-library/

            Parameters:
                    image: Numpy array

            Returns:
                    Shifted numpy array
    """
    image = np.asarray(image)

    if image.ndim != 2:
        raise ValueError("Only 2D images can be shifted. Stopping.")

    s = image.shape

    return np.roll(np.roll(image, int(s[0] / 2), 0), int(s[1] / 2), 1)


def get_radial_spatial_frequency_map(spatial_frequency_diameter, grid_size, rotation_angle=0,
                                     inclination_angle=0) -> np.ndarray:
    """
    Return the radial frequency map.

            Parameters:
                    spatial_frequency_diameter: Diameter in units of spatial frequency
                    grid_size: Grid size of the map
                    rotation_angle: Rotation angle of the surface
                    inclination_angle: Inclination angle of the surface

            Returns:
                    Array containing the radial frequency map
    """
    spatial_frequency_x_map = np.tile(np.arange(grid_size, dtype=np.float64), (grid_size, 1)) - int(grid_size / 2)
    spatial_frequency_y_map = spatial_frequency_x_map.T

    spatial_frequency_x_map = spatial_frequency_x_map * np.cos(-rotation_angle) - spatial_frequency_y_map * np.sin(
        -rotation_angle)
    spatial_frequency_y_map = spatial_frequency_x_map * np.sin(-rotation_angle) + spatial_frequency_y_map * np.cos(
        -rotation_angle)
    spatial_frequency_y_map = spatial_frequency_y_map * np.cos(-inclination_angle)

    spatial_frequency_radial_map = np.sqrt(
        spatial_frequency_x_map ** 2 + spatial_frequency_y_map ** 2) * spatial_frequency_diameter

    return spatial_frequency_radial_map


def get_k_correlation_model(low_spatial_frequency_power: float, correlation_length: float, power_law_falloff: float,
                            spatial_frequency_radial_map: np.ndarray) -> np.ndarray:
    """
    Return a curve corresponding to the k-correlation model.

            Parameters:
                    low_spatial_frequency_power: Power of the low spatial frequency in units of meters^4
                    correlation_length: Correlation length in units of meters^-1
                    power_law_falloff: Power law falloff coefficient
                    spatial_frequency_radial_map: Map corresponding to the radial coordinates in frequency space

            Returns:
                    Array containing the curve
    """
    return low_spatial_frequency_power / (
            1 + (spatial_frequency_radial_map / correlation_length) ** 2) ** ((power_law_falloff + 1) / 2.)


def plot_mean_one_dimensional_psd(power_spectral_density_phase_error: np.ndarray):
    """
    Plot the mean one-dimensional power spectral density (PSD) distribution of an error corresponding to a phase error.

            Parameters:
                    power_spectral_density_phase_error: Array to plot the PSD for
    """
    angle_real = power_spectral_density_phase_error.real
    angle_imag = power_spectral_density_phase_error.imag

    polar_real = cv2.warpPolar(angle_real, (256, 1024), (angle_real.shape[0] / 2, angle_real.shape[1] / 2),
                               angle_real.shape[1] * 0.9 * 0.5,
                               cv2.WARP_POLAR_LINEAR)

    polar_imag = cv2.warpPolar(angle_imag, (256, 1024), (angle_imag.shape[0] / 2, angle_imag.shape[1] / 2),
                               angle_imag.shape[1] * 0.9 * 0.5,
                               cv2.WARP_POLAR_LINEAR)

    polar = polar_real + 1j * polar_imag

    psd = 0
    for row in polar:
        psd += abs((fft((row)))) ** 2

    plt.plot(psd)
    plt.title('Mean 1-Dimensional PSD')
    plt.xscale('log')
    plt.ylabel('Amplitude (Length$^3$)')
    plt.xlabel('Cycles per Diameter ($m^{-1}$)')
    plt.grid()
    plt.tight_layout()
    plt.show()


def get_power_spectral_density_error(wavelength: float,
                                     beam_diameter: float,
                                     rms: float,
                                     grid_size: int,
                                     seed: Union[int, None] = None,
                                     correlation_length: float = 212.26 / u.meter,
                                     power_law_falloff: float = 7.8,
                                     plot_psd: bool = False) -> np.ndarray:
    """
    Return an array corresponding to the error given a certain power spectral density distribution.
    Implemented as found in https://sourceforge.net/projects/proper-library/

            Parameters:
                    wavelength: Wavelength of the beam for which the phase error is calculated
                    beam_diameter: Beam diameter of the beam
                    rms: Root mean square (RMS) that the phase error should have
                    grid_size: Grid size of the error map
                    seed: Seed to generate the random numbers
                    correlation_length: Correlation length in units of meters^-1
                    power_law_falloff: Power law falloff coefficient
                    plot_psd: Boolean to indicate whether the mean 1-D PSD should be plotted

            Returns:
                    Array containing the error map
    """
    sampling = (BaseWavefront.get_extent_pupil_plane_meters(beam_diameter) / grid_size)
    low_spatial_frequency_power = 1 * u.meter ** 4
    spatial_frequency_diameter = 1. / (grid_size * sampling)

    spatial_frequency_radial_map = get_radial_spatial_frequency_map(spatial_frequency_diameter, grid_size)
    power_spectral_density_map = get_k_correlation_model(low_spatial_frequency_power, correlation_length,
                                                         power_law_falloff, spatial_frequency_radial_map)

    # Remove piston, i.e. lowest spatial frequency
    power_spectral_density_map[grid_size // 2, grid_size // 2] = 0

    # Generate random phase between -pi and pi
    if seed is not None:
        np.random.seed(seed)
    phase = 2 * np.pi * np.random.uniform(size=(grid_size, grid_size)) - np.pi

    # Calculate Fourier transform
    error_map = (fft2((prop_shift_center(
        np.sqrt(power_spectral_density_map) / spatial_frequency_diameter * np.exp(1j * phase))))) / np.size(
        power_spectral_density_map)

    error_map = error_map.real / (grid_size ** 2 * sampling ** 2)

    # Fix RMS to match expected value from PSD
    x_map, y_map = get_x_y_grid(grid_size, BaseWavefront.get_extent_pupil_plane_meters(beam_diameter) / 2)
    error_map *= get_aperture_function(x_map, y_map, beam_diameter / 2, is_real=True)
    error_map = error_map / np.sqrt(np.mean(np.square(error_map))) * rms

    # Convert to radians
    error_map = 2 * np.pi * error_map / wavelength

    # Plot 1 dimensional PSD
    if plot_psd:
        plot_mean_one_dimensional_psd(error_map)

    # TODO: check implementation with RMS and units and shift center
    return error_map
