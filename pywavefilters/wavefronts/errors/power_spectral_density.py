import numpy as np
from numpy.fft import fft2


def prop_shift_center(image):
    """Shift a n by n image by (x,y)=(n/2,n/2), either shifting from the image
    origin to the center or vice-verse.
    Implemented as found in https://sourceforge.net/projects/proper-library/

    Parameters
    ----------
    image : numpy ndarray
        2D image to be shifted

    Returns
    -------
    shifted image : numpy ndarray
         Shifted image
    """
    image = np.asarray(image)

    if image.ndim != 2:
        raise ValueError("Only 2D images can be shifted. Stopping.")

    s = image.shape

    return np.roll(np.roll(image, int(s[0] / 2), 0), int(s[1] / 2), 1)


def get_power_spectral_density_error(wavelength: float,
                                     beam_diameter: float,
                                     low_spatial_frequency_power: float,
                                     correlation_length: float,
                                     power_law_falloff: float,
                                     grid_size: int,
                                     rotation_angle: float = 0,
                                     inclination_angle: float = 0) -> np.ndarray:
    """
    Return an array corresponding to the error given a certain power spectral density distribution.
    Implemented as found in https://sourceforge.net/projects/proper-library/

            Returns:
                    Array containing an error
    """
    # sampling = (BaseWavefront.get_extent_pupil_plane_meters(beam_diameter) / grid_size).value
    sampling = 4e-4  # * u.meter

    spatial_frequency_diameter = 1. / (grid_size * sampling)
    spatial_frequency_x_map = np.tile(np.arange(grid_size, dtype=np.float64), (grid_size, 1)) - int(grid_size / 2)
    spatial_frequency_y_map = spatial_frequency_x_map.T

    spatial_frequency_x_map = spatial_frequency_x_map * np.cos(-rotation_angle) - spatial_frequency_y_map * np.sin(
        -rotation_angle)
    spatial_frequency_y_map = spatial_frequency_x_map * np.sin(-rotation_angle) + spatial_frequency_y_map * np.cos(
        -rotation_angle)
    spatial_frequency_y_map = spatial_frequency_y_map * np.cos(-inclination_angle)

    spatial_frequency_radial_map = np.sqrt(
        spatial_frequency_x_map ** 2 + spatial_frequency_y_map ** 2) * spatial_frequency_diameter

    # Calculate power spectral density map
    power_spectral_density_map = low_spatial_frequency_power / (
            1. + (spatial_frequency_radial_map / correlation_length) ** 2) ** ((power_law_falloff + 1) / 2.)

    # Remove piston, i.e. lowest spatial frequency
    power_spectral_density_map[grid_size // 2, grid_size // 2] = 0

    # Generate random phase between -pi and pi
    phase = 2 * np.pi * np.random.uniform(size=(grid_size, grid_size)) - np.pi

    # Calculate Fourier transform
    error_map = fft2(prop_shift_center(
        np.sqrt(power_spectral_density_map) / spatial_frequency_diameter * np.exp(1j * phase))) / np.size(
        power_spectral_density_map)
    error_map = error_map.real / (grid_size ** 2 * sampling ** 2)

    # Fix RMS to match expected value from PSD
    rms_power_spectral_density = np.sqrt(np.sum(power_spectral_density_map)) * spatial_frequency_diameter
    rms_error_map = np.std(error_map)
    error_map *= (rms_power_spectral_density / rms_error_map)
    error_map = prop_shift_center(error_map)

    return error_map


def get_true_power_spectral_density_distribution(power_spectral_density_error: np.ndarray):
    pass
