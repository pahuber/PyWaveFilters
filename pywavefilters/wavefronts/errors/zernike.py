from math import factorial

import numpy as np
from astropy import units as u

from pywavefilters.util.math import get_kronecker_delta, get_x_y_grid
from pywavefilters.wavefronts.wavefront import BaseWavefront


def get_noll_index(index_n: int, index_m: int) -> int:
    """
    Return a single index j for each pair n, m of Zernike indices according to Noll's convention.

            Parameters:
                    index_n: First index of Zernike polynomial
                    index_m: Second index of Zernike polynomial

            Returns:
                    Noll index of Zernike polynomial
    """
    index_noll = (index_n * (index_n + 1)) / 2 + abs(index_m)

    if index_m > 0 and (index_n % 4 == 0 or index_n % 4 == 1):
        index_noll += 0
    elif index_m < 0 and (index_n % 4 == 2 or index_n % 4 == 3):
        index_noll += 0
    elif index_m >= 0 and (index_n % 4 == 2 or index_n % 4 == 3):
        index_noll += 1
    elif index_m <= 0 and (index_n % 4 == 0 or index_n % 4 == 1):
        index_noll += 1

    return index_noll


def get_n_m_from_noll(zernike_mode_index: int) -> (int, int):
    """
    Return a tuple of indices corresponding to the usual Zernike polynomial indices n and m.

            Parameters:
                    zernike_mode_index: Noll index of Zernike polynomial

            Returns:
                    Noll index of Zernike polynomial
    """
    nmax = int(zernike_mode_index / 2)
    nmin = 0
    mmin = -nmax
    mmax = nmax

    for index_n in range(nmin, nmax + 1, 1):
        for index_m in range(mmin, mmax + 1, 1):
            if get_noll_index(index_n, index_m) == zernike_mode_index and (index_n - index_m) % 2 == 0 \
                    and abs(index_m) <= index_n:
                return index_n, index_m


def get_radial_zernike_polynomial(index_n: int,
                                  index_m: int,
                                  radial_map: np.ndarray,
                                  maximum_radius: float) -> float:
    """
    Return the radial part of the Zernike polynomial.

            Parameters:
                    index_n: First index of Zernike polynomial
                    index_m: Second index of Zernike polynomial
                    radial_map: Map corresponding to the radial coordinate
                    maximum_radius: Maximum radius

            Returns:
                    Float corresponding to radial part of Zernike polynomial
    """
    index_m = abs(index_m)

    if (index_n - index_m) % 2 == 0:
        radial_part = 0
        for index_k in np.arange(0, (index_n - index_m) / 2 + 1, 1):
            index_k = int(index_k)
            radial_part += ((-1) ** index_k * factorial(int(index_n - index_k))) / (
                    factorial(index_k) * factorial(int((index_n + index_m) / 2 - index_k)) *
                    factorial(int((index_n - index_m) / 2 - index_k))) * \
                           (radial_map / maximum_radius) ** (index_n - 2 * index_k)
        radial_part[radial_map / maximum_radius == 1] = 1
        radial_part[radial_map / maximum_radius > 1] = 0

        return radial_part

    elif (index_n - index_m) % 2 != 0:
        return 0


def get_zernike_polynomial(zernike_mode_index: int,
                           radial_map: np.ndarray,
                           angular_map: np.ndarray,
                           maximum_radius: float) -> float:
    """
    Return a Zernike polynomial.

            Parameters:
                    zernike_mode_index: Noll index of Zernike polynomial
                    radial_map: Map corresponding to the radial coordinate
                    angular_map: Map corresponding to the angular coordinate
                    maximum_radius: Maximum radius

            Returns:
                    A Zernike polynomial
    """
    index_n, index_m = get_n_m_from_noll(zernike_mode_index)

    sign = np.sign(index_m)
    if sign == 0:
        sign = 1

    norm = sign * np.sqrt((2 * (index_n + 1)) / (1 + get_kronecker_delta(index_m, 0)))

    if index_m >= 0:
        return norm * get_radial_zernike_polynomial(index_n, index_m, radial_map, maximum_radius) * \
               np.cos(index_m * angular_map)
    else:
        return norm * get_radial_zernike_polynomial(index_n, index_m, radial_map, maximum_radius) * \
               np.sin(abs(index_m) * angular_map)


def get_zernike_error(wavelength: float, beam_diameter: float, zernike_modes: list, grid_size: int) -> np.ndarray:
    """
    Return an array composed of a sum of several Zernike polynomial terms Z_j.

            Returns:
                    Array containing sum of Zernike polynomials
    """

    # TODO: add RMS implementation
    if zernike_modes is None:
        return 0 * u.meter

    extent = BaseWavefront.get_extent_pupil_plane_meters(beam_diameter) / 2
    x_map, y_map = get_x_y_grid(grid_size, extent)

    radial_map = np.sqrt(x_map ** 2 + y_map ** 2)
    angular_map = np.arctan2(y_map, x_map)

    wavefront_error = 0
    for element in zernike_modes:
        zernike_mode_index = element[0]
        mode_coefficient = element[1]
        wavefront_error += mode_coefficient * get_zernike_polynomial(zernike_mode_index, radial_map,
                                                                     angular_map,
                                                                     beam_diameter / 2)

    return 2 * np.pi * wavefront_error / wavelength
