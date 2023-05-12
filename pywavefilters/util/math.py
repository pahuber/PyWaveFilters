import numpy as np
from scipy.signal import czt


def get_kronecker_delta(index_a: int, index_b: int) -> int:
    """
    Method to calculate the Kronecker delta.

            Parameters:
                    index_a: First index of Kronecker delta
                    index_b: Second index of Kronecker delta

            Returns:
                    0 or 1 as specified by the definition of the Kronecker delta
    """

    if index_a == index_b:
        return 1
    else:
        return 0


def get_2d_chirp_z_transform(complex_amplitude: np.ndarray, grid_size: int,
                             maximum_frequency: int) -> np.ndarray:
    """
    Method to calculate the 2-dimensional chirp z-transform using scipy.signal.czt. As found on
    https://stackoverflow.com/questions/30791905/zoom-in-on-np-fft2-result

            Parameters:
                    complex_amplitude: Complex amplitude of the Basewavefront object
                    grid_size: Grid size
                    maximum_frequency: Maximum frequency used for the transform

            Returns:
                    Transformed complex amplitude
    """

    # TODO: clean up this function

    width, height = complex_amplitude.shape
    aw = 2j * maximum_frequency / ((grid_size - 1) * width)

    f1 = czt(complex_amplitude, m=grid_size, w=np.exp(aw), a=np.exp(0.5 * (grid_size - 1) * aw),
             axis=0)
    data = f1 * np.exp(-0.5j * maximum_frequency * np.linspace(-1, 1, grid_size))[:, np.newaxis]
    ah = 2j * maximum_frequency / ((grid_size - 1) * height)
    f2 = czt(data, m=grid_size, w=np.exp(ah), a=np.exp(0.5 * (grid_size - 1) * ah), axis=1)

    transformed_complex_amplitude = f2 * np.exp(-0.5j * maximum_frequency * np.linspace(-1, 1, grid_size))[
                                         np.newaxis, :]

    return transformed_complex_amplitude
