from matplotlib import pyplot as plt

from pywavefilters.wavefronts.wavefront import BaseWavefront, Wavefront


def plot_intensity_pupil_plane(wavefront: BaseWavefront, title=None):
    """
    Plot the wavefront intensity in the pupil plane with correct axis scaling.

            Parameters:
                    wavefront: Wavefront to be plotted
                    title: Optional title of the plot
    """
    if not wavefront.is_in_pupil_plane:
        raise Exception('Wavefront must be in pupil plane')

    half_extent = wavefront.extent_pupil_plane_meters.value / 2
    plt.imshow(wavefront.intensity.value, extent=[-half_extent, half_extent, -half_extent, half_extent])

    if title is None:
        plt.title('Intensity Pupil Plane')
    else:
        plt.title(title)

    colorbar = plt.colorbar()
    colorbar.set_label('Intensity (W/m$^2$)')

    plt.xlabel('Width (m)')
    plt.ylabel('Height (m)')

    plt.show()


def plot_intensity_focal_plane(wavefront: BaseWavefront, title=None, dimensionless=False):
    """
    Plot the wavefront intensity in the focal plane with correct axis scaling.

            Parameters:
                    wavefront: Wavefront to be plotted
                    title: Optional title of the plot
                    dimensionless: Boolean to specify whether to plot in units of meters or in dimensionless units
    """
    if wavefront.is_in_pupil_plane:
        raise Exception('Wavefront must be in focal plane')

    if dimensionless:
        half_extent = wavefront.extent_focal_plane_dimensionless / 2
    else:
        half_extent = wavefront.extent_focal_plane_meters.value / 2

    plt.imshow(wavefront.intensity.value, extent=[-half_extent, half_extent, -half_extent, half_extent])

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


def plot_initial_wavefront_error(wavefront: Wavefront, title=None):
    """
    Plot the initial wavefront error.

            Parameters:
                    wavefront: Wavefront to be plotted
                    title: Optional title of the plot
    """
    plt.imshow(wavefront.initial_wavefront_error.value)
    plt.colorbar()

    if title is None:
        plt.title('Initial Wavefront Error')
    else:
        plt.title(title)

    plt.show()
