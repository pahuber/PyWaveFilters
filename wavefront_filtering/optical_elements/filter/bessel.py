import numpy as np
from scipy.special import jv, kv


def get_system_of_equations(variables, v_number):
    u_variable, w_variable = variables
    bessel_order = 0
    return (v_number - np.sqrt(u_variable ** 2 + w_variable ** 2), (u_variable * jv(bessel_order + 1, u_variable)) /
            (jv(bessel_order, u_variable)) - (w_variable * kv(bessel_order + 1, w_variable)) / (
                kv(bessel_order, w_variable)))


def get_mode_function(radius: float,
                      angle: float,
                      u_variable: float,
                      w_variable: float,
                      core_radius: float,
                      even=True):
    bessel_order = 0
    radial_part = np.heaviside(core_radius - radius, 0) * jv(bessel_order, u_variable * radius / core_radius) / \
                  jv(bessel_order, u_variable) + np.heaviside(radius - core_radius, 0) * \
                  kv(bessel_order, w_variable * radius / core_radius) / kv(bessel_order, w_variable)
    if even:
        angular_part = np.cos(bessel_order * angle)
    else:
        angular_part = np.sin(bessel_order * angle)

    return radial_part * angular_part
