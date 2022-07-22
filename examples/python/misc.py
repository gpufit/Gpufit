"""
General helper functions for the examples.
"""

import numpy as np


def gaussian_peak_1d(x, p):
    """
    Generates a 1D Gaussian peak.
    See http://gpufit.readthedocs.io/en/latest/fit_model_functions.html#d-gaussian-function

    :param x: x grid position values
    :param p: parameters (amplitude, center position, width, offset)
    :return: Gaussian peak
    """
    p = p.flatten()
    return p[0] * np.exp(-(x - p[1])**2 / (2 * p[2]**2)) + p[3]


def gaussian_peak_2d(x, y, p):
    """
    Generates a 2D Gaussian peak.
    http://gpufit.readthedocs.io/en/latest/fit_model_functions.html#d-gaussian-function-cylindrical-symmetry

    x,y - x and y grid position values
    p - parameters (amplitude, x,y center position, width, offset)

    :param x:
    :param y:
    :param p:
    :return:
    """
    p = p.flatten()
    return p[0] * np.exp(-((x - p[1])**2 + (y - p[2])**2) / (2 * p[3]**2)) + p[4]
