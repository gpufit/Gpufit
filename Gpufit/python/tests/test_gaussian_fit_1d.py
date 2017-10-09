"""
    Equivalent to https://github.com/gpufit/Gpufit/blob/master/Gpufit/tests/Gauss_Fit_1D.cpp
"""

import unittest
import numpy as np
import pygpufit.gpufit as gf

def generate_gauss_1d(parameters, x):
    """
    Generates a 1D Gaussian curve.

    :param parameters: The parameters (a, x0, s, b)
    :param x: The x values
    :return: A 1D Gaussian curve.
    """

    a = parameters[0]
    x0 = parameters[1]
    s = parameters[2]
    b = parameters[3]

    y = a * np.exp(-np.square(x - x0) / (2 * s**2)) + b

    return y

class Test(unittest.TestCase):

    def test_gaussian_fit_1d(self):
        # constants
        n_fits = 1
        n_points = 5
        n_parameter = 4  # model will be GAUSS_1D

        # true parameters
        true_parameters = np.array((4, 2, 0.5, 1), dtype=np.float32)

        # generate data
        data = np.empty((n_fits, n_points), dtype=np.float32)
        x = np.arange(n_points, dtype=np.float32)
        data[0, :] = generate_gauss_1d(true_parameters, x)

        # tolerance
        tolerance = 0.001

        # max_n_iterations
        max_n_iterations = 10

        # model id
        model_id = gf.ModelID.GAUSS_1D

        # initial parameters
        initial_parameters = np.empty((n_fits, n_parameter), dtype=np.float32)
        initial_parameters[0, :] = (2, 1.5, 0.3, 0)

        # call to gpufit
        parameters, states, chi_squares, number_iterations, execution_time = gf.fit(data, None, model_id,
                                                                                    initial_parameters, tolerance, \
                                                                                    max_n_iterations, None, None, None)

        # print results
        for i in range(n_parameter):
            print(' p{} true {} fit {}'.format(i, true_parameters[i], parameters[0, i]))
        print('fit state : {}'.format(states))
        print('chi square: {}'.format(chi_squares))
        print('iterations: {}'.format(number_iterations))
        print('time: {} s'.format(execution_time))

        assert (chi_squares < 1e-6)
        assert (states == 0)
        assert (number_iterations <= max_n_iterations)
        for i in range(n_parameter):
            assert (abs(true_parameters[i] - parameters[0, i]) < 1e-6)

if __name__ == '__main__':

    if not gf.cuda_available():
        raise RuntimeError(gf.get_last_error())
    unittest.main()
