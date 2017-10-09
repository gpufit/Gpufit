"""
    Equivalent to https://github.com/gpufit/Gpufit/blob/master/Gpufit/tests/Linear_Fit_1D.cpp
"""

import unittest
import numpy as np
import pygpufit.gpufit as gf

class Test(unittest.TestCase):

    def test_gaussian_fit_1d(self):
        # constants
        n_fits = 1
        n_points = 2
        n_parameter = 2

        # true parameters
        true_parameters = np.array((0, 1), dtype=np.float32)

        # data values
        data = np.empty((n_fits, n_points), dtype=np.float32)
        data[0, :] = (0, 1)

        # max number iterations
        max_number_iterations = 10

        # initial parameters
        initial_parameters = np.empty((n_fits, n_parameter), dtype=np.float32)
        initial_parameters[0, :] = (0, 0)

        # model id
        model_id = gf.ModelID.LINEAR_1D

        # tolerance
        tolerance = 0.001

        # user info
        user_info = np.array((0, 1), dtype=np.float32)

        # call to gpufit
        parameters, states, chi_squares, number_iterations, execution_time = gf.fit(data, None, model_id,
                                                                                    initial_parameters, tolerance, \
                                                                                    None, None, None, user_info)

        # print results
        for i in range(n_parameter):
            print(' p{} true {} fit {}'.format(i, true_parameters[i], parameters[0, i]))
        print('fit state : {}'.format(states))
        print('chi square: {}'.format(chi_squares))
        print('iterations: {}'.format(number_iterations))
        print('time: {} s'.format(execution_time))

        assert (chi_squares < 1e-6)
        assert (states == 0)
        assert (number_iterations <= max_number_iterations)
        for i in range(n_parameter):
            assert (abs(true_parameters[i] - parameters[0, i]) < 1e-6)

if __name__ == '__main__':

    if not gf.cuda_available():
        raise RuntimeError(gf.get_last_error())
    unittest.main()
