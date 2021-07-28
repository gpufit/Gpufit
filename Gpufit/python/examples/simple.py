"""
Example of the Python binding of the Gpufit library which implements
Levenberg Marquardt curve fitting in CUDA
https://github.com/gpufit/Gpufit

Simple example demonstrating a minimal call of all needed parameters for the Python interface
http://gpufit.readthedocs.io/en/latest/bindings.html#python
"""

import numpy as np
import pygpufit.gpufit as gf

if __name__ == '__main__':

    # cuda available checks
    print('CUDA available: {}'.format(gf.cuda_available()))
    print('CUDA versions runtime: {}, driver: {}'.format(*gf.get_cuda_version()))

    # number of fits, number of points per fit
    number_fits = 10
    number_points = 10

    # model ID and number of parameter
    model_id = gf.ModelID.GAUSS_1D
    number_parameter = 5

    # initial parameters
    initial_parameters = np.zeros((number_fits, number_parameter), dtype=np.float32)

    # data
    data = np.zeros((number_fits, number_points), dtype=np.float32)

    # run Gpufit
    parameters, states, chi_squares, number_iterations, execution_time = gf.fit(data, None, model_id, initial_parameters)