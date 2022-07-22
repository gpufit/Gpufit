"""
Example of the Python binding of the Gpufit library which implements
Levenberg Marquardt curve fitting in CUDA
https://github.com/gpufit/Gpufit
http://gpufit.readthedocs.io/en/latest/bindings.html#python

Multiple fits of a 2D Gaussian peak function with Poisson distributed noise
This example additionally requires numpy.
"""

import numpy as np
import pygpufit.gpufit as gf


def generate_gauss_2d(p, xi, yi):
    """
    Generates a 2D Gaussian peak.
    http://gpufit.readthedocs.io/en/latest/api.html#gauss-2d

    :param p: Parameters (amplitude, x,y center position, width, offset)
    :param xi: x positions
    :param yi: y positions
    :return: The Gaussian 2D peak.
    """

    arg = -(np.square(xi - p[1]) + np.square(yi - p[2])) / (2 * p[3] * p[3])
    y = p[0] * np.exp(arg) + p[4]

    return y


if __name__ == '__main__':

    # cuda available checks
    print('CUDA available: {}'.format(gf.cuda_available()))
    if not gf.cuda_available():
        raise RuntimeError(gf.get_last_error())
    print('CUDA versions runtime: {}, driver: {}'.format(*gf.get_cuda_version()))

    # number of fits and fit points
    number_fits = 10000
    size_x = 12
    number_points = size_x * size_x
    number_parameters = 5

    # set input arguments

    # true parameters
    true_parameters = np.array((10, 5.5, 5.5, 3, 10), dtype=np.float32)

    # initialize random number generator
    np.random.seed(0)

    # initial parameters (relative randomized, positions relative to width)
    initial_parameters = np.tile(true_parameters, (number_fits, 1))
    initial_parameters[:, (1, 2)] += true_parameters[3] * (-0.2 + 0.4 * np.random.rand(number_fits, 2))
    initial_parameters[:, (0, 3, 4)] *= 0.8 + 0.4 * np.random.rand(number_fits, 3)

    # generate x and y values
    g = np.arange(size_x)
    yi, xi = np.meshgrid(g, g, indexing='ij')
    xi = xi.astype(np.float32)
    yi = yi.astype(np.float32)

    # generate data
    data = generate_gauss_2d(true_parameters, xi, yi)
    data = np.reshape(data, (1, number_points))
    data = np.tile(data, (number_fits, 1))

    # add Poisson noise
    data = np.random.poisson(data)
    data = data.astype(np.float32)

    # tolerance
    tolerance = 0.0001

    # maximum number of iterations
    max_number_iterations = 20

    # estimator ID
    estimator_id = gf.EstimatorID.MLE

    # model ID
    model_id = gf.ModelID.GAUSS_2D

    # run Gpufit
    parameters, states, chi_squares, number_iterations, execution_time = gf.fit(data, None, model_id,
                                                                                initial_parameters,
                                                                                tolerance, max_number_iterations, None,
                                                                                estimator_id, None)

    # print fit results
    converged = states == 0
    print('*Gpufit*')

    # print summary
    print('\nmodel ID:        {}'.format(model_id))
    print('number of fits:  {}'.format(number_fits))
    print('fit size:        {} x {}'.format(size_x, size_x))
    print('mean chi_square: {:.2f}'.format(np.mean(chi_squares[converged])))
    print('iterations:      {:.2f}'.format(np.mean(number_iterations[converged])))
    print('time:            {:.2f} s'.format(execution_time))

    # get fit states
    number_converged = np.sum(converged)
    print('\nratio converged         {:6.2f} %'.format(number_converged / number_fits * 100))
    print('ratio max it. exceeded  {:6.2f} %'.format(np.sum(states == 1) / number_fits * 100))
    print('ratio singular hessian  {:6.2f} %'.format(np.sum(states == 2) / number_fits * 100))
    print('ratio neg curvature MLE {:6.2f} %'.format(np.sum(states == 3) / number_fits * 100))

    # mean, std of fitted parameters
    converged_parameters = parameters[converged, :]
    converged_parameters_mean = np.mean(converged_parameters, axis=0)
    converged_parameters_std = np.std(converged_parameters, axis=0)
    print('\nparameters of 2D Gaussian peak')
    for i in range(number_parameters):
        print('p{} true {:6.2f} mean {:6.2f} std {:6.2f}'.format(i, true_parameters[i], converged_parameters_mean[i],
                                                                 converged_parameters_std[i]))
