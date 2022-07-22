"""
Example of the Python binding of the Gpufit library which implements
Levenberg Marquardt curve fitting in CUDA
https://github.com/gpufit/Gpufit

Multiple fits of a 2D Gaussian peak function with Poisson distributed noise
repeated for a different total number of fits each time and plotting the results
http://gpufit.readthedocs.io/en/latest/bindings.html#python

This example additionally requires numpy (http://www.numpy.org/) and matplotlib (http://matplotlib.org/).
"""

import numpy as np
import matplotlib.pyplot as plt
import pygpufit.gpufit as gf


def gaussians_2d(x, y, p):
    """
    Generates many 2D Gaussians peaks for a given set of parameters
    """

    n_fits = p.shape[0]

    y = np.zeros((n_fits, x.shape[0], x.shape[1]), dtype=np.float32)

    # loop over each fit
    for i in range(n_fits):
        pi = p[i, :]
        arg = -(np.square(xi - pi[1]) + np.square(yi - pi[2])) / (2 * pi[3] * pi[3])
        y[i, :, :] = pi[0] * np.exp(arg) + pi[4]

    return y


if __name__ == '__main__':

    # cuda available checks
    print('CUDA available: {}'.format(gf.cuda_available()))
    if not gf.cuda_available():
        raise RuntimeError(gf.get_last_error())
    print('CUDA versions runtime: {}, driver: {}'.format(*gf.get_cuda_version()))

    # number of fit points
    size_x = 5
    number_points = size_x * size_x

    # set input arguments

    # true parameters
    mean_true_parameters = np.array((100, 2, 2, 1, 10), dtype=np.float32)

    # average noise level
    average_noise_level = 10

    # initialize random number generator
    np.random.seed(0)

    # tolerance
    tolerance = 0.0001

    # maximum number of iterations
    max_number_iterations = 10

    # model ID
    model_id = gf.ModelID.GAUSS_2D

    # loop over different number of fits
    n_fits_all = np.around(np.logspace(2, 6, 20)).astype(np.int)

    # generate x and y values
    g = np.arange(size_x)
    yi, xi = np.meshgrid(g, g, indexing='ij')
    xi = xi.astype(np.float32)
    yi = yi.astype(np.float32)

    # loop
    speed = np.zeros(n_fits_all.size)
    for i in range(n_fits_all.size):
        n_fits = n_fits_all[i]

        # vary positions of 2D Gaussian peaks slightly
        test_parameters = np.tile(mean_true_parameters, (n_fits, 1))
        test_parameters[:, (1, 2)] += mean_true_parameters[3] * (-0.2 + 0.4 * np.random.rand(n_fits, 2))

        # generate data
        data = gaussians_2d(xi, yi, test_parameters)
        data = np.reshape(data, (n_fits, number_points))

        # add noise
        data += np.random.normal(scale=average_noise_level, size=data.shape)

        # initial parameters (randomized relative (to width for position))
        initial_parameters = np.tile(mean_true_parameters, (n_fits, 1))
        initial_parameters[:, (1, 2)] += mean_true_parameters[3] * (-0.2 + 0.4 * np.random.rand(n_fits, 2))
        initial_parameters[:, (0, 3, 4)] *= 0.8 + 0.4 * np.random.rand(n_fits, 3)

        # run Gpufit
        parameters, states, chi_squares, number_iterations, execution_time = gf.fit(data, None, model_id,
                                                                                    initial_parameters, tolerance,
                                                                                    max_number_iterations)

        # analyze result
        converged = states == 0
        speed[i] = n_fits / execution_time
        precision_x0 = np.std(parameters[converged, 1] - test_parameters[converged, 1], axis=0, dtype=np.float64)

        # display result
        '{} fits '.format(n_fits)
        print('{:7} fits     iterations: {:6.2f} | time: {:6.3f} s | speed: {:8.0f} fits/s' \
              .format(n_fits, np.mean(number_iterations[converged]), execution_time, speed[i]))

# plot
plt.semilogx(n_fits_all, speed, 'bo-')
plt.grid(True)
plt.xlabel('number of fits per function call')
plt.ylabel('fits per second')
plt.legend(['Gpufit'], loc='upper left')
ax = plt.gca()
ax.set_xlim(n_fits_all[0], n_fits_all[-1])

plt.show()
