"""
Spline fit 1D example

Requires pyGpufit, pyGpuSpline (https://github.com/gpufit/Gpuspline), Numpy and Matplotlib.
"""

import numpy as np
from matplotlib import pyplot as plt
import pygpuspline.gpuspline as gs
import pygpufit.gpufit as gf
import misc


def generate_test_psf(x, p):
    """
    Test PSF that consists of the sum of Gaussian with different centers and different widths

    p[0] - ampltiude first Gaussian
    p[1] - center (both Gaussians shifted a bit towards left and right
    p[2] - Standard deviation (second Gaussian is a bit wider)
    p[3] - constant background

    :param x: x values to calculate PSF on
    :param p: parameters that define the PSF
    :return:
    """

    distance = x[-1] / 12
    arg1 = ((x - p[1] - distance)**2) / (2*p[2]**2)
    arg2 = ((x - p[1] + distance)**2) / (2*p[2]**2)
    ex = np.exp(-arg1) + 0.5 * np.exp(-arg2)

    y = ex / np.amax(ex)  # normalized to [0, 1]
    y = p[0] * y + p[3]  # scale with amplitude and background

    return y


if __name__ == '__main__':
    # initialize random number generator
    rng = np.random.default_rng(0)

    # data size
    size_x = 25

    # fit parameter
    tolerance = 1e-30
    max_n_iterations = 100
    estimator_id = gf.EstimatorID.LSE
    model_id = gf.ModelID.SPLINE_1D

    # derived values
    x = np.arange(size_x, dtype=np.float32)
    SF = 2  # scaling factor
    x_spline = np.arange(SF * size_x, dtype=np.float32)
    x2 = x_spline / SF

    # generate PSF (two Gaussians)
    psf_parameters = np.array([100, (size_x-1)/2, 1.5, 10], dtype=np.float32)
    psf = generate_test_psf(x2, psf_parameters)
    psf_normalized = (psf - psf_parameters[3]) / psf_parameters[0]

    # calculate spline coefficients of the PSF template
    coefficients = gs.spline_coefficients(psf_normalized)
    n_intervals = psf_normalized.shape[0] - 1

    # add noise to PSF data (no shift)
    snr = 10
    amplitude = psf_parameters[0]
    noise_std_dev = amplitude / (snr * np.log(10.0))
    noise = noise_std_dev * rng.standard_normal(psf.shape, dtype=np.float32)
    noisy_psf = psf + noise
    noisy_psf = np.reshape(noisy_psf, (1, noisy_psf.shape[0]))

    # set user info
    user_info = np.hstack((n_intervals, coefficients.flatten()))

    # true fit parameters (amplitude, center shift, offset)
    true_fit_parameters = np.array([psf_parameters[0], 0, psf_parameters[3]], dtype=np.float32)
    true_fit_parameters = true_fit_parameters.reshape((1, true_fit_parameters.shape[0]))

    # set initial fit parameters
    pos_shift = -1.2 * SF
    amp_shift = 30
    off_shift = 20

    fit_initial_parameters = true_fit_parameters + np.array([amp_shift, pos_shift, off_shift], dtype=np.float32)  # small deviation

    gauss_fit_initial_parameters = psf_parameters + np.array([amp_shift, pos_shift, 0, off_shift], dtype=np.float32)
    gauss_fit_initial_parameters = np.reshape(gauss_fit_initial_parameters, (1, gauss_fit_initial_parameters.shape[0]))

    # call to gpufit with spline fit
    parameters_spline, states_spline, chi_squares_spline, n_iterations_spline, time_spline = gf.fit(noisy_psf, None, model_id, fit_initial_parameters, tolerance, max_n_iterations, None, estimator_id, user_info)
    if not np.all(states_spline == 0):
        pass #raise RuntimeError('Spline fit failed')

    # call to gpufit with gauss1d fit
    parameters_gauss, states_gauss, chi_squares_gauss, n_iterations_gauss, time_gauss = gf.fit(noisy_psf, None, gf.ModelID.GAUSS_1D, gauss_fit_initial_parameters, tolerance, max_n_iterations, None, estimator_id, x2)
    if not np.all(states_gauss == 0):
        raise RuntimeError('Gaussian fit failed')

    # get data to plot
    a = true_fit_parameters[0]
    x = x_spline - true_fit_parameters[1]
    b = true_fit_parameters[2]
    spline_model  = a * gs.spline_values(coefficients, x) + b
    a = fit_initial_parameters[0]
    x = x_spline - fit_initial_parameters[1]
    b = fit_initial_parameters[2]
    initial_spline_fit = a * gs.spline_values(coefficients, x) + b
    a = parameters_spline[0]
    x = x_spline - parameters_spline[1]
    b = parameters_spline[2]
    final_fit_gpufit = a * gs.spline_values(coefficients, x) + b
    gauss_final_fit = misc.gaussian_peak_1d(x2, parameters_gauss[:, 0])

    # make a figure of function values
    fig, ax = plt.subplots()
    fig.tight_layout()
    ax.plot(x2, gauss_final_fit, '--.y')

# plot(x2, gauss_final_fit, '--.y', 'MarkerSize', 8, 'LineWidth', 2);
# hold on;
# plot(x2, noisy_psf(:, 1), 'ks', 'MarkerSize', 8, 'LineWidth', 2);
# plot(x2, initial_spline_fit,'--sg', 'MarkerSize', 8, 'LineWidth', 2);
# plot(x2, final_fit_cpufit,'-xc', 'MarkerSize', 8, 'LineWidth', 2);
# plot(x2, final_fit_gpufit,'--+b', 'MarkerSize', 8, 'LineWidth', 2);
# plot(x2, spline_model, ':r', 'MarkerSize', 8, 'LineWidth', 1.5);
# ylim([0, max(initial_spline_fit)]);
# legend(...
#     'final gauss fit',...
#     'noisy data',...
#     'initial spline fit',...
#     'final spline fit cpu',...
#     'final spline fit gpu',...
#     'true parameters spline');
