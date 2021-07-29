"""
Spline fit 2D rectangular example

Requires pyGpufit, pyGpuSpline (https://github.com/gpufit/Gpuspline), Numpy and Matplotlib.
"""

import numpy as np
from matplotlib import pyplot as plt
import pygpuspline.gpuspline as gs
import pygpufit.gpufit as gf
import misc


def calculate_psf(x, y, p):
    """
    Test PSF consists of an elliptic 2D Gaussian
    
    p[0] - amplitude
    p[1] - center x
    p[2] - center y
    p[3] - Standard deviation
    p[4] - constant background    
    :param x: 
    :param y: 
    :param p: 
    :return: 
    """
    sx = p[3] + 0.1
    sy = p[3] - 0.2

    arg_ex = np.exp(-1 / 2 * ((x - p[1]) / sx) ** 2 - 1 / 2 * ((y - p[2]) / sy) ** 2)

    z = p[0] * arg_ex + p[4]  # scale with amplitude and background

    return z


if __name__ == '__main__':
    # initialize random number generator
    rng = np.random.default_rng(0)

    # data size
    size_x = 12
    size_y = 14

    # tolerances
    tolerance = 1e-30
    max_n_iterations = 100
    estimator_id = gf.EstimatorID.LSE

    # derived values
    x = np.arange(size_x, dtype=np.float32)
    y = np.arange(size_y, dtype=np.float32)

    SF = 2  # scaling factor

    x_spline = np.arange(SF * size_x, dtype=np.float32)
    x_spline.shape = x_spline.shape + (1,)  # Nx1
    x2 = x_spline / SF

    y_spline = np.arange(SF * size_y, dtype=np.float32)
    y_spline.shape = (1,) + y_spline.shape  # 1xN
    y2 = y_spline / SF

    # generate PSF
    psf_parameters = np.array([100, (size_x - 1) / 2, (size_y - 1) / 2, 1, 10], dtype=np.float32)

    # calculate PSF template
    # calculate PSF on fine grid
    psf = calculate_psf(x2, y2, psf_parameters)
    # PSF template (normalized to minimum = 0 and maximum = 1)
    psf_normalized = (psf - psf_parameters[4]) / psf_parameters[0]

    # calculate spline coefficients of the PSF template
    coefficients = gs.spline_coefficients(psf_normalized)
    n_intervals = np.array(psf_normalized.shape, dtype=np.float32) - 1

    # add noise to PSF data
    snr = 50
    amplitude = psf_parameters[0]
    noise_std_dev = amplitude / (snr * np.log(10.0))
    noise = noise_std_dev * rng.standard_normal(psf.shape, dtype=np.float32)
    noisy_psf = psf + noise

    # set user info
    user_info = np.hstack((x2.size, y2.size, n_intervals, coefficients.flatten()))

    # true fit parameters (amplitude, center x shift, center y shift, offset)
    true_fit_parameters = np.array([psf_parameters[0], 0, 0, psf_parameters[4]], dtype=np.float32)
    true_fit_parameters = true_fit_parameters.reshape((1, true_fit_parameters.shape[0]))

    # set initial fit parameters
    pos_shift_x = 1.
    pos_shift_y = -2.1
    amp_shift = 20
    off_shift = 13

    spline_fit_initial_parameters = true_fit_parameters + np.array(
        [amp_shift, pos_shift_x * SF, pos_shift_y * SF, off_shift], dtype=np.float32)

    gauss_fit_initial_parameters = np.zeros((1, 5), dtype=np.float32)
    gauss_fit_initial_parameters[0, 0] = (psf_parameters[0] + amp_shift)
    gauss_fit_initial_parameters[0, 1] = (psf_parameters[1] + pos_shift_x) * SF
    gauss_fit_initial_parameters[0, 2] = (psf_parameters[2] + pos_shift_y) * SF
    gauss_fit_initial_parameters[0, 3] = (psf_parameters[3] + 0) * SF
    gauss_fit_initial_parameters[0, 4] = (psf_parameters[4] + off_shift)

    # reshape data
    linear_noisy_psf = np.reshape(noisy_psf, (1, noisy_psf.size))
    # this will only work if size_x,y aren't changed above (try to cut off data symmetrical to make a square data array for gauss fit)
    linear_noisy_psf_gauss = np.reshape(noisy_psf[:, 2:-2], (1, 24 ** 2))

    # call to gpufit with spline fit
    parameters_spline, states_spline, chi_squares_spline, n_iterations_spline, time_spline = gf.fit(linear_noisy_psf,
                                                                                                    None,
                                                                                                    gf.ModelID.SPLINE_2D,
                                                                                                    spline_fit_initial_parameters,
                                                                                                    tolerance,
                                                                                                    max_n_iterations,
                                                                                                    None, estimator_id,
                                                                                                    user_info)
    if not np.all(states_spline == 0):
        raise RuntimeError('Not all spline fits converged.')

    # call to gpufit with gauss1d fit
    parameters_gauss, states_gauss, chi_squares_gauss, n_iterations_gauss, time_gauss = gf.fit(linear_noisy_psf_gauss,
                                                                                               None,
                                                                                               gf.ModelID.GAUSS_2D,
                                                                                               gauss_fit_initial_parameters,
                                                                                               tolerance,
                                                                                               max_n_iterations, None,
                                                                                               estimator_id)
    if not np.all(states_gauss == 0):
        raise RuntimeError('Not all Gaussian fits converged.')

    # get data to plot
    a = spline_fit_initial_parameters[0, 0]
    xx = x_spline - spline_fit_initial_parameters[0, 1]
    yy = y_spline - spline_fit_initial_parameters[0, 2]
    b = spline_fit_initial_parameters[0, 3]
    initial_spline_fit = a * gs.spline_values(coefficients, xx, yy) + b
    final_spline_fit = a * gs.spline_values(coefficients, xx, yy) + b
    initial_gauss_fit = misc.gaussian_peak_2d(x_spline, y_spline, gauss_fit_initial_parameters)
    final_gauss_fit = misc.gaussian_peak_2d(x_spline, y_spline, parameters_gauss)

    # make a figure of psf, psf template, initial and final gauss fit, initial and final spline fit
    fig, axs = plt.subplots(2, 3)
    fig.tight_layout()
    axs = axs.flat
    axs[0].imshow(noisy_psf, cmap='hot')
    axs[0].set(title='noisy PSF')
    axs[1].imshow(initial_gauss_fit, cmap='hot')
    axs[1].set(title='initial Gaussian fit')
    axs[2].imshow(initial_spline_fit, cmap='hot')
    axs[2].set(title='initial spline fit')
    axs[3].imshow(psf_normalized, cmap='hot')
    axs[3].set(title='PSF template')
    axs[4].imshow(final_gauss_fit, cmap='hot')
    axs[4].set(title='final Gaussian fit')
    axs[5].imshow(final_spline_fit, cmap='hot')
    axs[5].set(title='final spline fit')

    # min_noisy_psf = min(min(noisy_psf))
    # max_noisy_psf = max(max(noisy_psf))
    # min_temp = min(min([initial_gauss_fit, initial_spline_fit, psf_normalized, final_gauss_fit, final_spline_fit]))
    # max_temp = max(max([initial_gauss_fit, initial_spline_fit, psf_normalized, final_gauss_fit, final_spline_fit]))
    # min_value = min(min_noisy_psf, min_temp)
    # max_value = max(max_noisy_psf, max_temp)
    # clims = [min_value max_value]
    # subplot(231) imagesc(x, y, noisy_psf, clims)            colorbar title('noisy psf') axis square
    # subplot(232) imagesc(x2, y2, initial_gauss_fit, clims)  colorbar title('initial gauss fit') axis square
    # subplot(233) imagesc(x2, y2, initial_spline_fit, clims) colorbar title('initial spline fit') axis square
    # subplot(234) imagesc(x2, y2, psf_normalized)            colorbar title('psf template') axis square
    # subplot(235) imagesc(x2, y2, final_gauss_fit, clims)    colorbar title('final gauss fit') axis square
    # subplot(236) imagesc(x2, y2, final_spline_fit, clims)   colorbar title('final spline fit') axis square
    # colormap('hot')
