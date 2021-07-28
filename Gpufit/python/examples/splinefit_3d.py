"""
Spline fit 3D example.

Requires pyGpufit, pyGpuSpline (https://github.com/gf.fit/Gpuspline), Numpy and Matplotlib.
"""

import numpy as np
from matplotlib import pyplot as plt
import pygpuspline.gpuspline as gs
import pygpufit.gpufit as gf
import misc

def calculate_psf(x, y, z, p):
    """
    
    :param x: 
    :param y: 
    :param z: 
    :param p: 
    :return: 
    """

    size_x = numel(x)
    size_y = numel(y)
    size_z = numel(z)
    
    s_max = p[4] * 5
    s_min = p[4] / 5
    
    sx = linspace(s_max, s_min, numel(z))
    sy = linspace(s_min, s_max, numel(z))
    sz = p[4] * 10
    
    f = np.zeros((size_x, size_y, size_z), dtype=np.float32)

    for x in range(size_x):
        for y in range(size_y):
            for z in range(size_z):

                arg_x =np.exp(-1 / 2 * ((x + 1 - p[1]) / sx(z + 1))**2)
                arg_y =np.exp(-1 / 2 * ((y - p[2]) / sy(z + 1))**2)
                arg_z =np.exp(-1 / 2 * ((z - p[3]) / sz)**2)

                f[x, y, z] = p[0] * arg_x * arg_y * arg_z + p(6)
    
    return f


if __name__ == '__main__':
    # initialize random number generator
    rng = np.random.default_rng(0)

    # data size
    size_x = 18
    size_y = 13
    size_z = 100
    
    # tolerances
    tolerance = 1e-30
    max_n_iterations = 100
    estimator_id = gf.EstimatorID.LSE
    
    # derived values
    x = np.arange(size_x, dtype=np.float32)
    y = np.arange(size_y, dtype=np.float32)
    z = np.arange(size_z, dtype=np.float32)
    
    # generate PSF
    psf_parameters = np.array([100, (size_x-1)/2, (size_y-1)/2, (size_z-1)/2+1, 1, 10], dtype=np.float32)
    psf = calculate_psf(x, y, z, psf_parameters)
    z_slice_index = 60
    
    # add noise
    snr = 10
    amplitude = psf_parameters[0]
    noise_std_dev = amplitude / (snr * np.log(10.0))
    noise = noise_std_dev * rng.standard_normal(psf.shape, dtype=np.float32)
    noisy_psf = psf + noise
    
    # calculate PSF template
    psf_template = (psf - psf_parameters(6)) / psf_parameters[0]
    
    # calculate spline coefficients of the PSF template
    coefficients = gs.spline_coefficients(psf_template)
    n_intervals = psf_template.shapoe - 1
    coefficients = reshape(coefficients, 64, n_intervals[0], n_intervals[1], n_intervals[2])
    
    # set user info
    user_info = np.hstack((size_x, size_y, 1, n_intervals, coefficients.flatten()))
    
    # true fit parameters
    true_fit_parameters = np.zeros((1, 5), dtype=np.float32)
    true_fit_parameters[0] = psf_parameters[0] # amplitude
    true_fit_parameters[1] = 0                 # center x shift
    true_fit_parameters[2] = 0                 # center y shift
    true_fit_parameters[3] = 1 - z_slice_index     # center z shift
    true_fit_parameters[4] = psf_parameters[5] # offset
    
    # set initial fit parameters
    pos_shift_x = 1.2
    pos_shift_y = -0.3
    pos_shift_z = -20
    amp_shift = -15
    off_shift = 2
    
    spline_fit_initial_parameters = true_fit_parameters + np.array([amp_shift, pos_shift_x, pos_shift_y, pos_shift_z, off_shift], dtype=np.float32)
    
    # reshape data
    linear_psf = reshape(noisy_psf(:,:,z_slice_index), numel(noisy_psf(:,:,z_slice_index)), 1)
    
    # call to gf.fit with spline fit
    parameters_spline, states_spline, chi_squares_spline, n_iterations_spline, time_spline
        = gf.fit(linear_psf, None, gf.ModelID.SPLINE_3D, spline_fit_initial_parameters, tolerance, max_n_iterations, None, estimator_id, user_info)
    if not np.all(states_spline == 0):
        raise RuntimeError('Spline fit did not converge')

    # get data to plot
    # spline with true parameters
    a_true = true_fit_parameters[0]
    x_true = x-true_fit_parameters[1]
    y_true = y-true_fit_parameters[2]
    z_true = -true_fit_parameters[3]
    b_true = true_fit_parameters[4]
    true_spline_fit = a_true * gs.spline_values(coefficients, x_true, y_true, z_true) + b_true
    
    # spline with initial fit parameters
    a_init = spline_fit_initial_parameters[0]
    x_init = x-spline_fit_initial_parameters[1]
    y_init = y-spline_fit_initial_parameters[2]
    z_init = -spline_fit_initial_parameters[3]
    b_init = spline_fit_initial_parameters[4]
    initial_spline_fit = a_init * gs.spline_values(coefficients, x_init, y_init, z_init) + b_init
    
    # spline with fit parameters
    a_fit = parameters_spline[0]
    x_fit = x-parameters_spline[1]
    y_fit = y-parameters_spline[2]
    z_fit = -parameters_spline[3]
    b_fit = parameters_spline[4]
    final_spline = a_fit * gs.spline_values(coefficients, x_fit, y_fit, z_fit) + b_fit

    # figure
    # current_slice = noisy_psf(:,:,z_slice_index)
    # min_noisy_psf = min(min(current_slice))
    # max_noisy_psf = max(max(current_slice))
    # min_temp = min(min([initial_spline_fit final_spline_gf.fit]))
    # max_temp = max(max([initial_spline_fit final_spline_gf.fit]))
    # min_value = min(min_noisy_psf, min_temp)
    # max_value = max(max_noisy_psf, max_temp)
    # clims = [min_value max_value]
    # subplot(2,2,1) imagesc(x, y', true_spline_fit, clims)     title(sprintf('true spline z=%.2f', true_fit_parameters[3])) axis image
    # subplot(2,2,2) imagesc(x, y', current_slice, clims)       title(sprintf('noisy psf z=%.2f', true_fit_parameters[3])) axis image
    # subplot(2,2,3) imagesc(x, y', initial_spline_fit, clims)  title(sprintf('initial spline fit z=%.2f', spline_fit_initial_parameters[3])) axis image
    # subplot(2,2,4) imagesc(x, y', final_spline_gf.fit, clims) title(sprintf('final spline gf.fit z=%.2f', parameters_spline[3])) axis image
    # colormap('hot')

