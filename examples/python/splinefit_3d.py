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

    size_x = x.size
    size_y = y.size
    size_z = z.size
    
    s_max = p[4] * 5
    s_min = p[4] / 5
    
    sx = np.linspace(s_max, s_min, size_z)
    sy = np.linspace(s_min, s_max, size_z)
    sz = p[4] * 10
    
    f = np.zeros((size_x, size_y, size_z), dtype=np.float32)

    for x in range(size_x):
        for y in range(size_y):
            for z in range(size_z):

                arg_x =np.exp(-1 / 2 * ((x + 1 - p[1]) / sx[z])**2)
                arg_y =np.exp(-1 / 2 * ((y - p[2]) / sy[z])**2)
                arg_z =np.exp(-1 / 2 * ((z - p[3]) / sz)**2)

                f[x, y, z] = p[0] * arg_x * arg_y * arg_z + p[5]
    
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
    z_slice_index = 61
    
    # add noise
    snr = 10
    amplitude = psf_parameters[0]
    noise_std_dev = amplitude / (snr * np.log(10.0))
    noise = noise_std_dev * rng.standard_normal(psf.shape, dtype=np.float32)
    noisy_psf = psf + noise
    
    # calculate PSF template
    psf_template = (psf - psf_parameters[5]) / psf_parameters[0]
    
    # calculate spline coefficients of the PSF template
    coefficients = gs.spline_coefficients(psf_template)
    n_intervals = np.array(psf_template.shape) - 1
    coefficients = np.reshape(coefficients, (64, n_intervals[0], n_intervals[1], n_intervals[2]))
    
    # set user info
    user_info = np.hstack((size_x, size_y, 1, n_intervals, coefficients.flatten()))
    
    # true fit parameters
    true_fit_parameters = np.zeros((1, 5), dtype=np.float32)
    true_fit_parameters[0, 0] = psf_parameters[0] # amplitude
    true_fit_parameters[0, 1] = 0                 # center x shift
    true_fit_parameters[0, 2] = 0                 # center y shift
    true_fit_parameters[0, 3] = 1 - z_slice_index     # center z shift
    true_fit_parameters[0, 4] = psf_parameters[5] # offset
    
    # set initial fit parameters
    pos_shift_x = 1.2
    pos_shift_y = -0.3
    pos_shift_z = -20
    amp_shift = -15
    off_shift = 2
    
    spline_fit_initial_parameters = true_fit_parameters + np.array([amp_shift, pos_shift_x, pos_shift_y, pos_shift_z, off_shift], dtype=np.float32)
    
    # reshape data
    linear_psf = np.array(noisy_psf[:,:,z_slice_index])  # need to make a copy, otherwise it would only make a view
    linear_psf = np.reshape(linear_psf, (1, noisy_psf.shape[0] * noisy_psf.shape[1]))
    
    # call to gf.fit with spline fit
    parameters_spline, states_spline, chi_squares_spline, n_iterations_spline, time_spline = gf.fit(linear_psf, None, gf.ModelID.SPLINE_3D, spline_fit_initial_parameters, tolerance, max_n_iterations, None, estimator_id, user_info)
    if not np.all(states_spline == 0):
        raise RuntimeError('Spline fit did not converge')

    # get data to plot
    # spline with true parameters
    a_true = true_fit_parameters[0, 0]
    x_true = x-true_fit_parameters[0, 1]
    y_true = y-true_fit_parameters[0, 2]
    z_true = -true_fit_parameters[0, 3]
    b_true = true_fit_parameters[0, 4]
    true_spline_fit = a_true * gs.spline_values(coefficients, x_true, y_true, z_true) + b_true
    
    # spline with initial fit parameters
    a_init = spline_fit_initial_parameters[0, 0]
    x_init = x-spline_fit_initial_parameters[0, 1]
    y_init = y-spline_fit_initial_parameters[0, 2]
    z_init = -spline_fit_initial_parameters[0, 3]
    b_init = spline_fit_initial_parameters[0, 4]
    initial_spline_fit = a_init * gs.spline_values(coefficients, x_init, y_init, z_init) + b_init
    
    # spline with fit parameters
    a_fit = parameters_spline[0, 0]
    x_fit = x-parameters_spline[0, 1]
    y_fit = y-parameters_spline[0, 2]
    z_fit = -parameters_spline[0, 3]
    b_fit = parameters_spline[0, 4]
    final_spline = a_fit * gs.spline_values(coefficients, x_fit, y_fit, z_fit) + b_fit

    # show results in a matplotlib figure
    current_slice = noisy_psf[:,:, z_slice_index]
    extent = [y[0], y[-1], x[0], x[-1]]

    fig, axs = plt.subplots(2, 2)
    fig.tight_layout()
    axs = axs.flat
    axs[0].imshow(true_spline_fit, cmap='hot', extent=extent)
    axs[0].set(title='true spline z={:.2f}'.format(true_fit_parameters[0, 3]))
    axs[1].imshow(current_slice, cmap='hot', extent=extent)
    axs[1].set(title='noisy PSF z={:.2f}'.format(true_fit_parameters[0, 3]))
    axs[2].imshow(initial_spline_fit, cmap='hot', extent=extent)
    axs[2].set(title='initial spline fit z={:.2f}'.format(spline_fit_initial_parameters[0, 3]))
    axs[3].imshow(final_spline, cmap='hot', extent=extent)
    axs[3].set(title='final spline fit z={:.2f}'.format(parameters_spline[0, 3]))

