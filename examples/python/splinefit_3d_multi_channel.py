"""
Spline fit 3D 4 channel example. Can for example be used in 4Pi-Storm microscopy.

Requires pyGpufit, pyGpuSpline (https://github.com/gpufit/Gpuspline), Numpy and Matplotlib.
"""

import numpy as np
from matplotlib import pyplot as plt
import pygpuspline.gpuspline as gs
import pygpufit.gpufit as gf
import misc

def merge_channels(x):
    """
    Takes a NxMx4 array and returns a 2Nx2M array by stacking the 4 2D images in a 2x2 grid.
    :param x: Input array
    :return: Output array
    """
    return np.vstack((np.hstack((x[:, :, 0], x[:, :, 1])), np.hstack((x[:, :, 2], x[:, :, 3]))))


def calculate_psf(size_x, size_y, size_z, p, n_channels):
    """
    calculate psf    
    :param size_x: 
    :param size_y: 
    :param size_z: 
    :param p: 
    :param n_channels: 
    :return: 
    """

    s_max = p[4] * 5
    s_min = p[4] / 5
    
    sx = np.linspace(s_max, s_min, size_z)
    sy = np.linspace(s_min, s_max, size_z)
    sz = 8
    
    delta_s = sx[0] - sx[1]
    sx = [sx, s_min - 1 * delta_s, s_min - 2 * delta_s, s_min - 3 * delta_s]
    sy = [sy, s_max + 1 * delta_s, s_max + 2 * delta_s, s_max + 3 * delta_s]
    
    x = np.linspace(0, np.pi, size_z)
    a = np.zeros((x.size, 4))
    a[:, 0] = np.sin(x + np.pi * 0 / 4) * 0.5 + 0.5
    a[:, 1] = np.sin(x + np.pi * 1 / 4) * 0.5 + 0.5
    a[:, 2] = np.sin(x + np.pi * 2 / 4) * 0.5 + 0.5
    a[:, 3] = np.sin(x + np.pi * 3 / 4) * 0.5 + 0.5
    
    f = np.zeros((size_x, size_y, size_z, n_channels), dtype=np.float32)

    for ch in range(n_channels):
        for xi in range(size_x):
            for yi in range(size_y):
                for zi in range(size_z):

                    arg_x = np.exp(-1 / 2 * ((xi - p[1]) / sx[zi + ch])**2)
                    arg_y = np.exp(-1 / 2 * ((yi - p[2]) / sy[zi + ch])**2)
                    arg_z = np.exp(-1 / 2 * ((zi + ch - p[3]) / sz)**2)

                    f[xi, yi, zi, ch] = a[zi, ch] * p[0] * arg_x * arg_y * arg_z + p[5]

                    return f
    

def spline_values_3d_multichannel(coefficients, n_intervals, n_channels, p):
    """
    
    :param coefficients: 
    :param n_intervals: 
    :param n_channels: 
    :param p: 
    :return: 
    """
    coefficients = np.reshape(coefficients, (64, n_intervals[0], n_intervals[1], n_intervals[2], n_channels))
    x = np.arange(n_intervals[0])-p[1]
    y = np.arange(n_intervals[1])-p[2]
    z = -p[3]
    f = p[0] * gs.spline_values(coefficients, x, y, z) + p[4]
    
    return f


if __name__ == '__main__':
    # initialize random number generator
    rng = np.random.default_rng(0)

    # data size
    size_x = 19
    size_y = 25
    size_z = 50
    n_channels = 4

    # tolerances
    tolerance = 1e-30
    max_n_iterations = 200
    estimator_id = gf.EstimatorID.LSE

    # derived values
    x = np.arange(size_x, dtype=np.float32)
    y = np.arange(size_y, dtype=np.float32)
    z = np.arange(size_z, dtype=np.float32)

    # generate PSF
    psf_parameters = np.array([100, (size_x - 1) / 2 + 1, (size_y - 1) / 2 - 1, (size_z - 1) / 2, 1, 10], dtype=np.float32)
    psf = calculate_psf(size_x, size_y, size_z, psf_parameters, n_channels)
    z_slice_index = 25

    # add noise
    snr = 5
    amplitude = psf_parameters[0]
    noise_std_dev = amplitude / (snr * np.log(10.0))
    noise = noise_std_dev * rng.standard_normal(psf.shape, dtype=np.float32)
    noisy_psf = psf + noise

    # calculate PSF template
    psf_normalized = (psf - psf_parameters[5]) / psf_parameters[0]

    # calculate spline coefficients of the PSF template
    coefficients = np.zeros([64, psf.shape[0]-1, psf.shape[1]-1, psf.shape[2]-1, n_channels], dtype=np.float32)
    for ch in range(n_channels):
        coefficients[:,:,:,:, ch] = gs.spline_coefficients(psf_normalized[:,:,:, ch])
    n_intervals = np.array(psf_normalized.shape) - 1
    n_intervals = n_intervals[:3]

    # set user info
    user_info = np.hstack((n_channels, size_x, size_y, 1, n_intervals, coefficients.flatten()))

    # true fit parameters
    true_fit_parameters = np.zeros((1, 5), dtype=np.float32)
    true_fit_parameters[0, 0] = psf_parameters[0] # amplitude
    true_fit_parameters[0, 1] = 0 # center x shift
    true_fit_parameters[0, 2] = 0 # center y shift
    true_fit_parameters[0, 3] = 1 - z_slice_index # z index
    true_fit_parameters[0, 4] = psf_parameters[5] # offset

    # set initial fit parameters
    pos_shift_x = -0.9
    pos_shift_y = 0.7
    pos_shift_z = 10
    amp_shift = -20
    off_shift = 5

    spline_fit_initial_parameters = true_fit_parameters + np.array([amp_shift, pos_shift_x, pos_shift_y, pos_shift_z, off_shift], dtype=np.float32)

    # reshape data
    linear_psf = noisy_psf[:,:,z_slice_index, :]
    linear_psf = np.reshape(linear_psf, (1, size_x * size_y * n_channels))

    # call to gpufit with spline fit
    parameters_spline, states_spline, chi_squares_spline, n_iterations_spline, time_spline = gf.fit(linear_psf, None, gf.ModelID.SPLINE_3D_MULTICHANNEL, spline_fit_initial_parameters, tolerance, max_n_iterations,
             None, estimator_id, user_info)
    # check if gpufit succeeded
    if not np.all(states_spline == 0):
        raise RuntimeError('Spline fit did not converge.')

    # get data to plot
    psf_4ch = merge_channels(psf[:,:, z_slice_index,:])
    
    noisy_psf_4ch = merge_channels(noisy_psf[:,:, z_slice_index,:])
    
    initial_spline_fit = spline_values_3d_multichannel(coefficients, n_intervals, n_channels,
                                                       spline_fit_initial_parameters)
    initial_spline_fit_4ch = merge_channels(initial_spline_fit)
    
    final_spline_gpufit = spline_values_3d_multichannel(coefficients, n_intervals, n_channels, parameters_spline)
    final_spline_4ch = merge_channels(final_spline_gpufit)
    
    # plot
    #
    # min_value = min([psf(:) noisy_psf(:) initial_spline_fit(:) final_spline_gpufit(:)])
    # max_value = max([psf(:) noisy_psf(:) initial_spline_fit(:) final_spline_gpufit(:)])
    # clims = [min_value max_value]
    #
    # subplot(2, 2, 1)
    # imagesc(x, y', psf_4ch, clims) title(sprintf('psf z = # .2 f', true_fit_parameters[3])) axis image
    # subplot(2, 2, 2)
    # imagesc(x, y, noisy_psf_4ch, clims) title(sprintf('noisy psf z = # .2 f', true_fit_parameters[3])) axis image
    # subplot(2, 2, 3)
    # imagesc(x, y, initial_spline_fit_4ch, clims) title(sprintf('initial spline fit z = # .2 f', spline_fit_initial_parameters[3])) axis image
    # subplot(2, 2, 4)
    # imagesc(x, y, final_spline_4ch, clims) title(sprintf('final gpufit z = # .2 f', parameters_spline[3])) axis image
    # colormap('hot')