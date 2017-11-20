"""
    Python binding for Gpufit, a Levenberg Marquardt curve fitting library written in CUDA
    See https://github.com/gpufit/Gpufit, http://gpufit.readthedocs.io/en/latest/bindings.html#python

    The binding is based on ctypes.
    See https://docs.python.org/3.5/library/ctypes.html, http://www.scipy-lectures.org/advanced/interfacing_with_c/interfacing_with_c.html
"""

import os
import time
from ctypes import cdll, POINTER, byref, c_int, c_float, c_char, c_char_p, c_size_t
import numpy as np
import sys

# define library loader (actual loading is lazy)
package_dir = os.path.dirname(os.path.realpath(__file__))
if os.name == 'nt':
    lib_path = os.path.join(package_dir, 'Gpufit.dll') # this will only work on Windows
elif os.name == 'posix':
    lib_path = os.path.join(package_dir, 'libGpufit.so') # this will only work on Linux
else:
    sys.exit('Operative System not recognized / not supported')

lib = cdll.LoadLibrary(lib_path)

# gpufit function in the dll
gpufit_func = lib.gpufit
gpufit_func.restype = c_int
gpufit_func.argtypes = [c_size_t, c_size_t, POINTER(c_float), POINTER(c_float), c_int, POINTER(c_float), c_float, c_int, POINTER(c_int), c_int, c_size_t, POINTER(c_char), POINTER(c_float), POINTER(c_int), POINTER(c_float), POINTER(c_int)]

# gpufit_get_last_error function in the dll
error_func = lib.gpufit_get_last_error
error_func.restype = c_char_p
error_func.argtypes = None

# gpufit_cuda_available function in the dll
cuda_available_func = lib.gpufit_cuda_available
cuda_available_func.restype = c_int
cuda_available_func.argtypes = None

# gpufit_get_cuda_version function in the dll
get_cuda_version_func = lib.gpufit_get_cuda_version
get_cuda_version_func.restype = c_int
get_cuda_version_func.argtypes = [POINTER(c_int), POINTER(c_int)]


class ModelID():

    GAUSS_1D = 0
    GAUSS_2D = 1
    GAUSS_2D_ELLIPTIC = 2
    GAUSS_2D_ROTATED = 3
    CAUCHY_2D_ELLIPTIC = 4
    LINEAR_1D = 5


class EstimatorID():

    LSE = 0
    MLE = 1


class Status():

    Ok = 0
    Error = 1


def fit(data, weights, model_id, initial_parameters, tolerance=None, max_number_iterations=None, \
           parameters_to_fit=None, estimator_id=None, user_info=None):
    """
    Calls the C interface fit function in the library.
    (see also http://gpufit.readthedocs.io/en/latest/bindings.html#python)

    All 2D NumPy arrays must be in row-major order (standard in NumPy), i.e. array.flags.C_CONTIGUOUS must be True
    (see also https://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html#internal-memory-layout-of-an-ndarray)

    :param data: The data - 2D NumPy array of dimension [number_fits, number_points] and data type np.float32
    :param weights: The weights - 2D NumPy array of the same dimension and data type as parameter data or None (no weights available)
    :param model_id: The model ID
    :param initial_parameters: Initial values for parameters - NumPy array of dimension [number_fits, number_parameters] and data type np.float32
    :param tolerance: The fit tolerance or None (will use default value)
    :param max_number_iterations: The maximal number of iterations or None (will use default value)
    :param parameters_to_fit: Which parameters to fit - NumPy array of length number_parameters and type np.int32 or None (will fit all parameters)
    :param estimator_id: The Estimator ID or None (will use default values)
    :param user_info: User info - NumPy array of type np.char or None (no user info available)
    :return: parameters, states, chi_squares, number_iterations, execution_time
    """

    # check all 2D NumPy arrays for row-major memory layout (otherwise interpretation of order of dimensions fails)
    if not data.flags.c_contiguous:
        raise RuntimeError('Memory layout of data array mismatch.')

    if weights is not None and not weights.flags.c_contiguous:
        raise RuntimeError('Memory layout of weights array mismatch.')

    if not initial_parameters.flags.c_contiguous:
        raise RuntimeError('Memory layout of initial_parameters array mismatch.')

    # size check: data is 2D and read number of points and fits
    if data.ndim != 2:
        raise RuntimeError('data is not two-dimensional')
    number_points = data.shape[1]
    number_fits = data.shape[0]

    # size check: consistency with weights (if given)
    if weights is not None and data.shape != weights.shape:
        raise  RuntimeError('dimension mismatch between data and weights')
        # the unequal operator checks, type, length and content (https://docs.python.org/3.7/reference/expressions.html#value-comparisons)

    # size check: initial parameters is 2D and read number of parameters
    if initial_parameters.ndim != 2:
        raise RuntimeError('initial_parameters is not two-dimensional')
    number_parameters = initial_parameters.shape[1]
    if initial_parameters.shape[0] != number_fits:
        raise RuntimeError('dimension mismatch in number of fits between data and initial_parameters')

    # size check: consistency with parameters_to_fit (if given)
    if parameters_to_fit is not None and parameters_to_fit.shape[0] != number_parameters:
        raise RuntimeError('dimension mismatch in number of parameters between initial_parameters and parameters_to_fit')

    # default value: tolerance
    if not tolerance:
        tolerance = 1e-4

    # default value: max_number_iterations
    if not max_number_iterations:
        max_number_iterations = 25

    # default value: estimator ID
    if not estimator_id:
        estimator_id = EstimatorID.LSE

    # default value: parameters_to_fit
    if parameters_to_fit is None:
        parameters_to_fit = np.ones(number_parameters, dtype=np.int32)

    # now only weights and user_info could be not given

    # type check: data, weights (if given), initial_parameters are all np.float32
    if data.dtype != np.float32:
        raise RuntimeError('type of data is not np.float32')
    if weights is not None and weights.dtype != np.float32:
        raise RuntimeError('type of weights is not np.float32')
    if initial_parameters.dtype != np.float32:
        raise RuntimeError('type of initial_parameters is not np.float32')

    # type check: parameters_to_fit is np.int32
    if parameters_to_fit.dtype != np.int32:
        raise RuntimeError('type of parameters_to_fit is not np.int32')

    # we don't check type of user_info, but we extract the size in bytes of it
    if user_info is not None:
        user_info_size = user_info.nbytes
    else:
        user_info_size = 0

    # pre-allocate output variables
    parameters = np.zeros((number_fits, number_parameters), dtype=np.float32)
    states = np.zeros(number_fits, dtype=np.int32)
    chi_squares = np.zeros(number_fits, dtype=np.float32)
    number_iterations = np.zeros(number_fits, dtype=np.int32)

    # conversion to ctypes types for optional C interface parameters using NULL pointer (None) as default argument
    if weights is not None:
        weights_p = weights.ctypes.data_as(gpufit_func.argtypes[3])
    else:
        weights_p = None
    if user_info is not None:
        user_info_p = user_info.ctypes.data_as(gpufit_func.argtypes[11])
    else:
        user_info_p = None

    # call into the library (measure time)
    t0 = time.clock()
    status = gpufit_func(
        gpufit_func.argtypes[0](number_fits), \
        gpufit_func.argtypes[1](number_points), \
        data.ctypes.data_as(gpufit_func.argtypes[2]), \
        weights_p, \
        gpufit_func.argtypes[4](model_id), \
        initial_parameters.ctypes.data_as(gpufit_func.argtypes[5]), \
        gpufit_func.argtypes[6](tolerance), \
        gpufit_func.argtypes[7](max_number_iterations), \
        parameters_to_fit.ctypes.data_as(gpufit_func.argtypes[8]), \
        gpufit_func.argtypes[9](estimator_id), \
        gpufit_func.argtypes[10](user_info_size), \
        user_info_p, \
        parameters.ctypes.data_as(gpufit_func.argtypes[12]), \
        states.ctypes.data_as(gpufit_func.argtypes[13]), \
        chi_squares.ctypes.data_as(gpufit_func.argtypes[14]), \
        number_iterations.ctypes.data_as(gpufit_func.argtypes[15]))
    t1 = time.clock()


    # check status
    if status != Status.Ok:
        # get error from last error and raise runtime error
        error_message = error_func()
        raise RuntimeError('status = {}, message = {}'.format(status, error_message))

    # return output values
    return parameters, states, chi_squares, number_iterations, t1 - t0


def get_last_error():
    """
    :return: Error message of last error.
    """
    return error_func()


def cuda_available():
    """
    :return: True if CUDA is available, False otherwise
    """
    return cuda_available_func() != 0


def get_cuda_version():
    """
    :return: Tuple with runtime and driver version as integers.
    """
    runtime_version = c_int(-1)
    driver_version = c_int(-1)
    status = get_cuda_version_func(byref(runtime_version), byref(driver_version))

    # check status
    if status != Status.Ok:
        # get error from last error and raise runtime error
        error_message = error_func()
        raise RuntimeError('status = {}, message = {}'.format(status, error_message))

    # decode versions
    runtime_version = runtime_version.value
    runtime_version = (runtime_version // 1000, runtime_version % 1000 // 10)
    driver_version = driver_version.value
    driver_version = (driver_version // 1000, driver_version % 1000 // 10)

    return runtime_version, driver_version
