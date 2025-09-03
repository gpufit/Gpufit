"""
Program written from scratch following example of pyGPUfit by Jakub Dokulil (jakub.dokulil@imp.ac.at).

For GPUfit see https://github.com/gpufit/Gpufit, http://gpufit.readthedocs.io/en/latest/bindings.html#python

"""
# %%

import os
import time
from ctypes import cdll, POINTER, byref, c_int, c_float, c_char, c_char_p, c_size_t
import numpy as np

# %%
package_dir = os.path.dirname(os.path.realpath(__file__))

if os.name == 'nt':
    lib_path = os.path.join(package_dir, 'cpufit.dll')  # library name on Windows
elif os.name == 'posix':
    lib_path = os.path.join(package_dir, 'libCpufit.so')  # library name on Unix
else:
    raise RuntimeError('OS {} not supported by pyGpufit.'.format(os.name))

lib = cdll.LoadLibrary(lib_path)

# %%
cpufit_func = lib.cpufit

# %%

cpufit_func.restype = c_int
cpufit_func.argtypes = [c_size_t, c_size_t, POINTER(c_float), POINTER(c_float), c_int, POINTER(c_float),
                        c_float, c_int, POINTER(c_int), c_int, c_size_t,
                        POINTER(c_char), POINTER(c_float), POINTER(c_int), POINTER(c_float), POINTER(c_int)]

# int cpufit
# (
#     std::size_t n_fits, - 1
#     std::size_t n_points, - number of points to be fited
#     REAL * data, - flattened image with data
#     REAL * weights, - wheights (see alesandros example)
#     int model_id, - we want 2 GAUSS_2D_ELLIPTIC
#     REAL * initial_parameters, - [p0, p1, ..., p5]
#     REAL tolerance, - lets set 1e-3
#     int max_n_iterations, 1200
#     int * parameters_to_fit,  - [1, 1, 1, 1, 1, 1]
#     int estimator_id, - 0
#     std::size_t user_info_size, - 0 
#     char * user_info, - NULL
#     REAL * output_parameters,
#     int * output_states,
#     REAL * output_chi_squares,
#     int * output_n_iterations
# )

error_func = lib.cpufit_get_last_error
error_func.restype = c_char_p
error_func.argtypes = None

class ModelID:
    GAUSS_1D = 0
    GAUSS_2D = 1
    GAUSS_2D_ELLIPTIC = 2
    GAUSS_2D_ROTATED = 3
    CAUCHY_2D_ELLIPTIC = 4
    LINEAR_1D = 5
    FLETCHER_POWELL = 6
    BROWN_DENNIS = 7
    SPLINE_1D = 8
    SPLINE_2D = 9
    SPLINE_3D = 10
    SPLINE_3D_MULTICHANNEL = 11
    SPLINE_3D_PHASE_MULTICHANNEL = 12
    SPLINE_4D = 13
    SPLINE_5D = 14

class EstimatorID:
    LSE = 0
    MLE = 1

class ConstraintType:
    FREE = 0
    LOWER = 1
    UPPER = 2
    LOWER_UPPER = 3

class Status:
    Ok = 0
    Error = 1

def _valid_id(cls, id):
    properties = [key for key in cls.__dict__.keys() if not key.startswith('__')]
    values = [cls.__dict__[key] for key in properties]
    return id in values

def fit(data, weights, model_id, initial_parameters, tolerance=None, max_number_iterations=None, \
        parameters_to_fit=None, estimator_id=None, user_info=None):
    """
    Calls the C interface fit function in the library.
    (see also )

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

    # call fit_constrained without any constraints
    return fit_constrained(data, weights, model_id, initial_parameters, tolerance=tolerance,
                           max_number_iterations=max_number_iterations, parameters_to_fit=parameters_to_fit,
                           estimator_id=estimator_id, user_info=user_info)


def fit_constrained(data, weights, model_id, initial_parameters, constraints=None, constraint_types=None,
                    tolerance=None, max_number_iterations=None, \
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
    :param constraints: Constraint bounds intervals - NumPy array of dimension [number_fits, 2*number_parameters] and data type np.float32
    :param constraint_types: Types of constraints for all parameters (including fixed parameters) - NumPy array of length number_parameters and type np.int32 or None (means no constraints) with values from class ConstraintType
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
        raise RuntimeError('dimension mismatch between data and weights')
        # the unequal operator checks, type, length and content (https://docs.python.org/3.7/reference/expressions.html#value-comparisons)

    # size check: initial parameters is 2D and read number of parameters
    if initial_parameters.ndim != 2:
        raise RuntimeError('initial_parameters is not two-dimensional')
    number_parameters = initial_parameters.shape[1]
    if initial_parameters.shape[0] != number_fits:
        raise RuntimeError('dimension mismatch in number of fits between data and initial_parameters')

    # size check: constraints is 2D and number of fits, 2x number of parameters if given
    if constraints is not None:
        if constraints.ndim != 2:
            raise RuntimeError('constraints not two-dimensional')
        if constraints.shape != (number_fits, 2*number_parameters):
            raise RuntimeError('constraints array has invalid shape')

    # size check: constraint_types has certain length (if given)
    if constraint_types is not None and constraint_types.shape[0] != number_parameters:
        raise RuntimeError('constraint_types should have length of number of parameters')

    # size check: consistency with parameters_to_fit (if given)
    if parameters_to_fit is not None and parameters_to_fit.shape[0] != number_parameters:
        raise RuntimeError(
            'dimension mismatch in number of parameters between initial_parameters and parameters_to_fit')

    # default value constraint types
    if constraint_types is None:
        constraint_types = np.full(number_parameters, ConstraintType.FREE, dtype=np.int32)

    # default value: tolerance
    if tolerance is None:
        tolerance = 1e-4

    # default value: max_number_iterations
    if max_number_iterations is None:
        max_number_iterations = 25

    # default value: estimator ID
    if estimator_id is None:
        estimator_id = EstimatorID.LSE

    # default value: parameters_to_fit
    if parameters_to_fit is None:
        parameters_to_fit = np.ones(number_parameters, dtype=np.int32)

    # now only weights and user_info could be not given

    # type check: data, weights (if given), initial_parameters, constraints (if given) are all np.float32
    if data.dtype != np.float32:
        raise RuntimeError('type of data is not np.float32')
    if weights is not None and weights.dtype != np.float32:
        raise RuntimeError('type of weights is not np.float32')
    if initial_parameters.dtype != np.float32:
        raise RuntimeError('type of initial_parameters is not np.float32')
    if constraints is not None and constraints.dtype != np.float32:
        raise RuntimeError('type of constraints is not np.float32')

    # type check: parameters_to_fit, constraint_types is np.int32
    if parameters_to_fit.dtype != np.int32:
        raise RuntimeError('type of parameters_to_fit is not np.int32')
    if constraint_types.dtype != np.int32:
        raise RuntimeError('type of constraint_types is not np.int32')

    # type check: valid model, estimator id, constraint_types
    if not _valid_id(ModelID, model_id):
        raise RuntimeError('Invalid model ID, use an attribute of ModelID')
    if not _valid_id(EstimatorID, estimator_id):
        raise RuntimeError('Invalid estimator ID, use an attribute of EstimatorID')
    if not all(_valid_id(ConstraintType, constraint_type) for constraint_type in constraint_types):
        raise RuntimeError('Invalid constraint type, use an attribute of ConstraintType')

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
        weights_p = weights.ctypes.data_as(cpufit_func.argtypes[3])
    else:
        weights_p = None
    if constraints is not None:
        constraints_p = constraints.ctypes.data_as(cpufit_func.argtypes[6])
    else:
        constraints_p = None
    if user_info is not None:
        user_info_p = user_info.ctypes.data_as(cpufit_func.argtypes[13])
    else:
        user_info_p = None

    # call into the library (measure time)
    t0 = time.perf_counter()
    status = cpufit_func(
        cpufit_func.argtypes[0](number_fits), \
        cpufit_func.argtypes[1](number_points), \
        data.ctypes.data_as(cpufit_func.argtypes[2]), \
        weights_p, \
        cpufit_func.argtypes[4](model_id), \
        initial_parameters.ctypes.data_as(cpufit_func.argtypes[5]), \
        # constraints_p, \
        # constraint_types.ctypes.data_as(cpufit_func.argtypes[7]), \
        cpufit_func.argtypes[6](tolerance), \
        cpufit_func.argtypes[7](max_number_iterations), \
        parameters_to_fit.ctypes.data_as(cpufit_func.argtypes[8]), \
        cpufit_func.argtypes[9](estimator_id), \
        cpufit_func.argtypes[10](user_info_size), \
        user_info_p, \
        parameters.ctypes.data_as(cpufit_func.argtypes[12]), \
        states.ctypes.data_as(cpufit_func.argtypes[13]), \
        chi_squares.ctypes.data_as(cpufit_func.argtypes[14]), \
        number_iterations.ctypes.data_as(cpufit_func.argtypes[15]))
    t1 = time.perf_counter()

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
