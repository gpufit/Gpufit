.. _external-bindings:

=================
External bindings
=================

This sections describes the Gpufit bindings to other programming languages. The bindings (e.g. to Python or Matlab) aim to
emulate the :ref:`c-interface` as closely as possible.

Most high level languages feature multidimensional numerical arrays. In the bindings implemented for Matlab and Python,
we adopt the convention that the input data should be organized as a 2D array, with one dimension corresponding to the
number of data points per fit, and the other corresponding to the number of fits. Internally, in memory, these arrays should
always be ordered such that the data values for each fit are kept together. In Matlab, for example, this means storing the
data in an array with dimensions [number_points_per_fit, number_fits]. In this manner, the data in memory is ordered in the
same way that is expected by the Gpufit C interface, and there is no need to copy or otherwise re-organize the data
before passing it to the GPU. The same convention is used for the weights, the initial model parameters, and the output parameters.

Unlike the C interface, the external bindings to not require the number of fits and the number of data points per fit to be 
specified explicitly. Instead, these numbers are inferred from the dimensions of the 2D input arrays.

Optional parameters with default values
---------------------------------------

The external bindings make some input parameters optional. The optional parameters are shown here.

:tolerance:
    default value 1e-4
:max_n_iterations:
    default value 25 iterations
:estimator_id:
    the default estimator is LSE as defined in constants.h_
:parameters_to_fit:
    by default all parameters are fit

For instructions on how to specify these parameters explicitly, see the sections below.
	
Python
------

The Gpufit binding for Python is a project named pyGpufit. This project contains a Python package named pygpufit, which
contains a module gpufit, and this module implements a method called fit. Calling this method is equivalent to
calling the C interface function :code:`gpufit()` of the Gpufit library. The package expects the input data to be
stored as NumPy array. NumPy follows row-major order by default.

Installation
++++++++++++

Wheel files for Python 2.X and 3.X on Windows 32/64 bit are included in the binary package. NumPy is required.

Install the wheel file with.

.. code-block:: bash

    pip install --no-index --find-links=LocalPathToWheelFile pyGpufit

Python Interface
++++++++++++++++

fit
...

The signature of the fit method (equivalent to calling the C interface function :code:`gpufit()`) is

.. code-block:: python

    def fit(data, weights, model_id:ModelID, initial_parameters, tolerance:float=None, max_number_iterations:int=None, parameters_to_fit=None, estimator_id:EstimatorID=None, user_info=None):

Optional parameters are passed in as None. The numbers of points, fits and parameters is deduced from the dimensions of
the input data and initial parameters arrays.

*Input parameters*

:data: Data
    2D NumPy array of shape (number_fits, number_points) and data type np.float32
:weights: Weights
    2D NumPy array of shape (number_fits, number_points) and data type np.float32 (same as data)

    :special: None indicates that no weights are available
:tolerance: Fit tolerance

    :type: float
    :special: If None, the default value will be used.
:max_number_iterations: Maximal number of iterations

    :type: int
    :special: If None, the default value will be used.
:estimator_id: estimator ID

    :type: EstimatorID which is an Enum in the same module and defined analogously to constants.h_.
    :special: If None, the default value is used.
:model_id: model ID

    :type: ModelID which is an Enum in the same module and defined analogously to constants.h_.
:initial_parameters: Initial parameters
    2D NumPy array of shape (number_fits, number_parameter)

    :array data type: np.float32
:parameters_to_fit: parameters to fit
    1D NumPy array of length number_parameter
    A zero indicates that this parameter should not be fitted, everything else means it should be fitted.

    :array data type: np.int32
    :special: If None, the default value is used.
:user_info: user info
    1D NumPy array of arbitrary type. The length in bytes is deduced automatically.

    :special: If None, no user_info is assumed.

*Output parameters*

:parameters: Fitted parameters for each fit
    2D NumPy array of shape (number_fits, number_parameter) and data type np.float32
:states: Fit result states for each fit
    1D NumPy array of length number_parameter of data type np.int32
    As defined in constants.h_:
:chi_squares: :math:`\chi^2` values for each fit
    1D NumPy array of length number_parameter of data type np.float32
:n_iterations: Number of iterations done for each fit
    1D NumPy array of length number_parameter of data type np.int32
:time: Execution time of call to fit
    In seconds.

Errors are raised if checks on parameters fail or if the execution of fit failed.

get_last_error
..............

The signature of the get_last_error method (equivalent to calling the C interface function *gpufit_get_last_error*) is

.. code-block:: python

    def get_last_error():

Returns a string representing the error message of the last occurred error.

cuda_available
..............

The signature of the cuda_available method (equivalent to calling the C interface function *gpufit_cuda_available*) is

.. code-block:: python

    def cuda_available():

Returns True if CUDA is available and False otherwise.

get_cuda_version
................

The signature of the get_cuda_version method (equivalent to calling the C interface function *gpufit_get_cuda_version*) is

.. code-block:: python

    def get_cuda_version():

*Output parameters*

:runtime version: Tuple of (Major version, Minor version)
:driver version: Tuple of (Major version, Minor version)

An error is raised if the execution failed (i.e. because CUDA is not available).

Python Examples
+++++++++++++++

2D Gaussian peak example
........................

An example can be found at `Python Gauss2D example`_. It is equivalent to :ref:`c-example-2d-gaussian`.

The essential imports are:

.. code-block:: python

    import numpy as np
    import pygpufit.gpufit as gf


First we test for availability of CUDA as well as CUDA driver and runtime versions.

.. code-block:: python

    # cuda available checks
    print('CUDA available: {}'.format(gf.cuda_available()))
    print('CUDA versions runtime: {}, driver: {}'.format(*gf.get_cuda_version()))

The true parameters describing an example 2D Gaussian peak functions are:

.. code-block:: python

    # true parameters
    true_parameters = np.array((10, 5.5, 5.5, 3, 10), dtype=np.float32)

A 2D grid of x and y positions can conveniently be generated using the np.meshgrid function:

.. code-block:: python

    # generate x and y values
    g = np.arange(size_x)
    yi, xi = np.meshgrid(g, g, indexing='ij')
    xi = xi.astype(np.float32)
    yi = yi.astype(np.float32)

Using these positions and the true parameter values a model function can be calculated as

.. code-block:: python

    def generate_gauss_2d(p, xi, yi):
        """
        Generates a 2D Gaussian peak.
        http://gpufit.readthedocs.io/en/latest/api.html#gauss-2d

        :param p: Parameters (amplitude, x,y center position, width, offset)
        :param xi: x positions
        :param yi: y positions
        :return: The Gaussian 2D peak.
        """

        arg = -(np.square(xi - p[1]) + np.square(yi - p[2])) / (2*p[3]*p[3])
        y = p[0] * np.exp(arg) + p[4]

        return y

The model function can be repeated and noise can be added using the np.tile and np.random.poisson functions.

.. code-block:: python

    # generate data
    data = generate_gauss_2d(true_parameters, xi, yi)
    data = np.reshape(data, (1, number_points))
    data = np.tile(data, (number_fits, 1))

    # add Poisson noise
    data = np.random.poisson(data)
    data = data.astype(np.float32)

The model and estimator IDs can be set as

.. code-block:: python

    # estimator ID
    estimator_id = gf.EstimatorID.MLE

    # model ID
    model_id = gf.ModelID.GAUSS_2D

When all input parameters are set we can call the C interface of Gpufit.

.. code-block:: python

    # run Gpufit
    parameters, states, chi_squares, number_iterations, execution_time = gf.fit(data, None, model_id, initial_parameters, tolerance, max_number_iterations, None, estimator_id, None)

And finally statistics about the results of the fits can be displayed where the mean and standard deviation of the
fitted parameters are limited to those fits that converged.

.. code-block:: python

    # print fit results

    # get fit states
    converged = states == 0
    number_converged = np.sum(converged)
    print('ratio converged         {:6.2f} %'.format(number_converged / number_fits * 100))
    print('ratio max it. exceeded  {:6.2f} %'.format(np.sum(states == 1) / number_fits * 100))
    print('ratio singular hessian  {:6.2f} %'.format(np.sum(states == 2) / number_fits * 100))
    print('ratio neg curvature MLE {:6.2f} %'.format(np.sum(states == 3) / number_fits * 100))
    print('ratio gpu not read      {:6.2f} %'.format(np.sum(states == 4) / number_fits * 100))

    # mean, std of fitted parameters
    converged_parameters = parameters[converged, :]
    converged_parameters_mean = np.mean(converged_parameters, axis=0)
    converged_parameters_std = np.std(converged_parameters, axis=0)

    for i in range(number_parameters):
        print('p{} true {:6.2f} mean {:6.2f} std {:6.2f}'.format(i, true_parameters[i], converged_parameters_mean[i], converged_parameters_std[i]))

    # print summary
    print('model ID: {}'.format(model_id))
    print('number of fits: {}'.format(number_fits))
    print('fit size: {} x {}'.format(size_x, size_x))
    print('mean chi_square: {:.2f}'.format(np.mean(chi_squares[converged])))
    print('iterations: {:.2f}'.format(np.mean(number_iterations[converged])))
    print('time: {:.2f} s'.format(execution_time))

	
Matlab
------

The Matlab binding for Gpufit is a Matlab script (gpufit.m_). This script checks the input data, sets default parameters, and
calls the C interface of the Gpufit library, via a compiled .mex file.

Please note, that before using the Matlab binding, the path to gpufit.m_ must be added to the Matlab path.

If other GPU-based computations are to be performed with Matlab in the same session, please use the Matlab GPU computing 
functionality first (for example with a call to gpuDevice or gpuArray) before calling the Gpufit Matlab binding. If this is not
done, Matlab will throw an error (Error using gpuArray An unexpected error occurred during CUDA execution. 
The CUDA error was: cannot set while device is active in this process).

Matlab Interface
++++++++++++++++

gpufit
......

Optional parameters are passed in as empty matrices (``[]``). The numbers of points, fits and parameters is deduced from the dimensions of
the input data and initial parameters matrices.

The signature of the gpufit function is

.. code-block:: matlab

    function [parameters, states, chi_squares, n_iterations, time] = gpufit(data, weights, model_id, initial_parameters, tolerance, max_n_iterations, parameters_to_fit, estimator_id, user_info)

*Input parameters*

:data: Data
    2D matrix of size [number_points, number_fits] and data type single
:weights: Weights
    2D matrix of size [number_points, number_fits] and data type single (same as data)

    :special: None indicates that no weights are available
:tolerance: Fit tolerance

    :type: single
    :special: If empty ([]), the default value will be used.
:max_number_iterations: Maximal number of iterations
    Will be converted to int32 if necessary

    :special: If empty ([]), the default value will be used.
:estimator_id: estimator ID

    :type: EstimatorID which is defined in EstimatorID.m analogously to constants.h_.
    :special: If empty ([]), the default value is used.
:model_id: model ID

    :type: ModelID which is defined in ModelID.m analogously to constants.h_.
:initial_parameters: Initial parameters
    2D matrix of size: [number_parameter, number_fits]

    :type: single
:parameters_to_fit: parameters to fit
    vector of length number_parameter, will be converted to int32 if necessary
    A zero indicates that this parameter should not be fitted, everything else means it should be fitted.

    :special: If empty ([]), the default value is used.
:user_info: user info
    vector of arbitrary type. The length in bytes is deduced automatically.

*Output parameters*

:parameters: Fitted parameters for each fit
    2D matrix of size: [number_parameter, number_fits] of data type single
:states: Fit result states for each fit
    vector of length number_parameter of data type int32
    As defined in constants.h_:
:chi_squares: :math:`\chi^2` values for each fit
    vector of length number_parameter of data type single
:n_iterations: Number of iterations done for each fit
    vector of length number_parameter of data type int32
:time: Execution time of call to gpufit
    In seconds.

Errors are raised if checks on parameters fail or if the execution of gpufit fails.

gpufit_cuda_available
.....................

The signature of the gputfit_cuda_available method (equivalent to calling the C interface function *gpufit_cuda_available*) is

.. code-block:: matlab

    function r = gpufit_cuda_available():

Returns True if CUDA is available and False otherwise.

Matlab Examples
+++++++++++++++

Simple example
..............

The most simple example is the `Matlab simple example`_. It is equivalent to :ref:`c-example-simple` and additionally
relies on default values for optional arguments.

2D Gaussian peak example
........................

An example can be found at `Matlab Gauss2D example`_. It is equivalent to :ref:`c-example-2d-gaussian`.

The true parameters describing an example 2D Gaussian peak functions are:

.. code-block:: matlab

    % true parameters
    true_parameters = single([10, 5.5, 5.5, 3, 10]);

A 2D grid of x and y positions can conveniently be generated using the ndgrid function:

.. code-block:: matlab

    % generate x and y values
    g = single(0 : size_x - 1);
    [x, y] = ndgrid(g, g);

Using these positions and the true parameter values a model function can be calculated as

.. code-block:: matlab

    function g = gaussian_2d(x, y, p)
    % Generates a 2D Gaussian peak.
    % http://gpufit.readthedocs.io/en/latest/api.html#gauss-2d
    %
    % x,y - x and y grid position values
    % p - parameters (amplitude, x,y center position, width, offset)

    g = p(1) * exp(-((x - p(2)).^2 + (y - p(3)).^2) / (2 * p(4)^2)) + p(5);

    end

The model function can be repeated and noise can be added using the repmat and poissrnd functions.

.. code-block:: matlab

    % generate data with Poisson noise
    data = gaussian_2d(x, y, true_parameters);
    data = repmat(data(:), [1, number_fits]);
    data = poissrnd(data);

The model and estimator IDs can be set as

.. code-block:: matlab

    % estimator id
    estimator_id = EstimatorID.MLE;

    % model ID
    model_id = ModelID.GAUSS_2D;

When all input parameters are set we can call the C interface of the Gpufit library.

.. code-block:: matlab

    %% run Gpufit
    [parameters, states, chi_squares, n_iterations, time] = gpufit(data, [], model_id, initial_parameters, tolerance, max_n_iterations, [], estimator_id, []);

And finally statistics about the results of the fits can be displayed where the mean and standard deviation of the
fitted parameters are limited to those fits that converged.

.. code-block:: matlab

    %% displaying results

    % get fit states
    converged = states == 0;
    number_converged = sum(converged);
    fprintf(' ratio converged         %6.2f %%\n', number_converged / number_fits * 100);
    fprintf(' ratio max it. exceeded  %6.2f %%\n', sum(states == 1) / number_fits * 100);
    fprintf(' ratio singular hessian  %6.2f %%\n', sum(states == 2) / number_fits * 100);
    fprintf(' ratio neg curvature MLE %6.2f %%\n', sum(states == 3) / number_fits * 100);
    fprintf(' ratio gpu not read      %6.2f %%\n', sum(states == 4) / number_fits * 100);

    % mean and std of fitted parameters
    converged_parameters = parameters(:, converged);
    converged_parameters_mean = mean(converged_parameters, 2);
    converged_parameters_std  = std(converged_parameters, [], 2);
    for i = 1 : number_parameters
        fprintf(' p%d true %6.2f mean %6.2f std %6.2f\n', i, true_parameters(i), converged_parameters_mean(i), converged_parameters_std(i));
    end

    % print summary
    fprintf('model ID: %d\n', model_id);
    fprintf('number of fits: %d\n', number_fits);
    fprintf('fit size: %d x %d\n', size_x, size_x);
    fprintf('mean chi-square: %6.2f\n', mean(chi_squares(converged)));
    fprintf('iterations: %6.2f\n', mean(n_iterations(converged)));
    fprintf('time: %6.2f s\n', time);
