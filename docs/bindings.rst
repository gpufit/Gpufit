.. _external-bindings:

=================
External bindings
=================

This sections describes the Gpufit bindings to other programming languages. The bindings (to Python, Matlab or Java) aim to
emulate the :ref:`c-interface` as closely as possible.

Most high level languages feature multidimensional numeric arrays. In the bindings implemented for Matlab and Python,
we adopt the convention that the input data should be organized as a 2D array, with one dimension corresponding to the
number of data points per fit, and the other corresponding to the number of fits. Internally, in memory, these arrays should
always be ordered such that the data values for each fit are kept together. In Matlab, for example, this means storing the
data in an array with dimensions [number_points_per_fit, number_fits]. In this manner, the data in memory is ordered in the
same way that is expected by the Gpufit C interface, and there is no need to copy or otherwise re-organize the data
before passing it to the GPU. The same convention is used for the weights, the initial model parameters, and the output parameters.

In Java we pre-allocate one dimensional FloatBuffers or IntBuffers for the data and the fit results. The user is responsible
for copying data into these buffers.

Unlike the C interface, the external bindings do not require the number of fits and the number of data points per fit to be 
specified explicitly. Instead, these numbers are inferred from the dimensions of the 2D input arrays.

Optional parameters with default values
---------------------------------------

The external bindings make some input parameters optional. The optional parameters are shown here. They are kept the same
for all bindings.

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

fit_constrained
...............

The :code:`fit_constrained` method is very similar to the :code:`fit` method with the additional possibility to
specify parameter constraints.

The signature of the :code:`fit_constrained` method (equivalent to calling the C interface function :code:`gpufit_constrained()`) is

.. code-block:: python

    def fit_constrained(data, weights, model_id:ModelID, initial_parameters, constraints=None, constraint_types=None, tolerance:float=None, max_number_iterations:int=None, parameters_to_fit=None, estimator_id:EstimatorID=None, user_info=None):

*Constraint input parameters*

:constraints: Constraint bound intervals for every parameter and every fit.
    2D NumPy array of shape (number_fits, 2*number_parameter) and data type np.float32
:contraint_types: Constraint types for every parameter
    1D NumPy array of length number_parameter
    Valid values are defined in gf.ConstraintType

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


2D Gaussian peak constrained fit example
........................................

An example for a constrained fit can be found at `Python Gauss2D constrained fit example`_. It differs from the previous
example only in that constraints are specified additionally (as 2D array of lower and upper bounds on parameters for every
fit) as well as constraint types (for all parameters including fixed parameters) that can take a value of ConstraintType (FREE, LOWER, UPPER or LOWER_UPPER)
in order to either do not enforce the constraints for a parameter or enforce them only at the lower or upper or both bounds.

The following code block demonstrates how the sigma of a 2D Gaussian peak can be constrained to the interval [2.9, 3.1] and the background and ampltiude to non-negative values.

.. code-block:: python

    # set constraints
    constraints = np.zeros((number_fits, 2*number_parameters), dtype=np.float32)
    constraints[:, 6] = 2.9
    constraints[:, 7] = 3.1
    constraint_types = np.array([gf.ConstraintType.LOWER, gf.ConstraintType.FREE, gf.ConstraintType.FREE, gf.ConstraintType.LOWER_UPPER, gf.ConstraintType.LOWER], dtype=np.int32)

    # run constrained Gpufit
    parameters, states, chi_squares, number_iterations, execution_time = gf.fit_constrained(data, None, model_id,
                                                                                initial_parameters, constraints, constraint_types,
                                                                                tolerance, max_number_iterations, None,
                                                                                estimator_id, None)

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

The signature of the :code:`gpufit` function is

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
    vector of suitable type (correct type is not checked and depends on the chosen fit model function or estimator). The length of user_info in bytes is determined automatically.

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

gpufit_constrained
..................

The :code:`gpufit_constrained` function is very similar to the :code:`gpufit` function with the additional possibility to specify
parameter constraints.

The signature of the :code:`gpufit_constrained` function is

.. code-block:: matlab

    function [parameters, states, chi_squares, n_iterations, time] = gpufit_constrained(data, weights, model_id, initial_parameters, constraints, constraint_types, tolerance, max_n_iterations, parameters_to_fit, estimator_id, user_info)

*Constraint input parameters*

:constraints: Constraint bound intervals for every parameter and every fit
    2D matrix of size [2*number_parameter, number_fits] of data type single
:contraint_types: Constraint types for every parameter
    Vector of length number_parameter, will be converted to int32 if necessary.
    Valid values are defined in ConstraintType.m.


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

2D Gaussian peak constrained fit example
........................................

An example for a constrained fit can be found at `Matlab Gauss2D constrained fit example`_. It differs from the previous
example only in that constraints are specified additionally (as 2D array of lower and upper bounds on parameters for every
fit) as well as constraint types (for all parameters including fixed parameters) that can take a value of ConstraintType (FREE, LOWER, UPPER or LOWER_UPPER)
in order to either do not enforce the constraints for a parameter or enforce them only at the lower or upper or both bounds.

The following code block demonstrates how the sigma of a 2D Gaussian peak can be constrained to the interval [2.9, 3.1] and the background and amplitude to non-negative values.

.. code-block:: matlab

    %% set constraints
    constraints = zeros([2*number_parameters, number_fits], 'single');
    constraints(7, :) = 2.9;
    constraints(8, :) = 3.1;
    constraint_types = int32([ConstraintType.LOWER, ConstraintType.FREE, ConstraintType.FREE, ConstraintType.LOWER_UPPER, ConstraintType.LOWER]);

    %% run constrained Gpufit
    [parameters, states, chi_squares, n_iterations, time] = gpufit_constrained(data, [], ...
        model_id, initial_parameters, constraints, constraint_types, tolerance, max_n_iterations, [], estimator_id, []);

Java
----

The Gpufit binding for Java consists of a small adapter C library named GpufitJNI and a Gpufit jar archive containing
a com.github.gpufit package. In these the class Gpufit has static methods largely equivalent to calling the C interface
function :code:`gpufit()` of the Gpufit library. The fit method expects the input to be given as a FitModel instance,
which among other things specifies the model and the estimator as enums. The results are returned as a FitResult instance.

Installation
++++++++++++

Build the Gpufit library and the GpufitJNI library from source as documented in :ref:`installation-and-testing`. Make sure
both libraries are in the Java library path, for example by using the -Djava.library.path comman line switch for the VM.

Build the Gpufit.jar from the sources using Gradle on Gpufit/java/gpufit/build.gradle. Make sure this jar is in the Java
class path of your application, e.g. by adding it as a dependency to your project

Java Interface
++++++++++++++

For a more complete description, see the Javadoc output of the Gpufit Java binding project.

Gpufit.fit
..........

The signature of the fit method (calls the C interface function :code:`gpufit()`) is

.. code-block:: java

    public static FitResult fit(FitModel fitModel, FitResult fitResult)

Input parameters are given as a FitModel, output parameters are stored in a FitResult. A FitResult can be re-used if
the number of fits and the number of parameters of the model didn't change. It must then also be given as second parameter.

*Input of the fit - Filling the FitModel*

.. code-block:: java

    public FitModel(int numberFits, int numberPoints, boolean withWeights, Model model, Float tolerance, Integer maxNumberIterations, Boolean[] parametersToFit, Estimator estimator, int userInfoSize)

:numberFits: Number of fits

:numberPoints: Number of data points per fit

:widthWeights: If true, a buffer for giving weights is pre-allocated, otherwise not

:model: An enum describing the model. See class Model for more information. Naming and id is equivalent to the C code.

:tolerance: Fit tolerance

    :special: If null, the default value will be used.

:maxNumberIterations: Maximal number of iterations

    :special: If null, the default value will be used.

:parametersToFit: Boolean array indicating which parameters should be fitted

    :special: If null, the default value will be used.

:estimator: Enum describing the estimator function. See class Estimator for more information. Naming and id is equivalent
    to the C code.

    :special: If None, the default value is used.

:userInfoSize: The size of the user info (in bytes).

    :special: Must be positive, otherwise the buffer for user info is not pre-allocated.

Afterwards the buffers for data, weights (if desired), initial parameters and user info (if desired) must be filled with
the appropriate content. The internal layout is the same as in the C part of Gpufit, i.e. the data represents an
1D number array of length of number fits times number data points per fit with an order of data points followed one
after another for all fits. In this batch. The initial parameters are number fits times number of parameters in the model
with the parameters for each fit changing fastest and the number of fits slowest.

*Fit output - The FitResult*

Memory for the fit output is either created automatically or a previous instance of FitResult can be reused to avoid
recreation.

.. code-block:: java

    public class FitResult {

        public final FloatBuffer parameters;
        public final IntBuffer states;
        public final FloatBuffer chiSquares;
        public final IntBuffer numberIterations;
        public float fitDuration;

:parameters: Fitted parameters for each fit
:states: Fit result states for each fit
    As defined in constants.h_:
:chi_squares: :math:`\chi^2` values for each fit
:n_iterations: Number of iterations done for each fit
:time: Execution time of call to fit
    In seconds.

Errors are raised if checks on parameters fail or if the execution of fit failed.

Gpufit.getLastError
...................

The signature of the get_last_error method (equivalent to calling the C interface function *gpufit_get_last_error*) is

.. code-block:: java

    public static native String getLastError()

Returns a string representing the error message of the last occurred error.

Gpufit.isCudaAvailable
......................

The signature of the cuda_available method (equivalent to calling the C interface function *gpufit_cuda_available*) is

.. code-block:: java

    public static native boolean isCudaAvailable()

Returns True if CUDA is available and False otherwise.

get_cuda_version
................

The signature of the get_cuda_version method (equivalent to calling the C interface function *gpufit_get_cuda_version*) is

.. code-block:: java

    public static CudaVersion getCudaVersion()

The output is a CudaVersion instance with two simple member variables.

:runtime version: String of "Major version.Minor version"
:driver version: String of "Major version.Minor version"

An error is raised if the execution failed (i.e. because CUDA is not available).

Java Example
++++++++++++

2D Gaussian peak example
........................

An example can be found at `Java Gauss2D example`_. It is equivalent to :ref:`c-example-2d-gaussian`.

First we test for availability of CUDA as well as CUDA driver and runtime versions.

.. code-block:: java

    // print general CUDA information
    System.out.println(String.format("CUDA available: %b", Gpufit.isCudaAvailable()));
    CudaVersion cudaVersion = Gpufit.getCudaVersion();
    System.out.println(String.format("CUDA versions runtime: %s, driver: %s", cudaVersion.runtime, cudaVersion.driver));

The model and estimator IDs can be set as

.. code-block:: Java

    Model model = Model.GAUSS_2D;
    Estimator estimator = Estimator.MLE;

The true parameters describing an example 2D Gaussian peak functions are:

.. code-block:: java

    // true parameters (order: amplitude, center-x, center-y, width, offset)
    float[] trueParameters = new float[]{10, 5.5f, 5.5f, 3, 10};

A 2D grid of x and y positions can conveniently be generated:

.. code-block:: java

    // generate x and y values
    float[] xi = new float[numberPoints];
    float[] yi = new float[numberPoints];
    for (int i = 0; i < sizeX; i++) {
        for (int j = 0; j < sizeX; j++) {
            xi[i * sizeX + j] = i;
            yi[i * sizeX + j] = j;
        }
    }

Using these positions and the true parameter values a model function can be calculated as

.. code-block:: java

    /**
     * Computes a 2D Gaussian peak given x and y values and parameters.
     *
     * See also: http://gpufit.readthedocs.io/en/latest/api.html#gauss-2d
     *
     * @param p Parameter array
     * @param x x values array
     * @param y y values array
     * @return Model values array
     */
    private static float[] generateGauss2D(float[] p, float[] x, float[] y) {
        // checks
        assert(x.length == y.length);
        assert(p.length == 5);

        // calculate data
        float[] data = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            float arg = -((x[i] - p[1]) * (x[i] - p[1]) + (y[i] - p[2]) * (y[i] - p[2])) / (2 * p[3] * p[3]);
            data[i] = p[0] * (float)Math.exp(arg) + p[4];
        }
        return data;
    }

The model function can be repeated and Poisson noise can be added.

.. code-block:: java

    // generate data
    float[] gauss2D = generateGauss2D(trueParameters, xi, yi);
    float[] data = new float[numberFits * numberPoints];
    for (int i = 0; i < numberFits; i++) {
        System.arraycopy(gauss2D, 0, data, i * numberPoints, numberPoints);
    }

    // add Poisson noise
    for (int i = 0; i < numberFits * numberPoints; i++) {
        data[i] = nextPoisson(data[i], rand);
    }

A FitModel containing all the input data including copying the data values from an array to a Java buffer can be done via

.. code-block:: java

    // assemble FitModel
    FitModel fitModel = new FitModel(numberFits, numberPoints, false, model, tolerance, maxNumberIterations, null, estimator, 0);

    // fill data and initial parameters in the fit model
    fitModel.data.clear();
    fitModel.data.put(data);
    fitModel.initialParameters.clear();
    fitModel.initialParameters.put(initialParameters);


When all input parameters are set we can call the C interface of Gpufit.

.. code-block:: java

    // fun Gpufit
    FitResult fitResult = Gpufit.fit(fitModel);

And finally statistics about the results of the fits can be displayed where the mean and standard deviation of the
fitted parameters are limited to those fits that converged.

.. code-block:: java

    // count FitState outcomes and get a list of those who converged
    boolean[] converged = new boolean[numberFits];
    int numberConverged = 0, numberMaxIterationExceeded = 0, numberSingularHessian = 0, numberNegativeCurvatureMLE = 0;
    for (int i = 0; i < numberFits; i++) {
        FitState fitState = FitState.fromID(fitResult.states.get(i));
        converged[i] = fitState.equals(FitState.CONVERGED);
        switch (fitState) {
            case CONVERGED:
                numberConverged++;
                break;
            case MAX_ITERATIONS:
                numberMaxIterationExceeded++;
                break;
            case SINGULAR_HESSIAN:
                numberSingularHessian++;
                break;
            case NEG_CURVATURE_MLE:
                numberNegativeCurvatureMLE++;
        }
    }

    // get mean and std of converged parameters
    float [] convergedParameterMean = new float[]{0, 0, 0, 0, 0};
    float [] convergedParameterStd = new float[]{0, 0, 0, 0, 0};
    for (int i = 0; i < numberFits; i++) {
        for (int j = 0; j < model.numberParameters; j++) {
            if (converged[i]) {
                convergedParameterMean[j] += fitResult.parameters.get(i * model.numberParameters + j);
            }
        }
    }
    for (int i = 0; i < model.numberParameters; i++) {
        convergedParameterMean[i] /= numberConverged;
    }
    for (int i = 0; i < numberFits; i++) {
        for (int j = 0; j < model.numberParameters; j++) {
            if (converged[i]) {
                float dev = fitResult.parameters.get(i * model.numberParameters + j) - convergedParameterMean[j];
                convergedParameterStd[j] += dev * dev;
            }
        }
    }
    for (int i = 0; i < model.numberParameters; i++) {
        convergedParameterStd[i] = (float)Math.sqrt(convergedParameterStd[i] / numberConverged);
    }

    // print fit results
    System.out.println("*Gpufit*");
    System.out.println(String.format("Model: %s", model.name()));
    System.out.println(String.format("Number of fits: %d", numberFits));
    System.out.println(String.format("Fit size: %d x %d", sizeX, sizeX));
    System.out.println(String.format("Mean Chi²: %.2f", meanFloatBuffer(fitResult.chiSquares, converged)));
    System.out.println(String.format("Mean  number iterations: %.2f", meanIntBuffer(fitResult.numberIterations, converged)));
    System.out.println(String.format("Time: %.2fs", fitResult.fitDuration));
    System.out.println(String.format("Ratio converged: %.2f %%", (float) numberConverged / numberFits * 100));
    System.out.println(String.format("Ratio max it. exceeded: %.2f %%", (float) numberMaxIterationExceeded / numberFits * 100));
    System.out.println(String.format("Ratio singular Hessian: %.2f %%", (float) numberSingularHessian / numberFits * 100));
    System.out.println(String.format("Ratio neg. curvature MLE: %.2f %%", (float) numberNegativeCurvatureMLE / numberFits * 100));

    System.out.println("\nParameters of 2D Gaussian peak");
    for (int i = 0; i < model.numberParameters; i++) {
        System.out.println(String.format("parameter %d, true: %.2f, mean %.2f, std: %.2f", i, trueParameters[i], convergedParameterMean[i], convergedParameterStd[i]));
    }
