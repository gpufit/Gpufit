.. _api-description:

======================
Gpufit API description
======================

The Gpufit source code compiles to a dynamic-link library (DLL), providing a C interface.
In the sections below, the C interface and its arguments are described in detail.

.. _c-interface:

C Interface
-----------

The C interface is defined in the Gpufit header file: gpufit.h_.

gpufit()
++++++++

This is the main fit function. A single call to the :code:`gpufit()` function executes a block of *N* fits.
The inputs to :code:`gpufit()` are scalars and pointers to arrays, and the outputs are also array pointers.

The inputs to the :code:`gpufit()` function are:

- the number of fits (*N*),
- the number of data points per fit (each fit has equal size),
- the fit data,
- an array of weight values that are used to weight the individual data points in the fit (optional),
- an ID number which specifies the fit model function,
- an array of initial parameters for the model functions,
- a tolerance value which determines when the fit has converged,
- the maximum number of iterations per fit,
- an array of flags which allow one or more fit parameters to be held constant,
- an ID number which specifies the fit estimator (e.g. least squares, etc.),
- the size of the user info data,
- the user info data, which may have multiple uses, for example to pass additional parameters to the fit functions,
  or to include independent variables (e.g. X values) with the fit data.

The outputs of :code:`gpufit()` are:

- the best fit model parameters for each fit,
- an array of flags indicating, for example, whether each fit converged,
- the final value of :math:`\chi^2` for each fit,
- the number of iterations needed for each fit to converge.

The :code:`gpufit()` function call is defined below.

.. code-block:: cpp

    int gpufit
    (
        size_t n_fits,
        size_t n_points,
        float * data,
        float * weights,
        int model_id,
        float * initial_parameters,
        float tolerance,
        int max_n_iterations,
        int * parameters_to_fit,
        int estimator_id,
        size_t user_info_size,
        char * user_info,
        float * output_parameters,
        int * output_states,
        float * output_chi_squares,
        int * output_n_iterations
    ) ;

.. _api-input-parameters:

Description of input parameters
...............................

:n_fits: Number of fits to be performed

    :type: size_t

:n_points: Number of data points per fit

    Gpufit is designed such that each fit must have the same number of data points per fit.

    :type: size_t

:data: Pointer to data values

    A pointer to the data values. The data must be passed in as a 1D array of floating point values, with the data
    for each fit concatenated one after another. In the case of multi-dimensional data, the data must be flattened
    to a 1D array. The number of elements in the array is equal to the product n_fits * n_points.

    :type: float *
    :length: n_points * n_fits

:weights: Pointer to weights

    The weights array includes unique weighting values for each fit. It is used only by the least squares estimator (LSE).
    The size of the weights array and its organization is identical to that for the data array.
    For statistical weighting, this parameter should be set equal to the inverse of the variance of the data
    (i.e. weights = 1.0 / variance ). The weights array is an optional input.

    :type: float *
    :length: n_points * n_fits
    :special: Use a NULL pointer to indicate that no weights are provided. In this case all data values will be weighted equally.

:model_id: Model ID

    Determines the model which is used for all fits in this call. See :ref:`fit-model-functions` for more details.

    As defined in constants.h_:

        :0: GAUSS_1D
        :1: GAUSS_2D
        :2: GAUSS_2D_ELLIPTIC
        :3: GAUSS_2D_ROTATED
        :4: CAUCHY_2D_ELLIPTIC
        :5: LINEAR_1D


    :type: int

:initial_parameters: Pointer to initial parameter values

    A 1D array containing the initial model parameter values for each fit. If the number of parameters of the fit model
    is defined by *n_parameters*, then the size of this array is *n_fits * n_parameters*.
	
    The parameter values for each fit are concatenated one after another. If there are *M* parameters per fit,
    the parameters array is organized as follows: [(parameter 1), (parameter 2), ..., (parameter M), (parameter 1),
    (parameter 2), ..., (parameter M), ...].

    :type: float *
    :length: n_fits * n_parameters

:tolerance: Fit tolerance threshold

    The fit tolerance determines when the fit has converged. After each fit iteration, the change in the absolute value
    of :math:`\chi^2` is calculated. The fit has converged when one of two conditions are met. First, if the change
    in the absolute value of :math:`\chi^2` is less than the tolerance value, the fit has converged.
    Alternatively, if the change in :math:`\chi^2` is less than the product of tolerance and the absolute value of
    :math:`\chi^2` [tolerance * abs(:math:`\chi^2`)], then the fit has converged.
	
    Setting a lower value for the tolerance results in more precise values for the fit parameters, but requires more fit
    iterations to reach convergence.
	
    A typical value for the tolerance settings is between 1.0E-3 and 1.0E-6.

    :type: float

:max_n_iterations: Maximum number of iterations

    The maximum number of fit iterations permitted. If the fit has not converged after this number of iterations,
    the fit returns with a status value indicating that the maximum number of iterations was reached.

    :type: int

:parameters_to_fit: Pointer to array indicating which model parameters should be held constant during the fit

    This is an array of ones or zeros, with a length equal to the number of parameters of the fit model function.
    Each entry in the array is a flag which determines whether or not the corresponding model parameter will be held
    constant during the fit. To allow a parameter to vary during the fit, set the entry in *parameters_to_fit* equal
    to one. To hold the value constant, set the entry to zero.
	
    An array of ones, e.g. [1,1,1,1,1,...] will allow all parameters to vary during the fit.

    :type: int *
    :length: n_parameters

:estimator_id: Estimator ID

    Determines the fit estimator which is used. See :ref:`estimator-functions` for more details.

    As defined in constants.h_:

        :0: LSE
        :1: MLE

    :type: int

:user_info_size: Size of user information data

    Size of the user information data array, in bytes.

    :type: size_t

:user_info: Pointer to user information data

    This parameter is intended to provide flexibility to the Gpufit interface. The user information data is a generic
    block of memory which is passed in to the :code:`gpufit()` function, and which is accessible in shared GPU memory by the
    fit model functions and the estimator functions. Possible uses for the user information data are to pass in values 
    for independent variables (e.g. X values) or to supply additional data to the fit model function or estimator. For 
    a coded example which makes use of the user information data, see :ref:`linear-regression-example`. The user 
    information data is an optional parameter - if no user information is required this parameter may be set to NULL.

    :type: char *
    :length: user_info_size
    :special: Use a NULL pointer to indicate that no user information is available.

.. _api-output-parameters:

Description of output parameters
................................

:output_parameters: Pointer to array of best-fit model parameters

    For each fit, this array contains the best-fit model parameters. The array is organized identically to the input
    parameters array.

    :type: float *
    :length: n_fits * n_parameters

:output_states: Pointer to array of fit result state IDs

    For each fit the result of the fit is indicated by a state ID. The state ID codes are defined below.
    A state ID of 0 indicates that the fit converged successfully.

    As defined in constants.h_:

        :0: The fit converged, tolerance is satisfied, the maximum number of iterations is not exceeded
        :1: Maximum number of iterations exceeded
        :2: During the Gauss-Jordan elimination the Hessian matrix is indicated as singular
        :3: Non-positive curve values have been detected while using MLE (MLE requires only positive curve values)
        :4: State not read from GPU Memory

    :type: int *
    :length: n_fits

:output_chi_squares: Pointer to array of :math:`\chi^2` values

    For each fit, this array contains the final :math:`\chi^2` value, as returned by the estimator function (see :ref:`estimator-functions`). 

    :type: float *
    :length: n_fits

:output_n_iterations: Pointer to array of iteration counts

    For each fit, this array contains the number of fit iterations which were performed. 

    :type: int *
    :length: n_fits

:return value: Status code

    The return value of the function call indicates whether an error occurred. As defined in constants.h_.

    :0: No error
    :-1: Error

gpufit_portable_interface()
+++++++++++++++++++++++++++

This function is a simple wrapper around the :code:`gpufit()` function, providing an alternative means of passing the function parameters.

.. code-block:: cpp

    int gpufit_portable_interface(int argc, void *argv[]);

Description of parameters
.........................

:argc: The length of the argv pointer array

:argv: Array of pointers to *gpufit* parameters, as defined above. For reference, the type of each element of the *argv* array is listed below.

	:argv[0]: Number of fits
	
		:type: size_t *
		
	:argv[1]: Number of points per fit
	
		:type: size_t *	
		
	:argv[2]: Fit data
	
		:type: float *	
		
	:argv[3]: Fit weights
	
		:type: float *	
		
	:argv[4]: Fit model ID
	
		:type: int *	
		
	:argv[5]: Initial parameters
	
		:type: float *	
		
	:argv[6]: Fit tolerance
	
		:type: float *	
		
	:argv[7]: Maximum number of iterations
	
		:type: int *	
		
	:argv[8]: Parameters to fit
	
		:type: int *	
		
	:argv[9]: Fit estimator ID
	
		:type: int *	
		
	:argv[10]: User info size
	
		:type: size_t *	
		
	:argv[11]: User info data
	
		:type: char *	
		
	:argv[12]: Output parameters
	
		:type: float *	
		
	:argv[13]: Output states
	
		:type: int *	
		
	:argv[14]: Output :math:`\chi^2` values
	
		:type: float *	
		
	:argv[15]: Output number of iterations
	
		:type: int *	
	

:return value: This function simply returns the :code:`gpufit()` return status code.

gpufit_get_last_error()
+++++++++++++++++++++++

A function that returns a string representation of the last error.

.. code-block:: cpp

    char const * gpufit_get_last_error();

:return value: Error message corresponding to the most recent error, or an empty string if no error occurred.

    'CUDA driver version is insufficient for CUDA runtime version'
        The graphics driver version installed on the computer is not supported by the CUDA Toolkit version which was used
        to build Gpufit.dll. Update the graphics driver or re-build Gpufit using a compatible CUDA Toolkit version.

gpufit_cuda_available()
+++++++++++++++++++++++

A function that calls a simple CUDA function to check if CUDA is available.

.. code-block:: cpp

    int gpufit_cuda_available();

:return value: Returns 0 if CUDA is not available (no suitable device found, or driver version insufficient).
               Use the function *gpufit_get_last_error()* to check the error message.
               Returns 1 if CUDA is available.
               
gpufit_get_cuda_version()
+++++++++++++++++++++++++

A function that returns the CUDA runtime version in *runtime_version* and the
installed CUDA driver version in *driver_version*.

.. code-block:: cpp

    int gpufit_get_cuda_version(int * runtime_version, int * driver_version);

:runtime_version: Pointer to the CUDA runtime version number. Format is Minor version times 10 plus Major version times 1000.
                  (is 0 if the CUDA runtime version is incompatible with the installed CUDA driver version)

:driver_version: Pointer to the CUDA driver version number. Format is Minor version times 10 plus Major version times 1000.
                 (is 0 if no CUDA enabled graphics card was detected)

:return value: Status code

    The return value of the function call indicates whether an error occurred.

    :0: No error
    :-1: Error. Use the function *gpufit_get_last_error()* to check the error message.



