.. _gpufit-customization:

=============
Customization
=============

This sections explains how to add custom fit model functions and custom fit estimators within the Gpufit library.
Functions calculating the estimator and model values are CUDA device functions that are defined in CUDA header files
using the C syntax. For each function and estimator there exists a separate file. Therefore, to add an additional model
or estimator a new CUDA header file containing the new model or estimator function must be created and included in the
library.

Please note, that in order to add a model function or estimator, it is necessary to rebuild the Gpufit library 
from source. In future releases of Gpufit, it may be possible to include new fit functions or estimators at runtime.


Add a new fit model function
----------------------------

To add a new fit model, the model function itself as well as analytic expressions for its partial derivatives 
must to be known. A function calculating the values of the model as well as a function calculating the
values of the partial derivatives of the model, with respect to the model parameters and possible grid 
coordinates, must be implemented.

Additionally, a new model ID must be defined and included in the list of available model IDs, and the number 
of model parameters and dimensions must be specified as well.

Detailed step by step instructions for adding a model function are given below.

1. Define an additional model ID in file constants.h_. When using the language bindings, the model ID must also be added for Python (in gpufit.py), Matlab (in ModelID.m), Java (in Model.java). 
2. Implement a CUDA device function within a newly created .cuh file in folder Gpufit/Gpufit/models according to the following template.

.. code-block:: cuda

    __device__ void ... (               // ... = function name
        float const * parameters,
        int const n_fits,
        int const n_points,
        float * value,
        float * derivative,
        int const point_index,
        int const fit_index,
        int const chunk_index,
        char * user_info,
        std::size_t const user_info_size)
    {
        ///////////////////////////// value //////////////////////////////

        value[point_index] = ... ;                      // formula calculating fit model values

        /////////////////////////// derivative ///////////////////////////
        float * current_derivative = derivative + point_index;

        current_derivative[0 * n_points] = ... ;  // formula calculating derivative values with respect to parameters[0]
        current_derivative[1 * n_points] = ... ;  // formula calculating derivative values with respect to parameters[1]
        .
        .
        .
    }

This code can be used as a pattern, where the placeholders ". . ." must be replaced by user code which calculates model
function values and partial derivative values of the model function for a particular set of parameters. See for example linear_1d.cuh_.

3.	Include the newly created .cuh file in models.cuh_
4.	Add a switch case in the CUDA device function ``calculate_model()`` in file models.cuh_ to allow calling the newly added model function.

.. code-block:: cpp

    switch (model_id)
    {
    case GAUSS_1D:
        calculate_gauss1d(parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
        break;
        .
        .
        .
    case ...:                       // model ID
        ...                         // function name
            (parameters, n_fits, n_points, value, derivative, point_index, fit_index, chunk_index, user_info, user_info_size);
        break;
        .
        .
        .
    default:
        break;
    }


5.	Add a switch case in function ``configure_model()`` in file models.cuh_.

.. code-block:: cpp

    switch (model_id)
    {
    case GAUSS_1D:              n_parameters = 4; n_dimensions = 1; break;
    .
    .
    .
    case ...:                   // model ID
        n_parameters = ...;     // number of model parameters
        n_dimensions = ...;     // number of model dimensions
        break;

    default:                                                        break;
    }

6.	After adding a new model function, if CMake is being used to configure the compiler, then CMake must be run again.  If not using CMake, the new model function file (the .cuh file) must be included in the project.
7.	Re-build the Gpufit project.

Add a new fit estimator
------------------------

To extend the Gpufit library with additional estimators, three CUDA device functions must be defined and integrated. The sections requiring modification are
the functions which calculate the estimator function values, and its gradient and hessian values. Also, a new estimator ID must be defined.
Detailed step by step instructions for adding an additional estimator is given below.

1. Define an additional estimator ID in constants.h_ When using the language bindings, the estimator ID must be added also for Python (in gpufit.py), Matlab (in EstimatorID.m), Java (in Estimator.java). 
2. Implement three functions within a newly created .cuh file in the folder Gpufit/Gpufit/estimators calculating :math:`\chi^2` values and
   its gradient and hessian according to the following template.

.. code-block:: cuda

    ///////////////////////////// Chi-square /////////////////////////////
    __device__ void ... (           // ... = function name Chi-square
        volatile float * chi_square,
        int const point_index,
        float const * data,
        float const * value,
        float const * weight,
        int * state,
        char * user_info,
        std::size_t const user_info_size)
    {
        chi_square[point_index] = ... ;            // formula calculating Chi-square summands
    }

    ////////////////////////////// gradient //////////////////////////////
    __device__ void ... (           // ... = function name gradient of Chi-square
        volatile float * gradient,
        int const point_index,
        int const parameter_index,
        float const * data,
        float const * value,
        float const * derivative,
        float const * weight,
        char * user_info,
        std::size_t const user_info_size)
    {
        gradient[point_index] = ... ;           // formula calculating summands of the gradient of Chi-square
                                                // model derivates are stored in derivative[parameter_index]
    }

    ////////////////////////////// hessian ///////////////////////////////
    __device__ void ... (           // function name hessian
        double * hessian,
        int const point_index,
        int const parameter_index_i,
        int const parameter_index_j,
        float const * data,
        float const * value,
        float const * derivative,
        float const * weight,
        char * user_info,
        std::size_t const user_info_size)
    {
        *hessian += ... ;            // formula calculating summands of the hessian of Chi-square
    }

This code can be used as a pattern, where the placeholders ". . ." must be replaced by user code which calculates the estimator
and the gradient and hessian values of the estimator given. For a concrete example, see lse.cuh_.

3. Include the newly created .cuh file in estimators.cuh_.

.. code-block:: cpp

    #include "....cuh"              // filename

4. Add a switch case in three CUDA device functions in the file estimators.cuh_.

  4a. Calculation of Chi-square:

    .. code-block:: cuda

        switch (estimator_id)
        {
        case LSE:
            calculate_chi_square_lse(chi_square, point_index, data, value, weight, state, user_info, user_info_size);
            break;
            .
            .
            .
        case ...:           // estimator ID
            ...             // function name Chi-square
                (chi_square, point_index, data, value, weight, state, user_info, user_info_size);
            break;

        default:
            break;
        }

  4b. Calculation of the gradients of Chi-square:

    .. code-block:: cuda

        switch (estimator_id)
        {
        case LSE:
            calculate_gradient_lse(gradient, point_index, parameter_index, data, value, derivative, weight, user_info, user_info_size);
            break;
            .
            .
            .
        case ...:           // estimator ID
            ...             // function name gradient
                (gradient, point_index, parameter_index, data, value, derivative, weight, user_info, user_info_size);
            break;

        default:
            break;
        }

  4c. Calculation of the Hessian:

    .. code-block:: cuda

        switch (estimator_id)
        {
        case LSE:
            calculate_hessian_lse
                (hessian, point_index, parameter_index_i, parameter_index_j, data, value, derivative, weight, user_info,user_info_size);
            break;
            .
            .
            .
        case ...:           // estimator ID
            ...             // function name hessian
                (hessian, point_index, parameter_index_i, parameter_index_j, data, value, derivative, weight, user_info, user_info_size);
            break;

        default:
            break;
        }

5.	After adding a new estimator, if CMake is being used to configure the compiler, then CMake must be run again. If not using CMake, the new estimator file (the .cuh file) must be included in the project.
6.	Re-build the Gpufit project.

Future releases
---------------

A current disadvantage of the Gpufit library, when compared with established CPU-based curve fitting packages,
is that in order to add or modify a fit model function or a fit estimator, the library must be recompiled.
We anticipate that this limitation can be overcome in future releases of the library, by employing 
run-time compilation of the CUDA code.
