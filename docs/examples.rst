===============
Examples in C++
===============

Example programs, written in C++, are part of the Gpufit project and can be built and run through the project environment.  
Here the example programs are described and the important steps in each program are highlighted.

Please note, that additionally, the C++ boost tests, and the Gpufit/Cpufit performance comparison test, may also be of value 
as example code demonstrating the use of Gpufit. However, a detailed description of these tests programs is not provided.

.. _c-example-simple:

Simple example (minimal call to :code:`gpufit()`)
-------------------------------------------------

This example demonstrates a simple, minimal program containing all of the required parameters for a call to the Gpufit function.  The example is contained
in the file Simple_Example.cpp_ and it can be built and executed within the project environment. Please note that it this code does not actually do anything other than 
make a single call to gpufit().

In the first section of the code, the *model ID* is set, memory space for initial parameters and data values is allocated, the *fit tolerance* is set, the *maximum number of iterations* is set, 
the *estimator ID* is set, and the *parameters to fit array* is initialized.  Note that in most applications, the data array will already exist and it will be unnecessary to allocate additional
space for data.  In this example, the *parameters to fit* array is initialized to all ones, indicating that all model parameters should be adjusted in the fit.

.. code-block:: cpp

    // number of fits, number of points per fit
    size_t const n_fits = 10;
    size_t const n_points_per_fit = 10;

    // model ID and number of model parameters
    int const model_id = GAUSS_1D;
    size_t const n_model_parameters = 5;

    // initial parameters
    std::vector< float > initial_parameters(n_fits * n_model_parameters);

    // data
    std::vector< float > data(n_points_per_fit * n_fits);

    // tolerance
    float const tolerance = 0.001f;

    // maximum number of iterations
    int const max_number_iterations = 10;

    // estimator ID
    int const estimator_id = LSE;

    // parameters to fit (all of them)
    std::vector< int > parameters_to_fit(n_model_parameters, 1);


In the next section of code, sufficient memory is allocated for the *fit results*, *output states*, *chi-square*, and *number of iterations* arrays. 

.. code-block:: cpp

    // output parameters
    std::vector< float > output_parameters(n_fits * n_model_parameters);
    std::vector< int > output_states(n_fits);
    std::vector< float > output_chi_square(n_fits);
    std::vector< int > output_number_iterations(n_fits);

Finally, a call to the C interface of Gpufit is made.  In this example, the optional inputs *weights* and *user_info* are not used.  The program 
then checks the return status from Gpufit.  If an error occurred, the last error message is obtained and an exception is thrown.

.. code-block:: cpp

    // call to gpufit (C interface)
    int const status = gpufit
        (
            n_fits,
            n_points_per_fit,
            data.data(),
            0,
            model_id,
            initial_parameters.data(),
            tolerance,
            max_number_iterations,
            parameters_to_fit.data(),
            estimator_id,
            0,
            0,
            output_parameters.data(),
            output_states.data(),
            output_chi_square.data(),
            output_number_iterations.data()
        );

    // check status
    if (status != STATUS_OK)
    {
        throw std::runtime_error(gpufit_get_last_error());
    }

In summary, the above example illustrates the basic details of the parameters which are passed to the :code:`gpufit()` function, such
as the size of the input and output variables, etc.  This example could be adapted for real applications by:

- choosing a model ID
- choosing an estimator ID
- setting the fit tolerance and maximum number of iterations
- using a data variable containing the data values to be fit
- providing initial parameters with suitable estimates of the true parameters
- processing the output data

In the following sections, examples are provided in which Gpufit is used to fit simulated datasets.

.. _c-example-2d-gaussian:

Example of 2D Gaussian fits
---------------------------

This example demonstrates the use of Gpufit to fit a dataset consisting of 2D Gaussian peaks.  The example is contained
in the file Gauss_Fit_2D_Example.cpp_ and it can be built and executed within the project environment.  The optional
inputs to gpufit(), *weights* and *user_info*, are not used.

This example features:

- Noisy data and random initial guesses for the fit parameters
- Use of the maximum likelihood estimator which is appropriate for data subject to Poisson noise

In this example, a set of simulated data is generated, consisting of 10\ :sup:`4` individual Gaussian peaks, with a size of 30 x 30 points.  
Random noise is added to the data.  The model function and the model parameters are described in :ref:`gauss-2d`.

In this example the true parameters used to generate the Gaussian data are defined in the following code block.

.. code-block:: cpp

    // true parameters
    std::vector< float > true_parameters{ 10.f, 14.5f, 14.5f, 3.f, 10.f}; // amplitude, center x/y positions, width, offset

These parameters define a 2D Gaussian peak centered at the middle of the grid (position 14.5, 14.5), with a width (standard deviation) of 3.0, an amplitude of 10
and a background of 10.  Note that, since we are not providing the independent variables (X values) in the call to Gpufit, the X and Y coordinates of the first 
data point are assumed to be 0.0, and increasing linearly from this point (i.e. :math:`0, 1, 2, ...`).

The guesses for the initial parameters are drawn from the true parameters with a uniformly distributed deviation
of about 20%. The initial guesses for the center coordinates are chosen with a deviation relative to the width of the Gaussian.

.. code-block:: cpp

    // initial parameters (randomized)
    std::vector< float > initial_parameters(n_fits * n_model_parameters);
    for (size_t i = 0; i < n_fits; i++)
    {
        for (size_t j = 0; j < n_model_parameters; j++)
        {
            if (j == 1 || j == 2)
            {
                initial_parameters[i * n_model_parameters + j] = true_parameters[j] + true_parameters[3] * (-0.2f + 0.4f * uniform_dist(rng));
            }
            else
            {
                initial_parameters[i * n_model_parameters + j] = true_parameters[j] * (0.8f + 0.4f*uniform_dist(rng));
            }
        }
    }

The 2D grid of *X* and *Y* values (each ranging from 0 to 29 with an increment of 1) is computed using a double for loop.

.. code-block:: cpp

    // generate x and y values
    std::vector< float > x(n_points_per_fit);
    std::vector< float > y(n_points_per_fit);
    for (size_t i = 0; i < size_x; i++)
    {
        for (size_t j = 0; j < size_x; j++) {
            x[i * size_x + j] = static_cast<float>(j);
            y[i * size_x + j] = static_cast<float>(i);
        }
    }

Next, a 2D Gaussian peak function (without noise) is calculated, once, using the true parameters.

.. code-block:: cpp

    void generate_gauss_2d(
        std::vector<float> const & x_coordinates,
        std::vector<float> const & y_coordinates,
        std::vector<float> const & gauss_params,
        std::vector<float> & output_values)
    {
        // Generates a Gaussian 2D function at a set of X and Y coordinates.  The Gaussian is defined by
        // an array of five parameters.

        // x_coordinates: Vector of X coordinates.
        // y_coordinates: Vector of Y coordinates.
        // gauss_params:  Vector of function parameters.
        // output_values: Output vector containing the values of the Gaussian function at the
        //                corresponding X, Y coordinates.

        // gauss_params[0]: Amplitude
        // gauss_params[1]: Center X position
        // guass_params[2]: Center Y position
        // gauss_params[3]: Gaussian width (standard deviation)
        // gauss_params[4]: Baseline offset

        // This code assumes that x_coordinates.size == y_coordinates.size == output_values.size

        for (size_t i = 0; i < x_coordinates.size(); i++)
        {

            float arg = -((x_coordinates[i] - gauss_params[1]) * (x_coordinates[i] - gauss_params[1])
                    + (y_coordinates[i] - gauss_params[2]) * (y_coordinates[i] - gauss_params[2]))
                    / (2.f * gauss_params[3] * gauss_params[3]);

            output_values[i] = gauss_params[0] * exp(arg) + gauss_params[4];

        }
    }

The variable temp_gauss is used to store the values of the Gaussian peak.  This variable is then used
as a template to generate a set of Gaussian peaks with random, Poisson-distributed noise.

.. code-block:: cpp

    // generate data with noise
    std::vector< float > temp_gauss(n_points_per_fit);
    // compute the model function
    generate_gauss_2d(x, y, true_parameters.begin(), temp_gauss);

    std::vector< float > data(n_fits * n_points_per_fit);
    for (size_t i = 0; i < n_fits; i++)
    {
        // generate Poisson random numbers
        for (size_t j = 0; j < n_points_per_fit; j++)
        {
            std::poisson_distribution< int > poisson_dist(temp_gauss[j]);
            data[i * n_points_per_fit + j] = static_cast<float>(poisson_dist(rng));
        }
    }

Thus, in this example, the data for each fit differs only in the random noise. This, and the
randomized initial guesses for each fit, result in each fit returning slightly different best-fit parameters.

Next, the model and estimator IDs are set, corresponding to the 2D Gaussian fit model function, and the MLE estimator.

.. code-block:: cpp

    // estimator ID
    int const estimator_id = MLE;

    // model ID
    int const model_id = GAUSS_2D;

Next, the gpufit function is called via the :ref:`c-interface`. Parameters weights, user_info and user_info_size are set to 0, 
indicating that they are not used in this example.

.. code-block:: cpp

    // call to gpufit (C interface)
    int const status = gpufit
        (
            n_fits,
            n_points_per_fit,
            data.data(),
            0,
            model_id,
            initial_parameters.data(),
            tolerance,
            max_number_iterations,
            parameters_to_fit.data(),
            estimator_id,
            0,
            0,
            output_parameters.data(),
            output_states.data(),
            output_chi_square.data(),
            output_number_iterations.data()
        );

    // check status
    if (status != STATUS_OK)
    {
        throw std::runtime_error(gpufit_get_last_error());
    }

After the fits are complete, the return value is checked to ensure that no error occurred.  

Output statistics
+++++++++++++++++

The last part of this example obtains statistics describing the fit results, and testing whether the fits converged, etc.

The output_states variable contains a state code which indicates whether the fit converged, or if an error occured 
(see the Gpufit API documentation, :ref:`api-output-parameters`, for details).  In this example, a histogram of all possible fit states 
is obtained by iterating over the state of each fit.

.. code-block:: cpp

    // get fit states
    std::vector< int > output_states_histogram(5, 0);
    for (std::vector< int >::iterator it = output_states.begin(); it != output_states.end(); ++it)
    {
        output_states_histogram[*it]++;
    }

In computing the mean and standard deviation of the results, only the converged fits are taken into account. The following code 
contains an example of the calculation of the means of the output parameters, iterating over all fits and all model parameters.

.. code-block:: cpp

    // compute mean of fitted parameters for converged fits
    std::vector< float > output_parameters_mean(n_model_parameters, 0);
    for (size_t i = 0; i != n_fits; i++)
    {
        if (output_states[i] == STATE_CONVERGED)
        {
            for (size_t j = 0; j < n_model_parameters; j++)
            {
                output_parameters_mean[j] += output_parameters[i * n_model_parameters + j];
            }
        }
    }
    // normalize
    for (size_t j = 0; j < n_model_parameters; j++)
    {
        output_parameters_mean[j] /= output_states_histogram[0];
    }

In summary, the above example illustrates a simple call to :code:`gpufit()` using a real dataset.  When the fit is complete, the 
fit results are obtained and the output states are checked.  Additionally, this example calculates some basic statistics 
describing the results.  The code also illustrates how the input and output parameters are organized in memory.

.. _linear-regression-example:	

Linear Regression Example
-------------------------

This example demonstrates the use of Gpufit to compute linear fits to a randomly generated dataset.  The example is contained
in the file Linear_Regression_Example.cpp_ and it can be built and executed within the project environment.  This example
illustrates how independent variables may be used in the fitting process, by taking advantage of the user_info parameter.  
In this example, a set of 10\ :sup:`4` individual fits are calculated.  Each simulated dataset consists of 20 randomly generated 
data values.  The *X* coordinates of the data points do not have a uniform spacing, but increase non-linearly. 
The user information data is used to pass the *X* values to :code:`gpufit()`.  The fits are unweighted, and the model function 
and model parameters are described in :ref:`linear-1d`.

For details of how user_info is used to store the values of the independent variable for this fit model function, 
see the section of the Gpufit documentation describing the model functions, :ref:`fit-model-functions`.

This example features:

- Noisy data and random initial guesses for the parameters
- Unequally spaced *X* position values, passed to :code:`gpufit()` using the user_info parameter.

The following code illustrates how the *X* positions of the data points are stored in the user_info variable, for this
model function. The user_info points at a vector of float values. Note, however, that the way in which user_info
is used by a model function may vary from function to function.

.. code-block:: cpp

    // custom x positions for the data points of every fit, stored in user_info
    std::vector< float > user_info(n_points_per_fit);
    for (size_t i = 0; i < n_points_per_fit; i++)
    {
        user_info[i] = static_cast<float>(pow(2, i));
    }

    // size of user_info in bytes
    size_t const user_info_size = n_points_per_fit * sizeof(float);

By providing the data coordinates for only one fit in user_info, the model function will use the same coordinates for
all fits in the dataset, as described in :ref:`fit-model-functions`.  

In the next section, the initial parameters for each fit are set to random values, uniformly distributed around the true parameter value.

.. code-block:: cpp

    // true parameters
    std::vector< float > true_parameters { 5, 2 }; // offset, slope

    // initial parameters (randomized)
    std::vector< float > initial_parameters(n_fits * n_model_parameters);
    for (size_t i = 0; i != n_fits; i++)
    {
        // random offset
        initial_parameters[i * n_model_parameters + 0] = true_parameters[0] * (0.8f + 0.4f * uniform_dist(rng));
        // random slope
        initial_parameters[i * n_model_parameters + 1] = true_parameters[0] * (0.8f + 0.4f * uniform_dist(rng));
    }

The data is then generated as the value of a linear function plus additive, normally distributed, random noise.

.. code-block:: cpp

    // generate data
    std::vector< float > data(n_points_per_fit * n_fits);
    for (size_t i = 0; i != data.size(); i++)
    {
        size_t j = i / n_points_per_fit; // the fit
        size_t k = i % n_points_per_fit; // the position within a fit

        float x = user_info[k];
        float y = true_parameters[0] + x * true_parameters[1];
        data[i] = y + normal_dist(rng);
    }

In the following code, the model and estimator IDs for the fit are initialized.

.. code-block:: cpp

    // estimator ID
    int const estimator_id = LSE;

    // model ID
    int const model_id = LINEAR_1D;

Finally, a call is made to :code:`gpufit()` (:ref:`c-interface`).  The weights parameter is set to 0, indicating that 
the fits are unweighted.

.. code-block:: cpp

    // call to gpufit (C interface)
    int const status = gpufit
        (
            n_fits,
            n_points_per_fit,
            data.data(),
            0,
            model_id,
            initial_parameters.data(),
            tolerance,
            max_number_iterations,
            parameters_to_fit.data(),
            estimator_id,
            user_info_size,
            reinterpret_cast< char * >( user_info.data() ),
            output_parameters.data(),
            output_states.data(),
            output_chi_square.data(),
            output_number_iterations.data()
        );

After the fits have been executed and the return value is checked to ensure that no error occurred, statistics 
describing the fit results are calculated and displayed, as in the previous example (see `Output statistics`_).
