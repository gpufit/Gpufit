========
Examples
========

C++ Examples_ are part of the library codebase and can be built and run through the project environment. Here the examples are
described and important steps are highlighted.

Please note, that additionally, the C++ Tests_ included in the codebase may also be of value as examples demonstrating the 
use of Gpufit. However, a detailed description of the the test code is not provided.

.. _c-example-simple:

Simple skeleton example
-----------------------

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

This simple example can be adapted for real applications by:

- choosing a model ID
- choosing an estimator ID
- setting the fit tolerance and maximum number of iterations
- using a data variable containing the data values to be fit
- providing initial parameters with suitable estimates of the true parameters
- processing the output data

In the following sections, examples are provided in which Gpufit is used to fit simulated datasets.

.. _c-example-2d-gaussian:

Fit 2D Gaussian functions example
---------------------------------

This example demonstrates the use of Gpufit to fit a dataset consisting of 2D Gaussian peaks.  The example is contained
in the file Gauss_Fit_2D_Example.cpp_ and it can be built and executed within the project environment.  The optional
inputs to gpufit(), *weights* and *user_info*, are not used.

This example features:

- Noisy data and random initial guesses for the fit parameters
- Use of the maximum likelihood estimator which is appropriate for data subject to Poisson noise

In this example, a set of simulated data is generated, consisting of 10\ :sup:`4` 2D Gaussian peaks, with a size of 30 x 30 points.  
Random noise is added to the data.  The model function and the model parameters are described in :ref:`gauss-2d`.

In this example the true parameters used to generate the Gaussian data are defined in the following code block.

.. code-block:: cpp

    // true parameters
	std::vector< float > true_parameters{ 10.f, 15.5f, 15.5f, 3.f, 10.f}; // amplitude, center x/y positions, width, offset

These parameters define a 2D Gaussian peak centered at the middle of the grid (position 15.5, 15.5), with a width (standard deviation) of 3.0, an amplitude of 10
and a background of 10.

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

The 2D grid of x and y values (each ranging from 0 to 49 with an increment of 1) is computed with a double for loop.

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

Then a 2D Gaussian peak model function (without noise) is calculated once for the true parameters

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
			
			float arg = -(   (x_coordinates[i] - gauss_params[1]) * (x_coordinates[i] - gauss_params[1]) 
						   + (y_coordinates[i] - gauss_params[2]) * (y_coordinates[i] - gauss_params[2])   ) 
						 / (2.f * gauss_params[3] * gauss_params[3]);
						 
			output_values[i] = gauss_params[0] * exp(arg) + gauss_params[4];
			
		}
	}

Stored in variable temp, it is then used in every fit to generate Poisson distributed random numbers.

.. code-block:: cpp

	// generate data with noise
	std::vector< float > temp(n_points_per_fit);
	// compute the model function
	generate_gauss_2d(x, y, temp, true_parameters.begin());

	std::vector< float > data(n_fits * n_points_per_fit);
	for (size_t i = 0; i < n_fits; i++)
	{
		// generate Poisson random numbers
		for (size_t j = 0; j < n_points_per_fit; j++)
		{
			std::poisson_distribution< int > poisson_dist(temp[j]);
			data[i * n_points_per_fit + j] = static_cast<float>(poisson_dist(rng));
		}
	}

Thus, in this example the difference between data for each fit only in the random noise. This, and the
randomized initial guesses for each fit, result in each fit returning slightly different best-fit parameters.

We set the model and estimator IDs for the fit accordingly.

.. code-block:: cpp

	// estimator ID
	int const estimator_id = MLE;

	// model ID
	int const model_id = GAUSS_2D;

And call the gpufit :ref:`c-interface`. Parameters weights, user_info and user_info_size are set to 0, indicating that they
won't be used during the fits.

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

After the fits have been executed and the return value is checked to ensure that no error occurred, some statistics
about the fits are displayed.

Output statistics
+++++++++++++++++

A histogram of all possible fit states (see :ref:`api-output-parameters`) is obtained by iterating over the state of each fit.

.. code-block:: cpp

	// get fit states
	std::vector< int > output_states_histogram(5, 0);
	for (std::vector< int >::iterator it = output_states.begin(); it != output_states.end(); ++it)
	{
		output_states_histogram[*it]++;
	}

In the computation of the mean and standard deviation only converged fits are taken into account. Here is an example of computing
the means of the output parameters iterating over all fits and all parameters.

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

.. _linear-regression-example:
	
Linear Regression Example
-------------------------

This example features:

- Noisy data and random initial guesses for the parameters
- Unequal spaced x position values given as custom user_info

It is contained in Linear_Regression_Example.cpp_ and can be built and executed within the project environment.

In this example, a straight line is fitted to 10\ :sup:`4` noisy data sets. Each data set includes 20 data points.
Locations of data points are scaled non-linear (exponentially). The user information given implicates the x positions of the data
sets. The fits are unweighted and the model function and the model parameters are described in :ref:`linear-1d`.

The custom x positions of the linear model are stored in the user_info.

.. code-block:: cpp

	// custom x positions for the data points of every fit, stored in user_info
	std::vector< float > user_info(n_points_per_fit);
	for (size_t i = 0; i < n_points_per_fit; i++)
	{
		user_info[i] = static_cast<float>(pow(2, i));
	}

	// size of user_info in bytes
	size_t const user_info_size = n_points_per_fit * sizeof(float);

Because only n_points_per_fit values are specified, this means that the same custom x position values are used for every fit.

The initial parameters for every fit are set to random values uniformly distributed around the true parameter value.

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

The data is generated as the value of a linear function and some additive normally distributed noise term.

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

We set the model and estimator IDs for the fit accordingly.

.. code-block:: cpp

	// estimator ID
	int const estimator_id = LSE;

	// model ID
	int const model_id = LINEAR_1D;

And call the gpufit :ref:`c-interface`. Parameter weights is set to 0, indicating that they won't be used during the fits.

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

After the fits have been executed and the return value is checked to ensure that no error occurred, some statistics
about the fits are displayed (see `Output statistics`_).
