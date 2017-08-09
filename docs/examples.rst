========
Examples
========

C++ Examples_ are part of the library code base and can be built and run through the project environment. Here they are
described and important steps are highlighted.

Please note, that additionally, the C++ Tests_ contained in the code base also demonstrate the usage of |GF|. However, a
detailed description of the tests is not provided.

.. _c-example-simple:

Simple skeleton example
-----------------------

This example shows the minimal code providing all required parameters and the call to the C interface. It is contained
in Simple_Example.cpp_ and can be built and executed within the project environment. Please note, that it this code does
not do anything other than call gpufit().

In the first section of the code, the model ID is set, space for initial parameters and data values is reserved (in a normal
application, however, the data array would already exist), the fit tolerance is set, the maximal number of iterations is set, 
the estimator ID is set, and the parameters to fit array is initialized to indicate that all parameters should be fit.

.. code-block:: cpp

	// number of fits, number of points per fit
	size_t const number_fits = 10;
	size_t const number_points = 10;

	// model ID and number of parameter
	int const model_id = GAUSS_1D;
	size_t const number_parameters = 5;

	// initial parameters
	std::vector< float > initial_parameters(number_fits * number_parameters);

	// data
	std::vector< float > data(number_points * number_fits);

	// tolerance
	float const tolerance = 0.001f;

	// maximal number of iterations
	int const max_number_iterations = 10;

	// estimator ID
	int const estimator_id = LSE;

	// parameters to fit (all of them)
	std::vector< int > parameters_to_fit(number_parameters, 1);

In a next step, sufficient memory is reserved for all four output parameters.

.. code-block:: cpp

	// output parameters
	std::vector< float > output_parameters(number_fits * number_parameters);
	std::vector< int > output_states(number_fits);
	std::vector< float > output_chi_square(number_fits);
	std::vector< int > output_number_iterations(number_fits);

Finally, there is a call to the C interface of Gpufit (in this example, the optional 
inputs *weights* and *user info* are not used) and a check of the return status.
If an error occurred, the last error message is obtained and an exception is thrown.

.. code-block:: cpp

	// call to gpufit (C interface)
	int const status = gpufit
        (
            number_fits,
            number_points,
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

This simple example can easily be adapted to real applications by:

- choosing your own model ID
- choosing your own estimator ID
- choosing your own fit tolerance and maximal number of iterations
- filling the data structure with the data values to be fitted
- filling the initial parameters structure with suitable estimates of the true parameters
- processing the output data

The following two examples show |GF| can be used to fit real data.

.. _c-example-2d-gaussian:

Fit 2D Gaussian functions example
---------------------------------

This example features:

- Multiple fits using a 2D Gaussian function
- Noisy data and random initial guesses for the fit parameters
- A Poisson noise adapted maximum likelihood estimator

It is contained in Gauss_Fit_2D_Example.cpp_ and can be built and executed within the project environment.  The optional 
inputs to gpufit(), *weights* and *user info*, are not used.

In this example, a 2D Gaussian curve is fit to 10\ :sup:`4` noisy data sets having a size of 20 x 20 points each.
The model function and the model parameters are described in :ref:`gauss-2d`.

In this example the true parameters used to generate the Gaussian data are set to

.. code-block:: cpp

    // true parameters
	std::vector< float > true_parameters{ 10.f, 9.5f, 9.5f, 3.f, 10.f}; // amplitude, center x/y positions, width, offset

which defines a 2D Gaussian peak centered at the middle of the grid (position 9.5, 9.5), with a width (standard deviation) of 3.0, an amplitude of 10
and a background of 10.

The guesses for the initial parameters are drawn from the true parameters with a uniformly distributed deviation
of about 20%. The initial guesses for the center coordinates are chosen with a deviation relative to the width of the Gaussian.

.. code-block:: cpp

	// initial parameters (randomized)
	std::vector< float > initial_parameters(number_fits * number_parameters);
	for (size_t i = 0; i < number_fits; i++)
	{
		for (size_t j = 0; j < number_parameters; j++)
		{
			if (j == 1 || j == 2)
			{
				initial_parameters[i * number_parameters + j] = true_parameters[j] + true_parameters[3]  * (-0.2f + 0.4f * uniform_dist(rng));
			}
			else
			{
				initial_parameters[i * number_parameters + j] = true_parameters[j] * (0.8f + 0.4f*uniform_dist(rng));
			}
		}
	}

The 2D grid of x and y values (each ranging from 0 to 19 with an increment of 1) is computed with a double for loop.

.. code-block:: cpp

	// generate x and y values
	std::vector< float > x(number_points);
	std::vector< float > y(number_points);
	for (size_t i = 0; i < size_x; i++)
	{
		for (size_t j = 0; j < size_x; j++) {
			x[i * size_x + j] = static_cast<float>(j);
			y[i * size_x + j] = static_cast<float>(i);
		}
	}

Then a 2D Gaussian peak model function (without noise) is calculated once for the true parameters

.. code-block:: cpp

    void generate_gauss_2d(std::vector<float> &x, std::vector<float> &y, std::vector<float> &g, std::vector<float>::iterator &p)
    {
        // generates a Gaussian 2D peak function on a set of x and y values with some paramters p (size 5)
        // we assume that x.size == y.size == g.size, no checks done

        // given x and y values and parameters p computes a model function g
        for (size_t i = 0; i < x.size(); i++)
        {
            float arg = -((x[i] - p[1]) * (x[i] - p[1]) + (y[i] - p[2]) * (y[i] - p[2])) / (2 * p[3] * p[3]);
            g[i] = p[0] * exp(arg) + p[4];
        }
    }

Stored in variable temp, it is then used in every fit to generate Poisson distributed random numbers.

.. code-block:: cpp

	// generate data with noise
	std::vector< float > temp(number_points);
	// compute the model function
	generate_gauss_2d(x, y, temp, true_parameters.begin());

	std::vector< float > data(number_fits * number_points);
	for (size_t i = 0; i < number_fits; i++)
	{
		// generate Poisson random numbers
		for (size_t j = 0; j < number_points; j++)
		{
			std::poisson_distribution< int > poisson_dist(temp[j]);
			data[i * number_points + j] = static_cast<float>(poisson_dist(rng));
		}
	}

Thus, in this example the difference between data for each fit only in the random noise.  This, and the 
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
            number_fits,
            number_points,
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
	std::vector< float > output_parameters_mean(number_parameters, 0);
	for (size_t i = 0; i != number_fits; i++)
	{
		if (output_states[i] == STATE_CONVERGED)
		{
			for (size_t j = 0; j < number_parameters; j++)
			{
				output_parameters_mean[j] += output_parameters[i * number_parameters + j];
			}
		}
	}
	// normalize
	for (size_t j = 0; j < number_parameters; j++)
	{
		output_parameters_mean[j] /= output_states_histogram[0];
	}

.. _linear-regression-example:
	
Linear Regression Example
-------------------------

This example features:

- Multiple fits of a 1D Linear curve
- Noisy data and random initial guesses for the parameters
- Unequal spaced x position values given as custom user info

It is contained in Linear_Regression_Example.cpp_ and can be built and executed within the project environment.

In this example, a straight line is fitted to 10\ :sup:`4` noisy data sets. Each data set includes 20 data points.
Locations of data points are scaled non-linear (exponentially). The user information given implicates the x positions of the data
sets. The fits are unweighted and the model function and the model parameters are described in :ref:`linear-1d`.

The custom x positions of the linear model are stored in the user_info.

.. code-block:: cpp

	// custom x positions for the data points of every fit, stored in user info
	std::vector< float > user_info(number_points);
	for (size_t i = 0; i < number_points; i++)
	{
		user_info[i] = static_cast<float>(pow(2, i));
	}

	// size of user info in bytes
	size_t const user_info_size = number_points * sizeof(float);

Because only number_points values are specified, this means that the same custom x position values are used for every fit.

The initial parameters for every fit are set to random values uniformly distributed around the true parameter value.

.. code-block:: cpp

	// true parameters
	std::vector< float > true_parameters { 5, 2 }; // offset, slope

	// initial parameters (randomized)
	std::vector< float > initial_parameters(number_fits * number_parameters);
	for (size_t i = 0; i != number_fits; i++)
	{
		// random offset
		initial_parameters[i * number_parameters + 0] = true_parameters[0] * (0.8f + 0.4f * uniform_dist(rng));
		// random slope
		initial_parameters[i * number_parameters + 1] = true_parameters[0] * (0.8f + 0.4f * uniform_dist(rng));
	}

The data is generated as the value of a linear function and some additive normally distributed noise term.

.. code-block:: cpp

	// generate data
	std::vector< float > data(number_points * number_fits);
	for (size_t i = 0; i != data.size(); i++)
	{
		size_t j = i / number_points; // the fit
		size_t k = i % number_points; // the position within a fit

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
            number_fits,
            number_points,
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
