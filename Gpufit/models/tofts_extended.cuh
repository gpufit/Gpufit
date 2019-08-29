#ifdef USE_TOFTS_EXTENDED
#define GPUFIT_TOFTS_EXTENDED_CUH_INCLUDED

__device__ void calculate_tofts_extended (               // function name
	REAL const * parameters,
	int const n_fits,
	int const n_points,
	REAL * value,
	REAL * derivative,
	int const point_index,						 
	int const fit_index,
	int const chunk_index,
	char * user_info,							 // contains time and Cp values in 1 dimensional array
	std::size_t const user_info_size)
{
	// indices
	REAL* user_info_float = (REAL*)user_info;

	///////////////////////////// value //////////////////////////////

	// split user_info array into time and Cp
	REAL* T = user_info_float;
	REAL* Cp = user_info_float + n_points;

	// integral (trapezoidal rule)
	REAL convFunc = 0;
	for (int i = 1; i < point_index; i++) {
		convFunc += Cp[i] * exp((-parameters[0] / parameters[1]) * (T[i] - T[i - 1]));
	}

	value[point_index] = parameters[0] * convFunc * parameters[2] * Cp[point_index];                      // formula calculating fit model values
	// C(t)		       =   Ktrans	   * trapz(Cp(k)*e^(-Ktrans*t/ve) * vp * Cp(k)
	/////////////////////////// derivative ///////////////////////////
	REAL * current_derivative = derivative + point_index;

	current_derivative[0 * n_points] = convFunc;				// formula calculating derivative values with respect to parameters[0] (Ktrans)
	current_derivative[1 * n_points] = Cp[point_index];			// formula calculating derivative values with respect to parameters[1] (ve)
	current_derivative[2 * n_points] = 1;						// formula calculating derivative values with respect to parameters[2] (vp)

	// deallocate pointers
	delete T;
	delete Cp;
}
#endif
