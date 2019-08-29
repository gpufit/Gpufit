#ifdef USE_PATLAK
#define GPUFIT_PATLAK_CUH_INCLUDED

__device__ void calculate_patlak (               // function name
	REAL const * parameters,
	int const n_fits,
	int const n_points,
	REAL * value,
	REAL * derivative,
	int const point_index,						 
	int const fit_index,
	int const chunk_index,
	char * user_info,							 // contains time and Cp values in a 1 dimensional array
	std::size_t const user_info_size)
{
	// indices
	REAL* user_info_float = (REAL*)user_info;

	///////////////////////////// value //////////////////////////////

	// split user_info array into time and Cp
	REAL* T = user_info_float;
	REAL* Cp = user_info_float + n_points;

	// integral (trapezoidal rule)
	REAL convCp = 0;
	for (int i = 1; i < point_index; i++) {
		REAL spacing = T[i] - T[i - 1];
		convCp += (Cp[i - 1] + Cp[i]) / 2 * spacing;
	}

	value[point_index] = parameters[0] * convCp + parameters[1] * Cp[point_index];                      // formula calculating fit model values
	//	C(t)		   =   Ktrans	   * trapz(Cp(k))  + vp     *    Cp(k)

	/////////////////////////// derivative ///////////////////////////
	REAL * current_derivative = derivative + point_index;

	current_derivative[0 * n_points] = convCp;					// formula calculating derivative values with respect to parameters[0] (Ktrans)
	current_derivative[1 * n_points] = Cp[point_index];			// formula calculating derivative values with respect to parameters[1] (vp)

	// deallocate pointers
	delete T;
	delete Cp;
}
#endif
