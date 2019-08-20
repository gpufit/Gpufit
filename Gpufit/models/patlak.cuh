#ifndef GPUFIT_PATLAK_CUH_INCLUDED
#define GPUFIT_PATLAK_CUH_INCLUDED
#define REAL float


__device__ void calculate_patlak (               // function name
	REAL const * parameters,
	int const n_fits,
	int const n_points,
	REAL * value,
	REAL * derivative,
	int const point_index,						 // k
	int const fit_index,
	int const chunk_index,
	char * user_info,							 // likely contains time and Cp values in 1 dimensional array
	std::size_t const user_info_size)
{
	// indices
	REAL* user_info_float = (REAL*)user_info;
	//REAL x = 0;
	//if (user_info_size / sizeof(REAL) == n_points) {				// unnecessary since this case is always valid for this model? and setting independent variables is below.
	//	x = user_info_float[point_index];
	//}

	///////////////////////////// value //////////////////////////////

	// split user_info array into time and Cp
	REAL *T = new REAL[n_points];
	for (int i = 0; i < n_points - 1; i++)
		T[i] = user_info_float[i];

	REAL *Cp = new REAL[n_points];
	for (int i = n_points - 1; i < 2 * n_points - 1; i++)
		Cp[i] = user_info_float[i];

	// integral (trapezoidal rule)
	REAL area = 0;
	for (int i = 1; i < point_index; i++) {				// or is point_index i's limit? ya probably
		// 
		REAL spacing = T[i] - T[i - 1];
		area += Cp[i] * spacing + 0.5 * (Cp[i] - Cp[i - 1]);
	}
	delete[] T;
	delete[] Cp;

	value[point_index] = parameters[0] * area + parameters[1] * Cp[point_index];                      // formula calculating fit model values
	//	C(t)		   =   Ktrans	   * trapz(Cp(k))  + vp   *    Cp(k)

	/////////////////////////// derivative ///////////////////////////
	REAL * current_derivative = derivative + point_index;

	current_derivative[0 * n_points] = Cp[point_index];			// formula calculating derivative values with respect to parameters[0] (Ktrans)
	current_derivative[1 * n_points] = Cp[point_index];			// formula calculating derivative values with respect to parameters[1] (vp)
}
#endif