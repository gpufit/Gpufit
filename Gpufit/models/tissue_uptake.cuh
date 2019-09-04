#ifdef USE_TISSUE_UPTAKE
#define GPUFIT_TISSUE_UPTAKE_CUH_INCLUDED

__device__ REAL get_tuptake_value (
	REAL p0, //Ktrans
	REAL p1, //Ve
	REAL p2, //Vp
	int const point_index,
	REAL const * T,
	REAL const * Cp)
{
	// integral/convolution
	REAL convFunc = 0;
	for (int i = 1; i < point_index; i++) {
		REAL spacing = T[i] - T[i - 1];
		REAL Ct = Cp[i] * exp(-p0 * (T[point_index]-T[i]) / p1);
		REAL Ctprev = Cp[i - 1] * exp(-p0 * (T[point_index]-T[i-1]) / p1);
		convFunc += ((Ct + Ctprev) / 2 * spacing);
	}

	REAL function_value = 1;
	return function_value;
}

__device__ void calculate_tissue_uptake (               // function name
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
//	REAL convCp = 0;
//	for (int i = 1; i < point_index; i++) {
//		REAL spacing = T[i] - T[i - 1];
//		convCp += (Cp[i - 1] + Cp[i]) / 2 * spacing;
//	}

	value[point_index] = get_tuptake_value(parameters[0],parameters[1],parameters[2],point_index,T,Cp);   // formula calculating fit model values
	// C(t)		       =   integral(Cp(k) * Fp*exp(-t/Tp) + Ktrans*(1-exp(-t/Tp)), need some algebra for Tp

	/////////////////////////// derivative ///////////////////////////
	REAL * current_derivative = derivative + point_index;

	current_derivative[0 * n_points] = 1;					// formula calculating derivative values with respect to parameters[0] (Ktrans)
	current_derivative[1 * n_points] = 1;			// formula calculating derivative values with respect to parameters[1] (vp)
	current_derivative[2 * n_points] = 1;			// formula calculating derivative values with respect to parameters[1] (Fp)
}
#endif
