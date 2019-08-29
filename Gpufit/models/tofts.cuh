#ifdef USE_TOFTS
#define GPUFIT_TOFTS_CUH_INCLUDED

__device__ void calculate_tofts (               // function name
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

	// integral/convolution
	REAL convFunc = 0;
	for (int i = 1; i < point_index; i++) {
		REAL spacing = T[i] - T[i - 1];
		REAL Ct = Cp[i] * exp(-parameters[0] * (T[point_index]-T[i]) / parameters[1]);
		REAL Ctprev = Cp[i - 1] * exp(-parameters[0] * (T[point_index]-T[i-1]) / parameters[1]);
		convFunc += ((Ct + Ctprev) / 2 * spacing);
	}

	value[point_index] = parameters[0] * convFunc;                    // formula calculating fit model values
	// C(t)		       =   Ktrans	   * trapz(Cp(k)*e^(-Ktrans*t/ve))

	/////////////////////////// derivative ///////////////////////////
	REAL * current_derivative = derivative + point_index;

	// formula calculating derivative values with respect to parameters[0] (Ktrans)
	//current_derivative[0 * n_points] = convFunc;
	REAL derivativeFunction = 0;
	for (int i = 1; i < point_index; i++) {
		REAL spacing = T[i] - T[i - 1];
		REAL Ct = Cp[i] * (1-parameters[0]/parameters[1]*(T[point_index]-T[i])) * exp(-parameters[0] * (T[point_index]-T[i]) / parameters[1]);
		REAL Ctprev = Cp[i - 1] * (1-parameters[0]/parameters[1]*(T[point_index]-T[i-1])) * exp(-parameters[0] * (T[point_index]-T[i-1]) / parameters[1]);
		derivativeFunction += ((Ct + Ctprev) / 2 * spacing);
	}
	current_derivative[0 * n_points] = derivativeFunction;

	// formula calculating derivative values with respect to parameters[1] (Ve)
	derivativeFunction = 0;
	for (int i = 1; i < point_index; i++) {
		REAL spacing = T[i] - T[i - 1];
		REAL Ct = Cp[i] * (T[point_index]-T[i]) * exp(-parameters[0] * (T[point_index]-T[i]) / parameters[1]);
		REAL Ctprev = Cp[i - 1] * (T[point_index]-T[i-1]) * exp(-parameters[0] * (T[point_index]-T[i-1]) / parameters[1]);
		derivativeFunction += ((Ct + Ctprev) / 2 * spacing);
	}
	current_derivative[1 * n_points] = pow(parameters[0],2)/pow(parameters[1],2)*derivativeFunction;

	// deallocate pointers
	delete T;
	delete Cp;
}
#endif
