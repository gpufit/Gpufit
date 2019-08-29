#ifdef USE_TOFTS_EXTENDED
#define GPUFIT_TOFTS_EXTENDED_CUH_INCLUDED

__device__ REAL get_value (
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

	REAL function_value = p0 * convFunc + p2 * Cp[point_index];
	return function_value;
}

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
	REAL const * T = user_info_float;
	REAL const * Cp = user_info_float + n_points;

	// formula calculating fit model values
	value[point_index] = get_value(parameters[0],parameters[1],parameters[2],point_index,T,Cp);


	/////////////////////////// derivative ///////////////////////////
	REAL * current_derivative = derivative + point_index;
	bool numericalApproximationDerivative = true;

	if (numericalApproximationDerivative)
	{
		REAL h = 10e-5;
		REAL f_plus_h;
		REAL f_minus_h;
		//3 point method

		// parameters[0]' = (Ktrans)'
		f_plus_h = get_value(parameters[0]+h,parameters[1],parameters[2],point_index,T,Cp);
		f_minus_h = get_value(parameters[0]-h,parameters[1],parameters[2],point_index,T,Cp);
		current_derivative[0 * n_points] = 1/(2*h)*(f_plus_h-f_minus_h);

		// parameters[1]' = (Ve)'
		f_plus_h = get_value(parameters[0],parameters[1]+h,parameters[2],point_index,T,Cp);
		f_minus_h = get_value(parameters[0],parameters[1]-h,parameters[2],point_index,T,Cp);
		current_derivative[1 * n_points] = 1/(2*h)*(f_plus_h-f_minus_h);

		// parameters[2]' = (Vp)'
		f_plus_h = get_value(parameters[0],parameters[1],parameters[2]+h,point_index,T,Cp);
		f_minus_h = get_value(parameters[0],parameters[1],parameters[2]-h,point_index,T,Cp);
		current_derivative[2 * n_points] = 1/(2*h)*(f_plus_h-f_minus_h);
	}
	else
	{
		// formula calculating derivative values with respect to parameters[0] (Ktrans)
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

		// formula calculating derivative values with respect to parameters[2] (Vp)
		current_derivative[2 * n_points] = Cp[point_index];
	}
}
#endif
