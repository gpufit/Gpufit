#define GPUFIT_NEW_MODEL_CUH_INCLUDED

__device__ REAL get_value (
	///////////// value calculated here to be used for value and derivative below /////////////
	REAL p0, //Ktrans
	REAL p1, //Ve
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
	REAL function_value = p0 * convFunc;
	return function_value;
}

__device__ void calculate_new_model (               // function name
	REAL const * parameters,
	int const n_fits,
	int const n_points,
	REAL * value,
	REAL * derivative,
	int const point_index,
	int const fit_index,
	int const chunk_index,
	char * user_info,				// contains time and Cp values in a 1 dimensional array
	std::size_t const user_info_size)
{
	///////////////////////////// value //////////////////////////////

	// convert to REAL(float/double)
	REAL* user_info_float = (REAL*)user_info;
	// split user_info array into time and Cp
	REAL const * T = user_info_float;
	REAL const * Cp = user_info_float + n_points;

	// formula calculating fit model values
	value[point_index] = get_value(parameters[0],parameters[1],point_index,T,Cp);


	/////////////////////////// derivative ///////////////////////////
	REAL * current_derivative = derivative + point_index;

	//Numerical differentiation, 3 point method, error O(h^2)
	//h can be lowered to reduce error, but if set too low round
	//off error will cause error to dramatically increase
	REAL h = 10e-5;
	REAL f_plus_h;
	REAL f_minus_h;

	// parameters[0]' = (Ktrans)'
	f_plus_h = get_value(parameters[0]+h,parameters[1],point_index,T,Cp);
	f_minus_h = get_value(parameters[0]-h,parameters[1],point_index,T,Cp);
	current_derivative[0 * n_points] = 1/(2*h)*(f_plus_h-f_minus_h);

	// parameters[1]' = (Ve)'
	f_plus_h = get_value(parameters[0],parameters[1]+h,point_index,T,Cp);
	f_minus_h = get_value(parameters[0],parameters[1]-h,point_index,T,Cp);
	current_derivative[1 * n_points] = 1/(2*h)*(f_plus_h-f_minus_h);

//	//Numerical differentiation, 5 point method, error O(h^4)
//	//smaller error than 3 point, but more function evaluations, potentially slower
//	//REAL h = 10e-5;
//	//REAL f_plus_h;
//	//REAL f_minus_h;
//	REAL f_plus_2h;
//	REAL f_minus_2h;
//
//	// parameters[0]' = (Ktrans)'
//	f_plus_h = get_value(parameters[0]+h,parameters[1],point_index,T,Cp);
//	f_minus_h = get_value(parameters[0]-h,parameters[1],point_index,T,Cp);
//	f_plus_2h = get_value(parameters[0]+2*h,parameters[1],point_index,T,Cp);
//	f_minus_2h = get_value(parameters[0]-2*h,parameters[1],point_index,T,Cp);
//	current_derivative[0 * n_points] = 1/(12*h)*(f_minus_2h-8*f_minus_h+8*f_plus_h-f_plus_2h);
//
//	// parameters[1]' = (Ve)'
//	f_plus_h = get_value(parameters[0],parameters[1]+h,point_index,T,Cp);
//	f_minus_h = get_value(parameters[0],parameters[1]-h,point_index,T,Cp);
//	f_plus_2h = get_value(parameters[0],parameters[1]+2*h,point_index,T,Cp);
//	f_minus_2h = get_value(parameters[0],parameters[1]-2*h,point_index,T,Cp);
//	current_derivative[1 * n_points] = 1/(12*h)*(f_minus_2h-8*f_minus_h+8*f_plus_h-f_plus_2h);

}
