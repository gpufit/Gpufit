#ifdef USE_2CXM
#define GPUFIT_TWO_COMPARTMENT_EXCHANGE_CUH_INCLUDED

__device__ REAL get_2cx_value (
	REAL p0, //Ktrans
	REAL p1, //Ve
	REAL p2, //Vp
	REAL p3, //Fp
	int const point_index,
	REAL const * T,
	REAL const * Cp)
{
	// integral/convolution
	REAL PS;
	if(p0>=p3) {
		PS = 10e8;
	} else {
		PS = p3 / ((p3 / p0) - 1);
	}

	REAL convFunc = 0;
	REAL Tp = p2 / (PS + p3);
	REAL Te = p1 / PS;
	REAL Tb = p2 / p3;
	REAL Kpos = 0.5 * (1/Tp + 1/Te + sqrt(pow(1/Tp + 1/Te,2) - 4 * 1/Te * 1/Tb));
	REAL Kneg = 0.5 * (1/Tp + 1/Te - sqrt(pow(1/Tp + 1/Te,2) - 4 * 1/Te * 1/Tb));
	REAL Eneg = (Kpos - 1/Tb) / (Kpos - Kneg);
	for (int i = 1; i < point_index; i++) {
		REAL spacing = T[i] - T[i - 1];
		REAL Ct =     Cp[i]     * (exp(-(T[point_index] - T[i])   * Kpos) + Eneg * (exp(-(T[point_index] - T[i])   * Kneg) - exp(-Kpos)));//(p2 * exp(-(T[point_index] - T[i])/Tp) + p0 * (1 - exp(-(T[point_index] - T[i])/Tp)));
		REAL Ctprev = Cp[i - 1] * (exp(-(T[point_index] - T[i-1]) * Kpos) + Eneg * (exp(-(T[point_index] - T[i-1]) * Kneg) - exp(-Kpos))); //(p2 * exp(-(T[point_index] - T[i-1])/Tp) + p0 * (1 - exp(-(T[point_index] - T[i-1])/Tp)));
		convFunc += ((Ct + Ctprev) / 2 * spacing);
	}
	REAL function_value = p3 * convFunc;
	return function_value;
}

__device__ void calculate_two_compartment_exchange (               // function name
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

	value[point_index] = get_2cx_value(parameters[0],parameters[1],parameters[2],parameters[3],point_index,T,Cp);                      // formula calculating fit model values
	// C(t)		       =   Fp * integral( K(t) * Cp)

	/////////////////////////// derivative ///////////////////////////
	REAL * current_derivative = derivative + point_index;

	//Numerical differentiation, 5 point method, error O(h^4)
	//smaller error than 3 point, but more function evaluations, potentially slower
	REAL h = 10e-5;
	REAL f_plus_h;
	REAL f_minus_h;
	REAL f_plus_2h;
	REAL f_minus_2h;

	// parameters[0]' = (Ktrans)'
	f_plus_h = get_2cx_value(parameters[0]+h,parameters[1],parameters[2],parameters[3],point_index,T,Cp);
	f_minus_h = get_2cx_value(parameters[0]-h,parameters[1],parameters[2],parameters[3],point_index,T,Cp);
	f_plus_2h = get_2cx_value(parameters[0]+2*h,parameters[1],parameters[2],parameters[3],point_index,T,Cp);
	f_minus_2h = get_2cx_value(parameters[0]-2*h,parameters[1],parameters[2],parameters[3],point_index,T,Cp);
	current_derivative[0 * n_points] = 1/(12*h)*(f_minus_2h-8*f_minus_h+8*f_plus_h-f_plus_2h);	// formula calculating derivative values with respect to parameters[0] (Ktrans)

	// parameters[1]' = (Ve)'
	f_plus_h = get_2cx_value(parameters[0],parameters[1]+h,parameters[2],parameters[3],point_index,T,Cp);
	f_minus_h = get_2cx_value(parameters[0],parameters[1]-h,parameters[2],parameters[3],point_index,T,Cp);
	f_plus_2h = get_2cx_value(parameters[0],parameters[1]+2*h,parameters[2],parameters[3],point_index,T,Cp);
	f_minus_2h = get_2cx_value(parameters[0],parameters[1]-2*h,parameters[2],parameters[3],point_index,T,Cp);
	current_derivative[1 * n_points] = 1/(12*h)*(f_minus_2h-8*f_minus_h+8*f_plus_h-f_plus_2h); // formula calculating derivative values with respect to parameters[1] (ve)

	// parameters[2]' = (Vp)'
	f_plus_h = get_2cx_value(parameters[0],parameters[1],parameters[2]+h,parameters[3],point_index,T,Cp);
	f_minus_h = get_2cx_value(parameters[0],parameters[1],parameters[2]-h,parameters[3],point_index,T,Cp);
	f_plus_2h = get_2cx_value(parameters[0],parameters[1],parameters[2]+2*h,parameters[3],point_index,T,Cp);
	f_minus_2h = get_2cx_value(parameters[0],parameters[1],parameters[2]-2*h,parameters[3],point_index,T,Cp);
	current_derivative[2 * n_points] = 1/(12*h)*(f_minus_2h-8*f_minus_h+8*f_plus_h-f_plus_2h);	// formula calculating derivative values with respect to parameters[1] (vp)

	// parameters[3]' = (Fp)'
	f_plus_h = get_2cx_value(parameters[0],parameters[1],parameters[2],parameters[3]+h,point_index,T,Cp);
	f_minus_h = get_2cx_value(parameters[0],parameters[1],parameters[2],parameters[3]-h,point_index,T,Cp);
	f_plus_2h = get_2cx_value(parameters[0],parameters[1],parameters[2],parameters[3]+2*h,point_index,T,Cp);
	f_minus_2h = get_2cx_value(parameters[0],parameters[1],parameters[2],parameters[3]-2*h,point_index,T,Cp);
	current_derivative[3 * n_points] = 1/(12*h)*(f_minus_2h-8*f_minus_h+8*f_plus_h-f_plus_2h);	// formula calculating derivative values with respect to parameters[1] (Fp)
}
#endif
