#ifdef USE_TISSUE_UPTAKE
#define GPUFIT_TISSUE_UPTAKE_CUH_INCLUDED

__device__ REAL get_tuptake_value (
	REAL p0, //Ktrans
	REAL p1, //Vp
	REAL p2, //Fp
	int const point_index,
	REAL const * T,
	REAL const * Cp)
{
	// integral/convolution
	REAL convFunc = 0;
	REAL Tp = p1 / (p2 / ((p2 / p0) - 1) + p2);
	for (int i = 1; i < point_index; i++) {
		REAL spacing = T[i] - T[i - 1];
		REAL Ct = Cp[i] * (p2 * exp(-(T[point_index] - T[i])/Tp) + p0 * (1 - exp(-(T[point_index] - T[i])/Tp)));
		REAL Ctprev = Cp[i - 1] * (p2 * exp(-(T[point_index] - T[i-1])/Tp) + p0 * (1 - exp(-(T[point_index] - T[i-1])/Tp)));
		convFunc += ((Ct + Ctprev) / 2 * spacing);
	}

	return convFunc;
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

	value[point_index] = get_tuptake_value(parameters[0],parameters[1],parameters[2],point_index,T,Cp);   // formula calculating fit model values
	// C(t)		       =   integral(Cp(k) * Fp*exp(-t/Tp) + Ktrans*(1-exp(-t/Tp))
	// where Tp = Vp*Fp/(Fp/Ktrans - 1)

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
	f_plus_h = get_tuptake_value(parameters[0]+h,parameters[1],parameters[2],point_index,T,Cp);
	f_minus_h = get_tuptake_value(parameters[0]-h,parameters[1],parameters[2],point_index,T,Cp);
	f_plus_2h = get_tuptake_value(parameters[0]+2*h,parameters[1],parameters[2],point_index,T,Cp);
	f_minus_2h = get_tuptake_value(parameters[0]-2*h,parameters[1],parameters[2],point_index,T,Cp);
	current_derivative[0 * n_points] = 1/(12*h)*(f_minus_2h-8*f_minus_h+8*f_plus_h-f_plus_2h);	// formula calculating derivative values with respect to parameters[0] (Ktrans)

	// parameters[1]' = (Vp)'
	f_plus_h = get_tuptake_value(parameters[0],parameters[1]+h,parameters[2],point_index,T,Cp);
	f_minus_h = get_tuptake_value(parameters[0],parameters[1]-h,parameters[2],point_index,T,Cp);
	f_plus_2h = get_tuptake_value(parameters[0],parameters[1]+2*h,parameters[2],point_index,T,Cp);
	f_minus_2h = get_tuptake_value(parameters[0],parameters[1]-2*h,parameters[2],point_index,T,Cp);
	current_derivative[1 * n_points] = 1/(12*h)*(f_minus_2h-8*f_minus_h+8*f_plus_h-f_plus_2h); // formula calculating derivative values with respect to parameters[1] (vp)

	// parameters[2]' = (Fp)'
	f_plus_h = get_tuptake_value(parameters[0],parameters[1],parameters[2]+h,point_index,T,Cp);
	f_minus_h = get_tuptake_value(parameters[0],parameters[1],parameters[2]-h,point_index,T,Cp);
	f_plus_2h = get_tuptake_value(parameters[0],parameters[1],parameters[2]+2*h,point_index,T,Cp);
	f_minus_2h = get_tuptake_value(parameters[0],parameters[1],parameters[2]-2*h,point_index,T,Cp);
	current_derivative[2 * n_points] = 1/(12*h)*(f_minus_2h-8*f_minus_h+8*f_plus_h-f_plus_2h);	// formula calculating derivative values with respect to parameters[1] (Fp)
}
#endif
