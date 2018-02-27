#ifndef RAMSEY_VAR_P_CUH_INCLUDED
#define RAMSEY_VAR_P_CUH_INCLUDED

__device__ void calculate_ramsey_var_p(
	double const * parameters,
    int const n_fits,
    int const n_points,
    double * value,
    double * derivative,
    int const point_index,
    int const fit_index,
    int const chunk_index,
    char * user_info,
    std::size_t const user_info_size)
{
	double * user_info_double = (double*)user_info;
    double x = 0.0f;
    if (!user_info_double)
    {
        x = point_index;
    }
    else if (user_info_size / sizeof(double) == n_points)
    {
        x = user_info_double[point_index];
    }
    else if (user_info_size / sizeof(double) > n_points)
    {
        int const chunk_begin = chunk_index * n_fits * n_points;
        int const fit_begin = fit_index * n_points;
        x = user_info_double[chunk_begin + fit_begin + point_index];
    }
	
	
	///////////////////////////// values //////////////////////////////
	
	// parameters: [A1 A2 c f1 f2 p t2star x1 x2] exp(-(x./t2star)^p)*(A1*cos(2*pi*f1*(x - x1)) + A2*cos(2*pi*f2*(x-x2))) + c
	double const * p = parameters;
	
	double const pi = 3.14159;
	double const t2arg = pow(x/p[6], p[5]);
	double const ex = exp(-t2arg);
	double const phasearg1 = 2*pi*p[3]*(x - p[7]);
	double const phasearg2 = 2*pi*p[4]*(x - p[8]);
	double const cos1 = cos(phasearg1);
	double const sin1 = sin(phasearg1);
	double const cos2 = cos(phasearg2);
	double const sin2 = sin(phasearg2);
	//double const xmin = x/p[6] - 1;
	//double const log = xmin - xmin*xmin/2.f + xmin*xmin*xmin/3.f - xmin*xmin*xmin*xmin/4.f;
	
	value[point_index] = ex*(p[0]*cos1 + p[1]*cos2) + p[2]; // formula calculating fit model values
	
	/////////////////////////// derivatives ///////////////////////////
	double * current_derivative = derivative + point_index;
	current_derivative[0 * n_points ] = ex*cos1 ; 
	current_derivative[1 * n_points ] = ex*cos2;
	current_derivative[2 * n_points ] = 1.f;
	current_derivative[3 * n_points ] = -p[0]*2*pi*(x-p[7])*ex*sin1;
	current_derivative[4 * n_points ] = -p[1]*2*pi*(x-p[8])*ex*sin2;
	current_derivative[5 * n_points ] = -log(x/p[6] + 0.000001)*ex*t2arg*(p[0]*cos1 + p[1]*cos2);
	current_derivative[6 * n_points ] = p[5]*1.f/(p[6]*p[6])*x*ex*pow(x/p[6],p[5]-1)*(p[0]*cos1 + p[1]*cos2);
	current_derivative[7 * n_points ] = p[0]*2*pi*p[3]*sin1*ex;
	current_derivative[8 * n_points ] = p[1]*2*pi*p[4]*sin2*ex;
	
	
}

#endif