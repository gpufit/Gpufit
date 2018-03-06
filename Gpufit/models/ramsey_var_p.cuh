#ifndef RAMSEY_VAR_P_CUH_INCLUDED
#define RAMSEY_VAR_P_CUH_INCLUDED

__device__ void calculate_ramsey_var_p(
	float const * parameters,
    int const n_fits,
    int const n_points,
    float * value,
    float * derivative,
    int const point_index,
    int const fit_index,
    int const chunk_index,
    char * user_info,
    std::size_t const user_info_size)
{
	float * user_info_double = (float*)user_info;
    float x = 0.;
    if (!user_info_double)
    {
        x = point_index;
    }
    else if (user_info_size / sizeof(float) == n_points)
    {
        x = user_info_double[point_index];
    }
    else if (user_info_size / sizeof(float) > n_points)
    {
        int const chunk_begin = chunk_index * n_fits * n_points;
        int const fit_begin = fit_index * n_points;
        x = user_info_double[chunk_begin + fit_begin + point_index];
    }
	
	
	///////////////////////////// values //////////////////////////////
	
	// parameters: [A1 A2 c f1 f2 p t2star x1 x2] exp(-(x./t2star)^p)*(A1*cos(2*pi*f1*(x - x1)) + A2*cos(2*pi*f2*(x-x2))) + c
	float const * p = parameters;
	
    float const pi = 3.14159f;
    float const t2arg = pow(x / p[6], p[5]);
    float const ex = exp(-t2arg);
    float const phasearg1 = 2.f * pi*p[3] * (x - p[7]);
    float const phasearg2 = 2.f * pi*p[4] * (x - p[8]);
    float const cos1 = cos(phasearg1);
    float const sin1 = sin(phasearg1);
    float const cos2 = cos(phasearg2);
    float const sin2 = sin(phasearg2);
	//float const xmin = x/p[6] - 1;
	//float const log = xmin - xmin*xmin/2. + xmin*xmin*xmin/3. - xmin*xmin*xmin*xmin/4.;
	
	value[point_index] = ex*(p[0]*cos1 + p[1]*cos2) + p[2]; // formula calculating fit model values
	
	/////////////////////////// derivatives ///////////////////////////
	float * current_derivative = derivative + point_index;
    current_derivative[0 * n_points] = ex*cos1;
    current_derivative[1 * n_points] = ex*cos2;
    current_derivative[2 * n_points] = 1.f;
    current_derivative[3 * n_points] = -p[0] * 2.f * pi*(x - p[7])*ex*sin1;
    current_derivative[4 * n_points] = -p[1] * 2.f * pi*(x - p[8])*ex*sin2;
    current_derivative[5 * n_points] = -log(x / p[6] + 0.000001f)*ex*t2arg*(p[0] * cos1 + p[1] * cos2);
    current_derivative[6 * n_points] = p[5] * 1.f / (p[6] * p[6])*x*ex*pow(x / p[6], p[5] - 1.f)*(p[0] * cos1 + p[1] * cos2);
    current_derivative[7 * n_points] = p[0] * 2.f * pi*p[3] * sin1*ex;
    current_derivative[8 * n_points] = p[1] * 2.f * pi*p[4] * sin2*ex;
	
	
}

#endif