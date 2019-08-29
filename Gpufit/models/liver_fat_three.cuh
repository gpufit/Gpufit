#ifdef USE_DIXON_3
#define GPUFIT_LIVER_FAT_THREE_CUH_INCLUDED
#include <thrust/complex.h>

// https://thrust.github.io/doc/group__complex__numbers.html

// This constant changes which derivatives are used.
// Make it 1 or 0 to toggle it
#define HARD_DERIVATIVES_BUT_TAKE_REAL_PART 0

/**
 * Parameters contents:
 * 0:					M_w
 * 1:					M_f
 * 2:					R^*_2eff
 *
 *
 * user_info data:
 * 0:					TE_n (REAL)
 *
 * @param parameters
 * @param n_fits
 * @param n_points
 * @param value
 * @param derivative
 * @param point_index
 * @param fit_index
 * @param chunk_index
 * @param user_info
 * @param user_info_size
 * @return
 */
__device__ void  calculate_liver_fat_3(
    REAL const * parameters,
    int const n_fits,
    int const n_points,
    REAL * value,
    REAL * derivative,
    int const point_index,
    int const fit_index,
    int const chunk_index,
    char * user_info,
    std::size_t const user_info_size)
{
	// constants
	thrust::complex<REAL> const pi = thrust::complex<REAL>(3.14159);

    // indices

    REAL * user_info_float = (REAL*) user_info;
    thrust::complex<REAL> TEn = 0;

    // Maybe try TEn(point_index)... etc if  = doesn't work, but
    // Documentation does say there is a constructor for just straight real numbers
    if (!user_info_float)
    {
        TEn = point_index;
    }
    else if (user_info_size / sizeof(REAL) == n_points)
    {
        TEn = user_info_float[point_index];
    }
    else if (user_info_size / sizeof(REAL) > n_points)
    {
        int const chunk_begin = chunk_index * n_fits * n_points;
        int const fit_begin = fit_index * n_points;
        TEn = user_info_float[chunk_begin + fit_begin + point_index];
    }

	// Extract parameters
	thrust::complex<REAL> const M_w = thrust::complex<REAL>(parameters[0]);
	thrust::complex<REAL> const M_f = thrust::complex<REAL>(parameters[1]);
	thrust::complex<REAL> const j = thrust::complex<REAL>(0.f, 1);
	thrust::complex<REAL> const R2eff(parameters[2]);
	thrust::complex<REAL> const ppm_list[] = {-0.4764702, -0.4253742, -0.3883296, -0.332124, -0.3040212, -0.2375964, 0.0868632};
	thrust::complex<REAL> const weight_list[] = {0.08, 0.63, 0.07, 0.09, 0.07, 0.02, 0.04};

	// Complex Number builder
	// First calculate C_n
	//maybe define C_n as a real and imaginary part
	thrust::complex<REAL> C_n = thrust::complex<REAL>(0.f, 0.f);

	for (int i =0; i < 7; i++)
	{
		// C_n calculation
		// weight_list * e ^ (j * 2 * pi * ppm_list * TEn)
		C_n += weight_list[i] * thrust::exp(j * 2 * pi * ppm_list[i] * TEn);
	}

	///////////////////////////// value //////////////////////////////

    thrust::complex<REAL> S_n = (M_w + C_n * M_f) * thrust::exp(-R2eff * TEn);
    value[point_index] = thrust::abs(S_n);

    /////////////////////////// derivative ///////////////////////////
    REAL * current_derivative = derivative + point_index;
    REAL const minusRT = thrust::exp(-1 * R2eff * TEn).real();
    REAL const minus2RT = thrust::exp(-2.0f * R2eff * TEn).real();

#if HARD_DERIVATIVES_BUT_TAKE_REAL_PART
    thrust::complex<REAL> dM_w = (M_f * C_n + M_w) * minusRT / thrust::abs(M_w + M_f * C_n);
    thrust::complex<REAL> dM_f = C_n * (M_f * C_n + M_w) * minusRT / thrust::abs(M_w + M_f * C_n);
    thrust::complex<REAL> dR = TEn * - minusRT * (M_f * C_n + M_w);

    // If we use the complex ones and ignore the imaginary part:
    // Order needs to be dw df dr
    current_derivative[0 * n_points] = dM_w.real();
    current_derivative[1 * n_points] = dM_f.real();
    current_derivative[2 * n_points] = dR.real();

#else
/*
    thrust::complex<REAL> dM_f = C_n * minusRT;
    thrust::complex<REAL> dM_w = minusRT;
    thrust::complex<REAL> dR = TEn * minusRT * thrust::abs(M_w + M_f * C_n) * R2eff * TEn;

    // easy derivative
    current_derivative[0 * n_points] = thrust::abs(dM_w);
    current_derivative[1 * n_points] = thrust::abs(dM_f);
    current_derivative[2 * n_points] = thrust::abs(dR);
*/

    thrust::complex<REAL> CR = C_n.real();
    thrust::complex<REAL> CI = C_n.imag();
    thrust::complex<REAL> dM_f = (2 * CR * minus2RT * (CR * M_f + M_w) - 2 * M_f* CI * CI * minus2RT) / (2 * pow(minus2RT * (CR * M_f + M_w) * (CR * M_f + M_w) - M_f * M_f * CI * CI * minus2RT, 0.5));
    thrust::complex<REAL> dM_w = (minus2RT * (CR * M_f + M_w)) / pow(minus2RT * (CR * M_f + M_w) * (CR * M_f + M_w) - M_f * M_f * CI * CI * minus2RT, 0.5);
    thrust::complex<REAL> dR = (2 * M_f * M_f * CI * CI * TEn * minus2RT - 2 * TEn * minus2RT * (CR * M_f + M_w) * (CR * M_f + M_w)) / (2 * pow(minus2RT * (CR * M_f + M_w) * (CR * M_f + M_w) - M_f * M_f * CI * CI * minus2RT, 0.5));

    current_derivative[0 * n_points] = dM_w.real();
    current_derivative[1 * n_points] = dM_f.real();
    current_derivative[2 * n_points] = dR.real();

#endif
}

#endif
