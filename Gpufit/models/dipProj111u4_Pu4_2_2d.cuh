#ifndef DIPPROJ111U4_PU4_2_2D_INCLUDED
#define DIPPROJ111U4_PU4_2_2D_INCLUDED

/* Description of the calculate_gauss2d function
* ==============================================
*
* This function calculates the values of two-dimensional 111-projected model dipole functions
* and their partial derivatives with respect to the model parameters. The NV axis onto which we
* project is u4 = [0, -sqrt(2/3), sqrt(1/3) ]. The dipole moment we fit is assumed to be parallel
* to u4 (corresponding to a paramagnetic source).
*
* NOTE THAT WE CHOOSE DIMENSTIONS WITH   mu0 * 4*pi = 1  !!
*
* No independent variables are passed to this model function.  Hence, the
* (X, Y) coordinate of the first data value is assumed to be (0.0, 0.0).  For
* a fit size of M x N data points, the (X, Y) coordinates of the data are
* simply the corresponding array index values of the data array, starting from
* zero.
*
* Parameters:
*
* parameters: An input vector of model parameters.
*             p[0]: center coordinate x
*             p[1]: center coordinate y
*             p[2]: center coordinate z
*             p[3]: dipole moment magnitude mm
*
* n_fits: The number of fits. (not used)
*
* n_points: The number of data points per fit.
*
* value: An output vector of model function values.
*
* derivative: An output vector of model function partial derivatives.
*
* point_index: The data point index.
*
* fit_index: The fit index. (not used)
*
* chunk_index: The chunk index. (not used)
*
* user_info: An input vector containing user information. (not used)
*
* user_info_size: The size of user_info in bytes. (not used)
*
* Calling the calculate_gauss2d function
* ======================================
*
* This __device__ function can be only called from a __global__ function or an other
* __device__ function.
*
*/
__device__ void calculate_dipProj111u4_Pu4_2_2d(
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
    // indices

    int const n_points_x = sqrt((float)n_points);
    int const point_index_y = point_index / n_points_x;
    int const point_index_x = point_index - point_index_y * n_points_x;

    // parameters
    float const * p = parameters;

    // value
    float const rt2 = sqrt(2.0);
    float const rt3 = sqrt(3.0);

    float const Dx_1 = (point_index_x - p[0]);
    float const Dy_1 = (point_index_y - p[1]);
    float const Dz_1 = p[2];
    float const mm_1 = p[3];

    float const Dx_2 = (point_index_x - p[4]);
    float const Dy_2 = (point_index_y - p[5]);
    float const Dz_2 = p[6];
    float const mm_2 = p[7];


    float const AA_1 = rt2*Dy_1 - Dz_1;
    float const BB_1 = mm_1;
    float const CC_1 = ( Dx_1*Dx_1 + Dy_1*Dy_1 + Dz_1*Dz_1 );
    float const CC12_1 = sqrt(CC_1);
    float const CC32_1 = CC_1*CC12_1;
    float const CC52_1 = CC_1*CC_1*CC12_1;
    float const CC72_1 = CC_1*CC_1*CC_1*CC12_1;

    float const AA_2 = rt2*Dy_2 - Dz_2;
    float const BB_2 = mm_2;
    float const CC_2 = ( Dx_2*Dx_2 + Dy_2*Dy_2 + Dz_2*Dz_2 );
    float const CC12_2 = sqrt(CC_2);
    float const CC32_2 = CC_2*CC12_2;
    float const CC52_2 = CC_2*CC_2*CC12_2;
    float const CC72_2 = CC_2*CC_2*CC_2*CC12_2;

    float const prefac = 1;

    value[point_index] = prefac * ( AA_1*AA_1*BB_1/CC52_1 - BB_1/CC32_1  +  AA_2*AA_2*BB_2/CC52_2 - BB_2/CC32_2  );


    // derivatives

    float * current_derivative = derivative + point_index;

    float const dAdx0_1 = 0;
    float const dAdy0_1 = -rt2;
    float const dAdz0_1 = -1;
    float const dAdmm_1 = 0;
    float const dBdx0_1 = 0.;
    float const dBdy0_1 = 0.;
    float const dBdz0_1 = 0.;
    float const dBdmm_1 = 1.;
    float const dCdx0_1 = -2*Dx_1;
    float const dCdy0_1 = -2*Dy_1;
    float const dCdz0_1 = 2*Dz_1;
    float const dCdmm_1 = 0.;

    float const dAdx0_2 = 0;
    float const dAdy0_2 = -rt2;
    float const dAdz0_2 = -1;
    float const dAdmm_2 = 0;
    float const dBdx0_2 = 0.;
    float const dBdy0_2 = 0.;
    float const dBdz0_2 = 0.;
    float const dBdmm_2 = 1.;
    float const dCdx0_2 = -2*Dx_2;
    float const dCdy0_2 = -2*Dy_2;
    float const dCdz0_2 = 2*Dz_2;
    float const dCdmm_2 = 0.;

	current_derivative[0 * n_points] = prefac * ( 2*AA_1*BB_1*dAdx0_1/CC52_1 + AA_1*AA_1*dBdx0_1/CC52_1 - dBdx0_1/CC32_1 - 2.5*AA_1*AA_1*BB_1*dCdx0_1/CC72_1 + 1.5*BB_1*dCdx0_1/CC52_1 );
	current_derivative[1 * n_points] = prefac * ( 2*AA_1*BB_1*dAdy0_1/CC52_1 + AA_1*AA_1*dBdy0_1/CC52_1 - dBdy0_1/CC32_1 - 2.5*AA_1*AA_1*BB_1*dCdy0_1/CC72_1 + 1.5*BB_1*dCdy0_1/CC52_1 );
	current_derivative[2 * n_points] = prefac * ( 2*AA_1*BB_1*dAdz0_1/CC52_1 + AA_1*AA_1*dBdz0_1/CC52_1 - dBdz0_1/CC32_1 - 2.5*AA_1*AA_1*BB_1*dCdz0_1/CC72_1 + 1.5*BB_1*dCdz0_1/CC52_1 );
	current_derivative[3 * n_points] = prefac * ( 2*AA_1*BB_1*dAdmm_1/CC52_1 + AA_1*AA_1*dBdmm_1/CC52_1 - dBdmm_1/CC32_1 - 2.5*AA_1*AA_1*BB_1*dCdmm_1/CC72_1 + 1.5*BB_1*dCdmm_1/CC52_1 );

  current_derivative[4 * n_points] = prefac * ( 2*AA_2*BB_2*dAdx0_2/CC52_2 + AA_2*AA_2*dBdx0_2/CC52_2 - dBdx0_2/CC32_2 - 2.5*AA_2*AA_2*BB_2*dCdx0_2/CC72_2 + 1.5*BB_2*dCdx0_2/CC52_2 );
	current_derivative[5 * n_points] = prefac * ( 2*AA_2*BB_2*dAdy0_2/CC52_2 + AA_2*AA_2*dBdy0_2/CC52_2 - dBdy0_2/CC32_2 - 2.5*AA_2*AA_2*BB_2*dCdy0_2/CC72_2 + 1.5*BB_2*dCdy0_2/CC52_2 );
	current_derivative[6 * n_points] = prefac * ( 2*AA_2*BB_2*dAdz0_2/CC52_2 + AA_2*AA_2*dBdz0_2/CC52_2 - dBdz0_2/CC32_2 - 2.5*AA_2*AA_2*BB_2*dCdz0_2/CC72_2 + 1.5*BB_2*dCdz0_2/CC52_2 );
	current_derivative[7 * n_points] = prefac * ( 2*AA_2*BB_2*dAdmm_2/CC52_2 + AA_2*AA_2*dBdmm_2/CC52_2 - dBdmm_2/CC32_2 - 2.5*AA_2*AA_2*BB_2*dCdmm_2/CC72_2 + 1.5*BB_2*dCdmm_2/CC52_2 );
}
#endif
