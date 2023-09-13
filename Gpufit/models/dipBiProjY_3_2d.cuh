#ifndef DIPBIPROJY_3_2D_INCLUDED
#define DIPBIPROJY_3_2D_INCLUDED

/* Description of the calculate_gauss2d function
* ==============================================
*
* This function calculates the values of two-dimensional y-projected model dipole functions
* and their partial derivatives with respect to the model parameters. 
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
*             p[0]: center coordinate x, dipole 1
*             p[1]: center coordinate y, dipole 1
*             p[2]: center coordinate z, dipole 1
*             p[3]: dipole moment x, dipole 1
*             p[4]: dipole moment y, dipole 1		  
*             p[5]: dipole moment z, dipole 1
*             p[6]:  center coordinate x, dipole 2
*             p[7]:  center coordinate y, dipole 2
*             p[8]:  center coordinate z, dipole 2
*             p[9]:  dipole moment x, dipole 2
*             p[10]: dipole moment y, dipole 2		  
*             p[11]: dipole moment z, dipole 2
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
__device__ void calculate_dipBiProjY_3_2d(
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
    float const Dx_1 = (point_index_x - p[0]);
    float const Dy_1 = (point_index_y - p[1]);
    float const Dz_1 = p[2];
    float const px_1 = p[3];
    float const py_1 = p[4];
    float const pz_1 = p[5];
	float const Dx_2 = (point_index_x - p[6]);
    float const Dy_2 = (point_index_y - p[7]);
    float const Dz_2 = p[8];
    float const px_2 = p[9];
    float const py_2 = p[10];
    float const pz_2 = p[11];
    float const Dx_3 = (point_index_x - p[12]);
    float const Dy_3 = (point_index_y - p[13]);
    float const Dz_3 = p[14];
    float const px_3 = p[15];
    float const py_3 = p[16];
    float const pz_3 = p[17];
	
    float const AA_1 = Dy_1 * ( px_1*Dx_1 + py_1*Dy_1 + pz_1*Dz_1 );
    float const BB_1 = py_1;
    float const CC_1 = ( Dx_1*Dx_1 + Dy_1*Dy_1 + Dz_1*Dz_1 );
    float const CC12_1 = sqrt(CC_1);
    float const CC32_1 = CC_1*CC12_1;
    float const CC52_1 = CC_1*CC_1*CC12_1;
    float const CC72_1 = CC_1*CC_1*CC_1*CC12_1;
    float const AA_2 = Dy_2 * ( px_2*Dx_2 + py_2*Dy_2 + pz_2*Dz_2 );
    float const BB_2 = py_2;
    float const CC_2 = ( Dx_2*Dx_2 + Dy_2*Dy_2 + Dz_2*Dz_2 );
    float const CC12_2 = sqrt(CC_2);
    float const CC32_2 = CC_2*CC12_2;
    float const CC52_2 = CC_2*CC_2*CC12_2;
    float const CC72_2 = CC_2*CC_2*CC_2*CC12_2;	
    float const AA_3 = Dy_3 * ( px_3*Dx_3 + py_3*Dy_3 + pz_3*Dz_3 );
    float const BB_3 = py_3;
    float const CC_3 = ( Dx_3*Dx_3 + Dy_3*Dy_3 + Dz_3*Dz_3 );
    float const CC12_3 = sqrt(CC_3);
    float const CC32_3 = CC_3*CC12_3;
    float const CC52_3 = CC_3*CC_3*CC12_3;
    float const CC72_3 = CC_3*CC_3*CC_3*CC12_3;
	
    float const prefac = -1.632993161855452; 
    value[point_index] = prefac * ( 3*AA_1/CC52_1 - BB_1/CC32_1 + 3*AA_2/CC52_2 - BB_3/CC32_3 + 3*AA_3/CC52_3 - BB_3/CC32_3 );	
	

    // derivatives

    float * current_derivative = derivative + point_index;

    float const dAdx0_1 = -px_1*Dy_1;
    float const dAdy0_1 = -px_1*Dx_1 - 2*py_1*Dy_1 - pz_1*Dz_1;
    float const dAdz0_1 = pz_1*Dy_1;
    float const dAdpx_1 = Dx_1*Dy_1;
    float const dAdpy_1 = Dy_1*Dy_1;
    float const dAdpz_1 = Dz_1*Dy_1;
    float const dBdx0_1 = 0.;
    float const dBdy0_1 = 1.0;
    float const dBdz0_1 = 0.;
    float const dBdpx_1 = 0.;
    float const dBdpy_1 = 0.;
    float const dBdpz_1 = 0.;	
    float const dCdx0_1 = -2*Dx_1;
    float const dCdy0_1 = -2*Dy_1;
    float const dCdz0_1 = 2*Dz_1;
    float const dCdpx_1 = 0.;
    float const dCdpy_1 = 0.;
    float const dCdpz_1 = 0.;
	
    float const dAdx0_2 = -px_2*Dy_2;
    float const dAdy0_2 = -px_2*Dx_2 - 2*py_2*Dy_2 - pz_2*Dz_2;
    float const dAdz0_2 = pz_2*Dy_2;
    float const dAdpx_2 = Dx_2*Dy_2;
    float const dAdpy_2 = Dy_2*Dy_2;
    float const dAdpz_2 = Dz_2*Dy_2;	
    float const dBdx0_2 = 0.;
    float const dBdy0_2 = 1.0;
    float const dBdz0_2 = 0.;
    float const dBdpx_2 = 0.;
    float const dBdpy_2 = 0.;
    float const dBdpz_2 = 0.;
    float const dCdx0_2 = -2*Dx_2;
    float const dCdy0_2 = -2*Dy_2;
    float const dCdz0_2 = 2*Dz_2;
    float const dCdpx_2 = 0.;
    float const dCdpy_2 = 0.;
    float const dCdpz_2 = 0.;
	
	float const dAdx0_3 = -px_3*Dy_3;
    float const dAdy0_3 = -px_3*Dx_3 - 2*py_3*Dy_3 - pz_3*Dz_3;
    float const dAdz0_3 = pz_3*Dy_3;
    float const dAdpx_3 = Dx_3*Dy_3;
    float const dAdpy_3 = Dy_3*Dy_3;
    float const dAdpz_3 = Dz_3*Dy_3;
    float const dBdx0_3 = 0.;
    float const dBdy0_3 = 1.0;
    float const dBdz0_3 = 0.;
    float const dBdpx_3 = 0.;
    float const dBdpy_3 = 0.;
    float const dBdpz_3 = 0.;	
    float const dCdx0_3 = -2*Dx_3;
    float const dCdy0_3 = -2*Dy_3;
    float const dCdz0_3 = 2*Dz_3;
    float const dCdpx_3 = 0.;
    float const dCdpy_3 = 0.;
    float const dCdpz_3 = 0.;
	
	
	current_derivative[0 * n_points] = prefac * ( 3*dAdx0_1/CC52_1 - 7.5*AA_1*dCdx0_1/CC72_1 - dBdx0_1/CC32_1 + 1.5*BB_1*dCdx0_1/CC52_1 );
    current_derivative[1 * n_points] = prefac * ( 3*dAdy0_1/CC52_1 - 7.5*AA_1*dCdy0_1/CC72_1 - dBdy0_1/CC32_1 + 1.5*BB_1*dCdy0_1/CC52_1 );
    current_derivative[2 * n_points] = prefac * ( 3*dAdz0_1/CC52_1 - 7.5*AA_1*dCdz0_1/CC72_1 - dBdz0_1/CC32_1 + 1.5*BB_1*dCdz0_1/CC52_1 );
    current_derivative[3 * n_points] = prefac * ( 3*dAdpx_1/CC52_1 - 7.5*AA_1*dCdpx_1/CC72_1 - dBdpx_1/CC32_1 + 1.5*BB_1*dCdpx_1/CC52_1 );
    current_derivative[4 * n_points] = prefac * ( 3*dAdpy_1/CC52_1 - 7.5*AA_1*dCdpy_1/CC72_1 - dBdpy_1/CC32_1 + 1.5*BB_1*dCdpy_1/CC52_1 );
    current_derivative[5 * n_points] = prefac * ( 3*dAdpz_1/CC52_1 - 7.5*AA_1*dCdpz_1/CC72_1 - dBdpz_1/CC32_1 + 1.5*BB_1*dCdpz_1/CC52_1 );
	
	current_derivative[6 * n_points] = prefac * ( 3*dAdx0_2/CC52_2 - 7.5*AA_2*dCdx0_2/CC72_2 - dBdx0_2/CC32_2 + 1.5*BB_2*dCdx0_2/CC52_2 );
    current_derivative[7 * n_points] = prefac * ( 3*dAdy0_2/CC52_2 - 7.5*AA_2*dCdy0_2/CC72_2 - dBdy0_2/CC32_2 + 1.5*BB_2*dCdy0_2/CC52_2 );
    current_derivative[8 * n_points] = prefac * ( 3*dAdz0_2/CC52_2 - 7.5*AA_2*dCdz0_2/CC72_2 - dBdz0_2/CC32_2 + 1.5*BB_2*dCdz0_2/CC52_2 );
    current_derivative[9 * n_points] = prefac * ( 3*dAdpx_2/CC52_2 - 7.5*AA_2*dCdpx_2/CC72_2 - dBdpx_2/CC32_2 + 1.5*BB_2*dCdpx_2/CC52_2 );
    current_derivative[10* n_points] = prefac * ( 3*dAdpy_2/CC52_2 - 7.5*AA_2*dCdpy_2/CC72_2 - dBdpy_2/CC32_2 + 1.5*BB_2*dCdpy_2/CC52_2 );
    current_derivative[11* n_points] = prefac * ( 3*dAdpz_2/CC52_2 - 7.5*AA_2*dCdpz_2/CC72_2 - dBdpz_2/CC32_2 + 1.5*BB_2*dCdpz_2/CC52_2 );
	
	current_derivative[12 * n_points] = prefac * ( 3*dAdx0_3/CC52_3 - 7.5*AA_3*dCdx0_3/CC72_3 - dBdx0_3/CC32_3 + 1.5*BB_3*dCdx0_3/CC52_3 );
    current_derivative[13 * n_points] = prefac * ( 3*dAdy0_3/CC52_3 - 7.5*AA_3*dCdy0_3/CC72_3 - dBdy0_3/CC32_3 + 1.5*BB_3*dCdy0_3/CC52_3 );
    current_derivative[14 * n_points] = prefac * ( 3*dAdz0_3/CC52_3 - 7.5*AA_3*dCdz0_3/CC72_3 - dBdz0_3/CC32_3 + 1.5*BB_3*dCdz0_3/CC52_3 );
    current_derivative[15 * n_points] = prefac * ( 3*dAdpx_3/CC52_3 - 7.5*AA_3*dCdpx_3/CC72_3 - dBdpx_3/CC32_3 + 1.5*BB_3*dCdpx_3/CC52_3 );
    current_derivative[16 * n_points] = prefac * ( 3*dAdpy_3/CC52_3 - 7.5*AA_3*dCdpy_3/CC72_3 - dBdpy_3/CC32_3 + 1.5*BB_3*dCdpy_3/CC52_3 );
    current_derivative[17 * n_points] = prefac * ( 3*dAdpz_3/CC52_3 - 7.5*AA_3*dCdpz_3/CC72_3 - dBdpz_3/CC32_3 + 1.5*BB_3*dCdpz_3/CC52_3 );
}
#endif
