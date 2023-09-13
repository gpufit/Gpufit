#ifndef DIPPROJ111YZ_1_2D_INCLUDED
#define DIPPROJ111YZ_1_2D_INCLUDED

/* Description of the calculate_gauss2d function
* ==============================================
*
* This function calculates the values of two-dimensional 111-projected model dipole functions
* and their partial derivatives with respect to the model parameters. The NV axis onto which we
* project lies in the Y-Z plane, and has unit vector [0, -sqrt(2/3), -sqrt(1/3) ]
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
*             p[3]: dipole moment x
*             p[4]: dipole moment x		  
*             p[5]: dipole moment z
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
__device__ void calculate_dipProj111YZ_1_2d(
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
	
    float const Dx = (point_index_x - p[0]);
    float const Dy = (point_index_y - p[1]);
    float const Dz = p[2];
    float const px = p[3];
    float const py = p[4];
    float const pz = p[5];
	
	float const aa1 = ( rt2*Dy + Dz );
	float const aa2 = ( px*Dx + py*Dy + pz*Dz );
    float const AA = aa1 * aa2;
    float const BB = rt2*py + pz;
    float const CC = ( Dx*Dx + Dy*Dy + Dz*Dz );
    float const CC12 = sqrt(CC);
    float const CC32 = CC*CC12;
    float const CC52 = CC*CC*CC12;
    float const CC72 = CC*CC*CC*CC12;
    float const prefac = -rt3; 

    value[point_index] = prefac * ( AA/CC52 - BB/CC32 / 3 );	
	

    // derivatives

    float * current_derivative = derivative + point_index;

    float const dAdx0 = -px*aa1;
    float const dAdy0 = -py*aa1 - rt2*aa2;
    float const dAdz0 = pz*aa1 + aa2;
    float const dAdpx = Dx*aa1;
    float const dAdpy = Dy*aa1;
    float const dAdpz = Dz*aa1;
	
    float const dBdx0 = 0.;
    float const dBdy0 = 0.;
    float const dBdz0 = 0.;
    float const dBdpx = 0.;
    float const dBdpy = rt2;
    float const dBdpz = 1.;
	
    float const dCdx0 = -2*Dx;
    float const dCdy0 = -2*Dy;
    float const dCdz0 = 2*Dz;
    float const dCdpx = 0.;
    float const dCdpy = 0.;
    float const dCdpz = 0.;
	
	current_derivative[0 * n_points] = prefac * ( dAdx0/CC52 - 2.5*AA*dCdx0/CC72 - dBdx0/CC32 /3.0 + 0.5*BB*dCdx0/CC52 );
    current_derivative[1 * n_points] = prefac * ( dAdy0/CC52 - 2.5*AA*dCdy0/CC72 - dBdy0/CC32 /3.0 + 0.5*BB*dCdy0/CC52 );
    current_derivative[2 * n_points] = prefac * ( dAdz0/CC52 - 2.5*AA*dCdz0/CC72 - dBdz0/CC32 /3.0 + 0.5*BB*dCdz0/CC52 );
    current_derivative[3 * n_points] = prefac * ( dAdpx/CC52 - 2.5*AA*dCdpx/CC72 - dBdpx/CC32 /3.0 + 0.5*BB*dCdpx/CC52 );
    current_derivative[4 * n_points] = prefac * ( dAdpy/CC52 - 2.5*AA*dCdpy/CC72 - dBdpy/CC32 /3.0 + 0.5*BB*dCdpy/CC52 );
    current_derivative[5 * n_points] = prefac * ( dAdpz/CC52 - 2.5*AA*dCdpz/CC72 - dBdpz/CC32 /3.0 + 0.5*BB*dCdpz/CC52 );
}
#endif
