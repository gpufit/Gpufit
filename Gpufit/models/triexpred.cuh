#ifndef GPUFIT_TRIEXPRED_CUH_INCLUDED
#define GPUFIT_TRIEXPRED_CUH_INCLUDED

/* Description of the calculate_triexp_red function
* ===================================================
*
* This function calculates the values of reduced triexponential functions
* and their partial derivatives with respect to the model parameters.
*
* The reduced triexponential function is: S/S0 = (1-a-b)*exp(-c*x)+a*exp(-d*x)+b*exp(-e*x)
* The derivatives are:
* dy/da = exp(-d*x) - exp(-c*x)
* dy/db = exp(-e*x) - exp(-c*x)
* dy/dc = (1-a-b)*(-x*exp(-c*x))    //a x e^(-c x) + b x e^(-c x) - x e^(-c x)
* dy/dd = a*(-x*exp(-d*x))
* dy/de = b*(-x*exp(-e*x))
*
* This function makes use of the user information data to pass in the
* independent variables (X values) corresponding to the data.  The X values
* must be of type REAL.
*
* Note that if no user information is provided, the (X) coordinate of the
* first data value is assumed to be (0.0).  In this case, for a fit size of
* M data points, the (X) coordinates of the data are simply the corresponding
* array index values of the data array, starting from zero.
*
* There are three possibilities regarding the X values:
*
*   No X values provided:
*
*       If no user information is provided, the (X) coordinate of the
*       first data value is assumed to be (0.0).  In this case, for a
*       fit size of M data points, the (X) coordinates of the data are
*       simply the corresponding array index values of the data array,
*       starting from zero.
*
*   X values provided for one fit:
*
*       If the user_info array contains the X values for one fit, then
*       the same X values will be used for all fits.  In this case, the
*       size of the user_info array (in bytes) must equal
*       sizeof(REAL) * n_points.
*
*   Unique X values provided for all fits:
*
*       In this case, the user_info array must contain X values for each
*       fit in the dataset.  In this case, the size of the user_info array
*       (in bytes) must equal sizeof(REAL) * n_points * nfits.
*
* Parameters:
*
* parameters: An input vector of model parameters.
*             p[0]: a   p[1]: b     p[2]: c
*             p[3]: d   p[4]: e
*
* n_fits: The number of fits.
*
* n_points: The number of data points per fit.
*
* value: An output vector of model function values.
*
* derivative: An output vector of model function partial derivatives.
*
* point_index: The data point index.
*
* fit_index: The fit index.
*
* chunk_index: The chunk index. Used for indexing of user_info.
*
* user_info: An input vector containing user information.
*
* user_info_size: The size of user_info in bytes.
*
* Calling the calculate_triexp_red function
* =======================================
*
* This __device__ function can be only called from a __global__ function or an other
* __device__ function.
*
*/

__device__ void calculate_triexp_red(
    REAL const* parameters,
    int const n_fits,
    int const n_points,
    REAL* value,
    REAL* derivative,
    int const point_index,
    int const fit_index,
    int const chunk_index,
    char* user_info,
    std::size_t const user_info_size)
{
    // indices

    REAL* user_info_float = (REAL*)user_info;
    REAL x = 0;
    if (!user_info_float)
    {
        x = point_index;
    }
    else if (user_info_size / sizeof(REAL) == n_points)
    {
        x = user_info_float[point_index];
    }
    else if (user_info_size / sizeof(REAL) > n_points)
    {
        int const chunk_begin = chunk_index * n_fits * n_points;
        int const fit_begin = fit_index * n_points;
        x = user_info_float[chunk_begin + fit_begin + point_index];
    }

    // parameters
    REAL const* p = parameters;

    // value
    // (1-a-b)*exp(-c*x)+a*exp(-d*x)+b*exp(-e*x)
    // p[0]: a   p[1]: b     p[2]: c    p[3]: d   p[4]: e

    value[point_index] = (1.0 - p[0] - p[1]) * exp(-p[2] * x) +
        p[0] * exp(-p[3] * x) +
        p[1] * exp(-p[4] * x);

    // derivatives
    /* dy/da = exp(-d * x) - exp(-c * x)
    dy/db = exp(-e*x)-exp(-c*x)
    dy/dc = (1-a-b)*(-x*exp(-c*x))
    dy/dd = a*(-x*exp(-d*x))
    dy/de = b*(-x*exp(-e*x))
    p[0]: a   p[1]: b     p[2]: c    p[3]: d   p[4]: e    */

    REAL* current_derivatives = derivative + point_index;
    current_derivatives[0 * n_points] = exp(-p[3] * x) - exp(-p[2] * x);
    current_derivatives[1 * n_points] = exp(-p[4] * x) - exp(-p[2] * x);
    //current_derivatives[2 * n_points] = (1 - p[0] - p[1]) * (-x * exp(-p[2] * x));
    // a x e^(-c x) + b x e^(-c x) - x e^(-c x)
    current_derivatives[2 * n_points] = p[0]*exp(-p[2]*x) + p[1]*exp(-p[2]*x) - x*exp(-p[2]*x);
    //current_derivatives[2 * n_points] = x * (p[0]+p[1]-1) * pow(2.71828, -p[2]*x);
    current_derivatives[3 * n_points] = p[0] * (-x * exp(-p[3] * x));
    current_derivatives[4 * n_points] = p[1] * (-x * exp(-p[4] * x));
}

#endif
