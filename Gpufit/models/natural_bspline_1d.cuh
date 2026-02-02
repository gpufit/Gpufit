#ifndef GPUFIT_NATURAL_BSPLINE1D_CUH_INCLUDED
#define GPUFIT_NATURAL_BSPLINE1D_CUH_INCLUDED

#include "natural_bspline_fast_cubic_basis_device.cuh"

/*
* Description of the calculate_natural_bspline1d function
* =======================================================
*
* This function calculates a value of a one-dimensional natural cubic B-spline model function
* and its partial derivatives with respect to the model parameters.
*
* The X coordinate of the first data value is assumed to be 0.0. For
* a fit size of N data points, the X coordinates of the data are
* simply the corresponding array index values of the data array, starting from zero.
*
* Parameters:
*
* parameters: An input vector of concatenated sets of model parameters.
*             p[0]: amplitude
*             p[1]: x-shift
*             p[2]: offset
*
* n_fits: The number of fits.
*
* n_points: The number of data points per fit.
*
* value:      output model values (for all points in this fit)
*
* derivative: output derivatives (with respect to each parameter)
*
* point_index: current point (x = point_index)
*
* fit_index:   fit number
*
* chunk_index: chunk number
*
* user_info:   passed-in buffer with spline meta-data:
*    user_info[0]                          = num_control_points
*    user_info[1...num_control_points+4]   = knot vector (float)
*    user_info[1+num_control_points+4 ...] = coefficients
*
* user_info_size: number of elements in user_info (bytes)
*
*/

__device__ void calculate_natural_bspline1d(
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
    // Read user_info buffer as REAL
    REAL const * ui = (REAL*)user_info;

    int const num_coeff = static_cast<int>(ui[0]);
    int const N_tpl = num_coeff - 2;   // template length
    int const num_knots = num_coeff + 4; // cubic
    REAL const * knots = ui + 1;
    REAL const * coeff = ui + 1 + num_knots;

    // Model parameters: [amp, shift, offset]
    REAL const * p = parameters;
    REAL amp    = p[0];
    REAL shift  = p[1];
    REAL offset = p[2];

    // Data point coordinate
    REAL x = static_cast<REAL>(point_index);
    REAL xq = x - p[1]; 

    bool clamped = false;
    if (xq < (REAL)0)           { xq = (REAL)0;           clamped = true; }
    if (xq > (REAL)(N_tpl - 1)) { xq = (REAL)(N_tpl - 1); clamped = true; }

    // Find knot span as in host code
    const int k = 3;
    int span;
    if (xq <= (REAL)0) 
    {               
        span = k;
    }
    else if (xq >= (REAL)(N_tpl - 1))
    {
        span = num_coeff - 1;
    }
    else
    {
        span = (int)xq + k;
    }

    // Evaluate basis and derivative (device function!)
    REAL basis[4], dbasis[4];
    nat_bspl_fast_cubic_basis_device(xq, span, knots, num_coeff, basis);
    nat_bspl_fast_cubic_basis_derivative_device(xq, span, knots, num_coeff, dbasis);

    // Compute value and d/dx
    REAL spline_val = 0, spline_dx = 0;
    for (int i = 0; i < 4; ++i)
    {
        int idx = span - k + i;
        if (idx >= 0 && idx < num_coeff)
        {
            spline_val += basis[i] * coeff[idx];
            spline_dx  += dbasis[i] * coeff[idx];
        }
    }

    if (clamped) spline_dx = (REAL)0;

    // Write value
    value[point_index] = amp * spline_val + offset;

    // Write derivatives [amp, shift, offset]
    REAL * der = derivative + point_index;
    der[0 * n_points] = spline_val;         // d/d(amp)
    der[1 * n_points] = -amp * spline_dx;   // d/d(shift)
    der[2 * n_points] = 1;                  // d/d(offset)
}

#endif
