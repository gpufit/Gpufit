#ifndef GPUFIT_NATURAL_BSPLINE_ND_CUH_INCLUDED
#define GPUFIT_NATURAL_BSPLINE_ND_CUH_INCLUDED

#include "bspline_fast_cubic_basis_evaluate_device.cuh"

// Max number of supported dimensions (adjust as needed)
#define NATURAL_BSPLINE_ND_MAX_DIMS 6
#define CUBIC_DEGREE 3
#define CUBIC_BASIS_SIZE 4

/*
* Description of the calculate_natural_bspline_nd function
* ========================================================
*
* parameters: [amp, center_0, ..., center_{D-1}, offset]
* n_fits:     number of fits (for Gpufit batch)
* n_points:   number of points per fit
* value:      output model values
* derivative: output derivatives
* point_index: index of the current point (for each axis, see below)
* fit_index:   index of the current fit
* chunk_index: chunk index
* user_info:   buffer holding all ND spline metadata in the following order:
*      user_info[0] = D (number of dims)
*      user_info[1 ... D] = number of data points per axis
*      user_info[1+D ... 1+2D] = number of control points per axis
*      user_info[1+2D ... 1+3D] = number of knots per axis
*      user_info[... next] = flattened knot vectors (all axes, all knots per axis)
*      user_info[... next] = flattened control point strides (for flattened coeff tensor)
*      user_info[... next] = coefficients (flattened, length = product of control points per axis)
* 
* For best performance, precompute and pack all these arrays on the host using your Gpuspline code.
*/

__device__ void calculate_natural_bspline_nd(
    REAL const * parameters,    // [amp, center_0..center_{D-1}, offset]
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
    // --- Unpack user_info ---
    REAL const * ui = (REAL*)user_info;

    int D = static_cast<int>(ui[0]);
    int d;
    int const * data_points = (int const *)(ui + 1);           // [D]
    int const * control_points = data_points + D;              // [D]
    int const * num_knots = control_points + D;                // [D]

    int offset_knots = 1 + 3 * D;
    int offset_strides = offset_knots;
    for (d = 0; d < D; ++d) offset_strides += num_knots[d];
    int offset_coeff = offset_strides + D;

    // --- Pointers to ND arrays ---
    REAL const * knots[NATURAL_BSPLINE_ND_MAX_DIMS];
    int n_ctrl[NATURAL_BSPLINE_ND_MAX_DIMS];
    int n_stride[NATURAL_BSPLINE_ND_MAX_DIMS];
    int n_span[NATURAL_BSPLINE_ND_MAX_DIMS];

    int acc = offset_knots;
    for (d = 0; d < D; ++d)
    {
        knots[d] = ui + acc;
        acc += num_knots[d];
        n_ctrl[d] = control_points[d];
        n_stride[d] = static_cast<int>(ui[offset_strides + d]);
    }
    REAL const * coeff = ui + offset_coeff;

    // --- Map point_index to ND coordinates (x[0], ..., x[D-1]) ---
    // For image or ND array, this is usually unravel_index(point_index, data_points)
    int coords[NATURAL_BSPLINE_ND_MAX_DIMS] = { 0 };
    int idx = point_index;
    for (d = 0; d < D; ++d)
    {
        coords[d] = idx % data_points[d];
        idx /= data_points[d];
    }

    // --- Unpack model parameters ---
    REAL amp = parameters[0];
    REAL center[NATURAL_BSPLINE_ND_MAX_DIMS];
    for (d = 0; d < D; ++d)
        center[d] = parameters[1 + d];
    REAL offset = parameters[1 + D];

    // --- Compute shifted input coords: pt[d] = coords[d] - center[d] ---
    REAL pt[NATURAL_BSPLINE_ND_MAX_DIMS];
    for (d = 0; d < D; ++d)
        pt[d] = static_cast<REAL>(coords[d]) - center[d];

    // --- For each axis: find knot span, evaluate basis and derivative ---
    int k = CUBIC_DEGREE;
    int span[NATURAL_BSPLINE_ND_MAX_DIMS];
    REAL B[NATURAL_BSPLINE_ND_MAX_DIMS][CUBIC_BASIS_SIZE];
    REAL dB[NATURAL_BSPLINE_ND_MAX_DIMS][CUBIC_BASIS_SIZE];

    for (d = 0; d < D; ++d)
    {
        int N = data_points[d];
        int M = n_ctrl[d];
        // Find knot span
        REAL xq = pt[d];
        if (xq <= 0.0)
            span[d] = k;
        else if (xq >= REAL(N - 1))
            span[d] = M - 1;
        else
            span[d] = static_cast<int>(xq) + k;
        // Basis
        evaluate_fast_cubic_basis_device(xq, span[d], knots[d], M, B[d]);
        evaluate_fast_cubic_basis_derivative_device(xq, span[d], knots[d], M, dB[d]);
    }

    // --- Tensor product sum ---
    // Setup for up to 6D (unroll for speed, expand if needed)
    REAL spline_val = 0;
    REAL spline_dx[NATURAL_BSPLINE_ND_MAX_DIMS] = {0};
    int stride[NATURAL_BSPLINE_ND_MAX_DIMS];
    int offset[NATURAL_BSPLINE_ND_MAX_DIMS];

    for (d = 0; d < D; ++d)
    {
        stride[d] = n_stride[d];
        offset[d] = span[d] - k;
    }

    // Only support up to 6D for hardcoded loops (can extend with templates if needed)
    if (D == 1)
    {
        for (int i0 = 0; i0 < 4; ++i0)
        {
            int idx0 = (offset[0] + i0) * stride[0];
            REAL c = coeff[idx0];
            REAL w = B[0][i0];
            spline_val += w * c;
            spline_dx[0] += dB[0][i0] * c;
        }
    }
    else if (D == 2)
    {
        for (int i0 = 0; i0 < 4; ++i0)
        for (int i1 = 0; i1 < 4; ++i1)
        {
            int idx = (offset[0] + i0) * stride[0] + (offset[1] + i1) * stride[1];
            REAL c = coeff[idx];
            REAL w = B[0][i0] * B[1][i1];
            spline_val += w * c;
            spline_dx[0] += dB[0][i0] * B[1][i1] * c;
            spline_dx[1] += B[0][i0] * dB[1][i1] * c;
        }
    }
    else if (D == 3)
    {
        for (int i0 = 0; i0 < 4; ++i0)
        for (int i1 = 0; i1 < 4; ++i1)
        for (int i2 = 0; i2 < 4; ++i2)
        {
            int idx = (offset[0] + i0) * stride[0] +
                      (offset[1] + i1) * stride[1] +
                      (offset[2] + i2) * stride[2];
            REAL c = coeff[idx];
            REAL w = B[0][i0] * B[1][i1] * B[2][i2];
            spline_val += w * c;
            spline_dx[0] += dB[0][i0] * B[1][i1] * B[2][i2] * c;
            spline_dx[1] += B[0][i0] * dB[1][i1] * B[2][i2] * c;
            spline_dx[2] += B[0][i0] * B[1][i1] * dB[2][i2] * c;
        }
    }
    else if (D == 4)
    {
        for (int i0 = 0; i0 < 4; ++i0)
        for (int i1 = 0; i1 < 4; ++i1)
        for (int i2 = 0; i2 < 4; ++i2)
        for (int i3 = 0; i3 < 4; ++i3)
        {
            int idx = (offset[0] + i0) * stride[0] +
                      (offset[1] + i1) * stride[1] +
                      (offset[2] + i2) * stride[2] +
                      (offset[3] + i3) * stride[3];
            REAL c = coeff[idx];
            REAL w = B[0][i0] * B[1][i1] * B[2][i2] * B[3][i3];
            spline_val += w * c;
            spline_dx[0] += dB[0][i0] * B[1][i1] * B[2][i2] * B[3][i3] * c;
            spline_dx[1] += B[0][i0] * dB[1][i1] * B[2][i2] * B[3][i3] * c;
            spline_dx[2] += B[0][i0] * B[1][i1] * dB[2][i2] * B[3][i3] * c;
            spline_dx[3] += B[0][i0] * B[1][i1] * B[2][i2] * dB[3][i3] * c;
        }
    }
    else if (D == 5)
    {
        for (int i0 = 0; i0 < 4; ++i0)
        for (int i1 = 0; i1 < 4; ++i1)
        for (int i2 = 0; i2 < 4; ++i2)
        for (int i3 = 0; i3 < 4; ++i3)
        for (int i4 = 0; i4 < 4; ++i4)
        {
            int idx = (offset[0] + i0) * stride[0] +
                      (offset[1] + i1) * stride[1] +
                      (offset[2] + i2) * stride[2] +
                      (offset[3] + i3) * stride[3] +
                      (offset[4] + i4) * stride[4];
            REAL c = coeff[idx];
            REAL w = B[0][i0] * B[1][i1] * B[2][i2] * B[3][i3] * B[4][i4];
            spline_val += w * c;
            spline_dx[0] += dB[0][i0] * B[1][i1] * B[2][i2] * B[3][i3] * B[4][i4] * c;
            spline_dx[1] += B[0][i0] * dB[1][i1] * B[2][i2] * B[3][i3] * B[4][i4] * c;
            spline_dx[2] += B[0][i0] * B[1][i1] * dB[2][i2] * B[3][i3] * B[4][i4] * c;
            spline_dx[3] += B[0][i0] * B[1][i1] * B[2][i2] * dB[3][i3] * B[4][i4] * c;
            spline_dx[4] += B[0][i0] * B[1][i1] * B[2][i2] * B[3][i3] * dB[4][i4] * c;
        }
    }
    else if (D == 6)
    {
        for (int i0 = 0; i0 < 4; ++i0)
        for (int i1 = 0; i1 < 4; ++i1)
        for (int i2 = 0; i2 < 4; ++i2)
        for (int i3 = 0; i3 < 4; ++i3)
        for (int i4 = 0; i4 < 4; ++i4)
        for (int i5 = 0; i5 < 4; ++i5)
        {
            int idx = (offset[0] + i0) * stride[0] +
                      (offset[1] + i1) * stride[1] +
                      (offset[2] + i2) * stride[2] +
                      (offset[3] + i3) * stride[3] +
                      (offset[4] + i4) * stride[4] +
                      (offset[5] + i5) * stride[5];
            REAL c = coeff[idx];
            REAL w = B[0][i0] * B[1][i1] * B[2][i2] * B[3][i3] * B[4][i4] * B[5][i5];
            spline_val += w * c;
            spline_dx[0] += dB[0][i0] * B[1][i1] * B[2][i2] * B[3][i3] * B[4][i4] * B[5][i5] * c;
            spline_dx[1] += B[0][i0] * dB[1][i1] * B[2][i2] * B[3][i3] * B[4][i4] * B[5][i5] * c;
            spline_dx[2] += B[0][i0] * B[1][i1] * dB[2][i2] * B[3][i3] * B[4][i4] * B[5][i5] * c;
            spline_dx[3] += B[0][i0] * B[1][i1] * B[2][i2] * dB[3][i3] * B[4][i4] * B[5][i5] * c;
            spline_dx[4] += B[0][i0] * B[1][i1] * B[2][i2] * B[3][i3] * dB[4][i4] * B[5][i5] * c;
            spline_dx[5] += B[0][i0] * B[1][i1] * B[2][i2] * B[3][i3] * B[4][i4] * dB[5][i5] * c;
        }
    }

    // --- Output model value ---
    value[point_index] = amp * spline_val + offset;

    // --- Output derivatives ---
    REAL * der = derivative + point_index;
    der[0 * n_points] = spline_val;           // d/d(amp)
    for (d = 0; d < D; ++d)
        der[(1 + d) * n_points] = -amp * spline_dx[d]; // d/d(center_d)
    der[(1 + D) * n_points] = 1;               // d/d(offset)
}

#endif
