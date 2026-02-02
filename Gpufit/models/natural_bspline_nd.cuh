#ifndef GPUFIT_NATURAL_BSPLINE_ND_CUH_INCLUDED
#define GPUFIT_NATURAL_BSPLINE_ND_CUH_INCLUDED

#include "natural_bspline_fast_cubic_basis_device.cuh"

#define NATURAL_BSPLINE_ND_MAX_DIMS 6
#define CUBIC_DEGREE 3
#define CUBIC_BASIS_SIZE 4

/*
user_info_REAL layout (mimic spline3d style: dims first, payload after)

user_info_REAL[0]                 = D
user_info_REAL[1 .. D]            = data_dims[d]        (used for unravel point_index)
user_info_REAL[1+D .. 1+2D]       = spline_dims[d]      (used for spline domain/clamp)
user_info_REAL[1+2D .. 1+3D]      = stride_ctrl[d]      (flattened coeff strides, axis0 fastest)
user_info_REAL[... next ...]      = knots for axis 0 (len = spline_dims[0] + 6)
                                   knots for axis 1 (len = spline_dims[1] + 6)
                                   ...
user_info_REAL[... next ...]      = coefficients (len = product over d of (spline_dims[d] + 2))

Notes:
- num_control_points[d] = spline_dims[d] + 2
- num_knots[d]          = spline_dims[d] + 6
- clamp xq to [0, spline_dims[d]-1]; if clamped => dB[d][*] = 0 (constant extension)
*/

__device__ void calculate_natural_bspline_nd(
    REAL const * parameters,    // [amp, shift_0..shift_{D-1}, offset]
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
    REAL const * ui = (REAL*)user_info;

    int const D = (int)(ui[0] + REAL(0.5)); // safe int cast from REAL
    if (D < 1 || D > NATURAL_BSPLINE_ND_MAX_DIMS)
    {
        value[point_index] = REAL(0);
        REAL * der0 = derivative + point_index;
        for (int j = 0; j < (2 + NATURAL_BSPLINE_ND_MAX_DIMS); ++j)
            der0[j * n_points] = REAL(0);
        return;
    }

    int data_dims[NATURAL_BSPLINE_ND_MAX_DIMS];
    int spline_dims[NATURAL_BSPLINE_ND_MAX_DIMS];
    int stride_ctrl[NATURAL_BSPLINE_ND_MAX_DIMS];

    int d;
    for (d = 0; d < D; ++d)
        data_dims[d] = (int)(ui[1 + d] + REAL(0.5));

    for (d = 0; d < D; ++d)
        spline_dims[d] = (int)(ui[1 + D + d] + REAL(0.5));

    for (d = 0; d < D; ++d)
        stride_ctrl[d] = (int)(ui[1 + 2 * D + d] + REAL(0.5));

    int const offset_knots = 1 + 3 * D;

    // knots pointers per axis
    REAL const * knots[NATURAL_BSPLINE_ND_MAX_DIMS];
    int acc = offset_knots;
    for (d = 0; d < D; ++d)
    {
        knots[d] = ui + acc;
        acc += (spline_dims[d] + 6);
    }

    // coefficients start
    REAL const * coeff = ui + acc;

    // ---- unravel point_index using data_dims ----
    int coords[NATURAL_BSPLINE_ND_MAX_DIMS] = { 0 };
    int idx = point_index;
    for (d = 0; d < D; ++d)
    {
        int const n = data_dims[d];
        coords[d] = (n > 0) ? (idx % n) : 0;
        idx = (n > 0) ? (idx / n) : 0;
    }

    // ---- parameters: amp, shifts, offset ----
    REAL const amp = parameters[0];
    REAL shift[NATURAL_BSPLINE_ND_MAX_DIMS];
    for (d = 0; d < D; ++d) shift[d] = parameters[1 + d];
    REAL const offset = parameters[1 + D];

    // ---- per-axis basis ----
    int const k = CUBIC_DEGREE;

    int span[NATURAL_BSPLINE_ND_MAX_DIMS];
    int base_idx[NATURAL_BSPLINE_ND_MAX_DIMS];

    REAL B[NATURAL_BSPLINE_ND_MAX_DIMS][CUBIC_BASIS_SIZE];
    REAL dB[NATURAL_BSPLINE_ND_MAX_DIMS][CUBIC_BASIS_SIZE];

    for (d = 0; d < D; ++d)
    {
        int const N = spline_dims[d];     // spline domain length
        int const M = N + 2;              // control points per axis

        // shifted coordinate in spline domain
        REAL xq = (REAL)coords[d] - shift[d];

        // clamp to [0, N-1], constant extension outside
        bool clamped = false;
        if (xq < REAL(0)) { xq = REAL(0); clamped = true; }
        if (xq > REAL(N - 1)) { xq = REAL(N - 1); clamped = true; }

        // span selection based on clamped xq
        if (xq <= REAL(0))
            span[d] = k;
        else if (xq >= REAL(N - 1))
            span[d] = M - 1;
        else
            span[d] = (int)xq + k;

        base_idx[d] = span[d] - k;

        nat_bspl_fast_cubic_basis_device(xq, span[d], knots[d], M, B[d]);

        if (clamped)
        {
            dB[d][0] = dB[d][1] = dB[d][2] = dB[d][3] = REAL(0);
        }
        else
        {
            nat_bspl_fast_cubic_basis_derivative_device(xq, span[d], knots[d], M, dB[d]);
        }
    }

    // ---- tensor product accumulation ----
    REAL spline_val = REAL(0);
    REAL spline_dshift[NATURAL_BSPLINE_ND_MAX_DIMS] = { 0 };

    if (D == 1)
    {
        for (int i0 = 0; i0 < 4; ++i0)
        {
            int const idxc = (base_idx[0] + i0) * stride_ctrl[0];
            REAL const c = coeff[idxc];
            spline_val += B[0][i0] * c;
            spline_dshift[0] += dB[0][i0] * c;
        }
    }
    else if (D == 2)
    {
        for (int i0 = 0; i0 < 4; ++i0)
        for (int i1 = 0; i1 < 4; ++i1)
        {
            int const idxc =
                (base_idx[0] + i0) * stride_ctrl[0] +
                (base_idx[1] + i1) * stride_ctrl[1];

            REAL const c = coeff[idxc];

            REAL const w = B[0][i0] * B[1][i1];
            spline_val += w * c;

            spline_dshift[0] += dB[0][i0] * B[1][i1] * c;
            spline_dshift[1] += B[0][i0] * dB[1][i1] * c;
        }
    }
    else if (D == 3)
    {
        for (int i0 = 0; i0 < 4; ++i0)
        for (int i1 = 0; i1 < 4; ++i1)
        for (int i2 = 0; i2 < 4; ++i2)
        {
            int const idxc =
                (base_idx[0] + i0) * stride_ctrl[0] +
                (base_idx[1] + i1) * stride_ctrl[1] +
                (base_idx[2] + i2) * stride_ctrl[2];

            REAL const c = coeff[idxc];

            REAL const w = B[0][i0] * B[1][i1] * B[2][i2];
            spline_val += w * c;

            spline_dshift[0] += dB[0][i0] * B[1][i1] * B[2][i2] * c;
            spline_dshift[1] += B[0][i0] * dB[1][i1] * B[2][i2] * c;
            spline_dshift[2] += B[0][i0] * B[1][i1] * dB[2][i2] * c;
        }
    }
    else if (D == 4)
    {
        for (int i0 = 0; i0 < 4; ++i0)
        for (int i1 = 0; i1 < 4; ++i1)
        for (int i2 = 0; i2 < 4; ++i2)
        for (int i3 = 0; i3 < 4; ++i3)
        {
            int const idxc =
                (base_idx[0] + i0) * stride_ctrl[0] +
                (base_idx[1] + i1) * stride_ctrl[1] +
                (base_idx[2] + i2) * stride_ctrl[2] +
                (base_idx[3] + i3) * stride_ctrl[3];

            REAL const c = coeff[idxc];

            REAL const w = B[0][i0] * B[1][i1] * B[2][i2] * B[3][i3];
            spline_val += w * c;

            spline_dshift[0] += dB[0][i0] * B[1][i1] * B[2][i2] * B[3][i3] * c;
            spline_dshift[1] += B[0][i0] * dB[1][i1] * B[2][i2] * B[3][i3] * c;
            spline_dshift[2] += B[0][i0] * B[1][i1] * dB[2][i2] * B[3][i3] * c;
            spline_dshift[3] += B[0][i0] * B[1][i1] * B[2][i2] * dB[3][i3] * c;
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
            int const idxc =
                (base_idx[0] + i0) * stride_ctrl[0] +
                (base_idx[1] + i1) * stride_ctrl[1] +
                (base_idx[2] + i2) * stride_ctrl[2] +
                (base_idx[3] + i3) * stride_ctrl[3] +
                (base_idx[4] + i4) * stride_ctrl[4];

            REAL const c = coeff[idxc];

            REAL const w = B[0][i0] * B[1][i1] * B[2][i2] * B[3][i3] * B[4][i4];
            spline_val += w * c;

            spline_dshift[0] += dB[0][i0] * B[1][i1] * B[2][i2] * B[3][i3] * B[4][i4] * c;
            spline_dshift[1] += B[0][i0] * dB[1][i1] * B[2][i2] * B[3][i3] * B[4][i4] * c;
            spline_dshift[2] += B[0][i0] * B[1][i1] * dB[2][i2] * B[3][i3] * B[4][i4] * c;
            spline_dshift[3] += B[0][i0] * B[1][i1] * B[2][i2] * dB[3][i3] * B[4][i4] * c;
            spline_dshift[4] += B[0][i0] * B[1][i1] * B[2][i2] * B[3][i3] * dB[4][i4] * c;
        }
    }
    else // D == 6
    {
        for (int i0 = 0; i0 < 4; ++i0)
        for (int i1 = 0; i1 < 4; ++i1)
        for (int i2 = 0; i2 < 4; ++i2)
        for (int i3 = 0; i3 < 4; ++i3)
        for (int i4 = 0; i4 < 4; ++i4)
        for (int i5 = 0; i5 < 4; ++i5)
        {
            int const idxc =
                (base_idx[0] + i0) * stride_ctrl[0] +
                (base_idx[1] + i1) * stride_ctrl[1] +
                (base_idx[2] + i2) * stride_ctrl[2] +
                (base_idx[3] + i3) * stride_ctrl[3] +
                (base_idx[4] + i4) * stride_ctrl[4] +
                (base_idx[5] + i5) * stride_ctrl[5];

            REAL const c = coeff[idxc];

            REAL const w = B[0][i0] * B[1][i1] * B[2][i2] * B[3][i3] * B[4][i4] * B[5][i5];
            spline_val += w * c;

            spline_dshift[0] += dB[0][i0] * B[1][i1] * B[2][i2] * B[3][i3] * B[4][i4] * B[5][i5] * c;
            spline_dshift[1] += B[0][i0] * dB[1][i1] * B[2][i2] * B[3][i3] * B[4][i4] * B[5][i5] * c;
            spline_dshift[2] += B[0][i0] * B[1][i1] * dB[2][i2] * B[3][i3] * B[4][i4] * B[5][i5] * c;
            spline_dshift[3] += B[0][i0] * B[1][i1] * B[2][i2] * dB[3][i3] * B[4][i4] * B[5][i5] * c;
            spline_dshift[4] += B[0][i0] * B[1][i1] * B[2][i2] * B[3][i3] * dB[4][i4] * B[5][i5] * c;
            spline_dshift[5] += B[0][i0] * B[1][i1] * B[2][i2] * B[3][i3] * B[4][i4] * dB[5][i5] * c;
        }
    }

    // ---- output value ----
    value[point_index] = amp * spline_val + offset;

    // ---- output derivatives (same convention as spline3d) ----
    REAL * der = derivative + point_index;

    der[0 * n_points] = spline_val; // d/d(amp)
    for (d = 0; d < D; ++d)
        der[(1 + d) * n_points] = -amp * spline_dshift[d]; // d/d(shift_d)
    der[(1 + D) * n_points] = REAL(1); // d/d(offset)
}

#endif
