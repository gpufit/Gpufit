#ifndef GPUFIT_NAT_BSPL_BASIS_CUH_INCLUDED
#define GPUFIT_NAT_BSPL_BASIS_CUH_INCLUDED

// Replace REAL with float or double as needed for your GPU build.

__device__ void nat_bspl_fast_cubic_basis_device(
    float x,                // query coordinate
    int span,               // knot interval
    const float* knots,     // pointer to knot vector (device)
    int num_control_points, // # control points for the axis
    float* basis            // output [4]
)
{
    const int k = 3; // Cubic
    float t = x - knots[span];

    if (span == k) {
        // First support interval
        // Paste: evaluate_fast_cubic_basis_first_span
        float t2 = t * t;
        float t3 = t2 * t;
        basis[0] = 1.0f - 3.0f * t + 3.0f * t2 - t3;
        basis[1] = 3.0f * t - 4.5f * t2 + 1.75f * t3;
        basis[2] = 1.5f * t2 - (11.0f / 12.0f) * t3;
        basis[3] = t3 / 6.0f;
    }
    else if (span == k + 1) {
        // Second support interval
        float t2 = t * t;
        float t3 = t2 * t;
        basis[0] = 0.25f - 0.75f * t + 0.75f * t2 - 0.25f * t3;
        basis[1] = (7.0f / 12.0f) + 0.25f * t - 1.25f * t2 + (7.0f / 12.0f) * t3;
        basis[2] = (1.0f / 6.0f) + 0.5f * t + 0.5f * t2 - 0.5f * t3;
        basis[3] = t3 / 6.0f;
    }
    else if (span >= k + 2 && span <= num_control_points - 3) {
        // Interior
        float t2 = t * t;
        float t3 = t2 * t;
        float omt = 1.0f - t;
        basis[0] = (1.0f / 6.0f) * omt * omt * omt;
        basis[1] = (1.0f / 6.0f) * (3.0f * t3 - 6.0f * t2 + 4.0f);
        basis[2] = (1.0f / 6.0f) * (-3.0f * t3 + 3.0f * t2 + 3.0f * t + 1.0f);
        basis[3] = (1.0f / 6.0f) * t3;
    }
    else if (span == num_control_points - 2) {
        // Second last span
        float omt = 1.0f - t;
        float omt2 = omt * omt;
        float omt3 = omt2 * omt;
        basis[0] = omt3 / 6.0f;
        basis[1] = (1.0f / 6.0f) + 0.5f * omt + 0.5f * omt2 - 0.5f * omt3;
        basis[2] = (7.0f / 12.0f) + 0.25f * omt - 1.25f * omt2 + (7.0f / 12.0f) * omt3;
        basis[3] = 0.25f - 0.75f * omt + 0.75f * omt2 - 0.25f * omt3;
    }
    else if (span == num_control_points - 1) {
        // Last span
        float omt = 1.0f - t;
        float omt2 = omt * omt;
        float omt3 = omt2 * omt;
        basis[0] = omt3 / 6.0f;
        basis[1] = 1.5f * omt2 - (11.0f / 12.0f) * omt3;
        basis[2] = 3.0f * omt - 4.5f * omt2 + 1.75f * omt3;
        basis[3] = 1.0f - 3.0f * omt + 3.0f * omt2 - omt3;
    }
}


__device__ void nat_bspl_fast_cubic_basis_derivative_device(
    float x,
    int span,
    const float* knots,
    int num_control_points,
    float* dbasis
)
{
    const int k = 3; // Cubic
    float t = x - knots[span];

    if (span == k) {
        float t2 = t * t;
        dbasis[0] = -3.0f + 6.0f * t - 3.0f * t2;
        dbasis[1] = 3.0f - 9.0f * t + 5.25f * t2;
        dbasis[2] = 3.0f * t - 2.75f * t2;
        dbasis[3] = 0.5f * t2;
    }
    else if (span == k + 1) {
        float t2 = t * t;
        dbasis[0] = -0.75f + 1.5f * t - 0.75f * t2;
        dbasis[1] = 0.25f - 2.5f * t + 1.75f * t2;
        dbasis[2] = 0.5f + t - 1.5f * t2;
        dbasis[3] = 0.5f * t2;
    }
    else if (span >= k + 2 && span <= num_control_points - 3) {
        float t2 = t * t;
        float omt = 1.0f - t;
        dbasis[0] = -0.5f * omt * omt;
        dbasis[1] = 1.5f * t2 - 2.0f * t;
        dbasis[2] = -1.5f * t2 + t + 0.5f;
        dbasis[3] = 0.5f * t2;
    }
    else if (span == num_control_points - 2) {
        float omt = 1.0f - t;
        float omt2 = omt * omt;
        dbasis[0] = -0.5f * omt2;
        dbasis[1] = -0.5f - omt + 1.5f * omt2;
        dbasis[2] = -0.25f + 2.5f * omt - 1.75f * omt2;
        dbasis[3] = 0.75f - 1.5f * omt + 0.75f * omt2;
    }
    else if (span == num_control_points - 1) {
        float omt = 1.0f - t;
        float omt2 = omt * omt;
        dbasis[0] = -0.5f * omt2;
        dbasis[1] = -3.0f * omt + 2.75f * omt2;
        dbasis[2] = -3.0f + 9.0f * omt - 5.25f * omt2;
        dbasis[3] = 3.0f - 6.0f * omt + 3.0f * omt2;
    }
}


#endif
