// natural_bspline_nd_fit_example_2d_gauss.cpp
//
// Test for Gpufit model: NATURAL_BSPLINE_2D
// Model parameters:
//   p[0] = amplitude
//   p[1 + d] = shift along axis d(NOT absolute peak position), for d = 0..D - 1
//   p[1 + D] = offset
//
// Strategy:
// 1) Build a ND natural B-spline "shape" from samples of a 2D Gaussian (on grid)
// 2) Generate noisy synthetic data from the *Gaussian* (not from the spline)
// 3) Fit that data using the NATURAL_BSPLINE_2D gpufit model
//
// Requires:
// - Gpufit compiled with your NATURAL_BSPLINE_2D model registered
// - Gpuspline C-API function available at runtime via splines.dll:
//     int calculate_coefficients_natural_bspline_nd(int num_dims, const int* dims, REAL* data, REAL* coefficients_out)
//     int calculate_values_natural_bspline_nd(int num_dims, const int* data_dims, const REAL* coefficients, int n_input_coords, const REAL* input_coords, int flag_fast_evaluate, REAL* output_values)
//
// Notes:
// - We build knots locally (deterministic, uniform grid), matching init_knot_vector().

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>

// Undef common Windows macro collisions
#ifdef ERROR
#undef ERROR
#endif
#ifdef OK
#undef OK
#endif

#endif

#include "../../Gpufit/gpufit.h"

#include <vector>
#include <random>
#include <iostream>
#include <chrono>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <algorithm>


#ifndef NATURAL_BSPLINE_ND_MAX_DIMS
#define NATURAL_BSPLINE_ND_MAX_DIMS 6
#endif


#ifdef _WIN32


typedef int(__cdecl *calculate_coefficients_fn)(
    int num_dims,
    const int* dims,
    REAL* data,
    REAL* knots_out,
    REAL* coeff_strides_out,
    REAL* coefficients_out
    );


typedef int(__cdecl *calculate_values_fn)(
    int num_dims,
    const int* data_dims,
    const REAL* coefficients,
    int n_input_coords,
    const REAL* input_coords,
    int flag_fast_evaluate,
    REAL* output_values
    );


typedef int(__cdecl *calculate_sizes_fn)(
    int num_dims,
    const int* dims,
    size_t* knots_len_out,
    size_t* coeff_strides_len_out,
    size_t* coefficients_len_out
    );


static HMODULE g_splines_dll = 0;


static FARPROC load_symbol_or_throw(const char* name)
{
    if (!g_splines_dll)
    {
        g_splines_dll = LoadLibraryA("splines.dll");
        if (!g_splines_dll)
            throw std::runtime_error("LoadLibraryA(\"splines.dll\") failed. Ensure splines.dll is next to the exe.");
    }

    FARPROC p = GetProcAddress(g_splines_dll, name);
    if (!p)
        throw std::runtime_error(std::string("GetProcAddress failed for symbol: ") + name);
    return p;
}


static calculate_coefficients_fn load_calculate_coefficients()
{
    return reinterpret_cast<calculate_coefficients_fn>(
        load_symbol_or_throw("calculate_coefficients_natural_bspline_nd"));
}


static calculate_values_fn load_calculate_values()
{
    return reinterpret_cast<calculate_values_fn>(
        load_symbol_or_throw("calculate_values_natural_bspline_nd"));
}


static calculate_sizes_fn load_calculate_sizes()
{
    return reinterpret_cast<calculate_sizes_fn>(
        load_symbol_or_throw("calculate_sizes_natural_bspline_nd"));
}


#endif


static void print_vector(const char* label, std::vector<REAL> const& v, int max_count = 32)
{
    std::cout << label << " (n=" << v.size() << "):\n";
    int n = (int)v.size();
    int m = (n < max_count) ? n : max_count;
    for (int i = 0; i < m; ++i)
    {
        std::cout << "  [" << i << "] " << v[i] << "\n";
    }
    if (m < n) std::cout << "  ...\n";
}


// ---- Gpuspline C interface (link from Gpuspline project) ----
extern "C" int calculate_coefficients_natural_bspline_nd(
    int num_dims,
    const int* dims,
    REAL* data,
    REAL* knots_out,
    REAL* coeff_strides_out,
    REAL* coefficients_out
    );


static void build_natural_bspline_1d_knots(int N, std::vector<REAL>& knots)
{
    const int k = 3; // cubic
    knots.resize(N + 6);

    for (int i = 0; i < k; ++i) knots[i] = REAL(0);
    for (int i = 0; i < N; ++i) knots[k + i] = (REAL)i;
    for (int i = 0; i < k; ++i) knots[k + N + i] = (REAL)(N - 1);
}


static void pack_user_info_natural_bspline_nd(
    int D,
    int const* data_dims,
    int const* spline_dims,
    std::vector<REAL> const& knots_concat,        // from DLL
    std::vector<REAL> const& coeff_strides_concat,// from DLL (len=D)
    std::vector<REAL> const& coefficients,
    std::vector<REAL>& user_info_real)
{

    // header length = 1 + 3D
    int const header_len = 1 + 3 * D;

    // expected sizes
    size_t expected_knots = 0;
    size_t expected_coeff = 1;
    for (int d = 0; d < D; ++d)
    {
        expected_knots += (size_t)(spline_dims[d] + 6);
        expected_coeff *= (size_t)(spline_dims[d] + 2);
    }

    if (knots_concat.size() != expected_knots)
        throw std::runtime_error("pack_user_info_natural_bspline_nd: knots size mismatch");
    if (coeff_strides_concat.size() != (size_t)D)
        throw std::runtime_error("pack_user_info_natural_bspline_nd: strides size mismatch");
    if (coefficients.size() != expected_coeff)
        throw std::runtime_error("pack_user_info_natural_bspline_nd: coefficients size mismatch");

    user_info_real.resize((size_t)header_len + expected_knots + expected_coeff);

    size_t off = 0;

    // D
    user_info_real[off++] = (REAL)D;

    // data_dims
    for (int d = 0; d < D; ++d) user_info_real[off++] = (REAL)data_dims[d];

    // spline_dims
    for (int d = 0; d < D; ++d) user_info_real[off++] = (REAL)spline_dims[d];

    // stride_ctrl (from DLL)
    for (int d = 0; d < D; ++d) user_info_real[off++] = coeff_strides_concat[(size_t)d];

    // knots (from DLL, concatenated)
    for (size_t i = 0; i < expected_knots; ++i)
        user_info_real[off++] = knots_concat[i];

    // coefficients
    for (size_t i = 0; i < expected_coeff; ++i)
        user_info_real[off++] = coefficients[i];

    if (off != user_info_real.size())
        throw std::runtime_error("pack_user_info_natural_bspline_nd: internal size mismatch");

}


static inline REAL gaussian_2d(
    REAL x, REAL y,
    REAL amp,
    REAL cx, REAL cy,
    REAL sx, REAL sy,
    REAL offset)
{
    REAL dx = x - cx;
    REAL dy = y - cy;
    REAL inv2sx2 = REAL(1) / (REAL(2) * sx * sx);
    REAL inv2sy2 = REAL(1) / (REAL(2) * sy * sy);
    return amp * std::exp(-(dx*dx)*inv2sx2 - (dy*dy)*inv2sy2) + offset;
}


static void generate_gaussian_shape_samples_2d(
    int nx, int ny,
    REAL sx, REAL sy,
    REAL cx, REAL cy,
    std::vector<REAL>& shape)
{
    shape.resize((size_t)nx * (size_t)ny);
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i)
        {
            REAL x = (REAL)i;
            REAL y = (REAL)j;
            shape[(size_t)i + (size_t)nx * (size_t)j] =
                gaussian_2d(x, y, REAL(1), cx, cy, sx, sy, REAL(0)); // amp=1, offset=0
        }
}


// Build integer-grid input_coords for reconstruction: (x,y) pairs
static void build_integer_grid_coords_2d(int nx, int ny, std::vector<REAL>& coords_xy)
{
    coords_xy.resize((size_t)nx * (size_t)ny * 2);
    size_t t = 0;
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i)
        {
            coords_xy[t++] = (REAL)i;
            coords_xy[t++] = (REAL)j;
        }
}


static void summarize_results(
    std::size_t n_fits,
    std::size_t n_params,
    std::vector<int> const& states,
    std::vector<REAL> const& true_params,
    std::vector<REAL> const& fitted_params,
    std::vector<REAL> const& chi_square,
    std::vector<int> const& n_iter)
{
    // --- fit state histogram ---
    std::vector<int> hist(5, 0);
    for (std::size_t i = 0; i < states.size(); ++i)
    {
        int s = states[i];
        if (s >= 0 && s < (int)hist.size())
            hist[s]++;
    }

    std::cout << "ratio converged              " << (REAL)hist[0] / (REAL)n_fits << "\n";
    std::cout << "ratio max iteration exceeded " << (REAL)hist[1] / (REAL)n_fits << "\n";
    std::cout << "ratio singular hessian       " << (REAL)hist[2] / (REAL)n_fits << "\n";
    std::cout << "ratio neg curvature MLE      " << (REAL)hist[3] / (REAL)n_fits << "\n";
    std::cout << "ratio gpu not read           " << (REAL)hist[4] / (REAL)n_fits << "\n";

    std::size_t const n_conv = (std::size_t)hist[0];
    if (n_conv == 0)
    {
        std::cout << "No converged fits.\n";
        return;
    }

    // --- means ---
    std::vector<REAL> fit_mean(n_params, 0);
    std::vector<REAL> true_mean(n_params, 0);

    for (std::size_t f = 0; f < n_fits; ++f)
    {
        if (states[f] != FitState::CONVERGED)
            continue;

        for (std::size_t j = 0; j < n_params; ++j)
        {
            fit_mean[j] += fitted_params[f * n_params + j];
            true_mean[j] += true_params[f * n_params + j];
        }
    }

    for (std::size_t j = 0; j < n_params; ++j)
    {
        fit_mean[j] /= (REAL)n_conv;
        true_mean[j] /= (REAL)n_conv;
    }

    // --- std of fitted parameters (population spread) ---
    std::vector<REAL> fit_std(n_params, 0);

    for (std::size_t f = 0; f < n_fits; ++f)
    {
        if (states[f] != FitState::CONVERGED)
            continue;

        for (std::size_t j = 0; j < n_params; ++j)
        {
            REAL d = fitted_params[f * n_params + j] - fit_mean[j];
            fit_std[j] += d * d;
        }
    }

    for (std::size_t j = 0; j < n_params; ++j)
        fit_std[j] = std::sqrt(fit_std[j] / (REAL)n_conv);

    // --- RMS error w.r.t. true parameters (NEW, important) ---
    std::vector<REAL> rms_err(n_params, 0);

    for (std::size_t f = 0; f < n_fits; ++f)
    {
        if (states[f] != FitState::CONVERGED)
            continue;

        for (std::size_t j = 0; j < n_params; ++j)
        {
            REAL diff =
                fitted_params[f * n_params + j]
                - true_params[f * n_params + j];
            rms_err[j] += diff * diff;
        }
    }

    for (std::size_t j = 0; j < n_params; ++j)
        rms_err[j] = std::sqrt(rms_err[j] / (REAL)n_conv);

    // --- print parameter summary ---
    for (std::size_t j = 0; j < n_params; ++j)
    {
        std::cout
            << "param " << j
            << " true_mean " << true_mean[j]
            << " fit_mean " << fit_mean[j]
            << " std " << fit_std[j]
            << " rms_err " << rms_err[j]
            << "\n";
    }

    // --- chi-square and iteration stats ---
    REAL chi_mean = 0;
    REAL it_mean = 0;

    for (std::size_t f = 0; f < n_fits; ++f)
    {
        if (states[f] != FitState::CONVERGED)
            continue;

        chi_mean += chi_square[f];
        it_mean += (REAL)n_iter[f];
    }

    chi_mean /= (REAL)n_conv;
    it_mean /= (REAL)n_conv;

    std::cout << "mean chi square " << chi_mean << "\n";
    std::cout << "mean iterations " << it_mean << "\n";
}


void natural_bspline_nd_fit_example_2d()
{
    std::size_t const n_fits = 10000;

    int const D = 2;

    int const spline_dims[2] = { 20, 16 };   // spline support (domain length per axis)
    int const data_dims[2] = { 80, 64 };   // fit data grid

    REAL const sx = (REAL)3.0;
    REAL const sy = (REAL)2.5;
    REAL const tpl_cx = (REAL)0.5 * (REAL)(spline_dims[0] - 1);
    REAL const tpl_cy = (REAL)0.5 * (REAL)(spline_dims[1] - 1);

    REAL const noise_sigma = (REAL)1.0;

    std::size_t const n_points_per_fit =
        (std::size_t)data_dims[0] * (std::size_t)data_dims[1];

    std::size_t const n_model_parameters = 1 + D + 1; // amp, shift_x, shift_y, offset

    // ---- build template samples (Gaussian on tpl grid) ----
    std::vector<REAL> shape_tpl;
    generate_gaussian_shape_samples_2d(
        spline_dims[0], spline_dims[1], sx, sy, tpl_cx, tpl_cy, shape_tpl);

    // ---- compute spline coefficients using ND DLL ----
    size_t knots_len = 0;
    size_t strides_len = 0;
    size_t coeff_len = 0;

    #ifdef _WIN32
    calculate_sizes_fn calc_sizes = load_calculate_sizes();
    int rc_sizes = calc_sizes(D, spline_dims, &knots_len, &strides_len, &coeff_len);
    if (rc_sizes != 0) throw std::runtime_error("calculate_sizes_natural_bspline_nd failed");
    #else
    throw std::runtime_error("Windows-only scaffold (dll).");
    #endif

    std::vector<REAL> knots(knots_len);
    std::vector<REAL> coeff_strides(strides_len);
    std::vector<REAL> coeff(coeff_len);

    {
        std::vector<REAL> shape_copy = shape_tpl;

        #ifdef _WIN32
        calculate_coefficients_fn calc_coeff = load_calculate_coefficients();
        int rc = calc_coeff(
            D,
            spline_dims,
            shape_copy.data(),
            knots.data(),
            coeff_strides.data(),
            coeff.data());
        if (rc != 0) throw std::runtime_error("calculate_coefficients_natural_bspline_nd failed");
        #else
        throw std::runtime_error("Windows-only scaffold (dll).");
        #endif
    }


    // ---- DEBUG: reconstruct tpl values at integer coords ----
    std::vector<REAL> coords_xy;
    build_integer_grid_coords_2d(spline_dims[0], spline_dims[1], coords_xy);

    int const n_coords = spline_dims[0] * spline_dims[1];
    std::vector<REAL> recon((size_t)n_coords);

    #ifdef _WIN32
    calculate_values_fn calc_vals = load_calculate_values();
    int rc_vals = calc_vals(
        D,
        spline_dims,
        coeff.data(),
        n_coords,
        coords_xy.data(),
        1, // fast
        recon.data());
    if (rc_vals != 0) throw std::runtime_error("calculate_values_natural_bspline_nd failed");
    #else
    throw std::runtime_error("Windows-only scaffold (dll).");
    #endif

    // error report (max_abs + rms)
    REAL max_abs_err = 0;
    REAL rms = 0;
    for (int idx = 0; idx < n_coords; ++idx)
    {
        REAL diff = recon[(size_t)idx] - shape_tpl[(size_t)idx];
        REAL a = std::abs(diff);
        if (a > max_abs_err) max_abs_err = a;
        rms += diff * diff;
    }
    rms = std::sqrt(rms / (REAL)n_coords);
    std::cout << "Reconstruction max_abs_err = " << max_abs_err << "\n";
    std::cout << "Reconstruction RMS_err     = " << rms << "\n";

    // ---- pack ND user_info bytes ----
    std::vector<REAL> user_info_real;
    pack_user_info_natural_bspline_nd(D, data_dims, spline_dims, knots, coeff_strides, coeff, user_info_real);


    char* user_info_bytes = reinterpret_cast<char*>(user_info_real.data());
    std::size_t user_info_size_bytes = user_info_real.size() * sizeof(REAL);

    // ---- RNG ----
    std::mt19937 rng(0);
    std::uniform_real_distribution<REAL> uni01(0, 1);
    std::normal_distribution<REAL> nrm(0, 1);

    // ---- true params + initial guesses ----
    std::vector<REAL> true_params(n_fits * n_model_parameters);
    std::vector<REAL> initial_params(n_fits * n_model_parameters);

    // Choose centers away from edges in DATA grid
    REAL const cx_min = (REAL)10.0;
    REAL const cx_max = (REAL)(data_dims[0] - 1) - (REAL)10.0;
    REAL const cy_min = (REAL)10.0;
    REAL const cy_max = (REAL)(data_dims[1] - 1) - (REAL)10.0;

    for (std::size_t f = 0; f < n_fits; ++f)
    {
        REAL amp_t = (REAL)100.0 * ((REAL)0.8 + (REAL)0.4 * uni01(rng));
        REAL cx_t = cx_min + (cx_max - cx_min) * uni01(rng);
        REAL cy_t = cy_min + (cy_max - cy_min) * uni01(rng);
        REAL off_t = (REAL)10.0  * ((REAL)0.8 + (REAL)0.4 * uni01(rng));

        REAL shift_x = cx_t - tpl_cx;
        REAL shift_y = cy_t - tpl_cy;

        size_t base = f * n_model_parameters;
        true_params[base + 0] = amp_t;
        true_params[base + 1] = shift_x;
        true_params[base + 2] = shift_y;
        true_params[base + 3] = off_t;

        // initial guesses
        initial_params[base + 0] = amp_t * ((REAL)0.8 + (REAL)0.4 * uni01(rng));
        initial_params[base + 1] = shift_x + (REAL)1.0 * ((REAL)-0.5 + uni01(rng));
        initial_params[base + 2] = shift_y + (REAL)1.0 * ((REAL)-0.5 + uni01(rng));
        initial_params[base + 3] = off_t  * ((REAL)0.8 + (REAL)0.4 * uni01(rng));
    }

    // ---- generate noisy synthetic data from GAUSSIAN truth on DATA grid ----
    std::vector<REAL> data(n_fits * n_points_per_fit);

    for (std::size_t f = 0; f < n_fits; ++f)
    {
        size_t base_p = f * n_model_parameters;
        REAL amp_t = true_params[base_p + 0];
        REAL cx_t = true_params[base_p + 1] + tpl_cx;
        REAL cy_t = true_params[base_p + 2] + tpl_cy;
        REAL off_t = true_params[base_p + 3];

        size_t base_d = f * n_points_per_fit;

        for (int y = 0; y < data_dims[1]; ++y)
            for (int x = 0; x < data_dims[0]; ++x)
            {
                REAL yy = (REAL)y;
                REAL xx = (REAL)x;
                REAL val = gaussian_2d(xx, yy, amp_t, cx_t, cy_t, sx, sy, off_t);
                data[base_d + (size_t)x + (size_t)data_dims[0] * (size_t)y] =
                    val + noise_sigma * nrm(rng);
            }
    }

    // ---- gpufit call ----
    REAL const tolerance = REAL(1e-3);
    int const max_number_iterations = 50;
    int const estimator_id = LSE;

    int const model_id = NATURAL_BSPLINE_2D; // your new registered model id

    std::vector<int> parameters_to_fit(n_model_parameters, 1);

    std::vector<REAL> output_parameters(n_fits * n_model_parameters);
    std::vector<int>  output_states(n_fits);
    std::vector<REAL> output_chi_square(n_fits);
    std::vector<int>  output_number_iterations(n_fits);

    auto t0 = std::chrono::high_resolution_clock::now();

    int status = gpufit(
        n_fits,
        n_points_per_fit,
        data.data(),
        0,
        model_id,
        initial_params.data(),
        tolerance,
        max_number_iterations,
        parameters_to_fit.data(),
        estimator_id,
        user_info_size_bytes,
        user_info_bytes,
        output_parameters.data(),
        output_states.data(),
        output_chi_square.data(),
        output_number_iterations.data());

    auto t1 = std::chrono::high_resolution_clock::now();

    if (status != ReturnState::OK)
        throw std::runtime_error(gpufit_get_last_error());

    std::cout << "execution time "
        << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
        << " ms\n";

    summarize_results(
        n_fits,
        n_model_parameters,
        output_states,
        true_params,
        output_parameters,
        output_chi_square,
        output_number_iterations);
}

int main()
{
    try
    {
        natural_bspline_nd_fit_example_2d();
        std::cout << "\nExample completed!\n";
    }
    catch (std::exception const& e)
    {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
