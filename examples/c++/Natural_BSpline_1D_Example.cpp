// natural_bspline_1d_fit_example.cpp
//
// Test for Gpufit model: NATURAL_BSPLINE_1D
// Model parameters:
//   p[0] amplitude
//   p[1] center
//   p[2] offset
//
// Strategy:
// 1) Build a 1D natural B-spline "shape" from samples of a Gaussian (on grid 0..N-1)
// 2) Generate noisy synthetic data from the *Gaussian* (not from the spline)
// 3) Fit that data using the NATURAL_BSPLINE_1D gpufit model
//
// Requires:
// - Gpufit compiled with your NATURAL_BSPLINE_1D model registered
// - Gpuspline C-API function available at link-time:
//     int calculate_coefficients_natural_bspline_1d(int N, REAL* data, REAL* coeff_out);
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

#ifdef _WIN32

typedef int(__cdecl *calculate_coefficients_fn)(
    int num_data_points,
    REAL* data,
    REAL* coefficients_out
    );

typedef int(__cdecl *calculate_values_fn)(
    int num_data_points,
    const REAL* coefficients,
    int n_input_coords,
    const REAL* input_coords,
    int flag_fast_evaluate,
    REAL* output_values
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
        load_symbol_or_throw("calculate_coefficients_natural_bspline_1d"));
}

static calculate_values_fn load_calculate_values()
{
    return reinterpret_cast<calculate_values_fn>(
        load_symbol_or_throw("calculate_values_natural_bspline_1d"));
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
extern "C" int calculate_coefficients_natural_bspline_1d(
    int num_data_points,
    REAL* data,
    REAL* coefficients_out
    );

// ---- helpers ----
static void build_natural_bspline_1d_knots(int N, std::vector<REAL>& knots)
{
    const int k = 3; // cubic
    knots.resize(N + 6);

    for (int i = 0; i < k; ++i) knots[i] = REAL(0);

    for (int i = 0; i < N; ++i) knots[k + i] = static_cast<REAL>(i);

    for (int i = 0; i < k; ++i) knots[k + N + i] = static_cast<REAL>(N - 1);
}

static void pack_user_info_natural_bspline_1d(
    int N,
    std::vector<REAL> const& knots,          // length N+6
    std::vector<REAL> const& coeff,          // length N+2
    std::vector<REAL>& user_info_real)
{
    int const num_coeff = N + 2;
    int const num_knots = N + 6;

    user_info_real.resize(1 + num_knots + num_coeff);
    user_info_real[0] = static_cast<REAL>(num_coeff);

    for (int i = 0; i < num_knots; ++i)
        user_info_real[1 + i] = knots[i];

    for (int i = 0; i < num_coeff; ++i)
        user_info_real[1 + num_knots + i] = coeff[i];
}

static void generate_gaussian_shape_samples(
    int N,
    REAL sigma,
    REAL center,
    std::vector<REAL>& shape)
{
    shape.resize(N);
    REAL const inv2sig2 = REAL(1) / (REAL(2) * sigma * sigma);

    for (int i = 0; i < N; ++i)
    {
        REAL x = static_cast<REAL>(i);
        REAL dx = x - center;
        shape[i] = std::exp(-(dx * dx) * inv2sig2);
    }
}

static inline REAL gaussian_1d(
    REAL x,
    REAL amp,
    REAL center,
    REAL sigma,
    REAL offset)
{
    REAL dx = x - center;
    REAL inv2sig2 = REAL(1) / (REAL(2) * sigma * sigma);
    return amp * std::exp(-(dx * dx) * inv2sig2) + offset;
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


void natural_bspline_1d_fit_example()
{
    // ---- sizes ----
    std::size_t const n_fits = 10000;
    int const N_tpl = 20;
    int const N_data = 80;
    REAL const sigma = (REAL)3.0;
    REAL const tpl_center = (REAL)0.5 * (REAL)(N_tpl - 1);
    REAL const noise_sigma = 1.0;
    std::size_t const n_points_per_fit = (std::size_t)N_data;
    std::size_t const n_model_parameters = 3;   // [amp, shift, offset]


    // This defines the fixed spline "template" S(x).
    std::vector<REAL> shape_tpl;
    generate_gaussian_shape_samples(N_tpl, sigma, tpl_center, shape_tpl);


    // ---- DEBUG: print the Gaussian samples used to create the spline model ----
    print_vector("Gaussian template samples (shape)", shape_tpl, N_tpl);


    // ---- compute spline coefficients using Gpuspline C interface ----
    std::vector<REAL> coeff(N_tpl + 2);
    {
        std::vector<REAL> shape_copy = shape_tpl; // API takes non-const REAL*

        #ifdef _WIN32
        calculate_coefficients_fn calc_coeff = load_calculate_coefficients();
        int rc = calc_coeff(N_tpl, shape_copy.data(), coeff.data());
        #else
        throw std::runtime_error("This example uses LoadLibrary/GetProcAddress; implement dlopen/dlsym for non-Windows.");
        #endif

        if (rc != 0) throw std::runtime_error("calculate_coefficients_natural_bspline_1d returned error");

        if (rc != 0) throw std::runtime_error("calculate_coefficients_natural_bspline_1d failed");
    }


    // ---- DEBUG: reconstruct values from coefficients using calculate_values ----
    std::vector<REAL> x_coords(N_tpl);
    for (int i = 0; i < N_tpl; ++i)
        x_coords[i] = (REAL)i;

    std::vector<REAL> recon(N_tpl, 0);

    #ifdef _WIN32
    calculate_values_fn calc_vals = load_calculate_values();
    int rc_vals = calc_vals(
        N_tpl,
        coeff.data(),
        N_tpl,
        x_coords.data(),
        1,              // flag_fast_evaluate (1 = fast, 0 = slow)
        recon.data()
        );
    if (rc_vals != 0)
        throw std::runtime_error("calculate_values_natural_bspline_1d returned error");
    #else
    throw std::runtime_error("calculate_values test is Windows-only in this example (dll).");
    #endif

    print_vector("Spline reconstructed values at x=0..N-1", recon, N_tpl);

    // Optional: print error vs Gaussian template
    std::vector<REAL> diff(N_tpl);
    REAL max_abs_err = 0;
    REAL rms = 0;
    for (int i = 0; i < N_tpl; ++i)
    {
        diff[i] = recon[i] - shape_tpl[i];
        REAL a = std::abs(diff[i]);
        if (a > max_abs_err) max_abs_err = a;
        rms += diff[i] * diff[i];
    }
    rms = std::sqrt(rms / (REAL)N_tpl);

    print_vector("Reconstruction error (recon - shape)", diff, N_tpl);
    std::cout << "Reconstruction max_abs_err = " << max_abs_err << "\n";
    std::cout << "Reconstruction RMS_err     = " << rms << "\n";



    // ---- build knots locally ----
    std::vector<REAL> knots;
    build_natural_bspline_1d_knots(N_tpl, knots);

    // ---- pack user_info for NATURAL_BSPLINE_1D model ----
    std::vector<REAL> user_info_real;
    pack_user_info_natural_bspline_1d(N_tpl, knots, coeff, user_info_real);

    char* user_info_bytes = reinterpret_cast<char*>(user_info_real.data());
    std::size_t user_info_size_bytes = user_info_real.size() * sizeof(REAL);

    // ---- RNG ----
    std::mt19937 rng(0);
    std::uniform_real_distribution<REAL> uni01(0, 1);

    // ---- true parameters + initial guesses ----
    std::vector<REAL> true_center(n_fits);
    std::vector<REAL> true_params(n_fits * n_model_parameters);
    std::vector<REAL> initial_params(n_fits * n_model_parameters);

    // choose centers away from the extreme edges for easier convergence
    REAL const center_min = REAL(10.0);
    REAL const center_max = REAL(N_data - 1) - REAL(10.0);

    for (std::size_t f = 0; f < n_fits; ++f)
    {
        REAL amp_t = REAL(100.0) * (REAL(0.8) + REAL(0.4) * uni01(rng));
        REAL center_t = center_min + (center_max - center_min) * uni01(rng);
        REAL offset_t = REAL(10.0) * (REAL(0.8) + REAL(0.4) * uni01(rng));

        REAL shift_t = center_t - tpl_center;

        true_params[f * n_model_parameters + 0] = amp_t;
        true_params[f * n_model_parameters + 1] = shift_t;
        true_params[f * n_model_parameters + 2] = offset_t;

        true_center[f] = center_t;

        // initial guesses
        initial_params[f * n_model_parameters + 0] = amp_t * (REAL(0.8) + REAL(0.4) * uni01(rng));
        initial_params[f * n_model_parameters + 1] = shift_t + REAL(1.0) * (REAL(-0.5) + uni01(rng)); // +/-0.5 px
        initial_params[f * n_model_parameters + 2] = offset_t * (REAL(0.8) + REAL(0.4) * uni01(rng));
    }

    // ---- generate synthetic noisy data from GAUSSIAN (ground truth) ----
    std::vector<REAL> data(n_fits * n_points_per_fit);

    for (std::size_t f = 0; f < n_fits; ++f)
    {
        REAL amp_t = true_params[f * n_model_parameters + 0];
        REAL center_t = true_center[f];
        REAL offset_t = true_params[f * n_model_parameters + 2];

        for (int j = 0; j < N_data; ++j)
        {
            REAL x = static_cast<REAL>(j);
            REAL y = gaussian_1d(x, amp_t, center_t, sigma, offset_t);

            std::normal_distribution<REAL> nrm(0, 1.0);
            data[f * n_points_per_fit + (std::size_t)j] = y + noise_sigma * nrm(rng);
        }
    }

    // ---- gpufit call ----
    REAL const tolerance = REAL(1e-3);
    int const max_number_iterations = 50;
    int const estimator_id = LSE;

    // IMPORTANT: must match your registered model ID
    int const model_id = NATURAL_BSPLINE_1D;

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
        0,                          // weights (nullptr)
        NATURAL_BSPLINE_1D,         // model_id
        initial_params.data(),
        tolerance,
        max_number_iterations,
        parameters_to_fit.data(),
        estimator_id,
        user_info_size_bytes,       // std::size_t
        user_info_bytes,            // char*
        output_parameters.data(),
        output_states.data(),
        output_chi_square.data(),
        output_number_iterations.data()
        );

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
        output_number_iterations
        );
}

int main()
{
    try
    {
        natural_bspline_1d_fit_example();
        std::cout << "\nExample completed!\n";
    }
    catch (std::exception const& e)
    {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
