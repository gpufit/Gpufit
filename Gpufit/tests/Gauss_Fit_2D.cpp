#define BOOST_TEST_MODULE Gpufit

#include "Gpufit/gpufit.h"

#include <boost/test/included/unit_test.hpp>

#include <array>

template<std::size_t SIZE>
void generate_gauss_2d(std::array< REAL , SIZE>& values, REAL const s = .5f)
{
    int const size_x = int(std::sqrt(SIZE));
    int const size_y = size_x;

    REAL const a = 4;
    REAL const x0 = (REAL(size_x) - 1) / 2;
    REAL const y0 = (REAL(size_y) - 1) / 2;
    REAL const b = 1;

    for (int point_index_y = 0; point_index_y < size_y; point_index_y++)
    {
        for (int point_index_x = 0; point_index_x < size_x; point_index_x++)
        {
            int const point_index = point_index_y * size_x + point_index_x;
            REAL const argx = ((point_index_x - x0)*(point_index_x - x0)) / (2 * s * s);
            REAL const argy = ((point_index_y - y0)*(point_index_y - y0)) / (2 * s * s);
            REAL const ex = exp(-argx) * exp(-argy);
            values[point_index] = a * ex + b;
        }
    }
}

void gauss_fit_2d()
{
    std::size_t const n_fits{ 1 };
    std::size_t const n_points{ 25 };
    std::array< REAL, n_points > data{};
    generate_gauss_2d(data);
    std::array< REAL, n_points > weights{};
    std::fill(weights.begin(), weights.end(), 1.f);
    std::array< REAL, 5 > initial_parameters{ { 3, 1.8f, 2.2f, .4f, 0 } };
    REAL tolerance{ .001f };
    int max_n_iterations{ 10 };
    std::array< int, 5 > parameters_to_fit{ { 1, 1, 1, 1, 1 } };
    std::array< REAL, 5 > output_parameters;
    int output_states;
    REAL output_chi_square;
    int output_n_iterations;

    int const status
        = gpufit
        (
            n_fits,
            n_points,
            data.data(),
            0,
            GAUSS_2D,
            initial_parameters.data(),
            tolerance,
            max_n_iterations,
            parameters_to_fit.data(),
            LSE,
            0,
            0,
            output_parameters.data(),
            &output_states,
            &output_chi_square,
            &output_n_iterations
        );

    BOOST_CHECK(status == 0);
    BOOST_CHECK(output_states == 0);
    BOOST_CHECK(output_n_iterations <= max_n_iterations);
    BOOST_CHECK(output_chi_square < 1e-6f);

    int const status_with_weights
        = gpufit
        (
            n_fits,
            n_points,
            data.data(),
            weights.data(),
            GAUSS_2D,
            initial_parameters.data(),
            tolerance,
            max_n_iterations,
            parameters_to_fit.data(),
            LSE,
            0,
            0,
            output_parameters.data(),
            &output_states,
            &output_chi_square,
            &output_n_iterations
        );

    BOOST_CHECK(status_with_weights == 0);
    BOOST_CHECK(output_states == 0);
    BOOST_CHECK(output_n_iterations <= max_n_iterations);
    BOOST_CHECK(output_chi_square < 1e-6f);
}

void gauss_fit_2d_large_dataset()
{
    std::size_t const n_fits{ 1 };
    std::size_t const n_points{ 2500 };
    std::array< REAL, n_points > data{};
    generate_gauss_2d(data, 5);
    std::array< REAL, n_points > weights{};
    std::fill(weights.begin(), weights.end(), 1.f);
    std::array< REAL, 5 > initial_parameters{ { 3, 24, 25, 4, 0 } };
    REAL tolerance{ .001f };
    int max_n_iterations{ 10 };
    std::array< int, 5 > parameters_to_fit{ { 1, 1, 1, 1, 1 } };
    std::array< REAL, 5 > output_parameters;
    int output_states;
    REAL output_chi_square;
    int output_n_iterations;

    int const status
        = gpufit
        (
            n_fits,
            n_points,
            data.data(),
            0,
            GAUSS_2D,
            initial_parameters.data(),
            tolerance,
            max_n_iterations,
            parameters_to_fit.data(),
            LSE,
            0,
            0,
            output_parameters.data(),
            &output_states,
            &output_chi_square,
            &output_n_iterations
        );

    BOOST_CHECK(status == 0);
    BOOST_CHECK(output_states == 0);
    BOOST_CHECK(output_n_iterations <= max_n_iterations);
    BOOST_CHECK(output_chi_square < 1e-6f);

    int const status_with_weights
        = gpufit
        (
            n_fits,
            n_points,
            data.data(),
            weights.data(),
            GAUSS_2D,
            initial_parameters.data(),
            tolerance,
            max_n_iterations,
            parameters_to_fit.data(),
            LSE,
            0,
            0,
            output_parameters.data(),
            &output_states,
            &output_chi_square,
            &output_n_iterations
        );

    BOOST_CHECK(status_with_weights == 0);
    BOOST_CHECK(output_states == 0);
    BOOST_CHECK(output_n_iterations <= max_n_iterations);
    BOOST_CHECK(output_chi_square < 1e-6f);
}

BOOST_AUTO_TEST_CASE( Gauss_Fit_2D )
{
    // 2d gauss fit
    gauss_fit_2d();

    // 2d gauss fit with large data set
    gauss_fit_2d_large_dataset();
}
