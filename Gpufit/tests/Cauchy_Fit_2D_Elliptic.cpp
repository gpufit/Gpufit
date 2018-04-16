#define BOOST_TEST_MODULE Gpufit

#include "Gpufit/gpufit.h"

#include <boost/test/included/unit_test.hpp>

#include <array>
#include <cmath>

template<std::size_t SIZE>
void generate_cauchy_2d_elliptic(std::array< REAL, SIZE>& values, std::array< REAL, 6 > const & parameters)
{
    int const size_x = int(std::sqrt(SIZE));
    int const size_y = size_x;

    REAL const a = parameters[0];
    REAL const x0 = parameters[1];
    REAL const y0 = parameters[2];
    REAL const sx = parameters[3];
    REAL const sy = parameters[4];
    REAL const b = parameters[5];

    for (int point_index_y = 0; point_index_y < size_y; point_index_y++)
    {
        for (int point_index_x = 0; point_index_x < size_x; point_index_x++)
        {
            int const point_index = point_index_y * size_x + point_index_x;
            REAL const argx = ((x0 - point_index_x) / sx) *((x0 - point_index_x) / sx) + 1;
            REAL const argy = ((y0 - point_index_y) / sy) *((y0 - point_index_y) / sy) + 1;
            values[point_index] = a / argx / argy + b;
        }
    }
}

BOOST_AUTO_TEST_CASE( Cauchy_Fit_2D_Elliptic )
{
    std::size_t const n_fits{ 1 } ;
    std::size_t const n_points{ 25 } ;

    std::array< REAL, 6 > const true_parameters{ { 4, 2, 2, 0.4f, 0.6f, 1 } };

    std::array< REAL, n_points > data{};
    generate_cauchy_2d_elliptic(data, true_parameters);

    std::array< REAL, n_points > weights{};
    std::fill(weights.begin(), weights.end(), 1.f);

    std::array< REAL, 6 > initial_parameters{ { 2, 1.8f, 2.2f, .5f, .5f, 0 } };

    REAL tolerance{ 1e-8f };

    int max_n_iterations{ 10 };

    std::array< int, 6 > parameters_to_fit{ { 1, 1, 1, 1, 1, 1 } };

    std::array< REAL, 6 > output_parameters;
    int output_state;
    REAL output_chi_square;
    int output_n_iterations;

    int const status
            = gpufit
            (
                n_fits,
                n_points,
                data.data(),
                weights.data(),
                CAUCHY_2D_ELLIPTIC,
                initial_parameters.data(),
                tolerance,
                max_n_iterations,
                parameters_to_fit.data(),
                LSE,
                0,
                0,
                output_parameters.data(),
                &output_state,
                &output_chi_square,
                &output_n_iterations
            ) ;

    BOOST_CHECK(status == 0);
    BOOST_CHECK(output_state == 0);
    BOOST_CHECK(output_n_iterations <= max_n_iterations);
    BOOST_CHECK(output_chi_square < 1e-6f);

    BOOST_CHECK(std::abs(output_parameters[0] - true_parameters[0]) < 1e-6f);
    BOOST_CHECK(std::abs(output_parameters[1] - true_parameters[1]) < 1e-6f);
    BOOST_CHECK(std::abs(output_parameters[2] - true_parameters[2]) < 1e-6f);
    BOOST_CHECK(std::abs(output_parameters[3] - true_parameters[3]) < 1e-6f);
    BOOST_CHECK(std::abs(output_parameters[4] - true_parameters[4]) < 1e-6f);
    BOOST_CHECK(std::abs(output_parameters[5] - true_parameters[5]) < 1e-6f);
}
