#define BOOST_TEST_MODULE Gpufit

#define PI 3.1415926535897f

#include "Gpufit/gpufit.h"

#include <boost/test/included/unit_test.hpp>

#include <array>
#include <cmath>

template<std::size_t SIZE>
void generate_gauss_2d_rotated(std::array< REAL, SIZE>& values, std::array< REAL, 7 > const & parameters)
{
    int const size_x = int(std::sqrt(SIZE));
    int const size_y = size_x;

    REAL const a = parameters[0];
    REAL const x0 = parameters[1];
    REAL const y0 = parameters[2];
    REAL const sx = parameters[3];
    REAL const sy = parameters[4];
    REAL const b = parameters[5];
    REAL const r = parameters[6];

    for (int point_index_y = 0; point_index_y < size_y; point_index_y++)
    {
        for (int point_index_x = 0; point_index_x < size_x; point_index_x++)
        {
            int const point_index = point_index_y * size_x + point_index_x;
            REAL const arga = ((point_index_x - x0) * cos(r)) - ((point_index_y - y0) * sin(r));
            REAL const argb = ((point_index_x - x0) * sin(r)) + ((point_index_y - y0) * cos(r));
            REAL const ex = exp((-.5f) * (((arga / sx) * (arga / sx)) + ((argb / sy) * (argb / sy))));
            values[point_index] = a * ex + b;
        }
    }
}

BOOST_AUTO_TEST_CASE( Gauss_Fit_2D_Rotated )
{
    std::size_t const n_fits{ 1 } ;
    std::size_t const n_points{ 64 } ;

    std::array< REAL, 7 > true_parameters{ { 10, 3.5f, 3.5f, .4f, .5f, 1, PI / 16 } };

    std::array< REAL, n_points > data{};
    generate_gauss_2d_rotated(data, true_parameters);

    std::array< REAL, n_points > weights{};
    std::fill(weights.begin(), weights.end(), 1.f);

    std::array< REAL, 7 > initial_parameters{ { 8, 3.4f, 3.6f, .4f, .5f, 2, 0 } };

    REAL tolerance{ .0000001f };

    int max_n_iterations{ 10 };

    std::array< int, 7 > parameters_to_fit{ { 1, 1, 1, 1, 1, 1, 1 } };

    std::array< REAL, 7 > output_parameters;
    int output_states;
    REAL output_chi_square;
    int output_n_iterations;

    int const status
            = gpufit
            (
                n_fits,
                n_points,
                data.data(),
                weights.data(),
                GAUSS_2D_ROTATED,
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
            ) ;

    BOOST_CHECK(status == 0);
    BOOST_CHECK(output_states == 0);
    BOOST_CHECK(output_n_iterations <= max_n_iterations);
    BOOST_CHECK(output_chi_square < 1e-6f);

    BOOST_CHECK(std::abs(output_parameters[0] - true_parameters[0]) < 2e-5f);
    BOOST_CHECK(std::abs(output_parameters[1] - true_parameters[1]) < 1e-6f);
    BOOST_CHECK(std::abs(output_parameters[2] - true_parameters[2]) < 1e-6f);
    BOOST_CHECK(std::abs(output_parameters[3] - true_parameters[3]) < 1e-6f);
    BOOST_CHECK(std::abs(output_parameters[4] - true_parameters[4]) < 1e-6f);
    BOOST_CHECK(std::abs(output_parameters[5] - true_parameters[5]) < 1e-6f);
    BOOST_CHECK(std::abs(output_parameters[6] - true_parameters[6]) < 1e-6f);
}
