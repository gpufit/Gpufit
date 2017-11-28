#define BOOST_TEST_MODULE Gpufit

#include "Gpufit/gpufit.h"

#include <boost/test/included/unit_test.hpp>

#include <array>
#include <cmath>

template<std::size_t SIZE>
void generate_gauss_2d_elliptic(std::array< float, SIZE>& values, std::array< float, 6 > const & parameters)
{
    int const size_x = int(std::sqrt(SIZE));
    int const size_y = size_x;

    float const a = parameters[0];
    float const x0 = parameters[1];
    float const y0 = parameters[2];
    float const sx = parameters[3];
    float const sy = parameters[4];
    float const b = parameters[5];

    for (int point_index_y = 0; point_index_y < size_y; point_index_y++)
    {
        for (int point_index_x = 0; point_index_x < size_x; point_index_x++)
        {
            int const point_index = point_index_y * size_x + point_index_x;
            float const argx = ((point_index_x - x0)*(point_index_x - x0)) / (2.f * sx * sx);
            float const argy = ((point_index_y - y0)*(point_index_y - y0)) / (2.f* sy * sy);
            float const ex = exp(-argx) * exp(-argy);
            values[point_index] = a * ex + b;
        }
    }
}

BOOST_AUTO_TEST_CASE( Gauss_Fit_2D_Elliptic )
{
    std::size_t const n_fits{ 1 } ;
    std::size_t const n_points{ 25 } ;

    std::array< float, 6 > const true_parameters{ { 4.f, 2.f, 2.f, 0.4f, 0.6f, 1.f } };

    std::array< float, n_points > data{};
    generate_gauss_2d_elliptic(data, true_parameters);

    std::array< float, n_points > weights{};
    std::fill(weights.begin(), weights.end(), 1.f);
    std::array< float, 6 > initial_parameters{ { 2.f, 1.8f, 2.2f, 0.5f, 0.5f, 0.f } };
    float tolerance{ 0.001f };
    int max_n_iterations{ 10 };
    std::array< int, 6 > parameters_to_fit{ { 1, 1, 1, 1, 1, 1 } };
    std::array< float, 6 > output_parameters;
    int output_state;
    float output_chi_square;
    int output_n_iterations;

    int const status
            = gpufit
            (
                n_fits,
                n_points,
                data.data(),
                weights.data(),
                GAUSS_2D_ELLIPTIC,
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
