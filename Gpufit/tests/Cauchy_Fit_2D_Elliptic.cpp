#define BOOST_TEST_MODULE Gpufit

#include "Gpufit/gpufit.h"

#include <boost/test/included/unit_test.hpp>

#include <array>

template<std::size_t SIZE>
void generate_cauchy_2d_elliptic(std::array< float, SIZE>& values)
{
    int const size_x = int(std::sqrt(SIZE));
    int const size_y = size_x;

    float const a = 4;
    float const x0 = (float(size_x) - 1.f) / 2.f;
    float const y0 = (float(size_y) - 1.f) / 2.f;
    float const sx = 0.4f;
    float const sy = 0.6f;
    float const b = 1.f;

    for (int point_index_y = 0; point_index_y < size_y; point_index_y++)
    {
        for (int point_index_x = 0; point_index_x < size_x; point_index_x++)
        {
            int const point_index = point_index_y * size_x + point_index_x;
            float const argx = ((x0 - point_index_x) / sx) *((x0 - point_index_x) / sx) + 1.f;
            float const argy = ((y0 - point_index_y) / sy) *((y0 - point_index_y) / sy) + 1.f;
            values[point_index] = a / argx / argy + b;
        }
    }
}

BOOST_AUTO_TEST_CASE( Cauchy_Fit_2D_Elliptic )
{
    std::size_t const n_fits{ 1 } ;
    std::size_t const n_points{ 25 } ;
    std::array< float, n_points > data{};
    generate_cauchy_2d_elliptic(data);
    std::array< float, n_points > weights{};
    std::fill(weights.begin(), weights.end(), 1.f);
    std::array< float, 6 > initial_parameters{ { 2.f, 1.8f, 2.2f, 0.5f, 0.5f, 0.f } };
    float tolerance{ 0.001f };
    int max_n_iterations{ 100 };
    std::array< int, 6 > parameters_to_fit{ { 1, 1, 1, 1, 1, 1 } };
    std::array< float, 6 > output_parameters;
    int output_states;
    float output_chi_square;
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
                &output_states,
                &output_chi_square,
                &output_n_iterations,
                0
            ) ;

    BOOST_CHECK( status == 0 ) ;
}
