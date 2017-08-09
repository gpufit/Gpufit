#define BOOST_TEST_MODULE Gpufit

#define PI 3.1415926535897f

#include "Gpufit/gpufit.h"

#include <boost/test/included/unit_test.hpp>

#include <array>

template<std::size_t SIZE>
void generate_gauss_2d_rotated(std::array< float, SIZE>& values)
{
    int const size_x = int(std::sqrt(SIZE));
    int const size_y = size_x;

    float const a = 10.f;
    float const x0 = (float(size_x) - 1.f) / 2.f;
    float const y0 = (float(size_y) - 1.f) / 2.f;
    float const sx = 0.4f;
    float const sy = 0.5f;
    float const b = 1.f;
    float const r = PI / 16.f;

    for (int point_index_y = 0; point_index_y < size_y; point_index_y++)
    {
        for (int point_index_x = 0; point_index_x < size_x; point_index_x++)
        {
            int const point_index = point_index_y * size_x + point_index_x;
            float const arga = ((point_index_x - x0) * cosf(r)) - ((point_index_y - y0) * sinf(r));
            float const argb = ((point_index_x - x0) * sinf(r)) + ((point_index_y - y0) * cosf(r));
            float const ex = exp((-0.5f) * (((arga / sx) * (arga / sx)) + ((argb / sy) * (argb / sy))));
            values[point_index] = a * ex + b;
        }
    }
}

BOOST_AUTO_TEST_CASE( Gauss_Fit_2D_Rotated )
{
    std::size_t const n_fits{ 1 } ;
    std::size_t const n_points{ 64 } ;
    std::array< float, n_points > data{};
    generate_gauss_2d_rotated(data);
    std::array< float, n_points > weights{};
    std::fill(weights.begin(), weights.end(), 1.f);
    std::array< float, 7 > initial_parameters{ { 8.f, 3.4f, 3.6f, 0.4f, 0.5f, 2.f, 0.f } };
    float tolerance{ 0.001f };
    int max_n_iterations{ 10 };
    std::array< int, 7 > parameters_to_fit{ { 1, 1, 1, 1, 1, 1, 1 } };
    std::array< float, 7 > output_parameters;
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

    BOOST_CHECK( status == 0 ) ;
}
