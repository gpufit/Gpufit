#define BOOST_TEST_MODULE Gpufit

#include "Gpufit/gpufit.h"

#include <boost/test/included/unit_test.hpp>

//#include <array>
#include <vector>
#include <random>
#include <iostream>
#include <chrono>
#include <numeric>
#include <math.h>
#include "../constants.h"

#include <stdexcept>

//template<std::size_t SIZE>
void generate_gauss_2d(
    std::vector< float > &  values,
    std::size_t const n_points,
    std::vector< float > p)
{
    int const size_x = int(std::sqrt(n_points));
    int const size_y = size_x;

    /*float const a = 4.f;
    float const x0 = (float(size_x) - 1.f) / 2.f;
    float const y0 = (float(size_y) - 1.f) / 2.f;
    float const b = 1.f;
    float const s = 0.5f*/

    for (int point_index_y = 0; point_index_y < size_y; point_index_y++)
    {
        for (int point_index_x = 0; point_index_x < size_x; point_index_x++)
        {
            int const point_index = point_index_y * size_x + point_index_x;
            float const argx = ((point_index_x - p[1])*(point_index_x - p[1])) / (2.f * p[3] * p[3]);
            float const argy = ((point_index_y - p[2])*(point_index_y - p[2])) / (2.f * p[3] * p[3]);
            float const ex = exp(-argx-argy);
            values[point_index] = p[0] * ex + p[4];
        }
    }
}

void compare_simulations()
{
    std::size_t const n_fits = 1;
    std::size_t const n_points = 25;
    std::vector< float > data_cpu(n_points);

    std::vector< float > parameters{ 4.f, 2.f, 2.f, 0.5f, 1.f };
    std::vector< float > data_gpu(n_fits * n_points);

    // CPU generation
    generate_gauss_2d(data_cpu, n_points, parameters);

    // GPU generation
    int const status
        = gpusimul
        (
            n_fits,
            n_points,
            GAUSS_2D,
            parameters.data(),
            0,
            0,
            data_gpu.data()
        );


    //generate_gauss_1d(data_in, data_out, x_data, true_parameters);

    for (int i = 0; i < n_points; i++)
    {
        std::cout << data_cpu[i] << ' ';
    }

    std::cout << "\n";

    for (int i = 0; i < n_points; i++)
    {
        std::cout << data_gpu.data()[i] << ' ';
    }


}

BOOST_AUTO_TEST_CASE( Gauss_Fit_2D )
{
    compare_simulations();
}
