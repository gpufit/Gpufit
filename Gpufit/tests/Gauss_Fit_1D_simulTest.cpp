#define BOOST_TEST_MODULE Gpufit

#include "Gpufit/gpufit.h"

#include <boost/test/included/unit_test.hpp>

#include <vector>

//template<std::size_t n_points>

void generate_gauss_1d_old(
    std::vector< float > & values,
    std::vector< float > & x_data,
    std::vector< float > & parameters )
{
    float const a = parameters[ 0 ];
    float const x0 = parameters[ 1 ];
    float const s = parameters[ 2 ];
    float const b = parameters[ 3 ];

    for ( int point_index = 0; point_index < 5; point_index++ )
    {
        float const x = x_data[point_index];
        float const argx = ( ( x - x0 )*( x - x0 ) ) / ( 2.f * s * s );
        float const ex = exp( -argx );
        values[ point_index ] = a * ex + b;
    }
}

/*void generate_gauss_1d(
    std::vector< float > & values_in,
    std::vector< float > & values_out,
    std::vector< float > & x_data,
    std::vector< float > & parameters)
{

    std::vector< int > parameters_to_fit{ 1, 1, 1, 1 };
    std::vector< float > output_parameters(4);
    std::vector< int > output_states(1);
    std::vector< float > output_chi_square(1);
    std::vector< int > output_n_iterations(1);
    //std::cout << parameters.data()[0] <<"\n";
    //std::cout << parameters.data()[1] <<"\n";
    //std::cout << parameters.data()[2] <<"\n";
    //std::cout << parameters.data()[3] <<"\n";

    int const status
        = gpusimul
        (
            1,
            5,
            values_in.data(),
            0,
            GAUSS_1D,
            parameters.data(),
            0.001f,
            5,
            parameters_to_fit.data(),
            LSE,
            0,
            0,
            output_parameters.data(),
            output_states.data(),
            output_chi_square.data(),
            output_n_iterations.data(),
            values_out.data()
        );
}*/

void compare_simulations()
{

    std::size_t const n_fits = 1;
    std::size_t const n_points = 5;

    std::vector< float > true_parameters = { 4.f, 2.f, 0.5f, 1.f } ;
    std::vector< float > x_data = { 0.f, 1.f, 2.f, 3.f, 4.f} ;
    std::vector< float > old_data = {0.f, 0.f, 0.f, 0.f, 0.f};
    std::vector< float > data_in = {0.f, 0.f, 0.f, 0.f, 0.f};
    std::vector< float > data_out(5);

    // CPU generation
    generate_gauss_1d_old(old_data, x_data, true_parameters);

    // GPU generation
    int const status
        = gpusimul
        (
            n_fits,
            n_points,
            GAUSS_1D,
            true_parameters.data(),
            0,
            0,
            data_out.data()
        );


    //generate_gauss_1d(data_in, data_out, x_data, true_parameters);

    for (int i = 0; i < n_points; i++)
    {
        std::cout << old_data[i] << ' ';
    }

    std::cout << "\n";

    for (int i = 0; i < n_points; i++)
    {
        std::cout << data_out.data()[i] << ' ';
    }
}

BOOST_AUTO_TEST_CASE( Gauss_Fit_1D_SimulTest )
{
    // single 1d gauss fit
    compare_simulations();
}
