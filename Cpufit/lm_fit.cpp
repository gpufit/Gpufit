#include "lm_fit.h"
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <utility>
#include <vector>
#include <numeric>

LMFit::LMFit(
    double const * const data,
    double const * const weights,
    Info const & info,
    double const * const initial_parameters,
    int const * const parameters_to_fit,
    char * const user_info,
    double * output_parameters,
    int * output_states,
    double * output_chi_squares,
    int * output_n_iterations,
    double * lambda_info
    ) :
    data_(data),
    weights_(weights),
    initial_parameters_(initial_parameters),
    parameters_to_fit_(parameters_to_fit),
    user_info_(user_info),
    output_parameters_(output_parameters),
    output_states_(output_states),
    output_chi_squares_(output_chi_squares),
    output_n_iterations_(output_n_iterations),
    lambda_info_(lambda_info),
    info_(info)
{}

LMFit::~LMFit()
{
}

void LMFit::run(double const tolerance)
{
    for (std::size_t fit_index = 0; fit_index < info_.n_fits_; fit_index++)
    {
        LMFitCPP gf_cpp(
            tolerance,
            fit_index,
            data_ + fit_index*info_.n_points_,
            weights_ ? weights_ + fit_index*info_.n_points_ : 0,
            info_,
            initial_parameters_ + fit_index*info_.n_parameters_,
            parameters_to_fit_,
            user_info_,
            output_parameters_ + fit_index*info_.n_parameters_,
            output_states_ + fit_index,
            output_chi_squares_ + fit_index,
            output_n_iterations_ + fit_index,
            lambda_info_ + fit_index * 10 * 1000);

        gf_cpp.run();
    }
}