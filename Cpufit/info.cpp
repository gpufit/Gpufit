#include "info.h"

Info::Info(void) :
    n_parameters_(0),
    n_parameters_to_fit_(0),
    max_n_iterations_(0),
    n_fits_(0),
    n_points_(0),
    user_info_size_(0)
{
}

Info::~Info(void)
{
}

void Info::set_number_of_parameters_to_fit(int const * parameters_to_fit)
{
    n_parameters_to_fit_ = n_parameters_;

    for (int i = 0; i < n_parameters_; i++)
    {
        if (!parameters_to_fit[i])
        {
            n_parameters_to_fit_--;
        }
    }
}