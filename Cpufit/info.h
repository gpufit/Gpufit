#ifndef CPUFIT_PARAMETERS_H_INCLUDED
#define CPUFIT_PARAMETERS_H_INCLUDED

#include <vector>

class Info
{
public:
    Info();
    virtual ~Info();
    void set_number_of_parameters_to_fit(int const * parameters_to_fit);

private:

public:
    int n_parameters_;
    int n_parameters_to_fit_;
    std::size_t n_fits_;
    std::size_t n_points_;
    int max_n_iterations_;
    int model_id_;
    int estimator_id_;
    std::size_t user_info_size_;
    
private:
};

#endif
