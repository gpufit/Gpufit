#include "cpufit.h"
#include "../Gpufit/constants.h"
#include "lm_fit.h"

#include <vector>
#include <numeric>
#include <algorithm>

LMFitCPP::LMFitCPP(
    float const tolerance,
    std::size_t const fit_index,
    float const * data,
    float const * weight,
    Info const & info,
    float const * initial_parameters,
    int const * parameters_to_fit,
    char * user_info,
    float * output_parameters,
    int * output_state,
    float * output_chi_square,
    int * output_n_iterations
    ) :
    fit_index_(fit_index),
    data_(data),
    weight_(weight),
    initial_parameters_(initial_parameters),
    tolerance_(tolerance),
    converged_(false),
    info_(info),
    parameters_to_fit_(parameters_to_fit),
    curve_(info.n_points_),
    derivatives_(info.n_points_*info.n_parameters_),
    hessian_(info.n_parameters_to_fit_*info.n_parameters_to_fit_),
    modified_hessian_(info.n_parameters_to_fit_*info.n_parameters_to_fit_),
    gradient_(info.n_parameters_to_fit_),
    delta_(info.n_parameters_to_fit_),
    prev_chi_square_(0),
    lambda_(0.001f),
    prev_parameters_(info.n_parameters_to_fit_),
    user_info_(user_info),
    parameters_(output_parameters),
    state_(output_state),
    chi_square_(output_chi_square),
    n_iterations_(output_n_iterations)
{}

void LMFitCPP::calc_derivatives_gauss2d(
    std::vector<float> & derivatives)
{
    std::size_t const  fit_size_x = std::size_t(std::sqrt(info_.n_points_));

    for (std::size_t y = 0; y < fit_size_x; y++)
        for (std::size_t x = 0; x < fit_size_x; x++)
        {
            float const argx = (x - parameters_[1]) * (x - parameters_[1]) / (2 * parameters_[3] * parameters_[3]);
            float const argy = (y - parameters_[2]) * (y - parameters_[2]) / (2 * parameters_[3] * parameters_[3]);
            float const ex = exp(-(argx + argy));

            derivatives[0 * info_.n_points_ + y*fit_size_x + x]
                = ex;
            derivatives[1 * info_.n_points_ + y*fit_size_x + x]
                = (parameters_[0] * (x - parameters_[1])*ex) / (parameters_[3] * parameters_[3]);
            derivatives[2 * info_.n_points_ + y*fit_size_x + x]
                = (parameters_[0] * (y - parameters_[2])*ex) / (parameters_[3] * parameters_[3]);
            derivatives[3 * info_.n_points_ + y*fit_size_x + x]
                = (parameters_[0]
                * ((x - parameters_[1])*(x - parameters_[1])
                + (y - parameters_[2])*(y - parameters_[2]))*ex)
                / (parameters_[3] * parameters_[3] * parameters_[3]);
            derivatives[4 * info_.n_points_ + y*fit_size_x + x]
                = 1;
        }
}

void LMFitCPP::calc_derivatives_gauss2delliptic(
    std::vector<float> & derivatives)
{
    std::size_t const  fit_size_x = std::size_t(std::sqrt(info_.n_points_));

    for (std::size_t y = 0; y < fit_size_x; y++)
        for (std::size_t x = 0; x < fit_size_x; x++)
        {
            float const argx = (x - parameters_[1]) * (x - parameters_[1]) / (2 * parameters_[3] * parameters_[3]);
            float const argy = (y - parameters_[2]) * (y - parameters_[2]) / (2 * parameters_[4] * parameters_[4]);
            float const ex = exp(-(argx +argy));

            derivatives[0 * info_.n_points_ + y*fit_size_x + x]
                = ex;
            derivatives[1 * info_.n_points_ + y*fit_size_x + x]
                = (parameters_[0] * (x - parameters_[1])*ex) / (parameters_[3] * parameters_[3]);
            derivatives[2 * info_.n_points_ + y*fit_size_x + x]
                = (parameters_[0] * (y - parameters_[2])*ex) / (parameters_[4] * parameters_[4]);
            derivatives[3 * info_.n_points_ + y*fit_size_x + x]
                = (parameters_[0] * (x - parameters_[1])*(x - parameters_[1])*ex) / (parameters_[3] * parameters_[3] * parameters_[3]);
            derivatives[4 * info_.n_points_ + y*fit_size_x + x]
                = (parameters_[0] * (y - parameters_[2])*(y - parameters_[2])*ex) / (parameters_[4] * parameters_[4] * parameters_[4]);
            derivatives[5 * info_.n_points_ + y*fit_size_x + x]
                = 1;
        }
}

void LMFitCPP::calc_derivatives_gauss2drotated(
    std::vector<float> & derivatives)
{
    std::size_t const  fit_size_x = std::size_t(std::sqrt(info_.n_points_));

    float const amplitude = parameters_[0];
    float const x0 = parameters_[1];
    float const y0 = parameters_[2];
    float const sig_x = parameters_[3];
    float const sig_y = parameters_[4];
    float const background = parameters_[5];
    float const rot_sin = sin(parameters_[6]);
    float const rot_cos = cos(parameters_[6]);

    for (std::size_t y = 0; y < fit_size_x; y++)
        for (std::size_t x = 0; x < fit_size_x; x++)
        {
            float const arga = ((x - x0) * rot_cos) - ((y - y0) * rot_sin);
            float const argb = ((x - x0) * rot_sin) + ((y - y0) * rot_cos);
            float const ex = exp((-0.5f) * (((arga / sig_x) * (arga / sig_x)) + ((argb / sig_y) * (argb / sig_y))));

            derivatives[0 * info_.n_points_ + y*fit_size_x + x]
                = ex;
            derivatives[1 * info_.n_points_ + y*fit_size_x + x]
                = ex * (amplitude * rot_cos * arga / (sig_x*sig_x) + amplitude * rot_sin *argb / (sig_y*sig_y));
            derivatives[2 * info_.n_points_ + y*fit_size_x + x]
                = ex * (-amplitude * rot_sin * arga / (sig_x*sig_x) + amplitude * rot_cos *argb / (sig_y*sig_y));
            derivatives[3 * info_.n_points_ + y*fit_size_x + x]
                = ex * amplitude * arga * arga / (sig_x*sig_x*sig_x);
            derivatives[4 * info_.n_points_ + y*fit_size_x + x]
                = ex * amplitude * argb * argb / (sig_y*sig_y*sig_y);
            derivatives[5 * info_.n_points_ + y*fit_size_x + x]
                = 1;
            derivatives[6 * info_.n_points_ + y*fit_size_x + x]
                = ex * amplitude * arga * argb * (1.0f / (sig_x*sig_x) - 1.0f / (sig_y*sig_y));
        }
}

void LMFitCPP::calc_derivatives_gauss1d(
    std::vector<float> & derivatives)
{
    float * user_info_float = (float*)user_info_;
    float x = 0.f;

    for (std::size_t point_index = 0; point_index < info_.n_points_; point_index++)
    {
        if (!user_info_float)
        {
            x = float(point_index);
        }
        else if (info_.user_info_size_ / sizeof(float) == info_.n_points_)
        {
            x = user_info_float[point_index];
        }
        else if (info_.user_info_size_ / sizeof(float) > info_.n_points_)
        {
            std::size_t const fit_begin = fit_index_ * info_.n_points_;
            x = user_info_float[fit_begin + point_index];
        }

        float argx = ((x - parameters_[1])*(x - parameters_[1])) / (2 * parameters_[2] * parameters_[2]);
        float ex = exp(-argx);

        derivatives[0 * info_.n_points_ + point_index] = ex;
        derivatives[1 * info_.n_points_ + point_index] = (parameters_[0] * (x - parameters_[1])*ex) / (parameters_[2] * parameters_[2]);
        derivatives[2 * info_.n_points_ + point_index] = (parameters_[0] * (x - parameters_[1])*(x - parameters_[1])*ex) / (parameters_[2] * parameters_[2] * parameters_[2]);
        derivatives[3 * info_.n_points_ + point_index] = 1;
    }
}

void LMFitCPP::calc_derivatives_cauchy2delliptic(
    std::vector<float> & derivatives)
{
    std::size_t const  fit_size_x = std::size_t(std::sqrt(info_.n_points_));

    for (std::size_t y = 0; y < fit_size_x; y++)
        for (std::size_t x = 0; x < fit_size_x; x++)
        {
            float const argx =
                ((parameters_[1] - x) / parameters_[3])
                *((parameters_[1] - x) / parameters_[3]) + 1.f;
            float const argy =
                ((parameters_[2] - y) / parameters_[4])
                *((parameters_[2] - y) / parameters_[4]) + 1.f;

            derivatives[0 * info_.n_points_ + y*fit_size_x + x]
                = 1.f / (argx*argy);
            derivatives[1 * info_.n_points_ + y*fit_size_x + x] =
                -2.f * parameters_[0] * (parameters_[1] - x)
                / (parameters_[3] * parameters_[3] * argx*argx*argy);
            derivatives[2 * info_.n_points_ + y*fit_size_x + x] =
                -2.f * parameters_[0] * (parameters_[2] - y)
                / (parameters_[4] * parameters_[4] * argy*argy*argx);
            derivatives[3 * info_.n_points_ + y*fit_size_x + x] =
                2.f * parameters_[0] * (parameters_[1] - x) * (parameters_[1] - x)
                / (parameters_[3] * parameters_[3] * parameters_[3] * argx*argx*argy);
            derivatives[4 * info_.n_points_ + y*fit_size_x + x] =
                2.f * parameters_[0] * (parameters_[2] - y) * (parameters_[2] - y)
                / (parameters_[4] * parameters_[4] * parameters_[4] * argy*argy*argx);
            derivatives[5 * info_.n_points_ + y*fit_size_x + x]
                = 1.f;
        }
}

void LMFitCPP::calc_derivatives_linear1d(
    std::vector<float> & derivatives)
{
    float * user_info_float = (float*)user_info_;
    float x = 0.f;

    for (std::size_t point_index = 0; point_index < info_.n_points_; point_index++)
    {
        if (!user_info_float)
        {
            x = float(point_index);
        }
        else if (info_.user_info_size_ / sizeof(float) == info_.n_points_)
        {
            x = user_info_float[point_index];
        }
        else if (info_.user_info_size_ / sizeof(float) > info_.n_points_)
        {
            std::size_t const fit_begin = fit_index_ * info_.n_points_;
            x = user_info_float[fit_begin + point_index];
        }

        derivatives[0 * info_.n_points_ + point_index] = 1.f;
        derivatives[1 * info_.n_points_ + point_index] = x;
    }
}

void LMFitCPP::calc_values_cauchy2delliptic(std::vector<float>& cauchy)
{
    int const size_x = int(std::sqrt(float(info_.n_points_)));
    int const size_y = size_x;

    for (int iy = 0; iy < size_y; iy++)
    {
        for (int ix = 0; ix < size_x; ix++)
        {
            float const argx =
                ((parameters_[1] - ix) / parameters_[3])
                *((parameters_[1] - ix) / parameters_[3]) + 1.f;
            float const argy =
                ((parameters_[2] - iy) / parameters_[4])
                *((parameters_[2] - iy) / parameters_[4]) + 1.f;

            cauchy[iy*size_x + ix] = parameters_[0] / (argx * argy) + parameters_[5];
        }
    }
}

void LMFitCPP::calc_values_gauss2d(std::vector<float>& gaussian)
{
    int const size_x = int(std::sqrt(float(info_.n_points_)));
    int const size_y = size_x;

    for (int iy = 0; iy < size_y; iy++)
    {
        for (int ix = 0; ix < size_x; ix++)
        {
            float argx = (ix - parameters_[1]) * (ix - parameters_[1]) / (2 * parameters_[3] * parameters_[3]);
            float argy = (iy - parameters_[2]) * (iy - parameters_[2]) / (2 * parameters_[3] * parameters_[3]);
            float ex = exp(-(argx +argy));

            gaussian[iy*size_x + ix] = parameters_[0] * ex + parameters_[4];
        }
    }
}

void LMFitCPP::calc_values_gauss2delliptic(std::vector<float>& gaussian)
{
    int const size_x = int(std::sqrt(float(info_.n_points_)));
    int const size_y = size_x;
    for (int iy = 0; iy < size_y; iy++)
    {
        for (int ix = 0; ix < size_x; ix++)
        {
            float argx = (ix - parameters_[1]) * (ix - parameters_[1]) / (2 * parameters_[3] * parameters_[3]);
            float argy = (iy - parameters_[2]) * (iy - parameters_[2]) / (2 * parameters_[4] * parameters_[4]);
            float ex = exp(-(argx + argy));

            gaussian[iy*size_x + ix]
                = parameters_[0] * ex + parameters_[5];
        }
    }
}
    
void LMFitCPP::calc_values_gauss2drotated(std::vector<float>& gaussian)
{
    int const size_x = int(std::sqrt(float(info_.n_points_)));
    int const size_y = size_x;

    float amplitude = parameters_[0];
    float background = parameters_[5];
    float x0 = parameters_[1];
    float y0 = parameters_[2];
    float sig_x = parameters_[3];
    float sig_y = parameters_[4];
    float rot_sin = sin(parameters_[6]);
    float rot_cos = cos(parameters_[6]);

    for (int iy = 0; iy < size_y; iy++)
    {
        for (int ix = 0; ix < size_x; ix++)
        {
            int const pixel_index = iy*size_x + ix;

            float arga = ((ix - x0) * rot_cos) - ((iy - y0) * rot_sin);
            float argb = ((ix - x0) * rot_sin) + ((iy - y0) * rot_cos);

            float ex
                = exp((-0.5f) * (((arga / sig_x) * (arga / sig_x)) + ((argb / sig_y) * (argb / sig_y))));

            gaussian[pixel_index] = amplitude * ex + background;
        }
    }
}

void LMFitCPP::calc_values_gauss1d(std::vector<float>& gaussian)
{
    float * user_info_float = (float*)user_info_;
    float x = 0.f;
    for (std::size_t point_index = 0; point_index < info_.n_points_; point_index++)
    {
        if (!user_info_float)
        {
            x = float(point_index);
        }
        else if (info_.user_info_size_ / sizeof(float) == info_.n_points_)
        {
            x = user_info_float[point_index];
        }
        else if (info_.user_info_size_ / sizeof(float) > info_.n_points_)
        {
            std::size_t const fit_begin = fit_index_ * info_.n_points_;
            x = user_info_float[fit_begin + point_index];
        }

        float argx
            = ((x - parameters_[1])*(x - parameters_[1]))
            / (2 * parameters_[2] * parameters_[2]);
        float ex = exp(-argx);
        gaussian[point_index] = parameters_[0] * ex + parameters_[3];
    }
}

void LMFitCPP::calc_values_linear1d(std::vector<float>& line)
{
    float * user_info_float = (float*)user_info_;
    float x = 0.f;
    for (std::size_t point_index = 0; point_index < info_.n_points_; point_index++)
    {
        if (!user_info_float)
        {
            x = float(point_index);
        }
        else if (info_.user_info_size_ / sizeof(float) == info_.n_points_)
        {
            x = user_info_float[point_index];
        }
        else if (info_.user_info_size_ / sizeof(float) > info_.n_points_)
        {
            std::size_t const fit_begin = fit_index_ * info_.n_points_;
            x = user_info_float[fit_begin + point_index];
        }
        line[point_index] = parameters_[0] + parameters_[1] * x;
    }
}

void LMFitCPP::calc_curve_values(std::vector<float>& curve, std::vector<float>& derivatives)
{           
    if (info_.model_id_ == GAUSS_1D)
    {
        calc_values_gauss1d(curve);
        calc_derivatives_gauss1d(derivatives);
    }
    else if (info_.model_id_ == GAUSS_2D)
    {
        calc_values_gauss2d(curve);
        calc_derivatives_gauss2d(derivatives);
    }
    else if (info_.model_id_ == GAUSS_2D_ELLIPTIC)
    {
        calc_values_gauss2delliptic(curve);
        calc_derivatives_gauss2delliptic(derivatives);
    }
    else if (info_.model_id_ == GAUSS_2D_ROTATED)
    {
        calc_values_gauss2drotated(curve);
        calc_derivatives_gauss2drotated(derivatives);
    }
    else if (info_.model_id_ == CAUCHY_2D_ELLIPTIC)
    {
        calc_values_cauchy2delliptic(curve);
        calc_derivatives_cauchy2delliptic(derivatives);
    }
    else if (info_.model_id_ == LINEAR_1D)
    {
        calc_values_linear1d(curve);
        calc_derivatives_linear1d(derivatives);
    }
}

void LMFitCPP::calculate_hessian(
    std::vector<float> const & derivatives,
    std::vector<float> const & curve)
{
    for (int jp = 0, jhessian = 0; jp < info_.n_parameters_; jp++)
    {
        if (parameters_to_fit_[jp])
        {
            for (int ip = 0, ihessian = 0; ip < jp + 1; ip++)
            {
                if (parameters_to_fit_[ip])
                {
                    std::size_t const ijhessian
                        = ihessian * info_.n_parameters_to_fit_ + jhessian;
                    std::size_t const jihessian
                        = jhessian * info_.n_parameters_to_fit_ + ihessian;
                    std::size_t const derivatives_index_i = ip*info_.n_points_;
                    std::size_t const derivatives_index_j = jp*info_.n_points_;
                    
                    double sum = 0.0;
                    for (std::size_t pixel_index = 0; pixel_index < info_.n_points_; pixel_index++)
                    {
                        if (info_.estimator_id_ == LSE)
                        {
                            if (!weight_)
                            {
                                sum
                                    += derivatives[derivatives_index_i + pixel_index]
                                    * derivatives[derivatives_index_j + pixel_index];
                            }
                            else
                            {
                                sum
                                    += derivatives[derivatives_index_i + pixel_index]
                                    * derivatives[derivatives_index_j + pixel_index]
                                    * weight_[pixel_index];
                            }
                        }
                        else if (info_.estimator_id_ == MLE)
                        {
                            sum
                                += data_[pixel_index] / (curve[pixel_index] * curve[pixel_index])
                                * derivatives[derivatives_index_i + pixel_index]
                                * derivatives[derivatives_index_j + pixel_index];
                        }
                    }
                    hessian_[ijhessian] = float(sum);
                    if (ijhessian != jihessian)
                    {
                        hessian_[jihessian]
                            = hessian_[ijhessian];
                    }
                    ihessian++;
                }
            }
            jhessian++;
        }
    }

}

void LMFitCPP::calc_gradient(
    std::vector<float> const & derivatives,
    std::vector<float> const & curve)
{

    for (int ip = 0, gradient_index = 0; ip < info_.n_parameters_; ip++)
    {
        if (parameters_to_fit_[ip])
        {
            std::size_t const derivatives_index = ip*info_.n_points_;
            double sum = 0.0;
            for (std::size_t pixel_index = 0; pixel_index < info_.n_points_; pixel_index++)
            {
                float deviant = data_[pixel_index] - curve[pixel_index];

                if (info_.estimator_id_ == LSE)
                {
                    if (!weight_)
                    {
                        sum
                            += deviant * derivatives[derivatives_index + pixel_index];
                    }
                    else
                    {
                        sum
                            += deviant * derivatives[derivatives_index + pixel_index] * weight_[pixel_index];
                    }

                }
                else if (info_.estimator_id_ == MLE)
                {
                    sum
                        += -derivatives[derivatives_index + pixel_index] * (1 - data_[pixel_index] / curve[pixel_index]);
                }
            }
            gradient_[gradient_index] = float(sum);
            gradient_index++;
        }
    }

}

void LMFitCPP::calc_chi_square(
    std::vector<float> const & values)
{
    double sum = 0.0;
    for (size_t pixel_index = 0; pixel_index < values.size(); pixel_index++)
    {
        float deviant = values[pixel_index] - data_[pixel_index];
        if (info_.estimator_id_ == LSE)
        {
            if (!weight_)
            {
                sum += deviant * deviant;
            }
            else
            {
                sum += deviant * deviant * weight_[pixel_index];
            }
        }
        else if (info_.estimator_id_ == MLE)
        {
            if (values[pixel_index] <= 0.f)
            {
                *state_ = FitState::NEG_CURVATURE_MLE;
                return;
            }
            if (data_[pixel_index] != 0.f)
            {
                sum
                    += 2 * (deviant - data_[pixel_index] * logf(values[pixel_index] / data_[pixel_index]));
            }
            else
            {
                sum += 2 * deviant;
            }
        }
    }
    *chi_square_ = float(sum);
}

void LMFitCPP::calc_curve_values()
{
	std::vector<float> & curve = curve_;
	std::vector<float> & derivatives = derivatives_;

	calc_curve_values(curve, derivatives);
}
    
void LMFitCPP::calc_coefficients()
{
    std::vector<float> & curve = curve_;
    std::vector<float> & derivatives = derivatives_;

    calc_chi_square(curve);

    if ((*chi_square_) < prev_chi_square_ || prev_chi_square_ == 0)
    {
        calculate_hessian(derivatives, curve);
        calc_gradient(derivatives, curve);
    }
}

void LMFitCPP::gauss_jordan()
{
    delta_ = gradient_;

    std::vector<float> & alpha = modified_hessian_;
    std::vector<float> & beta = delta_;

    int icol, irow;
    float big, dum, pivinv;

    std::vector<int> indxc(info_.n_parameters_to_fit_, 0);
    std::vector<int> indxr(info_.n_parameters_to_fit_, 0);
    std::vector<int> ipiv(info_.n_parameters_to_fit_, 0);

    for (int kp = 0; kp < info_.n_parameters_to_fit_; kp++)
    {
        big = 0.0;
        for (int jp = 0; jp < info_.n_parameters_to_fit_; jp++)
        {
            if (ipiv[jp] != 1)
            {
                for (int ip = 0; ip < info_.n_parameters_to_fit_; ip++)
                {
                    if (ipiv[ip] == 0)
                    {
                        if (fabs(alpha[jp*info_.n_parameters_to_fit_ + ip]) >= big)
                        {
                            big = fabs(alpha[jp*info_.n_parameters_to_fit_ + ip]);
                            irow = jp;
                            icol = ip;
                        }
                    }
                }
            }
        }
        ++(ipiv[icol]);


        if (irow != icol)
        {
            for (int ip = 0; ip < info_.n_parameters_to_fit_; ip++)
            {
                std::swap(alpha[irow*info_.n_parameters_to_fit_ + ip], alpha[icol*info_.n_parameters_to_fit_ + ip]);
            }
            std::swap(beta[irow], beta[icol]);
        }
        indxr[kp] = irow;
        indxc[kp] = icol;
        if (alpha[icol*info_.n_parameters_to_fit_ + icol] == 0.0)
        {
            *state_ = FitState::SINGULAR_HESSIAN;
            break;
        }
        pivinv = 1.0f / alpha[icol*info_.n_parameters_to_fit_ + icol];
        alpha[icol*info_.n_parameters_to_fit_ + icol] = 1.0;
        for (int ip = 0; ip < info_.n_parameters_to_fit_; ip++)
        {
            alpha[icol*info_.n_parameters_to_fit_ + ip] *= pivinv;
        }
        beta[icol] *= pivinv;

        for (int jp = 0; jp < info_.n_parameters_to_fit_; jp++)
        {
            if (jp != icol)
            {
                dum = alpha[jp*info_.n_parameters_to_fit_ + icol];
                alpha[jp*info_.n_parameters_to_fit_ + icol] = 0.0;
                for (int ip = 0; ip < info_.n_parameters_to_fit_; ip++)
                {
                    alpha[jp*info_.n_parameters_to_fit_ + ip] -= alpha[icol*info_.n_parameters_to_fit_ + ip] * dum;
                }
                beta[jp] -= beta[icol] * dum;
            }
        }
    }
}

void LMFitCPP::update_parameters()
{
    for (int parameter_index = 0, delta_index = 0; parameter_index < info_.n_parameters_; parameter_index++)
    {
        if (parameters_to_fit_[parameter_index])
        {
            prev_parameters_[parameter_index] = parameters_[parameter_index];
            parameters_[parameter_index] = parameters_[parameter_index] + delta_[delta_index++];
        }
    }
}

bool LMFitCPP::check_for_convergence()
{
    bool const fit_found
        = std::abs(*chi_square_ - prev_chi_square_)  < std::max(tolerance_, tolerance_ * std::abs(*chi_square_));

    return fit_found;
}

void LMFitCPP::evaluate_iteration(int const iteration)
{
    bool const max_iterations_reached = iteration == info_.max_n_iterations_ - 1;
    if (converged_ || max_iterations_reached)
    {
        (*n_iterations_) = iteration + 1;
        if (!converged_)
        {
            *state_ = FitState::MAX_ITERATION;
        }
    }
}

void LMFitCPP::prepare_next_iteration()
{
    if ((*chi_square_) < prev_chi_square_)
    {
        lambda_ *= 0.1f;
        prev_chi_square_ = (*chi_square_);
    }
    else
    {
        lambda_ *= 10.f;
        (*chi_square_) = prev_chi_square_;
        for (int parameter_index = 0, delta_index = 0; parameter_index < info_.n_parameters_; parameter_index++)
        {
            if (parameters_to_fit_[parameter_index])
            {
                parameters_[parameter_index] = prev_parameters_[parameter_index];
            }
        }
    }
}

void LMFitCPP::modify_step_width()
{
    modified_hessian_ = hessian_;
    size_t const n_parameters = (size_t)(sqrt((float)(hessian_.size())));
    for (size_t parameter_index = 0; parameter_index < n_parameters; parameter_index++)
    {
        modified_hessian_[parameter_index*n_parameters + parameter_index]
            = modified_hessian_[parameter_index*n_parameters + parameter_index]
            * (1.0f + (lambda_));
    }
}

void LMFitCPP::run()
{
    for (int i = 0; i < info_.n_parameters_; i++)
        parameters_[i] = initial_parameters_[i];

    *state_ = FitState::CONVERGED;
	calc_curve_values();
    calc_coefficients();
    prev_chi_square_ = (*chi_square_);
        
    for (int iteration = 0; (*state_) == 0; iteration++)
    {
        modify_step_width();
        
        gauss_jordan();

        update_parameters();

		calc_curve_values();
        calc_coefficients();

        converged_ = check_for_convergence();

        evaluate_iteration(iteration);

        prepare_next_iteration();

        if (converged_ || *state_ != FitState::CONVERGED)
        {
            break;
        }
    }
}
