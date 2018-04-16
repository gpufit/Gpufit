#ifndef TEST_UTILS_H_INCLUDED
#define TEST_UTILS_H_INCLUDED

#include<vector>
#include<random>
#include "Gpufit/gpufit.h"

#define CHK(x) if (!x) return false

extern std::mt19937 rng;

/*
Just to make sure that the content is erased after the resize.
*/
template<typename T> void clean_resize(std::vector<T> & v, std::size_t const n)
{
	v.resize(n);
	std::fill(v.begin(), v.end(), (T)0);
}

template<typename T> double max_relative_difference(std::vector<T> const & a, std::vector<T> const & b)
{
	double v = 0;

	auto it_a = a.begin();
	auto it_b = b.begin();

	while (it_a !=a.end())
	{
		T va = *it_a++;
		T vb = *it_b++;
		double d = static_cast<double>(std::abs(va - vb)) / (std::abs(va) + std::abs(vb));
		v = std::max(v, d);
	}
	return v;
}

template<typename T> double max_absolute_difference(std::vector<T> const & a, std::vector<T> const & b)
{
    double v = 0;

    auto it_a = a.begin();
    auto it_b = b.begin();

    while (it_a != a.end())
    {
        T va = *it_a++;
        T vb = *it_b++;
        double d = static_cast<double>(std::abs(va - vb));
        v = std::max(v, d);
    }
    return v;
}

template<typename T> bool close_or_equal(std::vector<T> const & a, std::vector<T> const & b, double relative_threshold = 1e-3, double absolute_threshold = 1e-6)
{
	if (a.empty() && b.empty())
	{
		return true;
	}
	if (a.size() != b.size())
	{
		return false;
	}
	double rd = max_relative_difference(a, b);
    double ad = max_absolute_difference(a, b);
    return rd < relative_threshold || ad < absolute_threshold;
}

/*
Calculates the standard deviation of a vector whose values are the differences of values of two others vectors of equal length.
Only use values if use[i] == 0.
*/
template<typename T> double calculate_standard_deviation(std::vector<T> const & a, std::vector<T> const & b, std::vector<int> const & use)
{
    std::size_t n = 0;
    double sq_diff = 0;

    for (std::size_t i = 0; i < a.size(); i++)
    {
        if (use[i] == 0)
        {
            n++;
            sq_diff += static_cast<double>((a[i] - b[i])) * (a[i] - b[i]);
        }
    }

    double std_dev = std::sqrt(sq_diff / n);
    return std_dev;
}

template<typename T> double calculate_mean(std::vector<T> const & a, std::vector<int> const & use)
{
    std::size_t n = 0;
    double s = 0;

    for (std::size_t i = 0; i < a.size(); i++)
    {
        if (use[i] == 0)
        {
            n++;
            s += static_cast<double>(a[i]);
        }
    }
    return s / n;
}

void generate_gauss_1d(std::vector< REAL > & v, std::vector< REAL > const & p);

void generate_gauss_2d(std::vector< REAL > & v, std::vector< REAL > const & p);

void generate_gauss_2d_elliptic(std::vector< REAL > & v, std::vector< REAL > const & p);

struct FitInput
{
	std::size_t n_fits;
	std::size_t n_points;
	std::size_t n_parameters;

	std::vector< REAL > data;
	std::vector< REAL > weights_; // size 0 means no weights

	int model_id;
	int estimator_id;

	std::vector< REAL > initial_parameters;
	std::vector< int > parameters_to_fit;

	REAL tolerance;
	int max_n_iterations;

	std::vector< REAL > user_info_; // user info is REAL

	REAL * weights()
	{
		if (!this->weights_.empty())
		{
			return this->weights_.data();
		}
		return 0;
	}

	char * user_info()
	{
		if (!this->user_info_.empty())
		{
			return reinterpret_cast<char *>(this->user_info_.data());
		}
		return 0;
	}

	std::size_t user_info_size()
	{
		return this->user_info_.size() * sizeof(REAL); // type of user_info is REAL
	}

	bool sanity_check()
	{
		CHK(this->data.size() == this->n_fits * this->n_points);
		if (!this->weights_.empty())
		{
			CHK(this->weights_.size() == this->n_fits * this->n_points);
		}
		CHK(this->initial_parameters.size() == this->n_fits * this->n_parameters);
		return true;
	}
};

struct FitOutput
{
	std::vector< REAL > parameters;
	std::vector< int > states;
	std::vector< REAL > chi_squares;
	std::vector< int > n_iterations;
};

#endif