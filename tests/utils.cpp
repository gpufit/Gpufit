#include "utils.h"

#include <stdexcept>

// initialize random number generator
std::mt19937 rng(0);

/*
    Given a parameter vector p with 4 entries, constructs a 1D Gaussian peak function with x values 0,..,v.size() - 1
*/
void generate_gauss_1d(std::vector< REAL > & v, std::vector< REAL > const & p)
{
	for (std::size_t i = 0; i < v.size(); i++)
	{
		REAL const argx = ((i - p[1]) * (i - p[1])) / (2 * p[2] * p[2]);
		REAL const ex = exp(-argx);
		v[i] = p[0] * ex + p[3];
	}
}

/*
    Given a parameters vector p with 5 entries, constructs a 2D Gaussian peak function with x, y values 0, .., sqrt(v.size()) - 1
*/
void generate_gauss_2d(std::vector< REAL > & v, std::vector< REAL > const & p)
{
    std::size_t const n = static_cast<std::size_t>(std::sqrt(v.size()));
    if (n * n != v.size())
    {
        throw std::runtime_error("v.size() is not a perfect square number");
    }
    
    for (std::size_t j = 0; j < n; j++)
    {
        REAL const argy = ((j - p[2]) * (j - p[2]));
        for (std::size_t i = 0; i < n; i++)
        {
            REAL const argx = ((i - p[1]) * (i - p[1]));
            REAL const ex = exp(-(argx + argy) / (2 * p[3] * p[3]));
            v[j * n + i] = p[0] * ex + p[3];
        }
    }
}

void generate_gauss_2d_elliptic(std::vector< REAL > & v, std::vector< REAL > const & p)
{
    std::size_t const n = static_cast<std::size_t>(std::sqrt(v.size()));
    if (n * n != v.size())
    {
        throw std::runtime_error("v.size() is not a perfect square number");
    }

    for (std::size_t j = 0; j < n; j++)
    {
        REAL const argy = ((j - p[2]) * (j - p[2])) / (2 * p[4] * p[4]);
        for (std::size_t i = 0; i < n; i++)
        {
            REAL const argx = ((i - p[1]) * (i - p[1])) / (2 * p[3] * p[3]);
            REAL const ex = exp(-(argx + argy));
            v[j * n + i] = p[0] * ex + p[3];
        }
    }
}
