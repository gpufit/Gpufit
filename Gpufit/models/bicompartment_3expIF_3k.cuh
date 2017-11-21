#ifndef GPUFIT_BICOMP3EXP3K_CUH_INCLUDED
#define GPUFIT_BICOMP3EXP3K_CUH_INCLUDED

/* Description of the calculate_gauss1d function
* ==============================================
*
* This function calculates the values of one-dimensional gauss model functions
* and their partial derivatives with respect to the model parameters.
*
* This function makes use of the user information data to pass in the
* independent variables (X values) corresponding to the data.  The X values
* must be of type float.
*
* There are three possibilities regarding the X values:
*
*   No X values provided:
*
*       If no user information is provided, the (X) coordinate of the
*       first data value is assumed to be (0.0).  In this case, for a
*       fit size of M data points, the (X) coordinates of the data are
*       simply the corresponding array index values of the data array,
*       starting from zero.
*
*   X values provided for one fit:
*
*       If the user_info array contains the X values for one fit, then
*       the same X values will be used for all fits.  In this case, the
*       size of the user_info array (in bytes) must equal
*       sizeof(float) * n_points.
*
*   Unique X values provided for all fits:
*
*       In this case, the user_info array must contain X values for each
*       fit in the dataset.  In this case, the size of the user_info array
*       (in bytes) must equal sizeof(float) * n_points * nfits.
*
* Parameters:
*
* parameters: An input vector of model parameters.
*             p[0]: amplitude
*             p[1]: center coordinate
*             p[2]: width (standard deviation)
*             p[3]: offset
*
* n_fits: The number of fits. (not used)
*
* n_points: The number of data points per fit.
*
* value: An output vector of model function values.
*
* derivative: An output vector of model function partial derivatives.
*
* point_index: The data point index.
*
* fit_index: The fit index. (not used)
*
* chunk_index: The chunk index. (not used)
*
* user_info: An input vector containing user information.
*
* user_info_size: The size of user_info in bytes.
*
* Calling the calculate_gauss1d function
* ======================================
*
* This __device__ function can be only called from a __global__ function or an other
* __device__ function.
*
*/

__device__ void calculate_bicompartment_3expIF_3k(
    float const * parameters,
    int const n_fits, // voxel number
    int const n_points, // time point number
    float * value, //output
    float * derivative, // output
    int const point_index, // current time point idx
    int const fit_index, // current voxel idx
    int const chunk_index,
    float * times,
    float * IFpar,
    float * IFvalue,
    char * user_info, // we use this to pass the time vector unequally spaced
    std::size_t const user_info_size)
{
    // indices

    float * times_float = (float*)times;
    float x = 0.0f;
    if (!times_float)
    {
        x = point_index;
    }
    else if (user_info_size / sizeof(float) == n_points)
    {
        x = times_float[point_index]; // current time point with custom time vector
    }
    else if (user_info_size / sizeof(float) > n_points)
    {
        int const chunk_begin = chunk_index * n_fits * n_points;
        int const fit_begin = fit_index * n_points;
        x = times_float[chunk_begin + fit_begin + point_index];
    }

    // derivative
    float * current_derivative = derivative + point_index;

    // parameters
    float const * p = parameters;   //p[0] = fv
                                    //p[1] =
                                    //p[2] =
                                    //p[3] =
                                    //p[4] =
    float delta0;
    float delta;
    float Ahat[3];
    float Abar[3];
    float sum = 0.f;
    float TAC = 0.f;
    float Jb = 0.f;
    float Jl = 0.f;
    float dk = (float)(log(2.)/109.8);

    Abar[0] = -IFpar[2]-IFpar[3];
    Abar[1] =  IFpar[2];
    Abar[2] =  IFpar[3];


    // value
    float const argx = (x - p[1]) * (x - p[1]) / (2 * p[2] * p[2]);
    float const ex = exp(-argx);
    value[point_index] = p[0] * ex + p[3];

    for (uint ii=1; ii<=3; ii+=2) { //i = 1:2:4 % 2 compartiments
        delta0  = p[ii+1] + IFpar[4];
        Ahat[0] = -IFpar[2]-IFpar[3]-(IFpar[1]/delta0);
        Ahat[1] =  IFpar[2];
        Ahat[2] =  IFpar[3];
        sum = 0.f;
        Jb  = 0.f;
        Jl  = 0.f;

        for (uint jj=0; jj<3; ++jj) {
            delta  = p[ii+1]+IFpar[4+jj];
            if (times_float[point_index]>=IFpar[0]) {
                sum += Ahat[jj] * (1.0f / delta) *
                        ( exp(IFpar[4+jj]*(times_float[point_index]-IFpar[0]))
                         -exp(-p[ii+1]*(times_float[point_index]-IFpar[0]))
                         );
                Jb  += Ahat[jj] * (1.0f / delta) *
                        ( exp(IFpar[4+jj]*(times_float[point_index]-IFpar[0]))
                         -exp(-p[ii+1]*(times_float[point_index]-IFpar[0]))
                        );
                Jl  += Abar[jj] * (1.0f / (delta*delta)) *
                        ( exp(-p[ii+1]*(times_float[point_index]-IFpar[0]))
                            -exp(IFpar[4+jj]*(times_float[point_index]-IFpar[0])))
                        + Abar[jj] * (1.0f / delta)*
                        (times_float[point_index]-IFpar[0]) *
                        exp(-p[ii+1]*(times_float[point_index]-IFpar[0]));
            }
        }

        if (times_float[point_index]>=IFpar[0]) {

            TAC += p[ii] * (sum + ((IFpar[1]*(times_float[point_index]-IFpar[0]))/delta0)
                            *exp(IFpar[4]*(times_float[point_index]-IFpar[0])));

            current_derivative[ii+1 * n_points] =
                        (1-p[0]) * (Jb + ((IFpar[1]*(times_float[point_index]-IFpar[0]))/delta0)
                                 *exp(IFpar[4]*(times_float[point_index]-IFpar[0])));

            current_derivative[ii+2 * n_points] = (1-p[0]) * (p[ii] * (Jl +
                                 (   exp(-p[ii+1]*(times_float[point_index]-IFpar[0]))
                                    -exp(IFpar[4]*(times_float[point_index]-IFpar[0])))
                                 * (IFpar[1] *(times_float[point_index]-IFpar[0]) *
                                    (1.0f / (delta0*delta0)) + 2*IFpar[1] *
                                    (1.0f / (delta0*delta0*delta0))) ));
        }


    TAC  *= exp(-dk*times[point_index]);
    current_derivative[0 * n_points] = IFvalue[point_index] - TAC;
    TAC  = ((1-p[0]) * TAC) + (p[0] * IFvalue[point_index]);
    if (TAC < 0.0) {
            TAC = 1e-16;
    }

    value[point_index] = TAC;
}

#endif
