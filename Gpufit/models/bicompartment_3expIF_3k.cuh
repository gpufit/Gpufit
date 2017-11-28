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
    //float * times,
    //float * IFpar,
    //float * IFvalue,
    char * user_info, // we use this to pass the time vector unequally spaced
    std::size_t const user_info_size)
{
    //float * user_info_float = (float*) user_info;

    // indices
    float const times_float[24] =
    { 0.08333333 ,0.25,0.41666666,0.58333334,0.75,0.91666666,
      1.08333333,1.25,1.41666667,1.58333333,1.75,1.91666667,
      2.25,2.75,3.5,4.5,5.5,7.,9.,12.5,17.5,22.5,27.5,35.
    };

    float const IFpar[7] =
    { 0.458300896195880,
      758028.906510941,
      3356.00773871079,
      7042.64861309165,
     -9.91821801288336,
      0.0134846319687693,
     -0.0585800774301212
    };

    float const IFvalue[24] =
    {0.,0.,24409.38070751,29004.28711479,
     16902.29913917,12060.98208071,10612.57271934,10197.68263123,
     10054.6062693,9978.15052684,9917.65977008,9861.25459037,
     9698.43991838,9541.40325626,9243.27864281,8965.43931906,
     8706.77619679,8242.85724982,7843.85989999,7087.59694672,
     6609.51306567,6344.98718338,6246.22490223,6414.56761034
    };

    //float const times_float[24] = {*times};
    float x = 0.0f;
    /*if (!times_float)
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
    }*/
    x = times_float[point_index];

    // derivative
    float * current_derivative = derivative + point_index;

    // parameters
    float const * p = parameters;   //p[0] = fv
                                    //p[1] =
                                    //p[2] =
                                    //p[3] =
                                    //p[4] =
    float Ahat[3];
    float Abar[3];
    float dk = (float)(log(2.)/109.8);
    float t0 = IFpar[0];
    float fv = p[0];
    float TAC = 0.;
    float dt = x-t0;

    Abar[0] = -IFpar[2]-IFpar[3];
    Abar[1] =  IFpar[2];
    Abar[2] =  IFpar[3];

    for (uint ii=1; ii<=3; ii+=2) { //i = 1:2:4 % 2 compartiments

        float Bi      =  p[ii];
        float Li      =  p[ii+1];
        float sumTerm     =  0.f;
        float Jb      =  0.f;
        float Jl      =  0.f;
        float delta0  =  Li + IFpar[4];
        Ahat[0]       = -IFpar[2]-IFpar[3]-(IFpar[1]/delta0);
        Ahat[1]       =  IFpar[2];
        Ahat[2]       =  IFpar[3];

        for (uint jj=0; jj<3; ++jj) {

            float Ahat_j = Ahat[jj];
            float Abar_j = Abar[jj];
            float lj     = IFpar[4+jj];
            float delta  = Li+lj;

            if (x>=t0) {
                sumTerm += Ahat_j * (1.0f / delta) *
                            ( exp(lj*dt) -exp(-Li*dt) );
                Jb  += Ahat_j * (1.0f / delta) *
                            ( exp(lj*dt) -exp(-Li*dt) );
                Jl  += Abar_j * (1.0f / (delta*delta)) *
                            ( exp(-Li*dt) -exp(lj*dt))
                            + Abar_j * (1.0f / delta)* dt * exp(-Li*dt);
            }
        }

        if (x>=t0)
        {
            TAC += Bi * (sumTerm + ((IFpar[1]*dt)/delta0)
                                *exp(IFpar[4]*dt));

            current_derivative[(ii) * n_points] =
                                        (1-fv) * (Jb + ((IFpar[1]*dt)/delta0)
                                                    *exp(IFpar[4]*dt));

            current_derivative[(ii+1) * n_points] =
                                           (1-fv) * (Bi * (Jl + ( exp(-Li*dt) -exp(IFpar[4]*dt))
                                                            * (IFpar[1] *dt *
                                                                (1.0f / (delta0*delta0))
                                                             + 2*IFpar[1] *
                                                                (1.0f / (delta0*delta0*delta0)))
                                                        )
                                                    );
        }
        else
        {
            current_derivative[(ii) * n_points] = 0;
            current_derivative[(ii+1) * n_points] = 0;
        }
    }

    TAC  *= exp(-dk*x);
    current_derivative[(0) * n_points] = IFvalue[point_index] - TAC;
    TAC  = ((1-fv) * TAC) + (fv * IFvalue[point_index]);
    if (TAC <= 1e-16)
    {
            TAC = 1e-16;
    }
    value[point_index] = TAC;
    // how to debug Jacobian
    // value[point_index] = current_derivative[4 * n_points];
}
#endif
