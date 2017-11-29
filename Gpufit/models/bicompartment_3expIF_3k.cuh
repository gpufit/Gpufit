#ifndef GPUFIT_BICOMP3EXP3K_CUH_INCLUDED
#define GPUFIT_BICOMP3EXP3K_CUH_INCLUDED
/* Description of the calculate_bicompartment_3expIF_3k function
* ==============================================
*
* This function calculates the values of one-dimensional compartmental model
* functions and their partial derivatives with respect to the model parameters.
*
* In literature you can find many example of compartmental models. The one
* implemented here is usually referred to as bi-compartmental model, and
* as such it is characterized by a bi-exponential impulse response function
* (IRF[t]). This model are used to model the tissue response after the
* injection of a tracer in blood. Final output of the model can be
* described as a convolution between the tissue IRF, and the blood input
* time series (IF[t]).
*
* This means that a measurment of the blood time series is a MUST HAVE
* requirement to compute the model output and to fit it to the data.
*
* Moreover, to avoid implementing a numeric convolution (maybe object of
* a future upgrade), here we decided to implement an analytic solution of
* the convolution between tissue IRF and blood input funciont:
*
*          value[t] = fv*IF[t] + (1-fv)* { conv(IRF[t], IF[t]) }
*
* This analytic solution requires a theoretical model of the input function
* as well. Here we used one of the models proposed by Feng[1] based on a
* 3-exponential model, described by 7 model parameters. This fitting of
* the input function is not implemented in Gpufit, yet, so the 7
* parameters required by this specific compartmental model need to be
* supplied as additional input.
*
* User information data
* ======================================
* This function makes use of the user information data to pass in the
* input function values (IF[t]), the input funcion parameters (IFpar), and
* if needed, the independent variable (X values, time vector) corresponding
* to the data. All of them MUST be float type.
*
* You have 2 option regarding the shape of the user_info input array:
*
*    1) Supply all the possible input data, such has:
*       -> std::vector< float > IFpar[7]
*       -> std::vector< float > IF[n_points]
*       -> std::vector< float > X[n_points]
*
*           Those three array need to be appended one to the other in one
*           single user_info input vector. You MUST respect the suggested
*           order. In main.cpp file you can have something like:
*
*            size_t const user_info_size = (times_float.size()
*                                           + IFvalue.size()
*                                           + IFparam.size()) * sizeof(float);
*            std::vector< float > user_info;
*            user_info.reserve(user_info_size / sizeof(float));
*            user_info.insert(user_info.end(), IFparam.begin(), IFparam.end());
*            user_info.insert(user_info.end(), IFvalue.begin(), IFvalue.end());
*            user_info.insert(user_info.end(),
*                               times_float.begin(), times_float.end());
*
*    2) Supply just the required input data, such has:
*       -> std::vector< float > IFpar[7]
*       -> std::vector< float > IF[n_points]
*
*           Those two array need to be appended one to the other in one
*           single user_info input vector. You MUST respect the suggested
*           order. In main.cpp file you can have something like:
*
*            size_t const user_info_size = ( IFvalue.size()
*                                           + IFparam.size()) * sizeof(float);
*            std::vector< float > user_info;
*            user_info.reserve(user_info_size / sizeof(float));
*            user_info.insert(user_info.end(), IFparam.begin(), IFparam.end());
*            user_info.insert(user_info.end(), IFvalue.begin(), IFvalue.end());
*            user_info.insert(user_info.end(),
*
*           If no user information is provided, the (X) coordinate of the
*           first data value is assumed to be (0.0).  In this case, for a
*           fit size of M data points, the (X) coordinates of the data are
*           simply the corresponding array index values of the data array,
*           starting from zero.
*
*   If the required input data regarding the blood time series are not
*   provided, it won't be possible to compute the model output and the
*   fitting .
*
* Parameters:
* ======================================
* parameters: An input vector of model parameters.
*             p[0]: fv (fraction of blood in tissue)
*             p[1]: K1
*             p[2]: k2
*             p[3]: k3
*             p[4]: k4
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
* Calling the calculate_bicompartment_3expIF_3k function
* ======================================
*
* This __device__ function can be only called from a __global__ function or an other
* __device__ function.
*
*
* References
* ======================================
* [1] D. Feng, S.-C. Huang, and X. Wang, “Models for computer simulation studies
*     of input functions for tracer kinetic modeling with positron emission
*     tomography,” International journal of bio-medical computing, vol. 32, no. 2,
*     pp. 95–110, 1993.
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
    char * user_info, // we use this to pass the time vector unequally spaced
    std::size_t const user_info_size)
{
    //independent variable
    float x   = 0.0f;
    float IF  = 0.0f;
    int const num_IFparams = 7;
    float IFpar[num_IFparams];
    float * user_info_float = (float*)user_info;

    if (user_info_size / sizeof(float) == (2*n_points + 7))
    {
        for(int ipar=0; ipar<7; ipar++)
        {
            IFpar[ipar] = user_info_float[ipar];
        }
        IF = user_info_float[num_IFparams + point_index];
        x  = user_info_float[num_IFparams + n_points + point_index];
    }
    else if (user_info_size / sizeof(float) == (n_points + 7))
    {
        for(int ipar=0; ipar<7; ipar++)
        {
            IFpar[ipar] = user_info_float[ipar];
        }
        IF = user_info_float[num_IFparams + point_index];
        x  = point_index;
    }
    else
    {
        float * current_derivative = derivative + point_index;
        current_derivative[0 * n_points] = 0.0f;
        current_derivative[1 * n_points] = 0.0f;
        current_derivative[2 * n_points] = 0.0f;
        current_derivative[3 * n_points] = 0.0f;
        current_derivative[4 * n_points] = 0.0f;

        value[point_index] = 0.0f;
        return;
    }

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
    current_derivative[(0) * n_points] = IF - TAC;
    TAC  = ((1-fv) * TAC) + (fv * IF);
    if (TAC <= 1e-16) { TAC = 1e-16; }

    value[point_index] = TAC;
    // how to debug Jacobian
    // value[point_index] = current_derivative[4 * n_points];
}
#endif
