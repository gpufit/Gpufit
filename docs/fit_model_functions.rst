.. _fit-model-functions:

Fit Model functions
-------------------

This section describes the fit model functions which are included with the Gpufit library. The model IDs usable
in the call of the Gpufit :ref:`c-interface` are defined in constants.h_.  Note that additional model functions may
be added as described in the documentation, see :ref:`gpufit-customization`.

Note: Handling of independent variables
+++++++++++++++++++++++++++++++++++++++

We note that, in the current version of the Gpufit library, the indepenent variables (e.g. *X* values) corresponding
to the fit data are not passed into the :code:`gpufit()` function.  The default behavior of the fit models functions 
is to assume the coordinates of the data values are uniformly spaced and monotonically increasing (see below).

However, the :code:`user_info` parameter of the Gpufit interface allows arbitrary data to be passed to the model
functions and estimators.  Hence, this mechanism may be used to supply independent variables to the fit.  In the 
current release, two model functions support this method: the linear regression model and the one-dimensional Gaussian
model.  This is described in more detail in the sections below, and an example is given in the Examples_ section of the 
documentation.

The `Linear regression`_ and `1D Gaussian function`_ models provide an option to pass in custom *X* coordinate values,
using the user information parameter.  In this case, the data type of the values must be single precision floating point.  
The user has two options for how to use this mechanism: one set of *X* values may be provided and used for all fits, or 
unique *X* values may be provided for all fits.  The first option allows for faster calculations, since it requires 
fewer data transfer operations.  If no *X* values are provided, the models assume that the data coordinates start with
zero, as described above.

When calling Gpufit by its :ref:`c-interface`, the user information size parameter must be set to the product of the 
number of values in the user information array and the size of the data type in bytes.  The number of the *X* coordinate
values must be equal to the total number of data points, or the number of data points per fit. 


.. _linear-1d:

Linear regression
+++++++++++++++++

A 1D linear function defined by two parameters (offset and slope).  The model ID of this function is ``LINEAR_1D``, and 
it is implemented in linear_1d.cuh_.

**Optional**: The *X* coordinate of each data point may be specified via the user information data parameter of the
Gpufit interface. The user information should then contain *X* coordinate values of type float in increasing order.

    :`Default X coordinates`:

        If user information is not provided, the *X* coordinate of the first data value is assumed to be (0.0).
        In this case, for a fit size of *M* data points, the *X* coordinates of the data are set equal to the indices of the
        data array, starting from zero (i.e. :math:`0, 1, 2, ..., M-1`).

    :`Identical X coordinate values for all fits`:

        If the number of values in the user information array is equal to the number of data points per fit, the same *X*
        coordinate values are used for all fits.

    :`Unique X coordinate values for each fit`:

        If the number of values in the user information array is equal to the total number of data points, unique *X*
        coordinate values are used for each fit.

.. math::

    g(x,\vec{p})=p_0+p_1 x

:`x`: (independent variable) *X* coordinate

    The *X* coordinate values may be specified in the user information data.  For details, see the linear regression code example, :ref:`linear-regression-example`.

:`p_0`: offset

:`p_1`: slope


.. _gauss-1d:

1D Gaussian function
++++++++++++++++++++

A 1D Gaussian function defined by four parameters. Its model ID is ``GAUSS_1D`` and it is implemented in gauss_1d.cuh_.
The user information data may be used to specify the X coordinate of each data point.  Here, :code:`p` is the vector of parameters (p0..p3) 
and the model function :code:`g` exists for each *X* coordinate of the input data.

**Optional**: The *X* coordinate of each data point may be specified via the user information data parameter of the
Gpufit interface.

    :`Default X coordinates`:

        If user information is not provided, the *X* coordinate of the first data value is assumed to be (0.0).
        In this case, for a fit size of *M* data points, the *X* coordinates of the data are set equal to the indices of the
        data array, starting from zero (i.e. :math:`0, 1, 2, ..., M-1`).

    :`Identical X coordinate values for all fits`:

        If the number of values in the user information array is equal to the number of data points per fit, the same *X*
        coordinate values are used for all fits.

    :`Unique X coordinate values for each fit`:

        If the number of values in the user information array is equal to the total number of data points, unique *X*
        coordinate values are used for each fit.

.. math::

    g(x,\vec{p})=p_0 e^{-\left(x-p_1\right)^2/\left(2p_2^2\right)}+p_3

:`x`: (independent variable) *X* coordinate

    The X coordinate values may be specified in the user information data. For details on how to do this, see the linear
    regression code example, :ref:`linear-regression-example`.

:`p_0`: amplitude

:`p_1`: center coordinate

:`p_2`: width (standard deviation)

:`p_3`: offset


.. _gauss-2d:

2D Gaussian function (cylindrical symmetry)
+++++++++++++++++++++++++++++++++++++++++++

A 2D Gaussian function defined by five parameters. Its model ID is ``GAUSS_2D`` and it is implemented in gauss_2d.cuh_.
Here, :code:`p` is the vector of parameters (p0..p4) and the model function :code:`g` exists for each x,y coordinate of the input data.

.. math::

    g(x,y,\vec{p})=p_0 e^{-\left(\left(x-p_1\right)^2+\left(y-p_2\right)^2\right)/\left(2p_3^2\right)}+p_4

:`x,y`: (independent variables) *X,Y* coordinates

    No independent variables are passed to this model function.
    Hence, the *(X,Y)* coordinates of the first data value are assumed to be (:math:`0.0, 0.0`).
    The fit size is *M x M* data points (M*M=number of data points in the interface), the *(X,Y)* coordinates of the data are simply the corresponding 2D array
    indices of the data array, starting from zero.

:`p_0`: amplitude

:`p_1`: center coordinate x

:`p_2`: center coordinate y

:`p_3`: width (standard deviation; equal width in x and y dimensions)

:`p_4`: offset


.. _gauss-2d-elliptic:

2D Gaussian function (elliptical)
+++++++++++++++++++++++++++++++++

A 2D elliptical Gaussian function defined by six parameters. Its model ID is ``GAUSS_2D_ELLIPTIC`` and it is implemented
in gauss_2d_elliptic.cuh_. Here, :code:`p` is the vector of parameters (p0..p5) and the model function :code:`g` exists for each x,y coordinate of the input data.

.. math::

    g(x,y,\vec{p})=p_0 e^{-\frac{1}{2}\left(\frac{\left(x-p_1\right)^2}{p_3^2}+\frac{\left(y-p_2\right)^2}{p_4^2}\right)}+p_5

:`x,y`: (independent variables) *X,Y* coordinates

    No independent variables are passed to this model function.
    Hence, the *(X,Y)* coordinates of the first data value are assumed to be (:math:`0.0, 0.0`).
    The fit size is *M x M* data points (M*M=number of data points in the interface), the *(X,Y)* coordinates of the data are simply the corresponding 2D array
    indices of the data array, starting from zero.

:`p_0`: amplitude

:`p_1`: center coordinate x

:`p_2`: center coordinate y

:`p_3`: width x (standard deviation)

:`p_4`: width y (standard deviation)

:`p_5`: offset


.. _gauss-2d-rotated:

2D Gaussian function (elliptical, rotated)
++++++++++++++++++++++++++++++++++++++++++

A 2D elliptical Gaussian function whose principal axis may be rotated with respect to the X and Y coordinate axes,
defined by seven parameters. Its model is ``GAUSS_2D_ROTATED`` and it is implemented in gauss_2d_rotated.cuh_.
Here, :code:`p` is the vector of parameters (p0..p6) and the model function :code:`g` exists for each x,y coordinate of the input data.

.. math::

    g(x,y,\vec{p})=p_0 e^{-\frac{1}{2}\left(\frac{\left((x-p_1)\cos{p_6}-(y-p_2)\sin{p_6}\right)^2}{p_3^2}+\frac{\left((x-p_1)\sin{p_6}+(y-p_2)\cos{p_6}\right)^2}{p_4^2}\right)}+p_5

:`x,y`: (independent variables) *X,Y* coordinates

    No independent variables are passed to this model function.
    Hence, the *(X,Y)* coordinates of the first data value are assumed to be (:math:`0.0, 0.0`).
    The fit size is *M x M* data points (M*M=number of data points in the interface), the *(X,Y)* coordinates of the data are simply the corresponding 2D array
    indices of the data array, starting from zero.

:`p_0`: amplitude

:`p_1`: center coordinate x

:`p_2`: center coordinate y

:`p_3`: width x (standard deviation)

:`p_4`: width y (standard deviation)

:`p_5`: offset

:`p_6`: rotation angle [radians]


.. _cauchy-2d-elliptic:

2D Cauchy function (elliptical)
+++++++++++++++++++++++++++++++

A 2D elliptical Cauchy function defined by six parameters. Its model ID is ``CAUCHY_2D_ELLIPTIC`` and it is implemented
in cauchy_2d_elliptic.cuh_. Here, :code:`p` is the vector of parameters (p0..p5) and the model function :code:`g` exists for each x,y
coordinate of the input data.

.. math::

    g(x,y,\vec{p})=p_0 \frac{1}{\left(\frac{x-p_1}{p_3}\right)^2+1} \frac{1}{\left(\frac{y-p_2}{p_4}\right)^2+1} + p_5

:`x,y`: (independent variables) *X,Y* coordinates

    No independent variables are passed to this model function.
    Hence, the *(X,Y)* coordinates of the first data value are assumed to be (:math:`0.0, 0.0`).
    The fit size is *M x M* data points (M*M=number of data points in the interface), the *(X,Y)* coordinates of the data are simply the corresponding 2D array
    indices of the data array, starting from zero.

:`p_0`: amplitude

:`p_1`: center coordinate x

:`p_2`: center coordinate y

:`p_3`: width x (standard deviation)

:`p_4`: width y (standard deviation)

:`p_5`: offset


1D Spline function
++++++++++++++++++

A 1D cubic spline function defined by 3 parameters and a set of cubic spline coefficients. See `Gpuspline on Github`_ for details
on how to compute the set of cubic spline coefficients from a data set so that it can be used here. The model ID is ``SPLINE_1D``
and it is implemented in spline_1d.cuh_. Here, :code:`p` is the vector of parameters (p0..p2) and :code:`c` the vector
of spline coefficients. The model function :code:`g` exists for each x coordinate of the input data.

.. math::

    g_i(x,\vec{p},\vec{c}_i)=p_2 + p_0 \sum_{m=0}^{3} c_{i,m}(x-i-p_1)^m

:`x`: (independent variables) *X* coordinates

    No independent variables are passed to this model function.
    Hence, the *X* coordinate of the first data value is assumed to be :math:`0.0`.
    The fit size is *M* data points (M=number of data points in the interface), the *X* coordinates of the data are
    simply the corresponding array indices of the data array, starting from zero.

:`p_0`: amplitude

:`p_1`: center coordinate

:`p_2`: offset

2D Spline function
++++++++++++++++++

A 2D cubic spline function defined by 4 parameters and a set of cubic spline coefficients. See `Gpuspline on Github`_ for details
on how to compute the set of cubic spline coefficients from a data set so that it can be used here. The model ID is ``SPLINE_2D``
and it is implemented in spline_2d.cuh_. Here, :code:`p` is the vector of parameters (p0..p3) and :code:`c` the vector
of spline coefficients. The model function :code:`g` exists for each x,y coordinate of the input data.

.. math::

    g_{i,j}(x,y,\vec{p},\vec{c}_{i,j})=p_3 + p_0 \sum_{m=0}^{3} \sum_{n=0}^{3} c_{i,j,m,n} (x-i-p_1)^m (y-j-p_2)^n

:`x,y`: (independent variables) *X,Y* coordinates

    No independent variables are passed to this model function.
    Hence, the *(X,Y)* coordinates of the first data value are assumed to be (:math:`0.0, 0.0`).
    The fit size is *M x N* data points (M*N=number of data points in the interface), the *(X,Y)* coordinates of the
    data are simply the corresponding 2D array indices of the data array, starting from zero.

:`p_0`: amplitude

:`p_1`: center coordinate x

:`p_2`: center coordinate y

:`p_3`: offset

3D Spline function
++++++++++++++++++

A 3D cubic spline function defined by 5 parameters and a set of cubic spline coefficients. See `Gpuspline on Github`_ for details
on how to compute the set of cubic spline coefficients from a data set so that it can be used here. The model ID is ``SPLINE_3D``
and it is implemented in spline_3d.cuh_. Here, :code:`p` is the vector of parameters (p0..p4) and :code:`c` the vector
of spline coefficients. The model function :code:`g` exists for each x,y,z coordinate of the input data.

.. math::

    g_{i,j,k}(x,y,z,\vec{p},\vec{c}_{i,j,k})=p_4 + p_0 \sum_{m=0}^{3} \sum_{n=0}^{3} \sum_{o=0}^{3} c_{i,j,k,m,n,o} (x-i-p_1)^m (y-j-p_2)^n (z-k-p_3)^o

:`x,y,z`: (independent variables) *X,Y,Z* coordinates

    No independent variables are passed to this model function.
    Hence, the *(X,Y,Z)* coordinates of the first data value are assumed to be (:math:`0.0, 0.0, 0.0`).
    The fit size is *M x N x O* data points (M*N*O=number of data points in the interface), the *(X,Y,Z)* coordinates of
    the data are simply the corresponding 3D array indices of the data array, starting from zero.

:`p_0`: amplitude

:`p_1`: center coordinate x

:`p_2`: center coordinate y

:`p_3`: center coordinate z

:`p_4`: offset

3D Multichannel Spline function
+++++++++++++++++++++++++++++++

A 3D cubic spline function with multiple channels defined by 5 parameters and a set of cubic spline coefficients.
See `Gpuspline on Github`_ for details on how to compute the set of cubic spline coefficients from a data set so that
it can be used here. The model ID is ``SPLINE_3D_MULTICHANNEL`` and it is implemented in spline_3d_multichannel.cuh_.
Here, :code:`p` is the vector of parameters (p0..p4) and :code:`c` the vector of spline coefficients.
The model function :code:`g` exists for each x,y,z coordinate of the input data.

.. math::

    g_{ch,i,j,k}(x,y,z,\vec{p},\vec{c}_{ch,i,j,k})=p_4 + p_0 \sum_{m=0}^{3} \sum_{n=0}^{3} \sum_{o=0}^{3} c_{ch,i,j,k,m,n,o} (x-i-p_1)^m (y-j-p_2)^n (z-k-p_3)^o

:`x,y,z`: (independent variables) *X,Y,Z* coordinates

    No independent variables are passed to this model function.
    Hence, the *(X,Y,Z)* coordinates of the first data value are assumed to be (:math:`0.0, 0.0, 0.0`).
    The fit size is *M x N x O* data points (M*N*O=number of data points in the interface), the *(X,Y,Z)* coordinates of
    the data are simply the corresponding 3D array indices of the data array, starting from zero.

:`p_0`: amplitude

:`p_1`: center coordinate x

:`p_2`: center coordinate y

:`p_3`: center coordinate z

:`p_4`: offset

3D Multichannel Spline function with variable phase
+++++++++++++++++++++++++++++++++++++++++++++++++++

A 3D cubic spline function with multiple channels and a variable phase defined by 6 parameters and a set of cubic spline
coefficients. See `Gpuspline on Github`_ for details on how to compute the set of cubic spline coefficients from a data set so that
it can be used here. The model ID is ``SPLINE_3D_PHASE_MULTICHANNEL`` and it is implemented in
spline_3d_phase_multichannel.cuh_. Here, :code:`p` is the vector of parameters (p0..p5) and :code:`c` the vector of
spline coefficients. The model function :code:`g` exists for each x,y,z coordinate of the input data.

.. math::

    g_{ch,i,j,k}(x,y,z,\vec{p},\vec{c}_{0,ch,i,j,k},\vec{c}_{1,ch,i,j,k},\vec{c}_{2,ch,i,j,k})= p_4 + p_0 \ (f_0 + cos(p5) \ f_1 + sin(p_5) \ f_2)

    f_0= \sum_{m=0}^{3} \sum_{n=0}^{3} \sum_{o=0}^{3} c_{0,ch,i,j,k,m,n,o} (x-i-p_1)^m (y-j-p_2)^n (z-k-p_3)^o

    f_1= \sum_{m=0}^{3} \sum_{n=0}^{3} \sum_{o=0}^{3} c_{1,ch,i,j,k,m,n,o} (x-i-p_1)^m (y-j-p_2)^n (z-k-p_3)^o

    f_2= \sum_{m=0}^{3} \sum_{n=0}^{3} \sum_{o=0}^{3} c_{2,ch,i,j,k,m,n,o} (x-i-p_1)^m (y-j-p_2)^n (z-k-p_3)^o

:`x,y,z`: (independent variables) *X,Y,Z* coordinates

    No independent variables are passed to this model function.
    Hence, the *(X,Y,Z)* coordinates of the first data value are assumed to be (:math:`0.0, 0.0, 0.0`).
    The fit size is *M x N x O* data points (M*N*O=number of data points in the interface), the *(X,Y,Z)* coordinates of
    the data are simply the corresponding 3D array indices of the data array, starting from zero.

:`p_0`: amplitude

:`p_1`: center coordinate x

:`p_2`: center coordinate y

:`p_3`: center coordinate z

:`p_4`: offset

:`p_5`: phase