.. _fit-model-functions:

Fit Model functions
-------------------

This section describes the fit model functions which are included with the Gpufit library. The model IDs usable
in the call of the Gpufit :ref:`c-interface` are defined in gpufit.h_.

Currently only the one-dimensional model functions `Linear regression`_ and `1D Gaussian function`_ provide an option to pass in custom *X* coordinate values
using the user information parameter of the Gpufit interface. The data type of the values must be single precision
floating point. If calling Gpufit by its :ref:`c-interface`, the user information size parameter must be set to the
product of the number of values in the user information array and the size of the data type in bytes. The number of the *X* coordinate
values must be equal to the total number of data points or the number of data points per fit. In the second case the
same *X* coordinates will be used for each fit.

Note that additional model functions may be added as described in the documentation, see :ref:`gpufit-customization`.

.. _linear-1d:

Linear regression
+++++++++++++++++

A 1D linear function defined by two parameters (offset and slope).  The model ID of this function is ``LINEAR_1D``, and it is implemented in linear_1d.cuh_.

**Optional**: The *X* coordinate of each data point may be specified via the user information data parameter of the
Gpufit interface.

    :`Default X coordinates`:

        If the user information is not provided, the *X* coordinate of the first data value is assumed to be (0.0).
        In this case, for a fit size of *M* data points, the *X* coordinates of the data are set equal to the indices of the
        data array, starting from zero (i.e. :math:`0, 1, 2, ..., M-1`).

    :`Unique X coordinate values for each fit`:

        If the number of values in the user information array is equal to the total number of data points, unique *X*
        coordinate values are used for each fit.

    :`Same X coordinate values for all fits`:

        If the number of values in the user information array is equal to the number of data points per fit, the same *X*
        coordinate values are used for all fits.

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
The user information data may be used to specify the X coordinate of each data point.
Here, p is the vector of parameters (p0..p3) and the model function g exists for each *X* coordinate of the input data.

**Optional**: The *X* coordinate of each data point may be specified via the user information data parameter of the
Gpufit interface.

    :`Default X coordinates`:

        If the user information is not provided, the *X* coordinate of the first data value is assumed to be (0.0).
        In this case, for a fit size of *M* data points, the *X* coordinates of the data are set equal to the indices of the
        data array, starting from zero (i.e. :math:`0, 1, 2, ..., M-1`).

    :`Unique X coordinate values for each fit`:

        If the number of values in the user information array is equal to the total number of data points, unique *X*
        coordinate values are used for each fit.

    :`Same X coordinate values for all fits`:

        If the number of values in the user information array is equal to the number of data points per fit, the same *X*
        coordinate values are used for all fits.

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
Here, p is the vector of parameters (p0..p4) and the model function g exists for each x,y coordinate of the input data.

.. math::

    g(x,y,\vec{p})=p_0 e^{-\left(\left(x-p_1\right)^2+\left(y-p_2\right)^2\right)/\left(2p_3^2\right)}+p_4

:`x,y`: (independent variables) *X,Y* coordinates
	
    No independent variables are passed to this model function.
    Hence, the *(X,Y)* coordinates of the first data value are assumed to be (:math:`0.0, 0.0`).
    For a fit size of *M x N* data points, the *(X,Y)* coordinates of the data are simply the corresponding 2D array
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
in gauss_2d_elliptic.cuh_. Here, p is the vector of parameters (p0..p5) and the model function g exists for each x,y coordinate of the input data.

.. math::

    g(x,y,\vec{p})=p_0 e^{-\frac{1}{2}\left(\frac{\left(x-p_1\right)^2}{p_3^2}+\frac{\left(y-p_2\right)^2}{p_4^2}\right)}+p_5

:`x,y`: (independent variables) *X,Y* coordinates

    No independent variables are passed to this model function.
    Hence, the *(X,Y)* coordinates of the first data value are assumed to be (:math:`0.0, 0.0`).
    For a fit size of *M x N* data points, the *(X,Y)* coordinates of the data are simply the corresponding
    2D array indices of the data array, starting from zero.

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
Here, p is the vector of parameters (p0..p6) and the model function g exists for each x,y coordinate of the input data.

.. math::

    g(x,y,\vec{p})=p_0 e^{-\frac{1}{2}\left(\frac{\left((x-p_1)\cos{p_6}-(y-p_2)\sin{p_6}\right)^2}{p_3^2}+\frac{\left((x-p_1)\sin{p_6}+(y-p_2)\cos{p_6}\right)^2}{p_4^2}\right)}+p_5

:`x,y`: (independent variables) *X,Y* coordinates

    No independent variables are passed to this model function.
    Hence, the *(X,Y)* coordinates of the first data value are assumed to be (:math:`0.0, 0.0`).
    For a fit size of *M x N* data points, the *(X,Y)* coordinates of the data are simply the corresponding
    2D array indices of the data array, starting from zero.

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
in cauchy_2d_elliptic.cuh_. Here, p is the vector of parameters (p0..p5) and the model function g exists for each x,y
coordinate of the input data.

.. math::

    g(x,y,\vec{p})=p_0 \frac{1}{\left(\frac{x-p_1}{p_3}\right)^2+1} \frac{1}{\left(\frac{y-p_2}{p_4}\right)^2+1} + p_5

:`x,y`: (independent variables) *X,Y* coordinates

    No independent variables are passed to this model function.
    Hence, the *(X,Y)* coordinates of the first data value are assumed to be (:math:`0.0, 0.0`).
    For a fit size of *M x N* data points, the *(X,Y)* coordinates of the data are simply the corresponding
    2D array indices of the data array, starting from zero.

:`p_0`: amplitude
	
:`p_1`: center coordinate x
	
:`p_2`: center coordinate y
	
:`p_3`: width x (standard deviation)
	
:`p_4`: width y (standard deviation)
	
:`p_5`: offset

