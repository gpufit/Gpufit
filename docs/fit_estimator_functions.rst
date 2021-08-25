.. _estimator-functions:

Estimator functions
-------------------

.. _estimator-lse:

Least squares estimator
+++++++++++++++++++++++

The least squares estimator computes the weighted sum of the squared deviation between the data values and the model at
the positions of the data points. The ID for this estimator is ``LSE``. It's implemented in lse.cuh_.

Least squares estimation is a common method, and the standard Levenberg-Marquardt algorithm described by Marquardt makes 
use of minimal least squares. The estimator is described as follows.

.. math::

    {\chi^2}(\vec{p}) = \sum_{n=0}^{N-1}{ \left(f_{n}(\vec{p})-z_{n}\right)^2\cdot w_n }

:`n`: The index of the data points (:math:`0,..,N-1`)

:`f_n`: The model function values at data position :math:`n`

:`z_n`: Data values at data position :math:`n`

:`\vec{p}`: Fit model function parameters

:`w_n`: Weight values for data at position :math:`n`


.. _estimator-mle:

Maximum likelihood estimator for data subject to Poisson statistics
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The maximum likelihood estimator (MLE) for Poisson distributed noise is relatively simple to implement. In the case of data with Poisson noise
is provides a more precise estimate when compared to an LSE estimator. The ID for this estimator is ``MLE``. It's implemented in mle.cuh_.

The estimator is described as follows.

.. math::

    {\chi^2}(\vec{p}) = 2\sum_{n=0}^{N-1}{(f_{n}(\vec{p})-z_{n})}-2\sum_{n=0,z_n\neq0}^{N-1}{z_n ln \left(\frac{f_{n}(\vec{p})}{z_n}\right)}

:`n`: The index of the data points (:math:`0,..,N-1`)

:`f_n`: The model function values at data position :math:`n`

:`z_n`: Data values at data position :math:`n`

:`\vec{p}`: Actual model function parameters

Note that this estimator does not provide any means to weight the data values. Rather, noise in the data is assumed to be purely Poissonian.