"""
picasso.fitting.gaussfit
~~~~~~~~~~~~~~~~~~~~~~~~

CPU 2D Gaussian PSF fitting: the CPU twin of
:mod:`picasso.fitting.gaussfit_cuda`.

:func:`fit_spots` takes and returns exactly what ``gaussfit_cuda.fit_spots``
does, so the two are interchangeable backends and ``picasso.localize`` can
dispatch between them on a single flag - the arrangement
:mod:`picasso.fitting.splinefit` and ``splinefit_cuda`` already use for the
spline models. The models, the Levenberg-Marquardt driver, the damping rule and
the estimators are the same port of Gpufit.

Three models, all evaluated at the pixel centre:

===================  ==========  ==============================================
model                parameters  layout
===================  ==========  ==============================================
:data:`SPHERICAL`             5  ``[peak, x, y, s, bg]``
:data:`ELLIPTIC`              6  ``[peak, x, y, sx, sy, bg]``
:data:`ROTATED`               7  ``[peak, x, y, sx, sy, bg, angle]``
===================  ==========  ==============================================

The amplitude is the Gaussian's *peak height*, not its integral;
``picasso.localize`` converts it to photons afterwards.


References
----------
The fitting algorithm and all three models are a port of Gpufit
(``models/gauss_2d*.cuh``, ``Cpufit/lm_fit_cpp.cpp``):

Przybylski, A., Thiel, B., Keller-Findeisen, J., Stock, B. & Bates, M.
"Gpufit: An open-source toolkit for GPU-accelerated curve fitting."
Scientific Reports 7, 15722 (2017).
https://doi.org/10.1038/s41598-017-15313-9
Licence (MIT): ``LICENSES/Gpufit-LICENSE.txt``.

:authors: Rafal Kowalewski
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import threading
from concurrent import futures
from typing import Callable, Literal

import numba
import numpy as np
from tqdm import tqdm

from picasso.fitting.splinefit import (  # noqa: F401  (re-exported)
    AsyncFit,
    FIT_STATE_CONVERGED,
    FIT_STATE_MAX_ITERATION,
    FIT_STATE_NEG_CURVATURE_MLE,
    FIT_STATE_SINGULAR_HESSIAN,
    _FASTMATH,
    _LAMBDA_DOWN,
    _LAMBDA_INITIAL,
    _LAMBDA_UP,
    _allocate_outputs,
    _lm_solve_step,
    n_workers,
    resolve_variance,
)

# Model identifiers. Defined here and imported by ``gaussfit_cuda`` so the two
# devices cannot disagree about what model 1 is, exactly as ``lmfit_cuda``
# imports its constants from ``splinefit``.
SPHERICAL = 0
ELLIPTIC = 1
ROTATED = 2

_N_PARAMS = {SPHERICAL: 5, ELLIPTIC: 6, ROTATED: 7}

# Convergence schedule, from the arguments the Gpufit path passed for these
# models. It happens to coincide with the spline single-start schedule, but the
# two are independent settings and are kept separate deliberately.
TOLERANCE = 1e-2
MAX_ITERATIONS = 20

# Convergence schedule of the CPU least-squares path, inherited from the
# retired ``picasso.gausslq``: ``TOLERANCE_LSQ_CPU`` was MINPACK's relative
# reduction in both the sum of squares (``ftol``) and the parameter vector
# (``xtol``). Kept separate from the schedule above so that least-squares
# results do not move now that the fit runs here. Used by
# ``picasso.localize.gauss_schedule``.
TOLERANCE_LSQ_CPU = 1e-2
MAX_ITERATIONS_LSQ_CPU = 200


@numba.njit(nogil=True, cache=True, fastmath=_FASTMATH)
def _estimator_terms(
    mle: bool, value: float, data: float, var: float
) -> tuple:
    """Per-pixel ``(chi_square, weight, factor, ok)``.

    ``weight`` multiplies the Hessian outer product and ``factor`` the gradient,
    so a caller accumulates ``grad_k += d_k * factor`` and
    ``hess_kl += weight * d_k * d_l``.

    ``var`` is the pixel's sCMOS readout variance in photoelectrons squared,
    and zero when no camera calibration is in use. See
    ``picasso.fitting.splinefit._estimator_terms`` for the shift it produces
    and why least squares is untouched by it.

    ``ok`` is False when a maximum-likelihood fit's *shifted* model value is
    not finite **or not positive**: the fit is abandoned rather than floored.
    See the module docstring, and ``picasso.fitting.splinefit.MU_FLOOR`` for
    why the spline models do the opposite."""
    if mle:
        shifted_value = value + var
        shifted_data = data + var
        if not (np.isfinite(shifted_value) and shifted_value > 0.0):
            return np.inf, 0.0, 0.0, False
        if shifted_data > 0.0:
            return (
                2.0
                * (
                    (shifted_value - shifted_data)
                    - shifted_data * np.log(shifted_value / shifted_data)
                ),
                shifted_data / (shifted_value * shifted_value),
                -(1.0 - shifted_data / shifted_value),
                True,
            )
        # An empty (or clipped) pixel: Gpufit's data == 0 term.
        return 2.0 * shifted_value, 0.0, -1.0, True
    # Least squares: the shift cancels, so it is never formed.
    deviation = value - data
    return deviation * deviation, 1.0, -deviation, True


# ----------------------------------------------------------------------
# Per-model accumulators
#
# Each fills ``hess`` and ``grad`` at ``theta`` and returns
# ``(chi_square, ok)``. ``spots`` is ``(n_spots, box, box)`` indexed
# ``[spot, row = y, column = x]``. These transcribe Gpufit's
# ``models/gauss_2d{,_elliptic,_rotated}.cuh`` and are the line-for-line CPU
# twins of the device accumulators in ``gaussfit_cuda``.
# ----------------------------------------------------------------------


@numba.njit(nogil=True, cache=True, fastmath=_FASTMATH)
def _accumulate_spherical(
    spots: np.ndarray,
    index: int,
    variance: np.ndarray,
    use_variance: bool,
    theta: np.ndarray,
    mle: bool,
    hess: np.ndarray,
    grad: np.ndarray,
) -> tuple:
    """Isotropic Gaussian, ``[amplitude, x, y, s, bg]`` (``GAUSS_2D``)."""
    box = spots.shape[1]
    amp = theta[0]
    cx = theta[1]
    cy = theta[2]
    sigma = theta[3]
    offset = theta[4]
    if not (
        np.isfinite(amp)
        and np.isfinite(cx)
        and np.isfinite(cy)
        and np.isfinite(sigma)
        and np.isfinite(offset)
        # A zero width divides by zero in every derivative.
        and abs(sigma) > 0.0
    ):
        return np.inf, False
    inv_s2 = 1.0 / (sigma * sigma)
    inv_s3 = 1.0 / (sigma * sigma * sigma)
    chi_square = 0.0
    g0 = g1 = g2 = g3 = g4 = 0.0
    h00 = h01 = h02 = h03 = h04 = 0.0
    h11 = h12 = h13 = h14 = 0.0
    h22 = h23 = h24 = 0.0
    h33 = h34 = 0.0
    h44 = 0.0
    for j in range(box):
        dy = j - cy
        for i in range(box):
            dx = i - cx
            ex = np.exp(-0.5 * (dx * dx + dy * dy) * inv_s2)
            value = amp * ex + offset
            data = spots[index, j, i]
            d0 = ex
            d1 = amp * ex * dx * inv_s2
            d2 = amp * ex * dy * inv_s2
            d3 = amp * ex * (dx * dx + dy * dy) * inv_s3
            # d4 == 1 (offset).
            var = 0.0
            if use_variance:
                var = variance[index, j, i]
            contrib, weight, factor, ok = _estimator_terms(
                mle, value, data, var
            )
            if not ok:
                return np.inf, False
            chi_square += contrib
            g0 += d0 * factor
            g1 += d1 * factor
            g2 += d2 * factor
            g3 += d3 * factor
            g4 += factor
            w0 = weight * d0
            w1 = weight * d1
            w2 = weight * d2
            w3 = weight * d3
            h00 += w0 * d0
            h01 += w0 * d1
            h02 += w0 * d2
            h03 += w0 * d3
            h04 += w0
            h11 += w1 * d1
            h12 += w1 * d2
            h13 += w1 * d3
            h14 += w1
            h22 += w2 * d2
            h23 += w2 * d3
            h24 += w2
            h33 += w3 * d3
            h34 += w3
            h44 += weight
    grad[0] = g0
    grad[1] = g1
    grad[2] = g2
    grad[3] = g3
    grad[4] = g4
    hess[0, 0] = h00
    hess[0, 1] = hess[1, 0] = h01
    hess[0, 2] = hess[2, 0] = h02
    hess[0, 3] = hess[3, 0] = h03
    hess[0, 4] = hess[4, 0] = h04
    hess[1, 1] = h11
    hess[1, 2] = hess[2, 1] = h12
    hess[1, 3] = hess[3, 1] = h13
    hess[1, 4] = hess[4, 1] = h14
    hess[2, 2] = h22
    hess[2, 3] = hess[3, 2] = h23
    hess[2, 4] = hess[4, 2] = h24
    hess[3, 3] = h33
    hess[3, 4] = hess[4, 3] = h34
    hess[4, 4] = h44
    return chi_square, True


@numba.njit(nogil=True, cache=True, fastmath=_FASTMATH)
def _accumulate_elliptic(
    spots: np.ndarray,
    index: int,
    variance: np.ndarray,
    use_variance: bool,
    theta: np.ndarray,
    mle: bool,
    hess: np.ndarray,
    grad: np.ndarray,
) -> tuple:
    """Elliptic Gaussian, ``[amplitude, x, y, sx, sy, bg]``."""
    box = spots.shape[1]
    amp = theta[0]
    cx = theta[1]
    cy = theta[2]
    sx = theta[3]
    sy = theta[4]
    offset = theta[5]
    if not (
        np.isfinite(amp)
        and np.isfinite(cx)
        and np.isfinite(cy)
        and np.isfinite(sx)
        and np.isfinite(sy)
        and np.isfinite(offset)
        and abs(sx) > 0.0
        and abs(sy) > 0.0
    ):
        return np.inf, False
    inv_sx2 = 1.0 / (sx * sx)
    inv_sy2 = 1.0 / (sy * sy)
    inv_sx3 = 1.0 / (sx * sx * sx)
    inv_sy3 = 1.0 / (sy * sy * sy)
    chi_square = 0.0
    g0 = g1 = g2 = g3 = g4 = g5 = 0.0
    h00 = h01 = h02 = h03 = h04 = h05 = 0.0
    h11 = h12 = h13 = h14 = h15 = 0.0
    h22 = h23 = h24 = h25 = 0.0
    h33 = h34 = h35 = 0.0
    h44 = h45 = 0.0
    h55 = 0.0
    for j in range(box):
        dy = j - cy
        for i in range(box):
            dx = i - cx
            ex = np.exp(-0.5 * (dx * dx * inv_sx2 + dy * dy * inv_sy2))
            value = amp * ex + offset
            data = spots[index, j, i]
            d0 = ex
            d1 = amp * ex * dx * inv_sx2
            d2 = amp * ex * dy * inv_sy2
            d3 = amp * ex * dx * dx * inv_sx3
            d4 = amp * ex * dy * dy * inv_sy3
            # d5 == 1 (offset).
            var = 0.0
            if use_variance:
                var = variance[index, j, i]
            contrib, weight, factor, ok = _estimator_terms(
                mle, value, data, var
            )
            if not ok:
                return np.inf, False
            chi_square += contrib
            g0 += d0 * factor
            g1 += d1 * factor
            g2 += d2 * factor
            g3 += d3 * factor
            g4 += d4 * factor
            g5 += factor
            w0 = weight * d0
            w1 = weight * d1
            w2 = weight * d2
            w3 = weight * d3
            w4 = weight * d4
            h00 += w0 * d0
            h01 += w0 * d1
            h02 += w0 * d2
            h03 += w0 * d3
            h04 += w0 * d4
            h05 += w0
            h11 += w1 * d1
            h12 += w1 * d2
            h13 += w1 * d3
            h14 += w1 * d4
            h15 += w1
            h22 += w2 * d2
            h23 += w2 * d3
            h24 += w2 * d4
            h25 += w2
            h33 += w3 * d3
            h34 += w3 * d4
            h35 += w3
            h44 += w4 * d4
            h45 += w4
            h55 += weight
    grad[0] = g0
    grad[1] = g1
    grad[2] = g2
    grad[3] = g3
    grad[4] = g4
    grad[5] = g5
    hess[0, 0] = h00
    hess[0, 1] = hess[1, 0] = h01
    hess[0, 2] = hess[2, 0] = h02
    hess[0, 3] = hess[3, 0] = h03
    hess[0, 4] = hess[4, 0] = h04
    hess[0, 5] = hess[5, 0] = h05
    hess[1, 1] = h11
    hess[1, 2] = hess[2, 1] = h12
    hess[1, 3] = hess[3, 1] = h13
    hess[1, 4] = hess[4, 1] = h14
    hess[1, 5] = hess[5, 1] = h15
    hess[2, 2] = h22
    hess[2, 3] = hess[3, 2] = h23
    hess[2, 4] = hess[4, 2] = h24
    hess[2, 5] = hess[5, 2] = h25
    hess[3, 3] = h33
    hess[3, 4] = hess[4, 3] = h34
    hess[3, 5] = hess[5, 3] = h35
    hess[4, 4] = h44
    hess[4, 5] = hess[5, 4] = h45
    hess[5, 5] = h55
    return chi_square, True


@numba.njit(nogil=True, cache=True, fastmath=_FASTMATH)
def _accumulate_rotated(
    spots: np.ndarray,
    index: int,
    variance: np.ndarray,
    use_variance: bool,
    theta: np.ndarray,
    mle: bool,
    hess: np.ndarray,
    grad: np.ndarray,
) -> tuple:
    """Rotated elliptic Gaussian, ``[amplitude, x, y, sx, sy, bg, angle]``.

    The angle derivative vanishes identically when ``sx == sy``, which makes
    the first Hessian singular; ``localize._initial_parameters_gauss`` breaks
    that symmetry in its seed on purpose, and this model relies on it."""
    box = spots.shape[1]
    amp = theta[0]
    cx = theta[1]
    cy = theta[2]
    sx = theta[3]
    sy = theta[4]
    offset = theta[5]
    angle = theta[6]
    if not (
        np.isfinite(amp)
        and np.isfinite(cx)
        and np.isfinite(cy)
        and np.isfinite(sx)
        and np.isfinite(sy)
        and np.isfinite(offset)
        and np.isfinite(angle)
        and abs(sx) > 0.0
        and abs(sy) > 0.0
    ):
        return np.inf, False
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    inv_sx2 = 1.0 / (sx * sx)
    inv_sy2 = 1.0 / (sy * sy)
    inv_sx3 = 1.0 / (sx * sx * sx)
    inv_sy3 = 1.0 / (sy * sy * sy)
    chi_square = 0.0
    g0 = g1 = g2 = g3 = g4 = g5 = g6 = 0.0
    h00 = h01 = h02 = h03 = h04 = h05 = h06 = 0.0
    h11 = h12 = h13 = h14 = h15 = h16 = 0.0
    h22 = h23 = h24 = h25 = h26 = 0.0
    h33 = h34 = h35 = h36 = 0.0
    h44 = h45 = h46 = 0.0
    h55 = h56 = 0.0
    h66 = 0.0
    for j in range(box):
        dy = j - cy
        for i in range(box):
            dx = i - cx
            arga = dx * cos_a - dy * sin_a
            argb = dx * sin_a + dy * cos_a
            ex = np.exp(-0.5 * (arga * arga * inv_sx2 + argb * argb * inv_sy2))
            value = amp * ex + offset
            data = spots[index, j, i]
            d0 = ex
            d1 = (
                amp * cos_a * arga * inv_sx2 + amp * sin_a * argb * inv_sy2
            ) * ex
            d2 = (
                -amp * sin_a * arga * inv_sx2 + amp * cos_a * argb * inv_sy2
            ) * ex
            d3 = amp * arga * arga * inv_sx3 * ex
            d4 = amp * argb * argb * inv_sy3 * ex
            # d5 == 1 (offset).
            d6 = amp * arga * argb * (inv_sx2 - inv_sy2) * ex
            var = 0.0
            if use_variance:
                var = variance[index, j, i]
            contrib, weight, factor, ok = _estimator_terms(
                mle, value, data, var
            )
            if not ok:
                return np.inf, False
            chi_square += contrib
            g0 += d0 * factor
            g1 += d1 * factor
            g2 += d2 * factor
            g3 += d3 * factor
            g4 += d4 * factor
            g5 += factor
            g6 += d6 * factor
            w0 = weight * d0
            w1 = weight * d1
            w2 = weight * d2
            w3 = weight * d3
            w4 = weight * d4
            w6 = weight * d6
            h00 += w0 * d0
            h01 += w0 * d1
            h02 += w0 * d2
            h03 += w0 * d3
            h04 += w0 * d4
            h05 += w0
            h06 += w0 * d6
            h11 += w1 * d1
            h12 += w1 * d2
            h13 += w1 * d3
            h14 += w1 * d4
            h15 += w1
            h16 += w1 * d6
            h22 += w2 * d2
            h23 += w2 * d3
            h24 += w2 * d4
            h25 += w2
            h26 += w2 * d6
            h33 += w3 * d3
            h34 += w3 * d4
            h35 += w3
            h36 += w3 * d6
            h44 += w4 * d4
            h45 += w4
            h46 += w4 * d6
            h55 += weight
            h56 += w6
            h66 += w6 * d6
    grad[0] = g0
    grad[1] = g1
    grad[2] = g2
    grad[3] = g3
    grad[4] = g4
    grad[5] = g5
    grad[6] = g6
    hess[0, 0] = h00
    hess[0, 1] = hess[1, 0] = h01
    hess[0, 2] = hess[2, 0] = h02
    hess[0, 3] = hess[3, 0] = h03
    hess[0, 4] = hess[4, 0] = h04
    hess[0, 5] = hess[5, 0] = h05
    hess[0, 6] = hess[6, 0] = h06
    hess[1, 1] = h11
    hess[1, 2] = hess[2, 1] = h12
    hess[1, 3] = hess[3, 1] = h13
    hess[1, 4] = hess[4, 1] = h14
    hess[1, 5] = hess[5, 1] = h15
    hess[1, 6] = hess[6, 1] = h16
    hess[2, 2] = h22
    hess[2, 3] = hess[3, 2] = h23
    hess[2, 4] = hess[4, 2] = h24
    hess[2, 5] = hess[5, 2] = h25
    hess[2, 6] = hess[6, 2] = h26
    hess[3, 3] = h33
    hess[3, 4] = hess[4, 3] = h34
    hess[3, 5] = hess[5, 3] = h35
    hess[3, 6] = hess[6, 3] = h36
    hess[4, 4] = h44
    hess[4, 5] = hess[5, 4] = h45
    hess[4, 6] = hess[6, 4] = h46
    hess[5, 5] = h55
    hess[5, 6] = hess[6, 5] = h56
    hess[6, 6] = h66
    return chi_square, True


# ----------------------------------------------------------------------
# Levenberg-Marquardt driver
# ----------------------------------------------------------------------


@numba.njit(nogil=True, cache=True, fastmath=_FASTMATH)
def _accumulate(
    model: int,
    spots: np.ndarray,
    index: int,
    variance: np.ndarray,
    use_variance: bool,
    theta: np.ndarray,
    mle: bool,
    hess: np.ndarray,
    grad: np.ndarray,
) -> tuple:
    """Dispatch to the accumulator of ``model``."""
    if model == SPHERICAL:
        return _accumulate_spherical(
            spots, index, variance, use_variance, theta, mle, hess, grad
        )
    if model == ELLIPTIC:
        return _accumulate_elliptic(
            spots, index, variance, use_variance, theta, mle, hess, grad
        )
    return _accumulate_rotated(
        spots, index, variance, use_variance, theta, mle, hess, grad
    )


@numba.njit(nogil=True, cache=True, fastmath=_FASTMATH)
def _fit_gauss_spot(
    spots: np.ndarray,
    index: int,
    variance: np.ndarray,
    use_variance: bool,
    model: int,
    init: np.ndarray,
    mle: bool,
    tolerance: float,
    max_iterations: int,
    thetas: np.ndarray,
    chi_squares: np.ndarray,
    states: np.ndarray,
    iterations: np.ndarray,
) -> None:
    """Fit one spot with the Levenberg-Marquardt driver ported from Gpufit.

    The same driver as :func:`picasso.fitting.splinefit._fit_spline_spot`,
    minus the axial multi-start - a Gaussian has no axial coordinate, so there
    is exactly one start per spot and nothing to rank.

    Results are written into the preallocated ``thetas``, ``chi_squares``,
    ``states`` and ``iterations`` at row ``index``.
    """
    n_params = init.shape[1]
    # All scratch is allocated once per spot, outside the iteration loop -
    # allocating inside it dominates the runtime of a threaded numba kernel.
    theta = np.empty(n_params)
    theta_previous = np.empty(n_params)
    grad = np.zeros(n_params)
    grad_ok = np.zeros(n_params)
    delta = np.empty(n_params)
    scaling = np.empty(n_params)
    hess = np.zeros((n_params, n_params))
    hess_ok = np.zeros((n_params, n_params))
    hess_damped = np.empty((n_params, n_params))
    indxc = np.empty(n_params, dtype=np.int32)
    indxr = np.empty(n_params, dtype=np.int32)
    ipiv = np.empty(n_params, dtype=np.int32)

    for p in range(n_params):
        theta[p] = init[index, p]
        scaling[p] = 0.0

    state = FIT_STATE_CONVERGED
    lam = _LAMBDA_INITIAL
    n_iterations = 0

    chi_square, ok = _accumulate(
        model, spots, index, variance, use_variance, theta, mle, hess, grad
    )
    if not ok:
        # The seed itself is unusable (non-finite or non-positive model).
        state = FIT_STATE_NEG_CURVATURE_MLE
    else:
        for p in range(n_params):
            grad_ok[p] = grad[p]
            for q in range(n_params):
                hess_ok[p, q] = hess[p, q]
    previous_chi_square = chi_square

    for iteration in range(max_iterations):
        if state != FIT_STATE_CONVERGED:
            break
        if not _lm_solve_step(
            hess_ok,
            grad_ok,
            scaling,
            lam,
            hess_damped,
            delta,
            n_params,
            indxc,
            indxr,
            ipiv,
        ):
            # The step is garbage, so it is not applied; the parameters of the
            # last accepted iteration stand.
            state = FIT_STATE_SINGULAR_HESSIAN
            break
        for p in range(n_params):
            theta_previous[p] = theta[p]
            theta[p] += delta[p]
        new_chi_square, ok = _accumulate(
            model, spots, index, variance, use_variance, theta, mle, hess, grad
        )
        n_iterations = iteration + 1
        if not ok:
            # The trial step left the model non-positive or non-finite. That is
            # a property of the *step*, not of the fit: undo it, damp harder and
            # try again, exactly as for a step that merely worsened chi-square.
            # Aborting here would return the seed unchanged, which for a wide
            # box shows up as sigma pinned to the seed width and integer
            # coordinates.
            chi_square = previous_chi_square
            for p in range(n_params):
                theta[p] = theta_previous[p]
            lam *= _LAMBDA_UP
            if iteration == max_iterations - 1:
                state = FIT_STATE_NEG_CURVATURE_MLE
            continue
        chi_square = new_chi_square
        if chi_square < previous_chi_square or previous_chi_square == 0.0:
            # Only an improving iteration refreshes the curvature the next step
            # is damped from (Gpufit skips the gradient/Hessian kernels
            # entirely on a failed iteration).
            for p in range(n_params):
                grad_ok[p] = grad[p]
                for q in range(n_params):
                    hess_ok[p, q] = hess[p, q]
        converged = abs(chi_square - previous_chi_square) < max(
            tolerance, tolerance * abs(chi_square)
        )
        if not converged and iteration == max_iterations - 1:
            state = FIT_STATE_MAX_ITERATION
        if chi_square < previous_chi_square:
            lam *= _LAMBDA_DOWN
            previous_chi_square = chi_square
        else:
            lam *= _LAMBDA_UP
            chi_square = previous_chi_square
            for p in range(n_params):
                theta[p] = theta_previous[p]
        if converged:
            break

    finite = np.isfinite(chi_square)
    if finite:
        for p in range(n_params):
            if not np.isfinite(theta[p]):
                finite = False
                break
    if finite:
        for p in range(n_params):
            thetas[index, p] = theta[p]
        chi_squares[index] = chi_square
        states[index] = state
        iterations[index] = n_iterations
    else:
        # The fit diverged. Report NaN parameters with an infinite chi-square;
        # ``localize.locs_from_fits_gauss`` turns those into NaN precisions.
        for p in range(n_params):
            thetas[index, p] = np.nan
        chi_squares[index] = np.inf
        if mle:
            states[index] = FIT_STATE_NEG_CURVATURE_MLE
        else:
            states[index] = FIT_STATE_SINGULAR_HESSIAN
        iterations[index] = n_iterations


# ----------------------------------------------------------------------
# Host layer
# ----------------------------------------------------------------------


def n_parameters(model: int) -> int:
    """Parameter count of a model."""
    return _N_PARAMS[model]


def _prepare(
    model: int,
    spots: np.ndarray,
    initial_parameters: np.ndarray,
    tolerance: float | None,
    max_iterations: int | None,
    variance: np.ndarray | None = None,
) -> tuple:
    """Validate and normalize the inputs both entry points share."""
    if model not in _N_PARAMS:
        raise ValueError(f"Unknown Gaussian model {model}.")
    spots = np.asarray(spots)
    if spots.ndim != 3 or spots.shape[1] != spots.shape[2]:
        raise ValueError(
            f"spots must have shape (n_spots, box, box), got {spots.shape}."
        )
    n_params = _N_PARAMS[model]
    initial_parameters = np.asarray(initial_parameters)
    if initial_parameters.shape != (len(spots), n_params):
        raise ValueError(
            "initial_parameters must have shape "
            f"{(len(spots), n_params)}, got {initial_parameters.shape}."
        )
    if tolerance is None:
        tolerance = TOLERANCE
    if max_iterations is None:
        max_iterations = MAX_ITERATIONS
    spots = np.ascontiguousarray(spots, dtype=np.float32)
    initial_parameters = np.ascontiguousarray(
        initial_parameters, dtype=np.float64
    )
    variance, use_variance = resolve_variance(variance, spots.shape, ndim=3)
    outputs = _allocate_outputs(len(spots), n_params)
    return (
        spots,
        variance,
        use_variance,
        initial_parameters,
        float(tolerance),
        int(max_iterations),
        outputs,
    )


def _kernel_args(
    variance: np.ndarray,
    use_variance: bool,
    model: int,
    initial_parameters: np.ndarray,
    mle: bool,
    tolerance: float,
    max_iterations: int,
    outputs: tuple,
) -> tuple:
    """Freeze everything :func:`_fit_gauss_spot` needs after ``index``.

    ``variance``/``use_variance`` lead, so they land immediately after
    ``index`` in the kernel signature and every ``_fit_gauss_spot(spots, index,
    *args)`` call site stays as it was."""
    thetas, chi_squares, states, iterations = outputs
    return (
        variance,
        bool(use_variance),
        int(model),
        initial_parameters,
        bool(mle),
        float(tolerance),
        int(max_iterations),
        thetas,
        chi_squares,
        states,
        iterations,
    )


def fit_spots(
    model: int,
    spots: np.ndarray,
    initial_parameters: np.ndarray,
    mle: bool = False,
    tolerance: float | None = None,
    max_iterations: int | None = None,
    progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    variance: np.ndarray | None = None,
) -> tuple:
    """Fit spots with a 2D Gaussian model on the CPU, serially.

    Signature-compatible with
    :func:`picasso.fitting.gaussfit_cuda.fit_spots`, so the two are
    interchangeable backends.

    Parameters
    ----------
    model : int
        :data:`SPHERICAL`, :data:`ELLIPTIC` or :data:`ROTATED`.
    spots : np.ndarray
        ``(n_spots, box, box)`` photon counts, indexed ``[spot, y, x]``.
    initial_parameters : np.ndarray
        ``(n_spots, n_params)`` seeds, from
        ``localize._initial_parameters_gauss``.
    mle : bool, optional
        Use the Poisson maximum-likelihood estimator instead of least squares.
    tolerance, max_iterations : optional
        ``None`` uses :data:`TOLERANCE` / :data:`MAX_ITERATIONS`.
    progress_callback : callable, "console" or None, optional
        ``"console"`` shows a tqdm bar; a callable is invoked with the
        cumulative number of spots fitted.

    Returns
    -------
    thetas, chi_squares, states, iterations
        Using Gpufit's state codes (see
        ``picasso.fitting.splinefit.FIT_STATE_CONVERGED``).
    """
    (
        spots,
        variance,
        use_variance,
        initial_parameters,
        tolerance,
        max_iterations,
        outputs,
    ) = _prepare(
        model, spots, initial_parameters, tolerance, max_iterations, variance
    )
    args = _kernel_args(
        variance,
        use_variance,
        model,
        initial_parameters,
        mle,
        tolerance,
        max_iterations,
        outputs,
    )
    use_tqdm = progress_callback == "console"
    index_range = range(len(spots))
    if use_tqdm:
        index_range = tqdm(index_range, desc="Fitting", unit="spot")
    for index in index_range:
        _fit_gauss_spot(spots, index, *args)
        if callable(progress_callback):
            progress_callback(index + 1)
    return outputs


def _worker(
    spots: np.ndarray, args: tuple, current: list, lock, stop: np.ndarray
) -> None:
    """Worker for asynchronous fitting: claim the next spot and fit it."""
    n_spots = len(spots)
    while True:
        with lock:
            if stop[0] or current[0] == n_spots:
                return
            index = current[0]
            current[0] += 1
        _fit_gauss_spot(spots, index, *args)


def fit_spots_async(
    model: int,
    spots: np.ndarray,
    initial_parameters: np.ndarray,
    mle: bool = False,
    tolerance: float | None = None,
    max_iterations: int | None = None,
    n_threads: int | None = None,
    variance: np.ndarray | None = None,
) -> AsyncFit:
    """Fit spots with a 2D Gaussian model on several CPU threads.

    Returns immediately with an
    :class:`picasso.fitting.splinefit.AsyncFit` the caller polls for progress,
    aborts, or checks for worker errors. See :func:`fit_spots` for the
    arguments.

    The kernels are ``nogil``, so this is *threads*, not processes - unlike
    the ``gausslq.fit_spots_parallel`` it replaces, which forked up to 60
    worker processes and pickled the spots into each of them.
    """
    (
        spots,
        variance,
        use_variance,
        initial_parameters,
        tolerance,
        max_iterations,
        outputs,
    ) = _prepare(
        model, spots, initial_parameters, tolerance, max_iterations, variance
    )
    args = _kernel_args(
        variance,
        use_variance,
        model,
        initial_parameters,
        mle,
        tolerance,
        max_iterations,
        outputs,
    )
    n_spots = len(spots)
    if n_threads is None:
        n_threads = n_workers()
    n_threads = max(1, min(int(n_threads), max(n_spots, 1)))
    lock = threading.Lock()
    current = [0]
    stop = np.zeros(1, dtype=np.uint8)
    executor = futures.ThreadPoolExecutor(n_threads)
    fs = [
        executor.submit(_worker, spots, args, current, lock, stop)
        for _ in range(n_threads)
    ]
    executor.shutdown(wait=False)
    return AsyncFit(current, *outputs, fs, stop)
