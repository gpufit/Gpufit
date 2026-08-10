"""
picasso.fitting.splinefit
~~~~~~~~~~~~~~~~~~~~~~~~~

CPU cubic-spline PSF fitting: a numba port of Gpufit's Levenberg-Marquardt
driver together with its four spline models, for both the least-squares and
the Poisson maximum-likelihood estimator.

This is the CPU twin of ``picasso.fitting.splinefit_cuda``, which fits
the same models on the GPU. The models, the estimators, the damping rule
and the convergence test are ported from the CUDA sources
(``Gpufit/models/spline_*.cuh``, ``Gpufit/estimators/{lse,mle}.cuh``,
``Gpufit/cuda_kernels.cu``) and from Gpufit's own serial reference
``Cpufit/lm_fit_cpp.cpp``, so a CPU fit and a GPU fit of the same spots
agree to within single/double-precision differences. Four models are
covered, matching the calibrations ``picasso.spline`` can produce:

===================================  ====  ============================
calibration model                    kind  parameters
===================================  ====  ============================
``spline-2d``                        2D    ``[N, x, y, bg]``
``spline-3d``                        3D    ``[N, x, y, z, bg]``
``spline-3d-multichannel``           3D    ``[N, x, y, z, bg]`` shared
``spline-3d-multichannel-link-xyz``  LINK  ``[x, y, z, N_0.., bg_0..]``
===================================  ====  ============================

The per-spot kernels are ``nogil=True`` (not ``parallel=True``) so that a
plain thread pool can run them concurrently while a shared counter reports
per-spot progress - the same arrangement as ``picasso.gaussmle``.

This module deliberately knows nothing about calibration dicts: it takes
plain arrays only, so that ``picasso.localize`` can import it without a
circular dependency. ``localize`` owns the dict handling
(``_spline_coeff_reshaped``, ``_spline_channel_affines``,
``_initial_parameters_spline``, ``crop_spline_calibration``).

References
----------
Przybylski, A., Thiel, B., Keller-Findeisen, J., Stock, B. & Bates, M.
"Gpufit: An open-source toolkit for GPU-accelerated curve fitting."
Scientific Reports 7, 15722 (2017).
https://doi.org/10.1038/s41598-017-15313-9
Licence (MIT): ``LICENSES/Gpufit-LICENSE.txt``.

Li, Y., Mund, M., Hoess, P. et al. "Real-time 3D single-molecule
localization using experimental point spread functions." Nature Methods
15, 367-369 (2018). https://doi.org/10.1038/nmeth.4661

Babcock, H. P. & Zhuang, X. "Analyzing Single Molecule Localization
Microscopy Data Using Cubic Splines." Scientific Reports 7, 552 (2017).
https://doi.org/10.1038/s41598-017-00622-w

Li, Y., Shi, W., Liu, S. et al. "Global fitting for high-accuracy
multi-channel single-molecule localization." Nature Communications 13,
3133 (2022). https://doi.org/10.1038/s41467-022-30719-4

:authors: Rafal Kowalewski
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import multiprocessing
import threading
from concurrent import futures
from concurrent.futures import Future
from typing import Callable, Literal, NamedTuple

import numba
import numpy as np
from tqdm import tqdm

from picasso import lib

# Model kinds. The lateral (2D) and axial (3D) models need separate kernels
# because numba types an array by its number of dimensions, so a
# ``(C, niy, nix, 4, 4)`` and a ``(C, niz, niy, nix, 4, 4, 4)`` coefficient
# table cannot flow through one function body. The photon-decoupled model gets
# its own kernel to exploit its block-sparse gradient.
KIND_2D = 0
KIND_3D = 1
KIND_LINK_XYZ = 2

# Convergence schedules. A single axial start converges loosely and quickly; a
# multi-start has to tell genuinely different axial minima apart on their
# chi-square, so it runs much tighter.
TOLERANCE_SINGLE_START = 1e-2
MAX_ITERATIONS_SINGLE_START = 20
TOLERANCE_MULTI_START = 1e-4
MAX_ITERATIONS_MULTI_START = 100

# Levenberg-Marquardt damping, from Gpufit's ``gpu_data.cu`` and
# ``cuda_kernels.cu``.
_LAMBDA_INITIAL = 1e-3
_LAMBDA_DOWN = 0.1  # on an improving iteration
_LAMBDA_UP = 10.0  # on a failed iteration

# Fit states, mirroring Gpufit's ``FitState`` enum (Gpufit/constants.h). They
# are reported verbatim in the CPU path so that ``states == 0`` means the same
# thing on both devices.
FIT_STATE_CONVERGED = 0
FIT_STATE_MAX_ITERATION = 1
FIT_STATE_SINGULAR_HESSIAN = 2
FIT_STATE_NEG_CURVATURE_MLE = 3

# Smallest model value the Poisson likelihood is evaluated at, in photons.
#
# A cubic-spline PSF undershoots slightly negative in the tails - that is
# ordinary ringing of a cubic through a peaked profile, not a broken
# calibration - so ``amplitude * phi + offset`` goes negative in the corners of
# a bright, low-background spot, and the Poisson likelihood is undefined there.
# Gpufit's MLE estimator reacts by aborting the fit with ``NEG_CURVATURE_MLE``,
# which is why ``localize.fit_spline_multichannel_ratiometric`` warns that its
# MLE chi-square is unusable: a large fraction of fits bail out. Reproducing
# that would make the CPU spline MLE useless on exactly the bright spots it is
# meant for.
#
# Instead such a pixel is floored, which is what Picasso's own
# maximum-likelihood code already does elsewhere: ``gaussmle._mlefit_sigma``
# drops a pixel whose model is below ``10e-3`` from its Newton update, and
# ``precision._spline_crlb`` floors the Fisher weight at
# ``_SPLINE_CRLB_MU_FLOOR``. The pixel still contributes a (bounded) penalty to
# the likelihood, so multi-start seeds stay comparable and solutions that need
# a negative model are disfavoured, but it contributes nothing to the gradient
# or the Hessian: its ``1 / mu`` weight would otherwise be enormous and, since
# Gpufit's scaling vector is a running maximum, would damp every parameter to a
# standstill for the rest of the fit. Well-behaved fits never reach the floor,
# so this does not change their result.
MU_FLOOR = 1e-3

# LLVM fast-math flags for the numerical kernels: everything except ``nnan``
# and ``ninf``. Those two let the compiler assume no NaN or infinity can occur,
# which silently folds the divergence guards (``np.isfinite``) to a constant
# True and lets a diverged fit be reported as a converged one. Plain
# ``fastmath=True`` enables them.
_FASTMATH = {"nsz", "arcp", "contract", "afn", "reassoc"}


def convergence_schedule(apply_seeds: bool) -> tuple:
    """``(tolerance, max_iterations)`` a spline fit uses, on either device.

    The single source of truth for both backends, so a CPU fit stops where the
    GPU fit would: ``picasso.fitting.splinefit_cuda`` reads it too, and
    :func:`fit_spots` / :func:`fit_spots_async` for the CPU kernels. Splitting
    it in two is how the devices silently drift apart - the convergence test is
    *relative* (``|dchi2| < tol * max(1, chi2)``), so a factor of 100 in the
    tolerance is a real difference in where the fit stops, felt most by the
    multi-start, which ranks its axial seeds on that chi-square."""
    if apply_seeds:
        return TOLERANCE_MULTI_START, MAX_ITERATIONS_MULTI_START
    return TOLERANCE_SINGLE_START, MAX_ITERATIONS_SINGLE_START


def resolve_schedule(
    apply_seeds: bool,
    tolerance: float | None = None,
    max_iterations: int | None = None,
) -> tuple:
    """:func:`convergence_schedule`, with explicit values taking precedence.

    ``None`` means "whatever this fit would use by default", which is the
    idiom the spline API uses throughout (``n_z_starts=None``,
    ``use_gpu=None``)."""
    default_tolerance, default_max_iterations = convergence_schedule(
        apply_seeds
    )
    if tolerance is None:
        tolerance = default_tolerance
    if max_iterations is None:
        max_iterations = default_max_iterations
    return float(tolerance), int(max_iterations)


# ----------------------------------------------------------------------
# Spline evaluation
#
# The coefficient table is the one ``precision._spline_coeff_reshaped``
# produces: ``(n_channels, niz, niy, nix, 4, 4, 4)`` indexed
# ``[c, k, j, i, z_power, y_power, x_power]`` (3D) or
# ``(n_channels, niy, nix, 4, 4)`` indexed ``[c, j, i, y_power, x_power]``
# (2D).
# ----------------------------------------------------------------------


@numba.njit(nogil=True, cache=True, fastmath=_FASTMATH)
def _interval(pos: float, n_intervals: int) -> int:
    """Spline interval containing ``pos``, clamped to the coefficient grid.

    Only the *index* is clamped. The caller keeps the true fractional
    coordinate ``pos - i``, which may fall outside ``[0, 1)``, so a position
    off the edge of the box **extrapolates** the boundary cubic rather than
    saturating at it."""
    i = int(np.floor(pos))
    if i < 0:
        i = 0
    elif i > n_intervals - 1:
        i = n_intervals - 1
    return i


@numba.njit(nogil=True, cache=True, fastmath=_FASTMATH)
def _floored_chi_square(data: float) -> float:
    """Likelihood penalty of a pixel below the :data:`MU_FLOOR` model value.

    Gpufit's Poisson term evaluated at the floor. It is large but finite, so it
    disfavours parameters that need a negative model without making the
    chi-square infinite - which would make the multi-start unable to rank its
    seeds."""
    if data > 0.0:
        return 2.0 * ((MU_FLOOR - data) - data * np.log(MU_FLOOR / data))
    return 2.0 * MU_FLOOR


@numba.njit(nogil=True, cache=True, fastmath=_FASTMATH)
def _estimator_terms(
    mle: bool, value: float, data: float, var: float
) -> tuple:
    """Per-pixel ``(chi_square, weight, factor, ok)``, flooring a low model.

    ``weight`` multiplies the Hessian outer product and ``factor`` the
    gradient, so a caller accumulates ``grad_k += d_k * factor`` and
    ``hess_kl += weight * d_k * d_l``. ``ok`` is False only for a non-finite
    model value under the maximum likelihood estimator, where the caller must
    abandon the fit; a merely non-positive value is floored instead of
    rejected, see :data:`MU_FLOOR`.

    The CPU twin of ``picasso.fitting.lmfit_cuda._estimator_terms`` and the
    flooring counterpart of ``picasso.fitting.gaussfit._estimator_terms``,
    which aborts instead - the Gaussian models cannot ring negative, so for
    them the floor is harmful rather than merely unnecessary."""
    if mle:
        value = value + var
        data = data + var
        if not np.isfinite(value):
            return np.inf, 0.0, 0.0, False
        if value < MU_FLOOR:
            # Below the floor the pixel is charged to the likelihood - so
            # multi-start seeds stay comparable and invalid regions are
            # disfavoured - but kept out of the gradient and Hessian, where its
            # 1/mu weight would be enormous and, through the monotone scaling
            # vector, damp every parameter to a standstill for the rest of the
            # fit.
            return _floored_chi_square(data), 0.0, 0.0, True
        if data > 0.0:
            return (
                2.0 * ((value - data) - data * np.log(value / data)),
                data / (value * value),
                -(1.0 - data / value),
                True,
            )
        # An empty (or clipped) pixel: Gpufit's data == 0 term.
        return 2.0 * value, 0.0, -1.0, True
    deviation = value - data
    # factor == data - value.
    return deviation * deviation, 1.0, -deviation, True


@numba.njit(nogil=True, cache=True, fastmath=_FASTMATH)
def _cubic(c0: float, c1: float, c2: float, c3: float, f: float) -> tuple:
    """Value and derivative of ``c0 + c1 f + c2 f^2 + c3 f^3`` (Horner)."""
    return (
        ((c3 * f + c2) * f + c1) * f + c0,
        (3.0 * c3 * f + 2.0 * c2) * f + c1,
    )


@numba.njit(nogil=True, cache=True, fastmath=_FASTMATH)
def _eval_spline_2d(
    coeff: np.ndarray, ch: int, pos_x: float, pos_y: float
) -> tuple:
    """Bicubic value and its native lateral derivatives at one point.

    ``coeff`` is ``(n_channels, niy, nix, 4, 4)``. Returns
    ``(phi, dphi_dx, dphi_dy)``, the derivatives being with respect to the
    *native* coordinate; the derivative with respect to the fitted shift is
    their negative."""
    niy = coeff.shape[1]
    nix = coeff.shape[2]
    xi = _interval(pos_x, nix)
    yi = _interval(pos_y, niy)
    fx = pos_x - xi
    fy = pos_y - yi
    # Horner along x for each y-power row: v_p is the row's value, d_p its
    # x-derivative.
    v0, d0 = _cubic(
        coeff[ch, yi, xi, 0, 0],
        coeff[ch, yi, xi, 0, 1],
        coeff[ch, yi, xi, 0, 2],
        coeff[ch, yi, xi, 0, 3],
        fx,
    )
    v1, d1 = _cubic(
        coeff[ch, yi, xi, 1, 0],
        coeff[ch, yi, xi, 1, 1],
        coeff[ch, yi, xi, 1, 2],
        coeff[ch, yi, xi, 1, 3],
        fx,
    )
    v2, d2 = _cubic(
        coeff[ch, yi, xi, 2, 0],
        coeff[ch, yi, xi, 2, 1],
        coeff[ch, yi, xi, 2, 2],
        coeff[ch, yi, xi, 2, 3],
        fx,
    )
    v3, d3 = _cubic(
        coeff[ch, yi, xi, 3, 0],
        coeff[ch, yi, xi, 3, 1],
        coeff[ch, yi, xi, 3, 2],
        coeff[ch, yi, xi, 3, 3],
        fx,
    )
    # ... then along y. The second Horner's value is phi and its derivative is
    # dphi/dy; running it over the x-derivative rows gives dphi/dx.
    phi, gy = _cubic(v0, v1, v2, v3, fy)
    gx, _ = _cubic(d0, d1, d2, d3, fy)
    return phi, gx, gy


@numba.njit(nogil=True, cache=True, fastmath=_FASTMATH)
def _eval_spline_3d(
    coeff: np.ndarray, ch: int, pos_x: float, pos_y: float, pos_z: float
) -> tuple:
    """Tricubic value and its native spatial derivatives at one point.

    ``coeff`` is ``(n_channels, niz, niy, nix, 4, 4, 4)``. Returns
    ``(phi, dphi_dx, dphi_dy, dphi_dz)`` with respect to the native
    coordinates. See :func:`_eval_spline_2d` for the evaluation scheme; the
    z-power axis is accumulated on top of it."""
    niz = coeff.shape[1]
    niy = coeff.shape[2]
    nix = coeff.shape[3]
    xi = _interval(pos_x, nix)
    yi = _interval(pos_y, niy)
    zi = _interval(pos_z, niz)
    fx = pos_x - xi
    fy = pos_y - yi
    fz = pos_z - zi
    phi = 0.0
    gx = 0.0
    gy = 0.0
    gz = 0.0
    pz = 1.0  # fz**zp
    dpz = 0.0  # d(fz**zp)/dfz
    for zp in range(4):
        v0, d0 = _cubic(
            coeff[ch, zi, yi, xi, zp, 0, 0],
            coeff[ch, zi, yi, xi, zp, 0, 1],
            coeff[ch, zi, yi, xi, zp, 0, 2],
            coeff[ch, zi, yi, xi, zp, 0, 3],
            fx,
        )
        v1, d1 = _cubic(
            coeff[ch, zi, yi, xi, zp, 1, 0],
            coeff[ch, zi, yi, xi, zp, 1, 1],
            coeff[ch, zi, yi, xi, zp, 1, 2],
            coeff[ch, zi, yi, xi, zp, 1, 3],
            fx,
        )
        v2, d2 = _cubic(
            coeff[ch, zi, yi, xi, zp, 2, 0],
            coeff[ch, zi, yi, xi, zp, 2, 1],
            coeff[ch, zi, yi, xi, zp, 2, 2],
            coeff[ch, zi, yi, xi, zp, 2, 3],
            fx,
        )
        v3, d3 = _cubic(
            coeff[ch, zi, yi, xi, zp, 3, 0],
            coeff[ch, zi, yi, xi, zp, 3, 1],
            coeff[ch, zi, yi, xi, zp, 3, 2],
            coeff[ch, zi, yi, xi, zp, 3, 3],
            fx,
        )
        slab, slab_y = _cubic(v0, v1, v2, v3, fy)
        slab_x, _ = _cubic(d0, d1, d2, d3, fy)
        phi += slab * pz
        gx += slab_x * pz
        gy += slab_y * pz
        gz += slab * dpz
        # Advance the z power basis: at step zp, pz == fz**zp and
        # dpz == zp * fz**(zp - 1).
        dpz = (zp + 1) * pz
        pz *= fz
    return phi, gx, gy, gz


# ----------------------------------------------------------------------
# Linear solve
#
# A port of Gpufit's Gauss-Jordan solver (``Cpufit/lm_fit_cpp.cpp``,
# ``LMFitCPP::solve_equation_system_gj``). It must NOT raise: these kernels run
# under ``nogil`` in a thread pool whose futures are never awaited, so an
# exception would kill the worker silently, stall the shared progress counter
# and hang the driver loop forever. ``np.linalg.solve`` is available in
# nopython mode but raises on a singular matrix, so it is not usable here.
# ----------------------------------------------------------------------


@numba.njit(nogil=True, cache=True)
def _solve_gj(
    a: np.ndarray,
    b: np.ndarray,
    n: int,
    indxc: np.ndarray,
    indxr: np.ndarray,
    ipiv: np.ndarray,
) -> bool:
    """Solve ``a x = b`` by Gauss-Jordan elimination with full pivoting.

    ``a`` and ``b`` are destroyed; the solution is left in ``b``. Only *rows*
    are interchanged, so the solution components stay in the original parameter
    order. ``indxc``, ``indxr`` and ``ipiv`` are caller-owned scratch of length
    ``n`` (allocated once per spot, not per iteration).

    Returns False when the pivot is zero or not finite, i.e. the damped Hessian
    is singular - Gpufit's ``SINGULAR_HESSIAN``. Deliberately compiled without
    ``fastmath``: the pivot search compares against exact zero.
    """
    for i in range(n):
        ipiv[i] = 0
    for i in range(n):
        big = 0.0
        irow = 0
        icol = 0
        for j in range(n):
            if ipiv[j] != 1:
                for k in range(n):
                    if ipiv[k] == 0:
                        if abs(a[j, k]) >= big:
                            big = abs(a[j, k])
                            irow = j
                            icol = k
        ipiv[icol] += 1
        if irow != icol:
            for lx in range(n):
                tmp = a[irow, lx]
                a[irow, lx] = a[icol, lx]
                a[icol, lx] = tmp
            tmp = b[irow]
            b[irow] = b[icol]
            b[icol] = tmp
        # Recorded but never read: Gpufit drops the Numerical-Recipes column
        # back-permutation because this elimination only interchanges rows.
        # Kept so the transcription stays line-for-line comparable with
        # ``LMFitCPP::solve_equation_system_gj``; the CUDA twin
        # (``lmfit_cuda._solve_gj_device``) omits them, since two unused
        # per-thread arrays are not free on a register-bound kernel.
        indxr[i] = irow
        indxc[i] = icol
        pivot = a[icol, icol]
        # ``not (abs(pivot) > 0)`` also rejects NaN, which a plain
        # ``pivot == 0`` test would let through.
        if not (abs(pivot) > 0.0):
            return False
        pivinv = 1.0 / pivot
        a[icol, icol] = 1.0
        for lx in range(n):
            a[icol, lx] *= pivinv
        b[icol] *= pivinv
        for ll in range(n):
            if ll != icol:
                dum = a[ll, icol]
                a[ll, icol] = 0.0
                for lx in range(n):
                    a[ll, lx] -= a[icol, lx] * dum
                b[ll] -= b[icol] * dum
    return True


@numba.njit(nogil=True, cache=True)
def _lm_solve_step(
    hess: np.ndarray,
    grad: np.ndarray,
    scaling: np.ndarray,
    lam: float,
    hess_damped: np.ndarray,
    delta: np.ndarray,
    n: int,
    indxc: np.ndarray,
    indxr: np.ndarray,
    ipiv: np.ndarray,
) -> bool:
    """One Levenberg-Marquardt step: damp the Hessian and solve for ``delta``.

    ``scaling`` is Gpufit's adaptive step-width vector: it holds the largest
    Hessian diagonal each parameter has ever shown and is therefore **monotone
    across iterations** (``cuda_modify_step_widths``). Resetting it every
    iteration is the classic way to get this wrong - the damping then no longer
    reflects the curvature the fit has already seen and badly-seeded fits stop
    converging.

    ``hess`` and ``grad`` are the *undamped* matrices of the last accepted
    iteration and are left untouched; the damped copy is rebuilt from them each
    time. Returns False if the damped system is singular."""
    for p in range(n):
        d = hess[p, p]
        if d > scaling[p]:
            scaling[p] = d
    for p in range(n):
        for q in range(n):
            hess_damped[p, q] = hess[p, q]
        hess_damped[p, p] += scaling[p] * lam
        delta[p] = grad[p]
    return _solve_gj(hess_damped, delta, n, indxc, indxr, ipiv)


# ----------------------------------------------------------------------
# Per-spot Levenberg-Marquardt kernels
#
# All three take the same 15 arguments so that one worker can drive any of
# them (as ``gaussmle._worker`` takes its kernel as an argument).
#
# ``spots`` is float32 ``(n_spots, n_channels, box, box)``, channel-major, and
# indexed ``[spot, channel, row = y, column = x]``: Gpufit's fast pixel axis is
# the column, which is x, and Picasso's spot cutter writes rows as y.
# ----------------------------------------------------------------------


@numba.njit(nogil=True, cache=True, fastmath=_FASTMATH)
def _accumulate_3d(
    spots: np.ndarray,
    index: int,
    variance: np.ndarray,
    use_variance: bool,
    coeff: np.ndarray,
    aff: np.ndarray,
    res: np.ndarray,
    theta: np.ndarray,
    mle: bool,
    hess: np.ndarray,
    grad: np.ndarray,
) -> tuple:
    """Chi-square, gradient and Hessian of the shared-amplitude 3D model.

    Parameters are ``[amplitude, x_shift, y_shift, z_shift, offset]``, the
    order the Gpufit models and ``localize._initial_parameters_spline`` use.
    The single-channel ``spline-3d`` model is the ``n_channels == 1``, identity
    affine, zero residual case of the multichannel one.

    Returns ``(chi_square, ok)``. ``ok`` is False when the parameters or the
    model value are not finite, i.e. the fit has diverged; ``hess``/``grad``
    are then meaningless and the caller must not accept them. Merely
    *non-positive* model values are floored rather than rejected - see
    :data:`MU_FLOOR`."""
    n_channels = spots.shape[1]
    box = spots.shape[2]
    amp = theta[0]
    x_shift = theta[1]
    y_shift = theta[2]
    z_shift = theta[3]
    offset = theta[4]
    if not (
        np.isfinite(amp)
        and np.isfinite(x_shift)
        and np.isfinite(y_shift)
        and np.isfinite(z_shift)
        and np.isfinite(offset)
    ):
        return np.inf, False
    # The spline is evaluated at ``position = pixel - parameter``; a single
    # camera frame is fitted, so the native z is simply -z_shift.
    pos_z = -z_shift
    chi_square = 0.0
    g0 = g1 = g2 = g3 = g4 = 0.0
    h00 = h01 = h02 = h03 = h04 = 0.0
    h11 = h12 = h13 = h14 = 0.0
    h22 = h23 = h24 = 0.0
    h33 = h34 = 0.0
    h44 = 0.0
    for ch in range(n_channels):
        # Each channel sees the shared lateral shift through its own affine and
        # sits on its own sub-pixel ROI offset, as in
        # spline_3d_multichannel.cuh. Constant over the box, so hoisted.
        a00 = aff[ch, 0]
        a01 = aff[ch, 1]
        a10 = aff[ch, 2]
        a11 = aff[ch, 3]
        sx = a00 * x_shift + a01 * y_shift + res[index, ch, 0]
        sy = a10 * x_shift + a11 * y_shift + res[index, ch, 1]
        for j in range(box):
            pos_y = j - sy
            for i in range(box):
                pos_x = i - sx
                phi, gx, gy, gz = _eval_spline_3d(
                    coeff, ch, pos_x, pos_y, pos_z
                )
                value = amp * phi + offset
                data = spots[index, ch, j, i]
                # d(value)/d(parameter). The lateral pair picks up the
                # transpose of the channel affine (shift = A @ theta, so
                # d(shift_x)/d(theta_x) = a00 and d(shift_y)/d(theta_x) = a10),
                # and the leading minus is the chain rule of
                # position = pixel - shift. Unlike the CRLB, whose diagonal is
                # sign-invariant, an LM step is not: dropping the minus sends
                # x, y and z the wrong way.
                d0 = phi
                d1 = -amp * (a00 * gx + a10 * gy)
                d2 = -amp * (a01 * gx + a11 * gy)
                d3 = -amp * gz
                # d4 == 1 (offset), folded into the accumulation below.
                var = 0.0
                if use_variance:
                    var = variance[index, ch, j, i]
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
def _accumulate_2d(
    spots: np.ndarray,
    index: int,
    variance: np.ndarray,
    use_variance: bool,
    coeff: np.ndarray,
    aff: np.ndarray,
    res: np.ndarray,
    theta: np.ndarray,
    mle: bool,
    hess: np.ndarray,
    grad: np.ndarray,
) -> tuple:
    """Chi-square, gradient and Hessian of the 2D spline model.

    Parameters are ``[amplitude, x_shift, y_shift, offset]``. There is no
    multichannel 2D model, so ``aff`` and ``res`` are accepted (to keep the
    kernel signatures uniform) but unused. See :func:`_accumulate_3d`."""
    box = spots.shape[2]
    amp = theta[0]
    x_shift = theta[1]
    y_shift = theta[2]
    offset = theta[3]
    if not (
        np.isfinite(amp)
        and np.isfinite(x_shift)
        and np.isfinite(y_shift)
        and np.isfinite(offset)
    ):
        return np.inf, False
    chi_square = 0.0
    g0 = g1 = g2 = g3 = 0.0
    h00 = h01 = h02 = h03 = 0.0
    h11 = h12 = h13 = 0.0
    h22 = h23 = 0.0
    h33 = 0.0
    for j in range(box):
        pos_y = j - y_shift
        for i in range(box):
            pos_x = i - x_shift
            phi, gx, gy = _eval_spline_2d(coeff, 0, pos_x, pos_y)
            value = amp * phi + offset
            data = spots[index, 0, j, i]
            d0 = phi
            d1 = -amp * gx
            d2 = -amp * gy
            # d3 == 1 (offset).
            var = 0.0
            if use_variance:
                var = variance[index, 0, j, i]
            contrib, weight, factor, ok = _estimator_terms(
                mle, value, data, var
            )
            if not ok:
                return np.inf, False
            chi_square += contrib
            g0 += d0 * factor
            g1 += d1 * factor
            g2 += d2 * factor
            g3 += factor
            w0 = weight * d0
            w1 = weight * d1
            w2 = weight * d2
            h00 += w0 * d0
            h01 += w0 * d1
            h02 += w0 * d2
            h03 += w0
            h11 += w1 * d1
            h12 += w1 * d2
            h13 += w1
            h22 += w2 * d2
            h23 += w2
            h33 += weight
    grad[0] = g0
    grad[1] = g1
    grad[2] = g2
    grad[3] = g3
    hess[0, 0] = h00
    hess[0, 1] = hess[1, 0] = h01
    hess[0, 2] = hess[2, 0] = h02
    hess[0, 3] = hess[3, 0] = h03
    hess[1, 1] = h11
    hess[1, 2] = hess[2, 1] = h12
    hess[1, 3] = hess[3, 1] = h13
    hess[2, 2] = h22
    hess[2, 3] = hess[3, 2] = h23
    hess[3, 3] = h33
    return chi_square, True


@numba.njit(nogil=True, cache=True, fastmath=_FASTMATH)
def _accumulate_link_xyz(
    spots: np.ndarray,
    index: int,
    variance: np.ndarray,
    use_variance: bool,
    coeff: np.ndarray,
    aff: np.ndarray,
    res: np.ndarray,
    theta: np.ndarray,
    mle: bool,
    hess: np.ndarray,
    grad: np.ndarray,
) -> tuple:
    """Chi-square, gradient and Hessian of the photon-decoupled 3D model.

    Parameters are ``[x_shift, y_shift, z_shift, N_0..N_{C-1},
    bg_0..bg_{C-1}]``: x, y and z are shared across channels while every
    channel fits its own photon count and background
    (``spline_3d_multichannel_link_xyz.cuh``).

    A pixel of channel ``ch`` only ever sees five of the ``3 + 2C`` parameters,
    so the gradient and Hessian are accumulated block-sparse - 15 updates per
    pixel instead of up to 120. See :func:`_accumulate_3d`."""
    n_channels = spots.shape[1]
    box = spots.shape[2]
    n_params = 3 + 2 * n_channels
    x_shift = theta[0]
    y_shift = theta[1]
    z_shift = theta[2]
    for p in range(n_params):
        if not np.isfinite(theta[p]):
            return np.inf, False
        grad[p] = 0.0
        for q in range(n_params):
            hess[p, q] = 0.0
    pos_z = -z_shift
    chi_square = 0.0
    for ch in range(n_channels):
        amp = theta[3 + ch]
        offset = theta[3 + n_channels + ch]
        # Global parameter indices of this channel's photon count and
        # background; the three shared position parameters are 0, 1, 2.
        ia = 3 + ch
        ib = 3 + n_channels + ch
        a00 = aff[ch, 0]
        a01 = aff[ch, 1]
        a10 = aff[ch, 2]
        a11 = aff[ch, 3]
        sx = a00 * x_shift + a01 * y_shift + res[index, ch, 0]
        sy = a10 * x_shift + a11 * y_shift + res[index, ch, 1]
        for j in range(box):
            pos_y = j - sy
            for i in range(box):
                pos_x = i - sx
                phi, gx, gy, gz = _eval_spline_3d(
                    coeff, ch, pos_x, pos_y, pos_z
                )
                value = amp * phi + offset
                data = spots[index, ch, j, i]
                d0 = -amp * (a00 * gx + a10 * gy)
                d1 = -amp * (a01 * gx + a11 * gy)
                d2 = -amp * gz
                # d(value)/d(N_ch) = phi, d(value)/d(bg_ch) = 1; both zero for
                # every other channel.
                var = 0.0
                if use_variance:
                    var = variance[index, ch, j, i]
                contrib, weight, factor, ok = _estimator_terms(
                    mle, value, data, var
                )
                if not ok:
                    return np.inf, False
                chi_square += contrib
                grad[0] += d0 * factor
                grad[1] += d1 * factor
                grad[2] += d2 * factor
                grad[ia] += phi * factor
                grad[ib] += factor
                w0 = weight * d0
                w1 = weight * d1
                w2 = weight * d2
                wa = weight * phi
                hess[0, 0] += w0 * d0
                hess[0, 1] += w0 * d1
                hess[0, 2] += w0 * d2
                hess[0, ia] += w0 * phi
                hess[0, ib] += w0
                hess[1, 1] += w1 * d1
                hess[1, 2] += w1 * d2
                hess[1, ia] += w1 * phi
                hess[1, ib] += w1
                hess[2, 2] += w2 * d2
                hess[2, ia] += w2 * phi
                hess[2, ib] += w2
                hess[ia, ia] += wa * phi
                hess[ia, ib] += wa
                hess[ib, ib] += weight
    # Only the upper triangle was filled (0 < 1 < 2 < ia < ib always holds);
    # mirror it once at the end rather than per pixel.
    for p in range(n_params):
        for q in range(p):
            hess[p, q] = hess[q, p]
    return chi_square, True


@numba.njit(nogil=True, cache=True, fastmath=_FASTMATH)
def _fit_spline_spot(
    spots: np.ndarray,
    index: int,
    variance: np.ndarray,
    use_variance: bool,
    kind: int,
    coeff2d: np.ndarray,
    coeff3d: np.ndarray,
    aff: np.ndarray,
    res: np.ndarray,
    init: np.ndarray,
    z_seeds: np.ndarray,
    apply_seeds: bool,
    mle: bool,
    tolerance: float,
    max_iterations: int,
    thetas: np.ndarray,
    chi_squares: np.ndarray,
    states: np.ndarray,
    iterations: np.ndarray,
) -> None:
    """Fit one spot with the Levenberg-Marquardt driver ported from Gpufit.

    ``kind`` selects the model (:data:`KIND_2D`, :data:`KIND_3D`,
    :data:`KIND_LINK_XYZ`). Both coefficient tables are always passed because
    numba types an array by its dimensionality and cannot form a union; the
    unused one is a dummy of the right rank (see :func:`_dummy_coefficients`).

    When ``apply_seeds`` is set, the whole fit is repeated from every axial
    seed in ``z_seeds`` and the best result is kept, ranked exactly as
    ``picasso.fitting.splinefit_cuda`` ranks the GPU multi-start: the
    lowest chi-square among *converged* fits, falling back to the lowest
    chi-square among merely finite ones. Running the seeds here rather than
    re-running whole passes keeps each spot's data in cache and reports
    progress once per spot rather than once per pass.

    Results are written into the preallocated ``thetas``, ``chi_squares``,
    ``states`` and ``iterations`` at row ``index``.
    """
    n_params = init.shape[1]
    # The 2D model has no z, and its parameter 3 is the offset - seeding it
    # would silently corrupt the background rather than the axial position.
    apply_seeds = apply_seeds and kind != KIND_2D
    z_col = 2 if kind == KIND_LINK_XYZ else 3
    # All scratch is allocated once per spot, outside the seed and iteration
    # loops - allocating inside them dominates the runtime of a threaded numba
    # kernel (the same reason gaussmle._mlefit_sigma hoists its buffers).
    theta = np.empty(n_params)
    theta_previous = np.empty(n_params)
    best = np.empty(n_params)
    best_finite = np.empty(n_params)
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

    best_chi = np.inf
    best_state = FIT_STATE_MAX_ITERATION
    best_iterations = 0
    have_best = False
    best_finite_chi = np.inf
    best_finite_state = FIT_STATE_MAX_ITERATION
    best_finite_iterations = 0
    have_best_finite = False

    n_seeds = z_seeds.shape[0] if apply_seeds else 1
    for seed in range(n_seeds):
        for p in range(n_params):
            theta[p] = init[index, p]
            scaling[p] = 0.0
        if apply_seeds:
            theta[z_col] = z_seeds[seed]

        state = FIT_STATE_CONVERGED
        lam = _LAMBDA_INITIAL
        n_iterations = 0

        if kind == KIND_2D:
            chi_square, ok = _accumulate_2d(
                spots,
                index,
                variance,
                use_variance,
                coeff2d,
                aff,
                res,
                theta,
                mle,
                hess,
                grad,
            )
        elif kind == KIND_3D:
            chi_square, ok = _accumulate_3d(
                spots,
                index,
                variance,
                use_variance,
                coeff3d,
                aff,
                res,
                theta,
                mle,
                hess,
                grad,
            )
        else:
            chi_square, ok = _accumulate_link_xyz(
                spots,
                index,
                variance,
                use_variance,
                coeff3d,
                aff,
                res,
                theta,
                mle,
                hess,
                grad,
            )
        if not ok:
            # The seed itself is unusable (non-finite parameters or model).
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
                # The step is garbage, so it is not applied; the parameters of
                # the last accepted iteration stand.
                state = FIT_STATE_SINGULAR_HESSIAN
                break
            for p in range(n_params):
                theta_previous[p] = theta[p]
                theta[p] += delta[p]
            if kind == KIND_2D:
                new_chi_square, ok = _accumulate_2d(
                    spots,
                    index,
                    variance,
                    use_variance,
                    coeff2d,
                    aff,
                    res,
                    theta,
                    mle,
                    hess,
                    grad,
                )
            elif kind == KIND_3D:
                new_chi_square, ok = _accumulate_3d(
                    spots,
                    index,
                    variance,
                    use_variance,
                    coeff3d,
                    aff,
                    res,
                    theta,
                    mle,
                    hess,
                    grad,
                )
            else:
                new_chi_square, ok = _accumulate_link_xyz(
                    spots,
                    index,
                    variance,
                    use_variance,
                    coeff3d,
                    aff,
                    res,
                    theta,
                    mle,
                    hess,
                    grad,
                )
            n_iterations = iteration + 1
            if not ok:
                # A property of the trial *step*, not of the fit: undo it, damp
                # harder and try again, exactly as for a step that merely
                # worsened chi-square. Aborting here would return the seed
                # unchanged. Kept identical to the driver in
                # ``picasso.fitting.gaussfit`` and its device twin.
                chi_square = previous_chi_square
                for p in range(n_params):
                    theta[p] = theta_previous[p]
                lam *= _LAMBDA_UP
                if iteration == max_iterations - 1:
                    state = FIT_STATE_NEG_CURVATURE_MLE
                continue
            chi_square = new_chi_square
            if chi_square < previous_chi_square or previous_chi_square == 0.0:
                # Only an improving iteration refreshes the curvature the next
                # step is damped from (Gpufit skips the gradient/Hessian
                # kernels entirely on a failed iteration).
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
        if not finite:
            continue
        # Ranking, identical to the GPU multi-start: prefer converged fits,
        # and among equals the lowest chi-square. For least squares every
        # finite fit counts as converged, mirroring
        # ``converged = finite & ((states == 0) if mle else True)``.
        if chi_square < best_finite_chi:
            best_finite_chi = chi_square
            best_finite_state = state
            best_finite_iterations = n_iterations
            have_best_finite = True
            for p in range(n_params):
                best_finite[p] = theta[p]
        ok_fit = (state == FIT_STATE_CONVERGED) if mle else True
        if ok_fit and chi_square < best_chi:
            best_chi = chi_square
            best_state = state
            best_iterations = n_iterations
            have_best = True
            for p in range(n_params):
                best[p] = theta[p]

    if have_best:
        for p in range(n_params):
            thetas[index, p] = best[p]
        chi_squares[index] = best_chi
        states[index] = best_state
        iterations[index] = best_iterations
    elif have_best_finite:
        for p in range(n_params):
            thetas[index, p] = best_finite[p]
        chi_squares[index] = best_finite_chi
        states[index] = best_finite_state
        iterations[index] = best_finite_iterations
    else:
        # Every seed diverged. Report the seed parameters with an infinite
        # chi-square; ``locs_from_fits_spline`` turns non-finite rows into NaN
        # precisions.
        for p in range(n_params):
            thetas[index, p] = np.nan
        chi_squares[index] = np.inf
        if mle:
            states[index] = FIT_STATE_NEG_CURVATURE_MLE
        else:
            states[index] = FIT_STATE_SINGULAR_HESSIAN
        iterations[index] = 0


# ----------------------------------------------------------------------
# Threaded driver with per-spot progress
#
# The kernels are ``nogil``, so real threads run them concurrently. Each worker
# claims the next spot index under a lock and bumps a shared counter; the
# caller polls that counter to report progress. This is the arrangement of
# ``gaussmle.gaussmle_async`` / ``gaussmle._worker``.
# ----------------------------------------------------------------------


def _dummy_coefficients() -> tuple:
    """Minimal stand-ins for the unused coefficient table.

    :func:`_fit_spline_spot` takes both a 2D and a 3D coefficient table because
    numba types an array by its rank and cannot form a union of the two; only
    the one matching ``kind`` is ever read."""
    dummy_2d = np.zeros((1, 1, 1, 4, 4))
    dummy_3d = np.zeros((1, 1, 1, 1, 4, 4, 4))
    return dummy_2d, dummy_3d


def _dummy_variance(ndim: int = 4) -> np.ndarray:
    """Stand-in for an absent sCMOS readout-variance map.

    Four bytes, and the same numba type as a real variance array - float32,
    C-contiguous, of the given rank - so a kernel compiles exactly one
    specialization whether or not a camera calibration is in use. Passing a
    scalar zero instead would type differently and force a second compile;
    passing a full-size array of zeros would cost a real read per pixel per
    iteration for nothing. The companion ``use_variance`` flag is a
    loop-invariant boolean, so the guard costs nothing either."""
    return np.zeros((1,) * ndim, dtype=np.float32)


def resolve_variance(
    variance: np.ndarray | None, expected_shape: tuple, ndim: int = 4
) -> tuple:
    """``(variance, use_variance)`` for a kernel argument list.

    ``None`` yields the dummy of :func:`_dummy_variance` and False. Anything
    else must match ``expected_shape`` - the variance patch is cut from the
    same ROIs as the spots, so a mismatch is a plumbing bug, not a user
    error."""
    if variance is None:
        return _dummy_variance(ndim), False
    variance = np.ascontiguousarray(variance, dtype=np.float32)
    if variance.shape != tuple(expected_shape):
        raise ValueError(
            "variance must have the same shape as spots "
            f"{tuple(expected_shape)}, got {variance.shape}."
        )
    return variance, True


def _kernel_args(
    variance: np.ndarray,
    use_variance: bool,
    kind: int,
    coefficients: np.ndarray,
    affines: np.ndarray,
    residuals: np.ndarray,
    initial_parameters: np.ndarray,
    z_seeds: np.ndarray,
    apply_seeds: bool,
    mle: bool,
    tolerance: float,
    max_iterations: int,
    thetas: np.ndarray,
    chi_squares: np.ndarray,
    states: np.ndarray,
    iterations: np.ndarray,
) -> tuple:
    """Freeze everything :func:`_fit_spline_spot` needs after ``index``.

    ``variance``/``use_variance`` lead, so they land immediately after
    ``index`` in the kernel signature and every
    ``_fit_spline_spot(spots, index, *args)`` call site stays as it was."""
    dummy_2d, dummy_3d = _dummy_coefficients()
    coeff2d = coefficients if kind == KIND_2D else dummy_2d
    coeff3d = dummy_3d if kind == KIND_2D else coefficients
    return (
        variance,
        bool(use_variance),
        kind,
        coeff2d,
        coeff3d,
        affines,
        residuals,
        initial_parameters,
        z_seeds,
        apply_seeds,
        mle,
        float(tolerance),
        int(max_iterations),
        thetas,
        chi_squares,
        states,
        iterations,
    )


def _allocate_outputs(n_spots: int, n_params: int) -> tuple:
    """Preallocated fit outputs, one row per spot."""
    thetas = np.full((n_spots, n_params), np.nan)
    chi_squares = np.full(n_spots, np.inf)
    states = np.zeros(n_spots, dtype=np.int32)
    iterations = np.zeros(n_spots, dtype=np.int32)
    return thetas, chi_squares, states, iterations


def _check_inputs(
    kind: int,
    spots: np.ndarray,
    coefficients: np.ndarray,
    affines: np.ndarray,
    residuals: np.ndarray,
    initial_parameters: np.ndarray,
) -> None:
    """Validate the array shapes the kernels rely on but cannot check."""
    if spots.ndim != 4:
        raise ValueError(
            "spots must be channel-major (n_spots, n_channels, box, box), got "
            f"shape {spots.shape}."
        )
    n_spots, n_channels, box, box_y = spots.shape
    if box != box_y:
        raise ValueError(f"spots must have a square box, got {box}x{box_y}.")
    expected_ndim = 5 if kind == KIND_2D else 7
    if coefficients.ndim != expected_ndim:
        raise ValueError(
            f"coefficients must have {expected_ndim} dimensions for this "
            f"model, got {coefficients.ndim}. Pass the output of "
            "precision._spline_coeff_reshaped."
        )
    if coefficients.shape[0] != n_channels:
        raise ValueError(
            f"coefficients have {coefficients.shape[0]} channels but spots "
            f"have {n_channels}."
        )
    if kind == KIND_2D:
        if n_channels != 1:
            raise ValueError(
                "There is no multichannel 2D spline model; got "
                f"{n_channels} channels."
            )
        n_params = 4
    elif kind == KIND_3D:
        n_params = 5
    else:
        n_params = 3 + 2 * n_channels
    if initial_parameters.shape != (n_spots, n_params):
        raise ValueError(
            f"initial_parameters must have shape {(n_spots, n_params)}, got "
            f"{initial_parameters.shape}."
        )
    if affines.shape != (n_channels, 4):
        raise ValueError(
            f"affines must have shape {(n_channels, 4)}, got {affines.shape}."
        )
    if residuals.shape[1:] != (n_channels, 2) or len(residuals) < max(
        n_spots, 1
    ):
        raise ValueError(
            "residuals must have shape (n_spots, n_channels, 2) = "
            f"{(n_spots, n_channels, 2)}, got {residuals.shape}."
        )


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
        _fit_spline_spot(spots, index, *args)


class AsyncFit(NamedTuple):
    """Handle on a spline fit running in the background.

    ``current[0]`` is the number of spots claimed so far, which the caller
    polls to report progress; the result arrays are filled in place as the
    workers run."""

    current: list
    theta: np.ndarray
    chi_squares: np.ndarray
    states: np.ndarray
    iterations: np.ndarray
    futures: list[Future]
    stop_flag: np.ndarray

    def results(self) -> tuple:
        """``(theta, chi_squares, states, iterations)``. Only meaningful once
        :meth:`finished` is True."""
        return self.theta, self.chi_squares, self.states, self.iterations

    def stop(self) -> None:
        """Ask the workers to stop claiming spots.

        Spots already in flight run to completion - a single fit is short - but
        nothing new is started, so an aborted fit stops burning CPU instead of
        quietly running to the end into arrays nobody will read."""
        self.stop_flag[0] = 1

    def finished(self) -> bool:
        """Whether every worker has exited."""
        return lib.n_futures_done(self.futures) == len(self.futures)

    def raise_errors(self) -> None:
        """Re-raise the first worker exception, if any.

        The workers write into preallocated arrays and their futures are never
        collected, so without this an exception inside a worker would be
        swallowed and the shared counter would simply stop advancing."""
        for future in self.futures:
            if future.done():
                exception = future.exception()
                if exception is not None:
                    raise exception


def n_workers() -> int:
    """Number of fitting threads, by the formula every CPU fitter uses."""
    # Python crashes when using >64 cores.
    return min(60, max(1, int(0.75 * multiprocessing.cpu_count())))


def fit_spots(
    kind: int,
    spots: np.ndarray,
    coefficients: np.ndarray,
    affines: np.ndarray,
    residuals: np.ndarray,
    initial_parameters: np.ndarray,
    z_seeds: np.ndarray,
    apply_seeds: bool,
    mle: bool = False,
    tolerance: float | None = None,
    max_iterations: int | None = None,
    progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    variance: np.ndarray | None = None,
) -> tuple:
    """Fit spots with a cubic-spline PSF model on the CPU, serially.

    Parameters
    ----------
    kind : int
        :data:`KIND_2D`, :data:`KIND_3D` or :data:`KIND_LINK_XYZ`.
    spots : np.ndarray
        Channel-major ``(n_spots, n_channels, box, box)`` photon counts,
        indexed ``[spot, channel, y, x]``. Single-channel models use
        ``n_channels == 1``.
    coefficients : np.ndarray
        ``(n_channels, niy, nix, 4, 4)`` (2D) or
        ``(n_channels, niz, niy, nix, 4, 4, 4)`` (3D), i.e. the output of
        ``precision._spline_coeff_reshaped``.
    affines : np.ndarray
        ``(n_channels, 4)`` per-channel lateral affine ``[a00, a01, a10, a11]``
        (``precision._spline_channel_affines``); the identity for a
        single-channel fit.
    residuals : np.ndarray
        ``(n_spots, n_channels, 2)`` sub-pixel ROI offsets
        (``precision._spline_crlb_residuals``); zeros for a single-channel fit.
    initial_parameters : np.ndarray
        ``(n_spots, n_params)`` seeds, from
        ``localize._initial_parameters_spline``.
    z_seeds : np.ndarray
        Axial seeds for the multi-start, in z-shift units.
    apply_seeds : bool
        Whether to run the multi-start at all; False keeps each spot's own
        initial z.
    mle : bool, optional
        Use the Poisson maximum-likelihood estimator instead of least squares.
    tolerance, max_iterations : optional
        Convergence schedule. ``None`` (the default) uses the one the GPU path
        uses for this kind of fit, see :func:`convergence_schedule`.
    variance : np.ndarray, optional
        Per-pixel sCMOS readout variance in photoelectrons squared, laid out
        exactly like ``spots``. ``None`` (the default) fits the plain Poisson
        model. See :func:`_estimator_terms`.
    progress_callback : callable, "console" or None, optional
        ``"console"`` shows a tqdm bar; a callable is invoked with the
        cumulative number of spots fitted.

    Returns
    -------
    thetas : np.ndarray
        ``(n_spots, n_params)`` fitted parameters.
    chi_squares : np.ndarray
        Chi-square at the optimum (twice the negative Poisson log-likelihood
        for ``mle``).
    states : np.ndarray
        Per-spot fit state, using Gpufit's codes (see
        :data:`FIT_STATE_CONVERGED`).
    iterations : np.ndarray
        Iterations used by the seed that won.
    """
    _check_inputs(
        kind, spots, coefficients, affines, residuals, initial_parameters
    )
    variance, use_variance = resolve_variance(variance, spots.shape, ndim=4)
    tolerance, max_iterations = resolve_schedule(
        apply_seeds, tolerance, max_iterations
    )
    n_spots = len(spots)
    thetas, chi_squares, states, iterations = _allocate_outputs(
        n_spots, initial_parameters.shape[1]
    )
    args = _kernel_args(
        variance,
        use_variance,
        kind,
        coefficients,
        affines,
        residuals,
        initial_parameters,
        z_seeds,
        apply_seeds,
        mle,
        tolerance,
        max_iterations,
        thetas,
        chi_squares,
        states,
        iterations,
    )
    use_tqdm = progress_callback == "console"
    index_range = range(n_spots)
    if use_tqdm:
        index_range = tqdm(index_range, desc="Fitting", unit="spot")
    for index in index_range:
        _fit_spline_spot(spots, index, *args)
        if callable(progress_callback):
            progress_callback(index + 1)
    return thetas, chi_squares, states, iterations


def fit_spots_async(
    kind: int,
    spots: np.ndarray,
    coefficients: np.ndarray,
    affines: np.ndarray,
    residuals: np.ndarray,
    initial_parameters: np.ndarray,
    z_seeds: np.ndarray,
    apply_seeds: bool,
    mle: bool = False,
    tolerance: float | None = None,
    max_iterations: int | None = None,
    n_threads: int | None = None,
    variance: np.ndarray | None = None,
) -> AsyncFit:
    """Fit spots with a cubic-spline PSF model on several CPU threads.

    Returns immediately with an :class:`AsyncFit` the caller polls for
    progress, aborts, or checks for worker errors. See :func:`fit_spots` for
    the arguments."""
    _check_inputs(
        kind, spots, coefficients, affines, residuals, initial_parameters
    )
    variance, use_variance = resolve_variance(variance, spots.shape, ndim=4)
    tolerance, max_iterations = resolve_schedule(
        apply_seeds, tolerance, max_iterations
    )
    n_spots = len(spots)
    thetas, chi_squares, states, iterations = _allocate_outputs(
        n_spots, initial_parameters.shape[1]
    )
    args = _kernel_args(
        variance,
        use_variance,
        kind,
        coefficients,
        affines,
        residuals,
        initial_parameters,
        z_seeds,
        apply_seeds,
        mle,
        tolerance,
        max_iterations,
        thetas,
        chi_squares,
        states,
        iterations,
    )
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
    return AsyncFit(current, thetas, chi_squares, states, iterations, fs, stop)


# ----------------------------------------------------------------------
# Reference helper (tests, debugging)
# ----------------------------------------------------------------------


def model_and_jacobian(
    kind: int,
    coefficients: np.ndarray,
    affines: np.ndarray,
    residuals_row: np.ndarray,
    theta: np.ndarray,
    box: int,
) -> tuple:
    """Model image and analytic Jacobian of one spot at ``theta``.

    Returns ``(mu, jacobian)`` with ``mu`` of shape ``(n_channels, box, box)``
    indexed ``[channel, y, x]`` and ``jacobian`` of shape
    ``(n_channels, box, box, n_params)``. This is the readable counterpart of
    what the fitting kernels accumulate; the tests use it to check the
    Jacobian against finite differences of ``mu``, which is what pins the sign
    convention of the position derivatives."""
    coefficients = np.ascontiguousarray(coefficients, dtype=np.float64)
    affines = np.ascontiguousarray(affines, dtype=np.float64)
    residuals_row = np.ascontiguousarray(residuals_row, dtype=np.float64)
    theta = np.ascontiguousarray(theta, dtype=np.float64)
    n_channels = coefficients.shape[0]
    n_params = len(theta)
    mu = np.zeros((n_channels, box, box))
    jacobian = np.zeros((n_channels, box, box, n_params))
    for ch in range(n_channels):
        a00, a01, a10, a11 = affines[ch]
        if kind == KIND_LINK_XYZ:
            x_shift, y_shift, z_shift = theta[0], theta[1], theta[2]
            amplitude = theta[3 + ch]
            offset = theta[3 + n_channels + ch]
        else:
            amplitude = theta[0]
            x_shift, y_shift = theta[1], theta[2]
            z_shift = theta[3] if kind == KIND_3D else 0.0
            offset = theta[-1]
        sx = a00 * x_shift + a01 * y_shift + residuals_row[ch, 0]
        sy = a10 * x_shift + a11 * y_shift + residuals_row[ch, 1]
        for j in range(box):
            for i in range(box):
                if kind == KIND_2D:
                    phi, gx, gy = _eval_spline_2d(
                        coefficients, ch, i - sx, j - sy
                    )
                    gz = 0.0
                else:
                    phi, gx, gy, gz = _eval_spline_3d(
                        coefficients, ch, i - sx, j - sy, -z_shift
                    )
                mu[ch, j, i] = amplitude * phi + offset
                dx = -amplitude * (a00 * gx + a10 * gy)
                dy = -amplitude * (a01 * gx + a11 * gy)
                if kind == KIND_LINK_XYZ:
                    jacobian[ch, j, i, 0] = dx
                    jacobian[ch, j, i, 1] = dy
                    jacobian[ch, j, i, 2] = -amplitude * gz
                    jacobian[ch, j, i, 3 + ch] = phi
                    jacobian[ch, j, i, 3 + n_channels + ch] = 1.0
                else:
                    jacobian[ch, j, i, 0] = phi
                    jacobian[ch, j, i, 1] = dx
                    jacobian[ch, j, i, 2] = dy
                    if kind == KIND_3D:
                        jacobian[ch, j, i, 3] = -amplitude * gz
                    jacobian[ch, j, i, -1] = 1.0
    return mu, jacobian
