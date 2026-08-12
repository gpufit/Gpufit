"""
picasso.fitting.lmfit_cuda
~~~~~~~~~~~~~~~~~~~~~~~~~~

CUDA device machinery shared by the numba fitting backends: the
Levenberg-Marquardt damping and linear solve, the least-squares and Poisson
maximum-likelihood estimators, and the host-side launch bookkeeping.

This is the device twin of the driver in :mod:`picasso.fitting.splinefit`,
which is itself a numba port of Gpufit's ``cuda_kernels.cu`` and
``Cpufit/lm_fit_cpp.cpp``. Every constant that defines *where a fit stops* -
the damping factors, the fit states, the Poisson model floor, the convergence
schedule - is imported from :mod:`picasso.fitting.splinefit` rather than
redefined here, so the two devices cannot drift apart. A fit is a sequence of
accept/reject decisions on a chi-square, so a schedule that differs by a factor
of ten is not a rounding difference, it is a different answer.

Three things about the CUDA target differ from the CPU kernels and are easy to
get wrong when transcribing:

``np.*`` scalar functions are not available in device code
    ``np.isfinite`` and ``np.floor`` are rejected outright by the CUDA target
    (verified against numba-cuda 0.30.2), so every one becomes ``math.*``.
    ``np.log`` happens to compile, but is spelled ``math.log`` here for
    consistency. The unit tests compile every device function with
    ``cuda.compile_ptx``, which is what catches this - the CUDA *simulator* runs
    plain Python, where all the NumPy forms work fine, so it cannot.

``fastmath`` is deliberately off
    ``picasso.fitting.splinefit`` uses a restricted LLVM flag set that
    excludes ``nnan`` and ``ninf``, because those let the compiler assume no
    NaN can occur and fold the divergence guards to a constant True. That
    specific hazard does not exist here - ``numba.cuda.jit`` takes a boolean ``fastmath`` that maps to
    NVVM's ``ftz``/``prec_div``/``prec_sqrt``/``fma``, not to LLVM's
    ``nnan``/``ninf``. It is still left off, because flushing denormals and
    approximating division would change results against the CPU backend for no
    gain in kernels that are bound on coefficient-table reads. Please do not
    "optimize" it on without benchmarking it first.

One thread per fit
    There is no shared memory, no cross-thread reduction and no atomic in any of
    these kernels, which is why results are bitwise reproducible across runs.
    It also means **no ``cuda.syncthreads()`` may ever be added**: the driver
    loop breaks out at different iterations in different lanes, so a barrier
    would be unmatched and hang the warp.

References
----------
The Levenberg-Marquardt driver, the damping rule, the Gauss-Jordan solve and
both estimators in this module are a port of Gpufit's ``cuda_kernels.cu``,
``estimators/{lse,mle}.cuh`` and ``Cpufit/lm_fit_cpp.cpp``:

Przybylski, A., Thiel, B., Keller-Findeisen, J., Stock, B. & Bates, M.
"Gpufit: An open-source toolkit for GPU-accelerated curve fitting."
Scientific Reports 7, 15722 (2017).
https://doi.org/10.1038/s41598-017-15313-9
Licence (MIT): ``LICENSES/Gpufit-LICENSE.txt``.

:authors: Rafal Kowalewski
:copyright: Copyright (c) 2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import math
import os
import warnings

import numpy as np
from numba import cuda, float64

from picasso.fitting.splinefit import (  # noqa: F401  (re-exported below)
    FIT_STATE_CONVERGED,
    FIT_STATE_MAX_ITERATION,
    FIT_STATE_NEG_CURVATURE_MLE,
    FIT_STATE_SINGULAR_HESSIAN,
    MU_FLOOR,
    _LAMBDA_DOWN,
    _LAMBDA_INITIAL,
    _LAMBDA_UP,
    convergence_schedule,
    resolve_schedule,
)

try:
    CUDA_AVAILABLE = bool(cuda.is_available())
except Exception:  # pragma: no cover - depends on the driver install
    CUDA_AVAILABLE = False

# Module-level constants become compile-time constants in device code; ``np.inf``
# and ``np.nan`` are not usable as attribute lookups there.
_INF = float(np.inf)
_NAN = float(np.nan)

# Threads per block. 128 matches the CRLB kernels; the wide photon-decoupled
# models drop to 64 to halve the per-block local working set. Note this is a
# working-set knob, not an occupancy fix: those kernels are register-bound at the
# hardware ceiling, and occupancy goes as registers x threads, so a smaller block
# yields the same threads per SM.
CUDA_THREADS = 128
CUDA_THREADS_WIDE = 64

# Device working set per launch, and a ceiling on the rows in one launch.
#
# The row ceiling is NOT the CRLB path's ``_SPLINE_CRLB_CUDA_MAX_ROWS``. A CRLB
# row is a single pass; a multi-start maximum-likelihood fit is up to
# ``n_seeds * max_iterations`` = 15 * 100 passes over ``box**2 * n_channels``
# pixels, three to four orders of magnitude more work per row. A display-attached
# Windows GPU kills any kernel running past the ~2 s TDR limit, and the symptom
# is a driver reset rather than a wrong number - so the budget below is scaled by
# the per-row work rather than being a flat constant. ``PICASSO_FIT_CUDA_MAX_ROWS``
# overrides the reference value for tuning on a specific card.
CUDA_CHUNK_BYTES = 256 << 20
_MAX_ROWS_REFERENCE = 4_000_000
# Per-row work, in passes, that ``_MAX_ROWS_REFERENCE`` is calibrated for.
_REFERENCE_WORK = 20


def max_rows_reference() -> int:
    """Row ceiling for a launch of the reference per-row cost."""
    override = os.environ.get("PICASSO_FIT_CUDA_MAX_ROWS")
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            warnings.warn(
                f"Ignoring PICASSO_FIT_CUDA_MAX_ROWS={override!r}: "
                "not an integer.",
                RuntimeWarning,
                stacklevel=2,
            )
    return _MAX_ROWS_REFERENCE


def chunk_rows(bytes_per_row: int, work_per_row: int) -> int:
    """Fits per kernel launch, bounded by device memory *and* by runtime.

    ``work_per_row`` is the number of model evaluations one fit can cost, i.e.
    ``n_seeds * max_iterations``. Scaling the ceiling by it keeps a long
    multi-start launch roughly as short in wall-clock as a cheap single-start
    one, which is what keeps a display-attached GPU under its watchdog timeout.
    """
    by_memory = max(1024, CUDA_CHUNK_BYTES // max(bytes_per_row, 1))
    by_runtime = max(
        1024,
        max_rows_reference() * _REFERENCE_WORK // max(int(work_per_row), 1),
    )
    return int(min(by_memory, by_runtime))


_gpu_fallback_warned = False


def warn_gpu_fallback(exc: Exception) -> None:
    """Report the first time a GPU fit falls back to the CPU.

    Having no CUDA device at all is not an error and is never reported - the CPU
    kernels are simply used. This is for the other case: a device is present but
    the attempt failed (out of memory, driver error), where the results are still
    correct but a silent fallback would hide a broken device path indefinitely.
    Warned once per process so a per-chunk failure cannot spam the log."""
    global _gpu_fallback_warned
    if _gpu_fallback_warned:
        return
    _gpu_fallback_warned = True
    warnings.warn(
        f"Fitting on the GPU failed ({exc!r}); falling back to the CPU. "
        "Results are unaffected, only slower.",
        RuntimeWarning,
        stacklevel=3,
    )


def require_cuda() -> None:
    """Raise unless a CUDA device is usable."""
    if not CUDA_AVAILABLE:
        raise RuntimeError(
            "GPU fitting was requested but no CUDA-capable GPU is available."
        )


# ----------------------------------------------------------------------
# Estimators
# ----------------------------------------------------------------------


@cuda.jit(device=True, inline=True)
def _floored_chi_square(data):
    """Likelihood penalty of a pixel below :data:`MU_FLOOR`.

    Gpufit's Poisson term evaluated at the floor: large but finite, so it
    disfavours parameters that need a negative model without making the
    chi-square infinite, which would leave the multi-start unable to rank its
    seeds. Port of ``splinefit._floored_chi_square``."""
    if data > 0.0:
        return 2.0 * ((MU_FLOOR - data) - data * math.log(MU_FLOOR / data))
    return 2.0 * MU_FLOOR


@cuda.jit(device=True, inline=True)
def _poisson_terms(value, data):
    """The Poisson terms for a strictly positive model value."""
    if data > 0.0:
        return (
            2.0 * ((value - data) - data * math.log(value / data)),
            data / (value * value),
            -(1.0 - data / value),
        )
    # An empty (or clipped) pixel: Gpufit's data == 0 term.
    return 2.0 * value, 0.0, -1.0


@cuda.jit(device=True, inline=True)
def _estimator_terms(mle, value, data, var):
    """Per-pixel ``(chi_square, weight, factor, ok)``, flooring a low model.

    ``weight`` multiplies the Hessian outer product and ``factor`` the gradient,
    so a caller accumulates ``grad_k += d_k * factor`` and
    ``hess_kl += weight * d_k * d_l``. This is the one place the least-squares
    and Poisson branches are written down for the device kernels.

    ``var`` is the pixel's sCMOS readout variance in photoelectrons squared,
    zero when no camera calibration is in use. See
    ``picasso.fitting.splinefit._estimator_terms``, the CPU twin, for what the
    shift by ``var`` means and why the least-squares branch is untouched by it.

    ``ok`` is False only for a non-finite model value under the maximum
    likelihood estimator, where the caller must abandon the fit. A merely
    *non-positive* value is floored instead of rejected - see :data:`MU_FLOOR`.

    **For the cubic-spline models only.** The floor is justified by the spline
    *basis*: a cubic rings slightly negative in the tails of a peaked profile,
    so a bright, low-background spot drives the model below zero in the corners
    of the box through no fault of the parameters. Do not use it for a model
    that cannot ring - see :func:`_estimator_terms_strict`.
    """
    if mle:
        shifted_value = value + var
        shifted_data = data + var
        if not math.isfinite(shifted_value):
            return _INF, 0.0, 0.0, False
        if shifted_value < MU_FLOOR:
            # Charged to the likelihood so seeds stay comparable, but kept out
            # of the gradient and Hessian: its 1/mu weight would be enormous
            # and, through the monotone scaling vector, would damp every
            # parameter to a standstill for the rest of the fit.
            return _floored_chi_square(shifted_data), 0.0, 0.0, True
        chi, weight, factor = _poisson_terms(shifted_value, shifted_data)
        return chi, weight, factor, True
    # Least squares: the shift cancels, so it is never formed.
    deviation = value - data
    return deviation * deviation, 1.0, -deviation, True


@cuda.jit(device=True, inline=True)
def _estimator_terms_strict(mle, value, data, var):
    """:func:`_estimator_terms`, but a non-positive model value aborts the fit.

    For models that cannot ring negative - the Gaussians, whose value only
    drops below zero if the *background* parameter does. There the floor is not
    just unnecessary but harmful: it zeroes that pixel's gradient and Hessian
    contribution, so nothing pushes the background back up, the chi-square stops
    moving and the relative convergence test then reports a badly wrong fit as
    converged. Gpufit aborts such a fit with ``NEG_CURVATURE_MLE`` and so does
    this; the caller reports the state and the parameters stay at the last
    accepted iterate.

    Note that with a camera calibration loaded the test is on the *shifted*
    mean ``value + var``, which is the quantity the approximation makes
    Poisson. A slightly negative background can therefore survive on a noisy
    pixel where it previously aborted the fit; that is correct under the model
    and only happens when a variance map is in use.
    """
    if mle:
        shifted_value = value + var
        if not (math.isfinite(shifted_value) and shifted_value >= MU_FLOOR):
            return _INF, 0.0, 0.0, False
        chi, weight, factor = _poisson_terms(shifted_value, data + var)
        return chi, weight, factor, True
    # Least squares: the shift cancels, so it is never formed.
    deviation = value - data
    return deviation * deviation, 1.0, -deviation, True


# ----------------------------------------------------------------------
# Linear solve
# ----------------------------------------------------------------------


@cuda.jit(device=True)
def _solve_gj_device(a, b, n, ipiv):
    """Gauss-Jordan solve of ``a[:n, :n] x = b[:n]``, in place, full pivoting.

    Port of ``splinefit._solve_gj``, which is itself Gpufit's
    ``LMFitCPP::solve_equation_system_gj``. Returns False on a zero or NaN pivot
    instead of raising - there is nothing to raise to in device code, and the
    caller reports the fit as :data:`FIT_STATE_SINGULAR_HESSIAN`.

    The CPU version also records the pivot permutation in ``indxc``/``indxr``.
    Those are written and never read: Gpufit drops the Numerical-Recipes column
    back-permutation because its elimination only ever interchanges rows. They
    are omitted here rather than transcribed, which saves two per-thread arrays
    in a kernel that is register-bound.
    """
    for i in range(n):
        ipiv[i] = 0
    for _ in range(n):
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


@cuda.jit(device=True)
def _lm_solve_step_device(hess, grad, scaling, lam, damped, delta, n, ipiv):
    """One Levenberg-Marquardt step: damp the Hessian and solve for ``delta``.

    ``scaling`` is Gpufit's adaptive step-width vector: it holds the largest
    Hessian diagonal each parameter has ever shown and is therefore **monotone
    across iterations** (``cuda_modify_step_widths``). Resetting it every
    iteration is the classic way to get this wrong - the damping then no longer
    reflects the curvature the fit has already seen, and badly-seeded fits stop
    converging.

    ``hess`` and ``grad`` are the *undamped* matrices of the last accepted
    iteration and are left untouched; the damped system is rebuilt into
    ``damped``/``delta``, which the solve destroys. Returns False if that system
    is singular."""
    for p in range(n):
        d = hess[p, p]
        if d > scaling[p]:
            scaling[p] = d
    for p in range(n):
        for q in range(n):
            damped[p, q] = hess[p, q]
        damped[p, p] += scaling[p] * lam
        delta[p] = grad[p]
    return _solve_gj_device(damped, delta, n, ipiv)


# ----------------------------------------------------------------------
# Levenberg-Marquardt driver
# ----------------------------------------------------------------------


def make_lm_driver(accumulate, n_params: int, z_col: int, seedable: bool):
    """Build the per-fit LM driver device function for one model.

    ``accumulate(spots, index, variance, use_variance, coeff, aff, res, theta,
    mle, hess, grad)`` must return ``(chi_square, ok)`` and fill the full
    symmetric ``hess`` and ``grad``, exactly like the
    ``picasso.fitting.splinefit._accumulate_*`` family.

    ``variance`` is laid out exactly like ``spots`` and holds the per-pixel
    sCMOS readout variance in photoelectrons squared; ``use_variance`` is False
    when no camera calibration is in use, and ``variance`` is then a
    ``(1, 1, 1, 1)`` dummy that exists only to keep the kernel's argument types
    stable across both states.

    The driver is generated rather than shared because ``cuda.local.array``
    needs a compile-time shape, so ``n_params`` has to be a closure constant.
    That also lets each model get scratch sized to itself: compiling one driver
    at the widest parameter count would give the common single-channel fit the
    register and local-memory profile of the six-channel one.

    ``z_col`` is the parameter index the axial multi-start seeds, and
    ``seedable`` is False for models with no axial coordinate - the 2D model's
    parameter 3 is the *background*, so seeding it would silently corrupt the
    fit rather than move it in z.

    This mirrors ``splinefit._fit_spline_spot`` line for line, including the
    seed ranking, so that the two devices agree on which seed wins.
    """
    n = n_params
    z_column = z_col
    can_seed = seedable

    @cuda.jit(device=True)
    def driver(
        spots,
        index,
        variance,
        use_variance,
        coeff,
        aff,
        res,
        init,
        z_seeds,
        apply_seeds,
        mle,
        tolerance,
        max_iterations,
        thetas,
        chi_squares,
        states,
        iterations,
    ):
        theta = cuda.local.array(n, float64)
        theta_previous = cuda.local.array(n, float64)
        best = cuda.local.array(n, float64)
        best_finite = cuda.local.array(n, float64)
        # ``grad`` doubles as the solve's right-hand side and therefore as the
        # step ``delta``: the solve writes it, the parameter update reads it,
        # and the next accumulation overwrites it. Likewise ``hess`` doubles as
        # the damped matrix the solve destroys - at solve time it holds the
        # previous accumulation, which has either been copied into ``hess_ok``
        # or is not needed. This saves a whole n x n float64 array per thread
        # without changing a single result.
        grad = cuda.local.array(n, float64)
        grad_ok = cuda.local.array(n, float64)
        scaling = cuda.local.array(n, float64)
        hess = cuda.local.array((n, n), float64)
        hess_ok = cuda.local.array((n, n), float64)
        ipiv = cuda.local.array(n, np.int32)

        # ``cuda.local.array`` is uninitialized where the CPU kernel used
        # ``np.zeros``. The accumulators provably overwrite ``hess``/``grad``
        # and nothing reads ``hess_ok``/``grad_ok`` before they are filled, but
        # zeroing all four is free next to the fit and removes a whole class of
        # heisenbug.
        for p in range(n):
            grad[p] = 0.0
            grad_ok[p] = 0.0
            for q in range(n):
                hess[p, q] = 0.0
                hess_ok[p, q] = 0.0

        seeded = apply_seeds and can_seed
        best_chi = _INF
        best_state = FIT_STATE_MAX_ITERATION
        best_iterations = 0
        have_best = False
        best_finite_chi = _INF
        best_finite_state = FIT_STATE_MAX_ITERATION
        best_finite_iterations = 0
        have_best_finite = False

        n_seeds = z_seeds.shape[0] if seeded else 1
        for seed in range(n_seeds):
            for p in range(n):
                theta[p] = init[index, p]
                scaling[p] = 0.0
            if seeded:
                theta[z_column] = z_seeds[seed]

            state = FIT_STATE_CONVERGED
            lam = _LAMBDA_INITIAL
            n_iterations = 0

            chi_square, ok = accumulate(
                spots,
                index,
                variance,
                use_variance,
                coeff,
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
                for p in range(n):
                    grad_ok[p] = grad[p]
                    for q in range(n):
                        hess_ok[p, q] = hess[p, q]
            previous_chi_square = chi_square

            for iteration in range(max_iterations):
                if state != FIT_STATE_CONVERGED:
                    break
                if not _lm_solve_step_device(
                    hess_ok, grad_ok, scaling, lam, hess, grad, n, ipiv
                ):
                    # The step is garbage, so it is not applied; the parameters
                    # of the last accepted iteration stand.
                    state = FIT_STATE_SINGULAR_HESSIAN
                    break
                for p in range(n):
                    theta_previous[p] = theta[p]
                    theta[p] += grad[p]
                new_chi_square, ok = accumulate(
                    spots,
                    index,
                    variance,
                    use_variance,
                    coeff,
                    aff,
                    res,
                    theta,
                    mle,
                    hess,
                    grad,
                )
                n_iterations = iteration + 1
                if not ok:
                    # A property of the trial *step*, not of the fit: undo it,
                    # damp harder and try again, exactly as for a step that
                    # merely worsened chi-square. Aborting here would return the
                    # seed unchanged. Kept identical to the CPU drivers in
                    # ``picasso.fitting.gaussfit`` and ``.splinefit``.
                    chi_square = previous_chi_square
                    for p in range(n):
                        theta[p] = theta_previous[p]
                    lam *= _LAMBDA_UP
                    if iteration == max_iterations - 1:
                        state = FIT_STATE_NEG_CURVATURE_MLE
                    continue
                chi_square = new_chi_square
                if (
                    chi_square < previous_chi_square
                    or previous_chi_square == 0.0
                ):
                    # Only an improving iteration refreshes the curvature the
                    # next step is damped from (Gpufit skips the
                    # gradient/Hessian kernels entirely on a failed iteration).
                    for p in range(n):
                        grad_ok[p] = grad[p]
                        for q in range(n):
                            hess_ok[p, q] = hess[p, q]
                limit = tolerance * abs(chi_square)
                if tolerance > limit:
                    limit = tolerance
                converged = abs(chi_square - previous_chi_square) < limit
                if not converged and iteration == max_iterations - 1:
                    state = FIT_STATE_MAX_ITERATION
                if chi_square < previous_chi_square:
                    lam *= _LAMBDA_DOWN
                    previous_chi_square = chi_square
                else:
                    lam *= _LAMBDA_UP
                    chi_square = previous_chi_square
                    for p in range(n):
                        theta[p] = theta_previous[p]
                if converged:
                    break

            finite = math.isfinite(chi_square)
            if finite:
                for p in range(n):
                    if not math.isfinite(theta[p]):
                        finite = False
                        break
            if not finite:
                continue
            # Ranking, identical to the CPU multi-start: prefer converged fits,
            # and among equals the lowest chi-square. For least squares every
            # finite fit counts as converged.
            if chi_square < best_finite_chi:
                best_finite_chi = chi_square
                best_finite_state = state
                best_finite_iterations = n_iterations
                have_best_finite = True
                for p in range(n):
                    best_finite[p] = theta[p]
            ok_fit = (state == FIT_STATE_CONVERGED) if mle else True
            if ok_fit and chi_square < best_chi:
                best_chi = chi_square
                best_state = state
                best_iterations = n_iterations
                have_best = True
                for p in range(n):
                    best[p] = theta[p]

        if have_best:
            for p in range(n):
                thetas[index, p] = best[p]
            chi_squares[index] = best_chi
            states[index] = best_state
            iterations[index] = best_iterations
        elif have_best_finite:
            for p in range(n):
                thetas[index, p] = best_finite[p]
            chi_squares[index] = best_finite_chi
            states[index] = best_finite_state
            iterations[index] = best_finite_iterations
        else:
            # Every seed diverged. ``locs_from_fits_spline`` turns non-finite
            # rows into NaN precisions.
            for p in range(n):
                thetas[index, p] = _NAN
            chi_squares[index] = _INF
            if mle:
                states[index] = FIT_STATE_NEG_CURVATURE_MLE
            else:
                states[index] = FIT_STATE_SINGULAR_HESSIAN
            iterations[index] = 0

    return driver


def make_fit_kernel(driver, cache: bool = False):
    """Wrap a driver from :func:`make_lm_driver` in a one-thread-per-fit kernel.

    ``cache`` is False by default and must stay False for kernels produced by a
    parameterized factory. Numba's on-disk cache keys on qualified name, code
    location and signature; two kernels generated from the same factory at
    different parameter counts agree on all three and differ only by a closure
    variable, which is exactly the case where the cache either refuses to store
    or returns the wrong artifact. Models whose driver is built once at a fixed
    size may pass True.
    """

    @cuda.jit(cache=cache)
    def kernel(
        spots,
        variance,
        use_variance,
        coeff,
        aff,
        res,
        init,
        z_seeds,
        apply_seeds,
        mle,
        tolerance,
        max_iterations,
        thetas,
        chi_squares,
        states,
        iterations,
    ):
        index = cuda.grid(1)
        if index >= spots.shape[0]:
            return
        driver(
            spots,
            index,
            variance,
            use_variance,
            coeff,
            aff,
            res,
            init,
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

    return kernel
