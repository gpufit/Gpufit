"""
picasso.fitting.gaussfit_cuda
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GPU Gaussian PSF fitting: the GPU twin of :mod:`picasso.fitting.gaussfit`, over
the same Levenberg-Marquardt driver as :mod:`picasso.fitting.splinefit_cuda`.

Every model identifier, parameter count and schedule constant is imported from
the CPU module, so the two devices cannot disagree about what a model is; only
the device code lives here. The models are transcribed from Gpufit's
``models/gauss_2d.cuh``, ``gauss_2d_elliptic.cuh`` and ``gauss_2d_rotated.cuh``.

===================  ==========  ==============================================
model                parameters  layout
===================  ==========  ==============================================
:data:`SPHERICAL`             5  ``[peak, x, y, s, bg]``
:data:`ELLIPTIC`              6  ``[peak, x, y, sx, sy, bg]``
:data:`ROTATED`               7  ``[peak, x, y, sx, sy, bg, angle]``
===================  ==========  ==============================================

The amplitude is the Gaussian's *peak height*; ``picasso.localize`` converts it
to photons afterwards. There is no axial multi-start here - these models have
no axial coordinate - so each spot is fitted once from its initial parameters.

References
----------
The fitting algorithm and all three Gaussian models are a port of Gpufit
(``models/gauss_2d*.cuh``, ``cuda_kernels.cu``):

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
from typing import Callable, Literal

import numpy as np
from numba import cuda
from tqdm import tqdm

from picasso.fitting import lmfit_cuda
from picasso.fitting.splinefit import _allocate_outputs, resolve_variance

# Everything that defines *what model this is* and *where a fit stops* comes
# from the CPU twin, so the two devices cannot drift apart - the same
# arrangement as ``lmfit_cuda`` importing its constants from ``splinefit``.
from picasso.fitting.gaussfit import (  # noqa: F401  (re-exported)
    ELLIPTIC,
    MAX_ITERATIONS,
    ROTATED,
    SPHERICAL,
    TOLERANCE,
    _N_PARAMS,
    n_parameters,
)
from picasso.fitting.lmfit_cuda import (
    CUDA_THREADS,
    _INF,
    _estimator_terms_strict,
    make_fit_kernel,
    make_lm_driver,
)


def _make_accumulate_spherical(ftype):
    """Isotropic Gaussian, ``[photons, x, y, s, bg]`` (Gpufit's ``GAUSS_2D``)."""
    half = ftype(0.5)

    @cuda.jit(device=True)
    def accumulate(
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
    ):
        box = spots.shape[2]
        amp = theta[0]
        cx = theta[1]
        cy = theta[2]
        sigma = theta[3]
        offset = theta[4]
        if not (
            math.isfinite(amp)
            and math.isfinite(cx)
            and math.isfinite(cy)
            and math.isfinite(sigma)
            and math.isfinite(offset)
            # A zero width divides by zero in every derivative.
            and abs(sigma) > 0.0
        ):
            return _INF, False
        inv_s2 = ftype(1.0 / (sigma * sigma))
        inv_s3 = ftype(1.0 / (sigma * sigma * sigma))
        chi_square = 0.0
        g0 = g1 = g2 = g3 = g4 = 0.0
        h00 = h01 = h02 = h03 = h04 = 0.0
        h11 = h12 = h13 = h14 = 0.0
        h22 = h23 = h24 = 0.0
        h33 = h34 = 0.0
        h44 = 0.0
        for j in range(box):
            dy = ftype(j - cy)
            for i in range(box):
                dx = ftype(i - cx)
                ex = math.exp(-half * (dx * dx + dy * dy) * inv_s2)
                value = amp * ex + offset
                data = spots[index, 0, j, i]
                d0 = ex
                d1 = amp * ex * dx * inv_s2
                d2 = amp * ex * dy * inv_s2
                d3 = amp * ex * (dx * dx + dy * dy) * inv_s3
                # d4 == 1 (offset).
                var = 0.0
                if use_variance:
                    var = variance[index, 0, j, i]
                contrib, weight, factor, ok = _estimator_terms_strict(
                    mle, value, data, var
                )
                if not ok:
                    return _INF, False
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

    return accumulate


def _make_accumulate_elliptic(ftype):
    """Elliptic Gaussian, ``[photons, x, y, sx, sy, bg]``."""
    half = ftype(0.5)

    @cuda.jit(device=True)
    def accumulate(
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
    ):
        box = spots.shape[2]
        amp = theta[0]
        cx = theta[1]
        cy = theta[2]
        sx = theta[3]
        sy = theta[4]
        offset = theta[5]
        if not (
            math.isfinite(amp)
            and math.isfinite(cx)
            and math.isfinite(cy)
            and math.isfinite(sx)
            and math.isfinite(sy)
            and math.isfinite(offset)
            and abs(sx) > 0.0
            and abs(sy) > 0.0
        ):
            return _INF, False
        inv_sx2 = ftype(1.0 / (sx * sx))
        inv_sy2 = ftype(1.0 / (sy * sy))
        inv_sx3 = ftype(1.0 / (sx * sx * sx))
        inv_sy3 = ftype(1.0 / (sy * sy * sy))
        chi_square = 0.0
        g0 = g1 = g2 = g3 = g4 = g5 = 0.0
        h00 = h01 = h02 = h03 = h04 = h05 = 0.0
        h11 = h12 = h13 = h14 = h15 = 0.0
        h22 = h23 = h24 = h25 = 0.0
        h33 = h34 = h35 = 0.0
        h44 = h45 = 0.0
        h55 = 0.0
        for j in range(box):
            dy = ftype(j - cy)
            for i in range(box):
                dx = ftype(i - cx)
                ex = math.exp(-half * (dx * dx * inv_sx2 + dy * dy * inv_sy2))
                value = amp * ex + offset
                data = spots[index, 0, j, i]
                d0 = ex
                d1 = amp * ex * dx * inv_sx2
                d2 = amp * ex * dy * inv_sy2
                d3 = amp * ex * dx * dx * inv_sx3
                d4 = amp * ex * dy * dy * inv_sy3
                # d5 == 1 (offset).
                var = 0.0
                if use_variance:
                    var = variance[index, 0, j, i]
                contrib, weight, factor, ok = _estimator_terms_strict(
                    mle, value, data, var
                )
                if not ok:
                    return _INF, False
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

    return accumulate


def _make_accumulate_rotated(ftype):
    """Rotated elliptic Gaussian, ``[photons, x, y, sx, sy, bg, angle]``.

    The angle derivative vanishes identically when ``sx == sy``, which makes the
    first Hessian singular; ``localize._initial_parameters_gauss`` breaks that
    symmetry in its seed on purpose, and this model relies on it."""
    half = ftype(0.5)

    @cuda.jit(device=True)
    def accumulate(
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
    ):
        box = spots.shape[2]
        amp = theta[0]
        cx = theta[1]
        cy = theta[2]
        sx = theta[3]
        sy = theta[4]
        offset = theta[5]
        angle = theta[6]
        if not (
            math.isfinite(amp)
            and math.isfinite(cx)
            and math.isfinite(cy)
            and math.isfinite(sx)
            and math.isfinite(sy)
            and math.isfinite(offset)
            and math.isfinite(angle)
            and abs(sx) > 0.0
            and abs(sy) > 0.0
        ):
            return _INF, False
        cos_a = ftype(math.cos(angle))
        sin_a = ftype(math.sin(angle))
        inv_sx2 = ftype(1.0 / (sx * sx))
        inv_sy2 = ftype(1.0 / (sy * sy))
        inv_sx3 = ftype(1.0 / (sx * sx * sx))
        inv_sy3 = ftype(1.0 / (sy * sy * sy))
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
            dy = ftype(j - cy)
            for i in range(box):
                dx = ftype(i - cx)
                arga = dx * cos_a - dy * sin_a
                argb = dx * sin_a + dy * cos_a
                ex = math.exp(
                    -half * (arga * arga * inv_sx2 + argb * argb * inv_sy2)
                )
                value = amp * ex + offset
                data = spots[index, 0, j, i]
                d0 = ex
                d1 = (
                    amp * cos_a * arga * inv_sx2 + amp * sin_a * argb * inv_sy2
                ) * ex
                d2 = (
                    -amp * sin_a * arga * inv_sx2
                    + amp * cos_a * argb * inv_sy2
                ) * ex
                d3 = amp * arga * arga * inv_sx3 * ex
                d4 = amp * argb * argb * inv_sy3 * ex
                # d5 == 1 (offset).
                d6 = amp * arga * argb * (inv_sx2 - inv_sy2) * ex
                var = 0.0
                if use_variance:
                    var = variance[index, 0, j, i]
                contrib, weight, factor, ok = _estimator_terms_strict(
                    mle, value, data, var
                )
                if not ok:
                    return _INF, False
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

    return accumulate


_ACCUMULATORS = {
    SPHERICAL: _make_accumulate_spherical,
    ELLIPTIC: _make_accumulate_elliptic,
    ROTATED: _make_accumulate_rotated,
}

_KERNEL_CACHE: dict = {}


def _get_kernel(model: int, single_precision: bool):
    """Memoized fit kernel for one model and precision."""
    key = (int(model), bool(single_precision))
    kernel = _KERNEL_CACHE.get(key)
    if kernel is None:
        if model not in _ACCUMULATORS:
            raise ValueError(f"Unknown Gaussian model {model}.")
        ftype = np.float32 if single_precision else np.float64
        accumulate = _ACCUMULATORS[model](ftype)
        # No axial coordinate, so no multi-start: ``z_col`` is unused.
        driver = make_lm_driver(
            accumulate, _N_PARAMS[model], 0, seedable=False
        )
        kernel = make_fit_kernel(driver)
        _KERNEL_CACHE[key] = kernel
    return kernel


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
    abort_callback: Callable[[], bool] | None = None,
    single_precision: bool = True,
    variance: np.ndarray | None = None,
) -> tuple:
    """Fit spots with a 2D Gaussian model on the GPU.

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
    progress_callback, abort_callback, single_precision
        As :func:`picasso.fitting.splinefit_cuda.fit_spots`.
    variance : np.ndarray, optional
        ``(n_spots, box, box)`` per-pixel sCMOS readout variance in
        photoelectrons squared, laid out exactly like ``spots``. ``None``
        (the default) fits the plain Poisson model.

    Returns
    -------
    thetas, chi_squares, states, iterations
        Using Gpufit's state codes (see ``splinefit.FIT_STATE_CONVERGED``).
    """
    lmfit_cuda.require_cuda()
    spots = np.asarray(spots)
    if spots.ndim != 3 or spots.shape[1] != spots.shape[2]:
        raise ValueError(
            "spots must have shape (n_spots, box, box), got " f"{spots.shape}."
        )
    n_params = _N_PARAMS[model]
    if initial_parameters.shape != (len(spots), n_params):
        raise ValueError(
            "initial_parameters must have shape "
            f"{(len(spots), n_params)}, got {initial_parameters.shape}."
        )
    if tolerance is None:
        tolerance = TOLERANCE
    if max_iterations is None:
        max_iterations = MAX_ITERATIONS

    n_spots, box = spots.shape[0], spots.shape[1]
    thetas, chi_squares, states, iterations = _allocate_outputs(
        n_spots, n_params
    )
    if n_spots == 0:
        return thetas, chi_squares, states, iterations

    # The shared driver and the accumulators index spots channel-major, as the
    # spline models do; a Gaussian fit is the single-channel case. This is a
    # reshape of a C-contiguous array, so it costs nothing.
    spots = np.ascontiguousarray(spots, dtype=np.float32).reshape(
        n_spots, 1, box, box
    )
    # The variance patch is cut from the same ROIs and rides along in exactly
    # the same layout, so the same free reshape applies.
    variance, use_variance = resolve_variance(
        variance, (n_spots, box, box), ndim=4
    )
    if use_variance:
        variance = variance.reshape(n_spots, 1, box, box)
    initial_parameters = np.ascontiguousarray(
        initial_parameters, dtype=np.float64
    )

    kernel = _get_kernel(model, single_precision)
    # The driver takes the spline models' coefficient, affine and residual
    # arrays; a Gaussian has no channel geometry, so these are unused
    # placeholders of the right rank.
    d_unused_coeff = cuda.to_device(np.zeros((1, 1, 1, 4, 4)))
    d_unused_aff = cuda.to_device(np.zeros((1, 4)))
    d_seeds = cuda.to_device(np.zeros(1))
    # Without a noise model the variance is a four-byte dummy, uploaded once
    # rather than per chunk.
    d_dummy_variance = None if use_variance else cuda.to_device(variance)

    bytes_per_row = 4 * box * box + 8 * (2 * n_params + 2) + 16
    if use_variance:
        bytes_per_row += 4 * box * box
    chunk = min(n_spots, lmfit_cuda.chunk_rows(bytes_per_row, max_iterations))

    use_tqdm = progress_callback == "console"
    do_callback = callable(progress_callback)
    pbar = (
        tqdm(total=n_spots, desc="Fitting", unit="spot") if use_tqdm else None
    )
    try:
        for start in range(0, n_spots, chunk):
            if abort_callback is not None and abort_callback():
                break
            stop = min(start + chunk, n_spots)
            n = stop - start
            d_spots = cuda.to_device(spots[start:stop])
            d_variance = (
                cuda.to_device(variance[start:stop])
                if use_variance
                else d_dummy_variance
            )
            d_init = cuda.to_device(initial_parameters[start:stop])
            d_res = cuda.to_device(np.zeros((n, 1, 2)))
            d_thetas = cuda.device_array((n, n_params), dtype=np.float64)
            d_chi = cuda.device_array(n, dtype=np.float64)
            d_states = cuda.device_array(n, dtype=np.int32)
            d_iterations = cuda.device_array(n, dtype=np.int32)
            blocks = (n + CUDA_THREADS - 1) // CUDA_THREADS
            kernel[blocks, CUDA_THREADS](
                d_spots,
                d_variance,
                use_variance,
                d_unused_coeff,
                d_unused_aff,
                d_res,
                d_init,
                d_seeds,
                False,
                bool(mle),
                float(tolerance),
                int(max_iterations),
                d_thetas,
                d_chi,
                d_states,
                d_iterations,
            )
            thetas[start:stop] = d_thetas.copy_to_host()
            chi_squares[start:stop] = d_chi.copy_to_host()
            states[start:stop] = d_states.copy_to_host()
            iterations[start:stop] = d_iterations.copy_to_host()
            if use_tqdm:
                pbar.update(n)
            elif do_callback:
                progress_callback(stop)
    finally:
        if use_tqdm:
            pbar.close()
    return thetas, chi_squares, states, iterations
