"""
picasso.fitting.splinefit_cuda
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GPU cubic-spline PSF fitting: the CUDA twin of
:mod:`picasso.fitting.splinefit`.

:func:`fit_spots` takes and returns exactly what ``splinefit.fit_spots`` does,
so the two are interchangeable backends and ``picasso.localize`` can dispatch
between them on a single flag. The models, the estimators, the damping rule and
the seed ranking all come from the shared code in
:mod:`picasso.fitting.lmfit_cuda`, which imports its constants from
:mod:`picasso.fitting.splinefit` - so the CPU and GPU fits stop in the same
place by construction rather than by agreement.

One thread fits one spot, including its whole axial multi-start. Gpufit instead
spreads a single fit across threads and reduces in shared memory; for Picasso's
workload - many spots of 49 to 98 points - one thread per spot removes every
reduction and all shared memory, and makes results bitwise reproducible.

Two transcription details are load-bearing and easy to get wrong:

Pixel loop order
    ``for j (y)`` outer, ``for i (x)`` inner, matching ``splinefit`` and the
    row-major ``[y, x]`` spot memory. The CRLB kernels in ``picasso.localize``
    loop the other way round. Copying them would silently change the
    floating-point summation order and break CPU/GPU agreement in a way that
    looks exactly like a broken model.

Coefficient layout
    ``precision._spline_coeff_reshaped``'s natural view,
    ``(n_channels, niz, niy, nix, 4, 4, 4)`` indexed
    ``[c, k, j, i, z_power, y_power, x_power]`` - the same array the CPU kernels
    and the CUDA CRLB kernels take. This is **not** the axis-reordered blob the
    old Gpufit path packed into ``user_info``; feeding that in would scramble
    the model without raising anything.

Precision
    The tricubic evaluation runs in single precision by default and everything
    downstream - chi-square, gradient, Hessian, parameter vector and the
    Gauss-Jordan solve - in double, unconditionally. The evaluation is where
    almost all the arithmetic is (64 coefficient reads and their Horner passes
    per pixel per channel) and calibrations are stored ``float32`` to begin
    with, so a single-precision tricubic carries ~1e-6 relative error, far below
    the shot noise the fit is limited by. The solve is the opposite case: with
    six channels the Hessian is 15x15 with near-collinear columns - the same
    conditioning that makes the CRLB path need a truncating pseudo-inverse - and
    the damping vector is a monotone running maximum, so a single bad diagonal
    would poison every later step. Pass ``single_precision=False`` to evaluate
    in double too; the tests use it to compare against the CPU kernels without
    a rounding excuse.

References
----------
The fitting algorithm and the spline models are a port of Gpufit
(``models/spline_*.cuh``, ``cuda_kernels.cu``):

Przybylski, A., Thiel, B., Keller-Findeisen, J., Stock, B. & Bates, M.
"Gpufit: An open-source toolkit for GPU-accelerated curve fitting."
Scientific Reports 7, 15722 (2017).
https://doi.org/10.1038/s41598-017-15313-9
Licence (MIT): ``LICENSES/Gpufit-LICENSE.txt``.

The cubic-spline PSF model and its use for single-molecule localization:

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

import math
from typing import Callable, Literal

import numpy as np
from numba import cuda, float64
from tqdm import tqdm

from picasso.fitting import lmfit_cuda
from picasso.fitting.splinefit import (
    KIND_2D,
    KIND_3D,
    KIND_LINK_XYZ,
    _allocate_outputs,
    _check_inputs,
    resolve_variance,
    resolve_schedule,
)
from picasso.fitting.lmfit_cuda import (
    CUDA_THREADS,
    CUDA_THREADS_WIDE,
    _INF,
    _estimator_terms,
    make_fit_kernel,
    make_lm_driver,
)

# Parameter counts per model. The lateral and axial models are fixed; the
# photon-decoupled model has ``3 + 2 * n_channels``.
_N_PARAMS_2D = 4
_N_PARAMS_3D = 5

# Parameter index the axial multi-start seeds.
_Z_COL_3D = 3
_Z_COL_LINK_XYZ = 2


# ----------------------------------------------------------------------
# Spline evaluation
# ----------------------------------------------------------------------


@cuda.jit(device=True, inline=True)
def _interval(pos, n_intervals):
    """Spline interval containing ``pos``, clamped to the coefficient grid.

    Only the *index* is clamped. The caller keeps the true fractional
    coordinate ``pos - i``, which may fall outside ``[0, 1)``, so a position off
    the edge of the box **extrapolates** the boundary cubic rather than
    saturating at it - what Gpufit's models do and what ``splinefit._interval``
    reproduces. Computed in double even when the evaluation is single precision,
    so that the interval never lands one cell off for a large coordinate."""
    i = int(math.floor(pos))
    if i < 0:
        i = 0
    elif i > n_intervals - 1:
        i = n_intervals - 1
    return i


def _make_eval_spline_2d(ftype):
    """Build the bicubic evaluator at the given working precision.

    Returns a device function ``(coeff, ch, pos_x, pos_y) -> (phi, gx, gy)``
    whose derivatives are with respect to the *native* coordinate; the
    derivative with respect to the fitted shift is their negative.

    The precision is a closure constant rather than an argument because numba
    infers arithmetic types from operands, and a bare Python float literal is
    always double - so keeping an expression in single precision means spelling
    every constant ``ftype(...)``. Confining that to the evaluator keeps it to a
    handful of lines instead of infecting the whole module.
    """
    two = ftype(2.0)
    three = ftype(3.0)

    @cuda.jit(device=True, inline=True)
    def _cubic(c0, c1, c2, c3, f):
        """Value and derivative of ``c0 + c1 f + c2 f^2 + c3 f^3`` (Horner)."""
        return (
            ((c3 * f + c2) * f + c1) * f + c0,
            (three * c3 * f + two * c2) * f + c1,
        )

    @cuda.jit(device=True)
    def eval_spline_2d(coeff, ch, pos_x, pos_y):
        niy = coeff.shape[1]
        nix = coeff.shape[2]
        xi = _interval(pos_x, nix)
        yi = _interval(pos_y, niy)
        # The subtraction is done in double and only then narrowed: doing it in
        # single would lose the low bits of the fractional coordinate for a
        # position far from the origin.
        fx = ftype(pos_x - xi)
        fy = ftype(pos_y - yi)
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
        # ... then along y. The second Horner's value is phi and its derivative
        # is dphi/dy; running it over the x-derivative rows gives dphi/dx.
        phi, gy = _cubic(v0, v1, v2, v3, fy)
        gx, _ = _cubic(d0, d1, d2, d3, fy)
        return phi, gx, gy

    return eval_spline_2d


def _make_eval_spline_3d(ftype):
    """Build the tricubic evaluator at the given working precision.

    Returns ``(coeff, ch, pos_x, pos_y, pos_z) -> (phi, gx, gy, gz)``. See
    :func:`_make_eval_spline_2d`; the z-power axis is accumulated on top of the
    same scheme."""
    zero = ftype(0.0)
    one = ftype(1.0)
    two = ftype(2.0)
    three = ftype(3.0)

    @cuda.jit(device=True, inline=True)
    def _cubic(c0, c1, c2, c3, f):
        return (
            ((c3 * f + c2) * f + c1) * f + c0,
            (three * c3 * f + two * c2) * f + c1,
        )

    @cuda.jit(device=True)
    def eval_spline_3d(coeff, ch, pos_x, pos_y, pos_z):
        niz = coeff.shape[1]
        niy = coeff.shape[2]
        nix = coeff.shape[3]
        xi = _interval(pos_x, nix)
        yi = _interval(pos_y, niy)
        zi = _interval(pos_z, niz)
        fx = ftype(pos_x - xi)
        fy = ftype(pos_y - yi)
        fz = ftype(pos_z - zi)
        phi = zero
        gx = zero
        gy = zero
        gz = zero
        pz = one  # fz**zp
        dpz = zero  # d(fz**zp)/dfz
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
            dpz = ftype(zp + 1) * pz
            pz *= fz
        return phi, gx, gy, gz

    return eval_spline_3d


# ----------------------------------------------------------------------
# Accumulators: chi-square, gradient and Hessian in one pass
# ----------------------------------------------------------------------


def _make_accumulate_2d(eval_spline_2d):
    """Accumulator for the 2D model, parameters ``[amplitude, x, y, offset]``.

    There is no multichannel 2D model, so ``aff`` and ``res`` are accepted to
    keep the signature uniform but unused. Transcription of
    ``splinefit._accumulate_2d``."""

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
        x_shift = theta[1]
        y_shift = theta[2]
        offset = theta[3]
        if not (
            math.isfinite(amp)
            and math.isfinite(x_shift)
            and math.isfinite(y_shift)
            and math.isfinite(offset)
        ):
            return _INF, False
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
                phi, gx, gy = eval_spline_2d(coeff, 0, pos_x, pos_y)
                value = amp * phi + offset
                data = spots[index, 0, j, i]
                d0 = phi
                d1 = -amp * gx
                d2 = -amp * gy
                # d3 == 1 (offset), folded into the accumulation below.
                var = 0.0
                if use_variance:
                    var = variance[index, 0, j, i]
                contrib, weight, factor, ok = _estimator_terms(
                    mle, value, data, var
                )
                if not ok:
                    return _INF, False
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

    return accumulate


def _make_accumulate_3d(eval_spline_3d):
    """Accumulator for the shared-amplitude 3D model.

    Parameters are ``[amplitude, x, y, z, offset]``, shared across every
    channel. The single-channel ``spline-3d`` model is the ``n_channels == 1``,
    identity-affine, zero-residual case of the multichannel one. Transcription
    of ``splinefit._accumulate_3d``."""

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
        n_channels = spots.shape[1]
        box = spots.shape[2]
        amp = theta[0]
        x_shift = theta[1]
        y_shift = theta[2]
        z_shift = theta[3]
        offset = theta[4]
        if not (
            math.isfinite(amp)
            and math.isfinite(x_shift)
            and math.isfinite(y_shift)
            and math.isfinite(z_shift)
            and math.isfinite(offset)
        ):
            return _INF, False
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
            # Each channel sees the shared lateral shift through its own affine
            # and sits on its own sub-pixel ROI offset. Constant over the box,
            # so hoisted.
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
                    phi, gx, gy, gz = eval_spline_3d(
                        coeff, ch, pos_x, pos_y, pos_z
                    )
                    value = amp * phi + offset
                    data = spots[index, ch, j, i]
                    # The lateral pair picks up the transpose of the channel
                    # affine (shift = A @ theta), and the leading minus is the
                    # chain rule of position = pixel - shift. Unlike the CRLB,
                    # whose diagonal is sign-invariant, an LM step is not:
                    # dropping the minus sends x, y and z the wrong way.
                    d0 = phi
                    d1 = -amp * (a00 * gx + a10 * gy)
                    d2 = -amp * (a01 * gx + a11 * gy)
                    d3 = -amp * gz
                    # d4 == 1 (offset).
                    var = 0.0
                    if use_variance:
                        var = variance[index, ch, j, i]
                    contrib, weight, factor, ok = _estimator_terms(
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


def _make_accumulate_link_xyz(eval_spline_3d, n_channels: int):
    """Accumulator for the photon-decoupled 3D model at a fixed channel count.

    Parameters are ``[x, y, z, N_0..N_{C-1}, bg_0..bg_{C-1}]``: x, y and z are
    shared while every channel fits its own photon count and background.

    A pixel of channel ``ch`` touches only five of the ``3 + 2C`` parameters, so
    the Jacobian is block sparse. ``splinefit._accumulate_link_xyz`` exploits
    that with 15 read-modify-writes into ``hess`` per pixel; on the GPU those
    would be 15 local-memory round trips, so here the same 15 quantities live in
    registers and are written out once per channel - the arrangement
    ``precision._spline_crlb_link_xyz_kernel`` already uses. The summation
    order is unchanged, so the two still agree to rounding.
    """
    n_ch = n_channels

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
        n_params = 3 + 2 * n_ch
        x_shift = theta[0]
        y_shift = theta[1]
        z_shift = theta[2]
        for p in range(n_params):
            if not math.isfinite(theta[p]):
                return _INF, False
            grad[p] = 0.0
            for q in range(n_params):
                hess[p, q] = 0.0
        pos_z = -z_shift
        chi_square = 0.0
        # Shared x/y/z block, accumulated across every channel.
        g0 = g1 = g2 = 0.0
        h00 = h01 = h02 = 0.0
        h11 = h12 = 0.0
        h22 = 0.0
        for ch in range(n_ch):
            amp = theta[3 + ch]
            offset = theta[3 + n_ch + ch]
            # Global parameter indices of this channel's photon count and
            # background; the three shared position parameters are 0, 1, 2.
            ia = 3 + ch
            ib = 3 + n_ch + ch
            a00 = aff[ch, 0]
            a01 = aff[ch, 1]
            a10 = aff[ch, 2]
            a11 = aff[ch, 3]
            sx = a00 * x_shift + a01 * y_shift + res[index, ch, 0]
            sy = a10 * x_shift + a11 * y_shift + res[index, ch, 1]
            # This channel's own photon (a) and background (b) entries.
            ga = gb = 0.0
            h0a = h1a = h2a = 0.0
            h0b = h1b = h2b = 0.0
            haa = hab = hbb = 0.0
            for j in range(box):
                pos_y = j - sy
                for i in range(box):
                    pos_x = i - sx
                    phi, gx, gy, gz = eval_spline_3d(
                        coeff, ch, pos_x, pos_y, pos_z
                    )
                    value = amp * phi + offset
                    data = spots[index, ch, j, i]
                    d0 = -amp * (a00 * gx + a10 * gy)
                    d1 = -amp * (a01 * gx + a11 * gy)
                    d2 = -amp * gz
                    # d(value)/d(N_ch) = phi, d(value)/d(bg_ch) = 1; both zero
                    # for every other channel.
                    var = 0.0
                    if use_variance:
                        var = variance[index, ch, j, i]
                    contrib, weight, factor, ok = _estimator_terms(
                        mle, value, data, var
                    )
                    if not ok:
                        return _INF, False
                    chi_square += contrib
                    g0 += d0 * factor
                    g1 += d1 * factor
                    g2 += d2 * factor
                    ga += phi * factor
                    gb += factor
                    w0 = weight * d0
                    w1 = weight * d1
                    w2 = weight * d2
                    wa = weight * phi
                    h00 += w0 * d0
                    h01 += w0 * d1
                    h02 += w0 * d2
                    h0a += w0 * phi
                    h0b += w0
                    h11 += w1 * d1
                    h12 += w1 * d2
                    h1a += w1 * phi
                    h1b += w1
                    h22 += w2 * d2
                    h2a += w2 * phi
                    h2b += w2
                    haa += wa * phi
                    hab += wa
                    hbb += weight
            grad[ia] = ga
            grad[ib] = gb
            hess[0, ia] = h0a
            hess[1, ia] = h1a
            hess[2, ia] = h2a
            hess[0, ib] = h0b
            hess[1, ib] = h1b
            hess[2, ib] = h2b
            hess[ia, ia] = haa
            hess[ia, ib] = hab
            hess[ib, ib] = hbb
        grad[0] = g0
        grad[1] = g1
        grad[2] = g2
        hess[0, 0] = h00
        hess[0, 1] = h01
        hess[0, 2] = h02
        hess[1, 1] = h11
        hess[1, 2] = h12
        hess[2, 2] = h22
        # Only the upper triangle was filled (0 < 1 < 2 < ia < ib always
        # holds); mirror it once at the end. The cross-channel photon and
        # background blocks are structurally zero and were cleared above.
        for p in range(n_params):
            for q in range(p):
                hess[p, q] = hess[q, p]
        return chi_square, True

    return accumulate


# ----------------------------------------------------------------------
# Kernel construction and cache
# ----------------------------------------------------------------------

_KERNEL_CACHE: dict = {}


def _build_kernel(kind: int, n_channels: int, single_precision: bool):
    """Compile the fit kernel for one model, channel count and precision."""
    ftype = np.float32 if single_precision else np.float64
    if kind == KIND_2D:
        accumulate = _make_accumulate_2d(_make_eval_spline_2d(ftype))
        driver = make_lm_driver(
            accumulate, _N_PARAMS_2D, _Z_COL_3D, seedable=False
        )
    elif kind == KIND_3D:
        accumulate = _make_accumulate_3d(_make_eval_spline_3d(ftype))
        driver = make_lm_driver(
            accumulate, _N_PARAMS_3D, _Z_COL_3D, seedable=True
        )
    elif kind == KIND_LINK_XYZ:
        accumulate = _make_accumulate_link_xyz(
            _make_eval_spline_3d(ftype), n_channels
        )
        driver = make_lm_driver(
            accumulate,
            3 + 2 * n_channels,
            _Z_COL_LINK_XYZ,
            seedable=True,
        )
    else:
        raise ValueError(f"Unknown spline model kind {kind}.")
    return make_fit_kernel(driver)


def _get_kernel(kind: int, n_channels: int, single_precision: bool):
    """Memoized :func:`_build_kernel`.

    Only the photon-decoupled model needs one kernel per channel count - its
    parameter count is ``3 + 2C``, and a device-local array needs a compile-time
    shape. The other two have a fixed parameter count and loop over channels at
    run time, so a single kernel serves every channel count."""
    key = (
        kind,
        int(n_channels) if kind == KIND_LINK_XYZ else 0,
        bool(single_precision),
    )
    kernel = _KERNEL_CACHE.get(key)
    if kernel is None:
        kernel = _build_kernel(kind, n_channels, single_precision)
        _KERNEL_CACHE[key] = kernel
    return kernel


def n_parameters(kind: int, n_channels: int) -> int:
    """Parameter count of a model."""
    if kind == KIND_2D:
        return _N_PARAMS_2D
    if kind == KIND_3D:
        return _N_PARAMS_3D
    return 3 + 2 * int(n_channels)


# ----------------------------------------------------------------------
# Host entry point
# ----------------------------------------------------------------------


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
    abort_callback: Callable[[], bool] | None = None,
    single_precision: bool = True,
    variance: np.ndarray | None = None,
) -> tuple:
    """Fit spots with a cubic-spline PSF model on the GPU.

    Array-for-array the same contract as ``splinefit.fit_spots`` - see its
    docstring for the layouts of ``spots``, ``coefficients``, ``affines``,
    ``residuals``, ``initial_parameters`` and ``z_seeds`` - plus:

    Parameters
    ----------
    abort_callback : callable or None, optional
        Polled between chunks; returning True stops the fit. A launched kernel
        cannot be cancelled, so chunk boundaries are the only points at which an
        abort can take effect. Spots not reached keep their NaN parameters and
        infinite chi-square.
    single_precision : bool, optional
        Evaluate the spline in single precision (the default). Everything
        downstream of the model is double regardless. See the module docstring.

    Returns
    -------
    thetas, chi_squares, states, iterations
        As ``splinefit.fit_spots``, using Gpufit's state codes.
    """
    lmfit_cuda.require_cuda()
    _check_inputs(
        kind, spots, coefficients, affines, residuals, initial_parameters
    )
    variance, use_variance = resolve_variance(variance, spots.shape, ndim=4)
    tolerance, max_iterations = resolve_schedule(
        apply_seeds, tolerance, max_iterations
    )
    n_spots, n_channels, box, _ = spots.shape
    n_params = initial_parameters.shape[1]
    thetas, chi_squares, states, iterations = _allocate_outputs(
        n_spots, n_params
    )
    if n_spots == 0:
        return thetas, chi_squares, states, iterations

    # The 2D model has no axial coordinate; its parameter 3 is the background,
    # so seeding it would corrupt the fit rather than move it in z. The driver
    # enforces this too - it is repeated here so the work estimate is right.
    seeded = bool(apply_seeds) and kind != KIND_2D
    n_seeds = len(z_seeds) if seeded else 1

    coeff_dtype = np.float32 if single_precision else np.float64
    spots = np.ascontiguousarray(spots, dtype=np.float32)
    coefficients = np.ascontiguousarray(coefficients, dtype=coeff_dtype)
    affines = np.ascontiguousarray(affines, dtype=np.float64)
    residuals = np.ascontiguousarray(residuals, dtype=np.float64)
    initial_parameters = np.ascontiguousarray(
        initial_parameters, dtype=np.float64
    )
    z_seeds = np.ascontiguousarray(z_seeds, dtype=np.float64)

    kernel = _get_kernel(kind, n_channels, single_precision)
    threads = CUDA_THREADS_WIDE if n_params > 8 else CUDA_THREADS

    # Constant over the whole run, so uploaded once.
    d_coeff = cuda.to_device(coefficients)
    d_aff = cuda.to_device(affines)
    d_seeds = cuda.to_device(z_seeds)
    # Without a noise model the variance is a four-byte dummy, uploaded once
    # rather than per chunk.
    d_dummy_variance = None if use_variance else cuda.to_device(variance)

    bytes_per_row = (
        4 * n_channels * box * box  # spots
        + (4 * n_channels * box * box if use_variance else 0)  # variance
        + 8 * 2 * n_channels  # residuals
        + 8 * n_params  # initial parameters
        + 8 * n_params  # fitted parameters
        + 8  # chi-square
        + 8  # state and iteration count
    )
    chunk = min(
        n_spots,
        lmfit_cuda.chunk_rows(bytes_per_row, n_seeds * max_iterations),
    )

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
            d_res = cuda.to_device(residuals[start:stop])
            d_init = cuda.to_device(initial_parameters[start:stop])
            d_thetas = cuda.device_array((n, n_params), dtype=np.float64)
            d_chi = cuda.device_array(n, dtype=np.float64)
            d_states = cuda.device_array(n, dtype=np.int32)
            d_iterations = cuda.device_array(n, dtype=np.int32)
            blocks = (n + threads - 1) // threads
            kernel[blocks, threads](
                d_spots,
                d_variance,
                use_variance,
                d_coeff,
                d_aff,
                d_res,
                d_init,
                d_seeds,
                seeded,
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
