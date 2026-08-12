"""
picasso.fitting.precision
~~~~~~~~~~~~~~~~~~~~~~~~~

Uncertainty estimates for fitted localizations - what Picasso reports as
``lpx``/``lpy`` (and the width and axial errors).

Two families live here, for the same quantity computed two ways:

*Closed forms.* Evaluated from a fit's reported parameters rather than from
the data: given photons, widths and background, each returns the standard
error the corresponding estimator attains. Which one applies depends on the
estimator that produced the fit:

==============================  ====================================
function                        estimator
==============================  ====================================
:func:`localization_precision`  least-squares Gaussian (position)
:func:`sigma_uncertainty_lsq`   least-squares Gaussian (width)
:func:`sigma_uncertainty_mle`   Poisson maximum likelihood (width)
==============================  ====================================

*Numerically inverted Fisher matrices.* The expensive counterpart: they
evaluate the model and its derivatives at every pixel of every fitting box and
invert the resulting information matrix, so they need the PSF model itself.
They are exact under the pixel noise model (including a per-pixel sCMOS
variance map, which the closed forms can only approximate by a flat
background offset):

==============================  ====================================
function                        model
==============================  ====================================
:func:`_gauss_crlb`             2D Gaussian, all three variants
:func:`_spline_crlb`            cubic-spline PSF, 2D/3D/multichannel
:func:`_spline_link_xyz_crlb`   photon-decoupled multichannel spline
==============================  ====================================

The spline bounds run on a CUDA GPU when one is available (``CUDA_AVAILABLE``)
and fall back to the numba CPU kernels otherwise; both compute the same thing,
so the choice is invisible to the caller.

The closed forms lived in ``picasso.gausslq`` and ``picasso.gaussmle`` until
0.11 (those modules are deprecated and go in Picasso 1.0); the Fisher-matrix
estimators lived in ``picasso.localize`` and moved here in 0.11, where
``picasso.localize`` still re-exports them under their old names.

References
----------
Mortensen, K. I., Churchman, L. S., Spudich, J. A. & Flyvbjerg, H.
"Optimized localization analysis for single-molecule tracking and
super-resolution microscopy." Nature Methods 7, 377-381 (2010).
https://doi.org/10.1038/nmeth.1447

Rieger, B. & Stallinga, S. "The lateral and axial localization uncertainty in
super-resolution light microscopy." ChemPhysChem 15, 664-670 (2014).
https://doi.org/10.1002/cphc.201300711

Kowalewski, R., Reinhardt, S. C. M. et al. Nature Communications (2026).
https://doi.org/10.1038/s41467-026-70198-5

:authors: Joerg Schnitzbauer, Maximilian Thomas Strauss, Rafal Kowalewski
:copyright: Copyright (c) 2016-2026 Jungmann Lab, MPI of Biochemistry
"""

from __future__ import annotations

import math
import warnings
from typing import Callable, Literal

import numba
import numpy as np
from numba import cuda
from tqdm import tqdm

from picasso import lib

# Check for CUDA availability for the spline CRLB kernels. Otherwise, CPU is
# used. ``picasso.localize`` keeps its own probe for the *fitting* path.
try:
    CUDA_AVAILABLE = bool(cuda.is_available())
except Exception:
    CUDA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Closed-form precisions
# ---------------------------------------------------------------------------


def localization_precision(
    photons: lib.FloatArray1D,
    s: lib.FloatArray1D,
    s_orth: lib.FloatArray1D,
    bg: lib.FloatArray1D,
    em: bool,
    readout_variance: lib.FloatArray1D | float = 0.0,
) -> lib.FloatArray1D:
    """Theoretical localization precision of a 2D unweighted Gaussian fit.

    Mortensen et al., Nature Methods, 2010.

    Edit v0.9.0: corrected formula for diagonal covariance Gaussian
    (i.e., sx != sy). The background term includes the orthogonal sigma.

    Parameters
    ----------
    photons : lib.FloatArray1D
        Number of photons collected for the localization.
    s : lib.FloatArray1D
        Size of the single-emitter image for each localization.
    s_orth : lib.FloatArray1D
        Size of the single-emitter image in the orthogonal direction
        for each localization.
    bg : lib.FloatArray1D
        Background signal for each localization (per pixel).
    em : bool
        Whether EMCCD was used for the localization. Its stochastic
        multiplication doubles the variance.
    readout_variance : lib.FloatArray1D or float, optional
        Mean sCMOS readout variance over the fitting box, in photoelectrons
        squared. Adds to the background term. Default 0.

    Returns
    -------
    lib.FloatArray1D
        Cramer-Rao lower bound for localization precision for each
        localization.

    Notes
    -----
    With a per-pixel sCMOS camera calibration, ``readout_variance`` adds the
    pixel readout noise to the background term, which is where a spatially
    uniform noise floor enters this closed form. That is an approximation:
    Mortensen's derivation assumes the background variance is *constant* over
    the fitting box, so replacing ``bg`` by ``bg + mean(var)`` is exact only
    when the variance map is flat there. With one hot pixel next to the
    emitter it under- or over-states the precision depending on where that
    pixel falls under the PSF. For sCMOS data with a calibration loaded,
    prefer a maximum-likelihood method, whose Cramer-Rao bound is evaluated
    pixel by pixel and is exact under the noise model
    (:func:`_gauss_crlb`).
    """
    bg = bg + readout_variance
    s2 = s**2
    sa2 = s2 + 1 / 12
    sa = sa2**0.5
    sa_orth2 = s_orth**2 + 1 / 12
    sa_orth = sa_orth2**0.5
    v = sa2 * (16 / 9 + (8 * np.pi * sa * sa_orth * bg) / photons) / photons
    if em:
        v *= 2
    with np.errstate(invalid="ignore"):
        return np.sqrt(v)


def sigma_uncertainty_lsq(
    sigma: lib.SeriesOrFloatArray1D,
    sigma_orth: lib.SeriesOrFloatArray1D,
    photons: lib.SeriesOrFloatArray1D,
    bg: lib.SeriesOrFloatArray1D,
    readout_variance: lib.SeriesOrFloatArray1D | float = 0.0,
) -> lib.FloatArray1D:
    """Standard error of a **least-squares** fitted sigma.

    From the 2D Gaussian least-squares model with a diagonal covariance
    matrix, Kowalewski, Reinhardt, et al. Nature Communications, 2026.

    Parameters
    ----------
    sigma : lib.SeriesOrFloatArray1D
        Fitted sigma values in camera pixels.
    sigma_orth : lib.SeriesOrFloatArray1D
        Fitted sigma values in the orthogonal direction in camera
        pixels.
    photons : lib.SeriesOrFloatArray1D
        Number of photons.
    bg : lib.SeriesOrFloatArray1D
        Background photons per pixel.
    readout_variance : lib.SeriesOrFloatArray1D or float, optional
        Mean sCMOS readout variance over the fitting box, in photoelectrons
        squared. Adds to the background term, with the same caveat as in
        :func:`localization_precision`. Default 0.

    Returns
    -------
    se_sigma : lib.FloatArray1D
        Standard error of fitted sigma values in camera pixels.
    """
    bg = bg + readout_variance
    sa2 = sigma**2 + 1 / 12
    sa4 = sa2**2
    sa = sa2**0.5
    sa2_orth = sigma_orth**2 + 1 / 12
    sa_orth = sa2_orth**0.5
    var_sa2 = (
        sa4
        / photons
        * (512 / 81 + (64 * np.pi * sa * sa_orth * bg) / (3 * photons))
    )
    var_sigma = var_sa2 / (4 * sigma**2)
    se_sigma = np.sqrt(var_sigma)
    return se_sigma


def sigma_uncertainty_mle(
    sigma: lib.SeriesOrFloatArray1D,
    sigma_orth: lib.SeriesOrFloatArray1D,
    photons: lib.SeriesOrFloatArray1D,
    bg: lib.SeriesOrFloatArray1D,
    readout_variance: lib.SeriesOrFloatArray1D | float = 0.0,
) -> lib.FloatArray1D:
    """Standard error of a **maximum-likelihood** fitted sigma.

    From the MLE 2D Gaussian / Poisson noise model, using the
    approximation of Rieger and Stallinga, ChemPhysChem, 2014.

    Parameters
    ----------
    sigma : lib.SeriesOrFloatArray1D
        Fitted sigma values in camera pixels.
    sigma_orth : lib.SeriesOrFloatArray1D
        Unused; see above.
    photons : lib.SeriesOrFloatArray1D
        Number of photons.
    bg : lib.SeriesOrFloatArray1D
        Background photons per pixel.
    readout_variance : lib.SeriesOrFloatArray1D or float, optional
        Mean sCMOS readout variance over the fitting box, in photoelectrons
        squared. Adds to the background term, with the same caveat as in
        :func:`localization_precision`. Default 0.

    Returns
    -------
    se_sigma : lib.FloatArray1D
        Standard error of fitted sigma values in camera pixels.
    """
    bg = bg + readout_variance
    sa2 = sigma**2 + 1 / 12
    tau = (2 * np.pi * sa2 * bg) / (photons)
    delta_sigma_sq = (sigma**2 / (4 * photons)) * (
        1 + 8 * tau + np.sqrt((8 * tau) / (1 + 2 * tau))
    )
    return np.sqrt(delta_sigma_sq)


# ---------------------------------------------------------------------------
# Shared constants for the Fisher-matrix estimators
# ---------------------------------------------------------------------------

_SPLINE_CRLB_MU_FLOOR = 1e-3  # photons; floors 1 / mu in the Fisher weight
# Target size of one (n_locs, P, P) float64 information-matrix block in the
# chunked CRLB solve; the peak is a few multiples of this.
_SPLINE_CRLB_CHUNK_BYTES = 64 << 20
_GAUSS_CRLB_MU_FLOOR = (
    1e-3  # photons; floors 1 / mu in the Poisson Fisher weight
)
# EMCCD stochastic multiplication doubles every pixel's variance (excess noise
# factor F^2 = 2), so every variance derived from a Poisson pixel model has to
# be scaled by it. Same factor as in :func:`localization_precision`.
_EM_EXCESS_NOISE_FACTOR = 2.0

# Largest channel count the photon-decoupled (link-XYZ) spline fit supports.
_LINK_XYZ_MAX_CHANNELS = 6

_LINK_XYZ_MODEL = "spline-3d-multichannel-link-xyz"


# ---------------------------------------------------------------------------
# Calibration and array adapters for the CRLB kernels
# ---------------------------------------------------------------------------


def _spline_n_channels(calibration: dict) -> int:
    """Number of channels a (multi)channel spline calibration encodes.

    Multichannel coefficients are ``(64, nix, niy, niz, n_channels)``, so the
    channel axis is the last one."""
    model = calibration["model"]
    coeff = np.asarray(calibration["coefficients"])
    if model in ("spline-3d-multichannel", _LINK_XYZ_MODEL):
        return int(calibration.get("n_channels", coeff.shape[-1]))
    return 1


def _spline_channel_major(
    spots: lib.FloatArray3D, n_channels: int
) -> np.ndarray:
    """Spots as ``(n_spots, n_channels, box, box)`` for the CPU kernels.

    Picasso stacks multichannel spots channel-*minor*, ``(n, box, box,
    n_channels)``, so those need a transpose - the same reordering
    the kernels' channel-major data layout needs. A
    single-channel stack only needs a length-1 axis inserted, which is a free
    reshape; transposing it instead would copy the entire spot stack for
    nothing."""
    if n_channels == 1:
        if spots.ndim == 4 and spots.shape[3] == 1:
            spots = spots[..., 0]
        return spots.reshape(len(spots), 1, spots.shape[1], spots.shape[2])
    if spots.ndim != 4:
        raise ValueError(
            "Multichannel spline fitting expects spots of shape "
            "(n_spots, box, box, n_channels)."
        )
    if spots.shape[3] != n_channels:
        raise ValueError(
            f"Spots have {spots.shape[3]} channels but the calibration has "
            f"{n_channels}."
        )
    return np.ascontiguousarray(spots.transpose(0, 3, 1, 2))


def _spline_channel_affines(calibration: dict, n_channels: int) -> np.ndarray:
    """Per-channel lateral 2x2 affine as ``(n_channels, 4)``
    ``[a00, a01, a10, a11]``.

    The multichannel fit evaluates one shared lateral shift through each
    channel's own transform, so the precision has to be computed the same way -
    otherwise the reported ``lpx``/``lpy`` come from a different model than the
    one that was fitted. Falls back to the identity per channel when the
    calibration carries no usable ``channel_transforms``, matching what the
    CUDA models do when the affine block is missing from ``user_info``."""
    aff = np.tile(np.array([1.0, 0.0, 0.0, 1.0]), (n_channels, 1))
    transforms = calibration.get("channel_transforms")
    if transforms is not None and len(transforms) == n_channels:
        for c, t in enumerate(transforms):
            lin = np.asarray(t, dtype=np.float64)[:, :2]
            aff[c] = (lin[0, 0], lin[0, 1], lin[1, 0], lin[1, 1])
    return aff


def _crlb_variance_channel_major(
    variance: np.ndarray | None, n_channels: int
) -> np.ndarray | None:
    """Normalize a readout-variance patch array for the CRLB kernels.

    They index ``var[m, ch, j, i]``, so the patches arrive in the same Picasso
    layout the spots do - ``(k, box, box)`` single-channel from ``get_spots``,
    or channel-*last* ``(k, box, box, n_channels)`` from
    :func:`get_spots_multichannel` - and go through the very same reordering.
    Special-casing the 4D form here instead would leave a multichannel patch
    channel-last while the kernel indexes it channel-major, which silently
    reads another channel's noise."""
    if variance is None:
        return None
    return _spline_channel_major(np.asarray(variance), n_channels)


def _crlb_variance_chunk(
    variance: lib.FloatArray4D | None, sl: slice
) -> lib.FloatArray4D:
    """A chunk of the readout-variance patches for a CRLB kernel.

    The kernels index ``var[m, ch, j, i]`` unconditionally, so a stand-in of
    the right rank has to exist even when there is no calibration - numba
    types an array by its rank, and a scalar would not compile."""
    if variance is None:
        return np.zeros((1, 1, 1, 1), dtype=np.float32)
    return np.ascontiguousarray(variance[sl], dtype=np.float32)


def _spline_crlb_residuals(
    residuals: np.ndarray | None, n_locs: int, n_channels: int
) -> np.ndarray:
    """``(n_locs, n_channels, 2)`` float64 ROI residuals for the CRLB kernels,
    or zeros when the fit was run without them - so the precision is evaluated
    under the same geometry that produced ``theta``."""
    rows = max(int(n_locs), 1)
    if residuals is None:
        return np.zeros((rows, n_channels, 2), dtype=np.float64)
    res = np.ascontiguousarray(residuals, dtype=np.float64)
    if res.shape != (n_locs, n_channels, 2):
        raise ValueError(
            "residuals must have shape (n_locs, n_channels, 2) = "
            f"{(n_locs, n_channels, 2)}, got {res.shape}."
        )
    if n_locs == 0:
        return np.zeros((rows, n_channels, 2), dtype=np.float64)
    return res


def _spline_coeff_reshaped(
    calibration: dict, dtype: type = np.float64
) -> np.ndarray:
    """Raw calibration coefficients as ``(n_channels, niz, niy, nix, 4, 4, 4)``
    (3D) or ``(n_channels, niy, nix, 4, 4)`` (2D), float64 for the numba
    kernels.

    ``dtype`` narrows the output; the CUDA kernels ask for float32 when the
    calibration itself is float32 (see :func:`_spline_crlb_coeff_dtype`), which
    halves the device-side table without losing any information."""
    model = calibration["model"]
    coeff = np.ascontiguousarray(calibration["coefficients"], dtype=dtype)
    if model in ("spline-3d-multichannel", _LINK_XYZ_MODEL):
        _, nix, niy, niz, n_channels = coeff.shape
        return np.stack(
            [
                np.ascontiguousarray(coeff[..., c]).reshape(
                    niz, niy, nix, 4, 4, 4
                )
                for c in range(n_channels)
            ]
        )
    # Single-channel models get their leading channel axis from reshape rather
    # than ``[None]``: the latter gives that axis stride 0, which numba (CPU and
    # CUDA alike) types as a non-contiguous array and then indexes through
    # computed strides - several times slower for no reason.
    if model == "spline-2d":
        _, nix, niy = coeff.shape
        return coeff.reshape(1, niy, nix, 4, 4)
    _, nix, niy, niz = coeff.shape
    return coeff.reshape(1, niz, niy, nix, 4, 4, 4)


def _spline_crlb_coeff_dtype(calibration: dict) -> type:
    """Precision to upload the spline coefficients to the GPU in.

    Calibrations are stored float32 (``picasso.io.load_spline_calibration``), so
    widening them for the device would cost bandwidth without adding
    information - the kernels widen each coefficient to float64 in-register
    anyway, which is what the CPU kernels do in bulk. Anything not already
    float32 is passed through as float64."""
    if np.asarray(calibration["coefficients"]).dtype == np.float32:
        return np.float32
    return np.float64


# ---------------------------------------------------------------------------
# Gaussian CRLB
# ---------------------------------------------------------------------------


def _gauss_crlb(
    theta: lib.FloatArray2D,
    box: int,
    em: bool,
    rotated: bool = False,
    variance: lib.FloatArray3D | None = None,
) -> lib.FloatArray2D:
    """Poisson Cramer-Rao lower bound for MLE Gaussian fits.

    Builds the Fisher information matrix ``I = Σ g gᵀ / μ`` (``g = ∂μ/∂θ``) of
    the Gaussian model that was actually optimized and returns the diagonal of
    its inverse — the variance an efficient maximum-likelihood estimator
    attains. Evaluated at the fitted parameters; the spot data is not needed.
    Mirrors :func:`_spline_crlb` and ``gaussmle._mlefit_sigmaxy_crlb``.

    Model (photon units, spots are gain-converted before fitting)::

        mu(i, j) = N / (2 pi sx sy) * E + bg

    where ``i`` indexes the x (column) coordinate, ``j`` the y (row)
    coordinate, and ``E`` is the (optionally rotated) unit-height Gaussian.
    Parametrizing the amplitude directly as the total photon count ``N``
    (Picasso's reported ``photons``) makes the returned variances line up with
    the reported columns, with the ``N``/``sx``/``sy`` coupling of ``mu``
    folded into the derivatives.

    Parameters
    ----------
    theta : lib.FloatArray2D
        Fitted parameters in the fit's order ``[photons (total N), x, y,
        sx, sy, bg]`` (elliptic) or ``[..., bg, angle (radians)]`` (rotated).
        Positions are box-local (pixel = parameter, matching
        :func:`_initial_parameters_gauss`).
    box : int
        Fit box side length (pixels).
    em : bool
        EMCCD excess noise: doubles every parameter's variance (halves the
        Fisher weight), as in :func:`localization_precision`.
    rotated : bool, optional
        If True, ``theta`` carries the seventh (angle) column and the CRLB
        includes it. Default False.
    variance : lib.FloatArray3D, optional
        ``(n_locs, box, box)`` per-pixel sCMOS readout variance in
        photoelectrons squared, indexed ``[spot, y, x]`` like the spots. The
        Fisher weight becomes ``1 / (mu + var)``, which is exactly Eq. 3.6 of
        Huang et al. (2013). Default None.

    Returns
    -------
    crlb : lib.FloatArray2D
        ``(n_locs, n_params)`` parameter variances (float64) in the same column
        order as ``theta`` (angle variance in rad²). Non-converged and
        numerically singular fits are NaN.
    """
    theta = np.asarray(theta, dtype=np.float64)
    n_locs = len(theta)
    n_params = 7 if rotated else 6

    N = theta[:, 0]
    x = theta[:, 1]
    y = theta[:, 2]
    sx = theta[:, 3]
    sy = theta[:, 4]
    ang = theta[:, 6] if rotated else None
    finite = np.isfinite(theta).all(axis=1)

    grid = np.arange(box, dtype=np.float64)
    crlb = np.full((n_locs, n_params), np.nan)
    if n_locs == 0:
        return crlb

    # One kernel spans all localizations; chunk only to bound peak memory of the
    # (chunk, n_params, box, box) gradient tensor.
    chunk = max(1, min(n_locs, 50_000))
    for start in range(0, n_locs, chunk):
        stop = min(start + chunk, n_locs)
        sl = slice(start, stop)
        # Per-pixel coordinates relative to the fitted center. Axis 1 = x
        # (column) pixel index, axis 2 = y (row) pixel index.
        Nc = N[sl][:, None, None]
        sxc = sx[sl][:, None, None]
        syc = sy[sl][:, None, None]
        dx = grid[None, :, None] - x[sl][:, None, None]
        dy = grid[None, None, :] - y[sl][:, None, None]

        if rotated:
            ct = np.cos(ang[sl])[:, None, None]
            st = np.sin(ang[sl])[:, None, None]
            u = dx * ct - dy * st
            w = dx * st + dy * ct
            E = np.exp(-0.5 * (u**2 / sxc**2 + w**2 / syc**2))
            s = Nc / (2.0 * np.pi * sxc * syc) * E  # signal = mu - bg
            gx = s * (u * ct / sxc**2 + w * st / syc**2)
            gy = s * (-u * st / sxc**2 + w * ct / syc**2)
            gsx = s * (u**2 / sxc**3 - 1.0 / sxc)
            gsy = s * (w**2 / syc**3 - 1.0 / syc)
            gang = s * (u * w * (1.0 / sxc**2 - 1.0 / syc**2))
        else:
            E = np.exp(-0.5 * (dx**2 / sxc**2 + dy**2 / syc**2))
            s = Nc / (2.0 * np.pi * sxc * syc) * E  # signal = mu - bg
            gx = s * (dx / sxc**2)
            gy = s * (dy / syc**2)
            gsx = s * (dx**2 / sxc**3 - 1.0 / sxc)
            gsy = s * (dy**2 / syc**3 - 1.0 / syc)

        gN = s / Nc
        gbg = np.ones_like(s)
        grads = [gN, gx, gy, gsx, gsy, gbg]
        if rotated:
            grads.append(gang)
        g = np.stack(grads, axis=1)  # (m, n_params, box, box)

        mu = np.maximum(s + theta[sl, 5][:, None, None], _GAUSS_CRLB_MU_FLOOR)
        if variance is not None:
            # This kernel builds its per-pixel tensors as [spot, x, y] - dx
            # varies along axis 1, dy along axis 2 - while a variance patch is
            # [spot, y, x], like the spots it was cut alongside. Transpose, or
            # the noise lands on the transposed pixel and nothing complains.
            mu = mu + np.transpose(variance[sl], (0, 2, 1))
        gw = g / mu[:, None, :, :]
        fisher = np.einsum("mpij,mqij->mpq", gw, g)  # (m, n_params, n_params)

        # Non-converged rows carry NaN parameters (hence NaN Fisher); set them to
        # the identity so the batched pinv stays well-defined, then mask below.
        bad = ~finite[sl]
        fisher[bad] = np.eye(n_params)
        with np.errstate(invalid="ignore", divide="ignore"):
            cov = np.linalg.pinv(fisher)
            var = np.diagonal(cov, axis1=1, axis2=2).copy()
        var[bad] = np.nan
        crlb[sl] = var

    if em:
        # EMCCD excess noise doubles every pixel's variance, hence the CRLB
        # (matches the factor-2 in localization_precision).
        crlb *= _EM_EXCESS_NOISE_FACTOR
    crlb = np.where(crlb > 0.0, crlb, np.nan)
    return crlb


# ---------------------------------------------------------------------------
# CPU (numba) spline information matrices
# ---------------------------------------------------------------------------


@numba.njit(parallel=True, cache=True, fastmath=True)
def _spline_infomats_3d(
    coeff,
    aff,
    res,
    box,
    amp,
    x_shift,
    y_shift,
    z_eval,
    offset,
    finite,
    mu_floor,
    mle,
    var,
    use_var,
    bread,
    meat,
):
    """Fill the per-localization information matrices of the 3D cubic-spline
    model. Parallel per-spot numba kernel. Non-converged rows are skipped
    (left as preset by the caller). Parameter order [x, y, z, amplitude, offset].

    ``aff`` is ``(n_channels, 4)`` ``[a00, a01, a10, a11]`` and ``res`` is
    ``(n_locs, n_channels, 2)`` sub-pixel ROI offsets, so that each channel is
    evaluated exactly where ``spline_3d_multichannel.cuh`` evaluates it - the
    shared shift mapped through that channel's affine, minus its ROI residual -
    and the x/y derivatives pick up the matching ``Aᵀ`` chain rule. Both reduce
    to the single-channel case at identity and zero.

    With ``mle`` True, ``bread`` (n, 5, 5) receives the Poisson Fisher matrix
    ``I = Σ g gᵀ / μ`` (its inverse is the MLE Cramer-Rao bound) and ``meat``
    is left untouched (weight 0). With ``mle`` False, the two matrices form the
    unweighted-least-squares sandwich covariance ``J⁻¹ M J⁻¹``: ``bread`` = the
    Gauss-Newton normal matrix ``J = Σ g gᵀ`` and ``meat`` = ``M = Σ μ g gᵀ``
    (Poisson pixel variance ``σ² = μ``). ``g = ∂μ/∂θ``.
    """
    n_channels, niz, niy, nix = (
        coeff.shape[0],
        coeff.shape[1],
        (coeff.shape[2]),
        coeff.shape[3],
    )
    n_locs = amp.shape[0]
    for m in numba.prange(n_locs):
        if not finite[m]:
            continue
        a = amp[m]
        o = offset[m]
        # bread accumulators (f*): Fisher when mle else Gauss-Newton normal J.
        f00 = f01 = f02 = f03 = f04 = 0.0
        f11 = f12 = f13 = f14 = 0.0
        f22 = f23 = f24 = 0.0
        f33 = f34 = 0.0
        f44 = 0.0
        # meat accumulators (s*): least-squares sandwich M = Σ μ g gᵀ (0 if mle).
        s00 = s01 = s02 = s03 = s04 = 0.0
        s11 = s12 = s13 = s14 = 0.0
        s22 = s23 = s24 = 0.0
        s33 = s34 = 0.0
        s44 = 0.0
        # z basis (one slice per localization)
        zc = z_eval[m]
        zi = int(np.floor(zc))
        zi = 0 if zi < 0 else (niz - 1 if zi > niz - 1 else zi)
        fz = zc - zi
        pz0, pz1, pz2, pz3 = 1.0, fz, fz * fz, fz * fz * fz
        dz1, dz2, dz3 = 1.0, 2.0 * fz, 3.0 * fz * fz
        for ch in range(n_channels):
            # This channel sees the shared lateral shift through its own affine
            # and sits on its own sub-pixel ROI offset - exactly as in
            # spline_3d_multichannel.cuh. Hoisted: constant over the box.
            a00 = aff[ch, 0]
            a01 = aff[ch, 1]
            a10 = aff[ch, 2]
            a11 = aff[ch, 3]
            sx = a00 * x_shift[m] + a01 * y_shift[m] + res[m, ch, 0]
            sy = a10 * x_shift[m] + a11 * y_shift[m] + res[m, ch, 1]
            for i in range(box):
                xco = i - sx
                xi = int(np.floor(xco))
                xi = 0 if xi < 0 else (nix - 1 if xi > nix - 1 else xi)
                fx = xco - xi
                px0, px1, px2, px3 = 1.0, fx, fx * fx, fx * fx * fx
                dx1, dx2, dx3 = 1.0, 2.0 * fx, 3.0 * fx * fx
                for j in range(box):
                    yco = j - sy
                    yi = int(np.floor(yco))
                    yi = 0 if yi < 0 else (niy - 1 if yi > niy - 1 else yi)
                    fy = yco - yi
                    py0, py1, py2, py3 = 1.0, fy, fy * fy, fy * fy * fy
                    dy1, dy2, dy3 = 1.0, 2.0 * fy, 3.0 * fy * fy
                    phi = gx = gy = gz = 0.0
                    for zp in range(4):
                        pzv = (
                            pz0
                            if zp == 0
                            else (
                                pz1 if zp == 1 else (pz2 if zp == 2 else pz3)
                            )
                        )
                        dzv = (
                            0.0
                            if zp == 0
                            else (
                                dz1 if zp == 1 else (dz2 if zp == 2 else dz3)
                            )
                        )
                        for yp in range(4):
                            pyv = (
                                py0
                                if yp == 0
                                else (
                                    py1
                                    if yp == 1
                                    else (py2 if yp == 2 else py3)
                                )
                            )
                            dyv = (
                                0.0
                                if yp == 0
                                else (
                                    dy1
                                    if yp == 1
                                    else (dy2 if yp == 2 else dy3)
                                )
                            )
                            for xp in range(4):
                                cf = coeff[ch, zi, yi, xi, zp, yp, xp]
                                pxv = (
                                    px0
                                    if xp == 0
                                    else (
                                        px1
                                        if xp == 1
                                        else (px2 if xp == 2 else px3)
                                    )
                                )
                                dxv = (
                                    0.0
                                    if xp == 0
                                    else (
                                        dx1
                                        if xp == 1
                                        else (dx2 if xp == 2 else dx3)
                                    )
                                )
                                phi += cf * pzv * pyv * pxv
                                gx += cf * pzv * pyv * dxv
                                gy += cf * pzv * dyv * pxv
                                gz += cf * dzv * pyv * pxv
                    mu = o + a * phi
                    # Huang et al.'s sCMOS shift: a pixel's readout
                    # variance adds to the model mean, so the Fisher
                    # weight becomes 1/(mu + var) and the least-
                    # squares sandwich meat becomes the true pixel
                    # variance.
                    if use_var:
                        mu += var[m, ch, j, i]
                    if mu < mu_floor:
                        mu = mu_floor
                    # bread weight wa (1/μ Fisher, else 1) and meat weight wb
                    # (μ for the least-squares sandwich, else unused).
                    if mle:
                        wa = 1.0 / mu
                        wb = 0.0
                    else:
                        wa = 1.0
                        wb = mu
                    # d(mu)/d(param); the CRLB diagonal is sign-invariant per
                    # parameter, so native-coordinate vs shift sign is irrelevant.
                    d0 = a * (a00 * gx + a10 * gy)
                    d1 = a * (a01 * gx + a11 * gy)
                    d2 = a * gz
                    d3 = phi
                    f00 += d0 * d0 * wa
                    f01 += d0 * d1 * wa
                    f02 += d0 * d2 * wa
                    f03 += d0 * d3 * wa
                    f04 += d0 * wa
                    f11 += d1 * d1 * wa
                    f12 += d1 * d2 * wa
                    f13 += d1 * d3 * wa
                    f14 += d1 * wa
                    f22 += d2 * d2 * wa
                    f23 += d2 * d3 * wa
                    f24 += d2 * wa
                    f33 += d3 * d3 * wa
                    f34 += d3 * wa
                    f44 += wa
                    s00 += d0 * d0 * wb
                    s01 += d0 * d1 * wb
                    s02 += d0 * d2 * wb
                    s03 += d0 * d3 * wb
                    s04 += d0 * wb
                    s11 += d1 * d1 * wb
                    s12 += d1 * d2 * wb
                    s13 += d1 * d3 * wb
                    s14 += d1 * wb
                    s22 += d2 * d2 * wb
                    s23 += d2 * d3 * wb
                    s24 += d2 * wb
                    s33 += d3 * d3 * wb
                    s34 += d3 * wb
                    s44 += wb
        bread[m, 0, 0] = f00
        bread[m, 0, 1] = bread[m, 1, 0] = f01
        bread[m, 0, 2] = bread[m, 2, 0] = f02
        bread[m, 0, 3] = bread[m, 3, 0] = f03
        bread[m, 0, 4] = bread[m, 4, 0] = f04
        bread[m, 1, 1] = f11
        bread[m, 1, 2] = bread[m, 2, 1] = f12
        bread[m, 1, 3] = bread[m, 3, 1] = f13
        bread[m, 1, 4] = bread[m, 4, 1] = f14
        bread[m, 2, 2] = f22
        bread[m, 2, 3] = bread[m, 3, 2] = f23
        bread[m, 2, 4] = bread[m, 4, 2] = f24
        bread[m, 3, 3] = f33
        bread[m, 3, 4] = bread[m, 4, 3] = f34
        bread[m, 4, 4] = f44
        if not mle:
            meat[m, 0, 0] = s00
            meat[m, 0, 1] = meat[m, 1, 0] = s01
            meat[m, 0, 2] = meat[m, 2, 0] = s02
            meat[m, 0, 3] = meat[m, 3, 0] = s03
            meat[m, 0, 4] = meat[m, 4, 0] = s04
            meat[m, 1, 1] = s11
            meat[m, 1, 2] = meat[m, 2, 1] = s12
            meat[m, 1, 3] = meat[m, 3, 1] = s13
            meat[m, 1, 4] = meat[m, 4, 1] = s14
            meat[m, 2, 2] = s22
            meat[m, 2, 3] = meat[m, 3, 2] = s23
            meat[m, 2, 4] = meat[m, 4, 2] = s24
            meat[m, 3, 3] = s33
            meat[m, 3, 4] = meat[m, 4, 3] = s34
            meat[m, 4, 4] = s44


@numba.njit(parallel=True, cache=True, fastmath=True)
def _spline_infomats_2d(
    coeff,
    box,
    amp,
    x_shift,
    y_shift,
    offset,
    finite,
    mu_floor,
    mle,
    var,
    use_var,
    bread,
    meat,
):
    """2D analogue of :func:`_spline_infomats_3d`. ``coeff`` is
    ``(n_channels, niy, nix, 4, 4)``; parameter order [x, y, amplitude, offset].
    """
    n_channels, niy, nix = coeff.shape[0], coeff.shape[1], coeff.shape[2]
    n_locs = amp.shape[0]
    for m in numba.prange(n_locs):
        if not finite[m]:
            continue
        a = amp[m]
        o = offset[m]
        # bread accumulators (f*): Fisher when mle else Gauss-Newton normal J.
        f00 = f01 = f02 = f03 = 0.0
        f11 = f12 = f13 = 0.0
        f22 = f23 = 0.0
        f33 = 0.0
        # meat accumulators (s*): least-squares sandwich M = Σ μ g gᵀ (0 if mle).
        s00 = s01 = s02 = s03 = 0.0
        s11 = s12 = s13 = 0.0
        s22 = s23 = 0.0
        s33 = 0.0
        for ch in range(n_channels):
            for i in range(box):
                xco = i - x_shift[m]
                xi = int(np.floor(xco))
                xi = 0 if xi < 0 else (nix - 1 if xi > nix - 1 else xi)
                fx = xco - xi
                px0, px1, px2, px3 = 1.0, fx, fx * fx, fx * fx * fx
                dx1, dx2, dx3 = 1.0, 2.0 * fx, 3.0 * fx * fx
                for j in range(box):
                    yco = j - y_shift[m]
                    yi = int(np.floor(yco))
                    yi = 0 if yi < 0 else (niy - 1 if yi > niy - 1 else yi)
                    fy = yco - yi
                    py0, py1, py2, py3 = 1.0, fy, fy * fy, fy * fy * fy
                    dy1, dy2, dy3 = 1.0, 2.0 * fy, 3.0 * fy * fy
                    phi = gx = gy = 0.0
                    for yp in range(4):
                        pyv = (
                            py0
                            if yp == 0
                            else (
                                py1 if yp == 1 else (py2 if yp == 2 else py3)
                            )
                        )
                        dyv = (
                            0.0
                            if yp == 0
                            else (
                                dy1 if yp == 1 else (dy2 if yp == 2 else dy3)
                            )
                        )
                        for xp in range(4):
                            cf = coeff[ch, yi, xi, yp, xp]
                            pxv = (
                                px0
                                if xp == 0
                                else (
                                    px1
                                    if xp == 1
                                    else (px2 if xp == 2 else px3)
                                )
                            )
                            dxv = (
                                0.0
                                if xp == 0
                                else (
                                    dx1
                                    if xp == 1
                                    else (dx2 if xp == 2 else dx3)
                                )
                            )
                            phi += cf * pyv * pxv
                            gx += cf * pyv * dxv
                            gy += cf * dyv * pxv
                    mu = o + a * phi
                    # Huang et al.'s sCMOS shift: a pixel's readout
                    # variance adds to the model mean, so the Fisher
                    # weight becomes 1/(mu + var) and the least-
                    # squares sandwich meat becomes the true pixel
                    # variance.
                    if use_var:
                        mu += var[m, ch, j, i]
                    if mu < mu_floor:
                        mu = mu_floor
                    if mle:
                        wa = 1.0 / mu
                        wb = 0.0
                    else:
                        wa = 1.0
                        wb = mu
                    d0, d1, d2 = a * gx, a * gy, phi
                    f00 += d0 * d0 * wa
                    f01 += d0 * d1 * wa
                    f02 += d0 * d2 * wa
                    f03 += d0 * wa
                    f11 += d1 * d1 * wa
                    f12 += d1 * d2 * wa
                    f13 += d1 * wa
                    f22 += d2 * d2 * wa
                    f23 += d2 * wa
                    f33 += wa
                    s00 += d0 * d0 * wb
                    s01 += d0 * d1 * wb
                    s02 += d0 * d2 * wb
                    s03 += d0 * wb
                    s11 += d1 * d1 * wb
                    s12 += d1 * d2 * wb
                    s13 += d1 * wb
                    s22 += d2 * d2 * wb
                    s23 += d2 * wb
                    s33 += wb
        bread[m, 0, 0] = f00
        bread[m, 0, 1] = bread[m, 1, 0] = f01
        bread[m, 0, 2] = bread[m, 2, 0] = f02
        bread[m, 0, 3] = bread[m, 3, 0] = f03
        bread[m, 1, 1] = f11
        bread[m, 1, 2] = bread[m, 2, 1] = f12
        bread[m, 1, 3] = bread[m, 3, 1] = f13
        bread[m, 2, 2] = f22
        bread[m, 2, 3] = bread[m, 3, 2] = f23
        bread[m, 3, 3] = f33
        if not mle:
            meat[m, 0, 0] = s00
            meat[m, 0, 1] = meat[m, 1, 0] = s01
            meat[m, 0, 2] = meat[m, 2, 0] = s02
            meat[m, 0, 3] = meat[m, 3, 0] = s03
            meat[m, 1, 1] = s11
            meat[m, 1, 2] = meat[m, 2, 1] = s12
            meat[m, 1, 3] = meat[m, 3, 1] = s13
            meat[m, 2, 2] = s22
            meat[m, 2, 3] = meat[m, 3, 2] = s23
            meat[m, 3, 3] = s33


@numba.njit(parallel=True, cache=True, fastmath=True)
def _spline_infomats_link_xyz_3d(
    coeff,
    aff,
    res,
    box,
    x_shift,
    y_shift,
    z_eval,
    photons,
    bg,
    finite,
    mu_floor,
    mle,
    var,
    use_var,
    bread,
    meat,
):
    """Per-localization information matrices for the photon-decoupled (link-XYZ)
    3D cubic-spline model. Parameter order
    ``[x, y, z, N_0..N_{c-1}, bg_0..bg_{c-1}]`` (P = 3 + 2*n_channels).

    Unlike :func:`_spline_infomats_3d` (shared amplitude/offset), each pixel of
    channel ``ch`` has ``mu = bg[ch] + N[ch] * phi_ch`` and a block-sparse
    gradient: the shared x/y/z columns scale by that channel's ``N[ch]``, while
    only channel ``ch``'s photon column (= phi) and background column (= 1) are
    non-zero. ``bread``/``meat`` play the same Fisher / least-squares-sandwich
    roles as in :func:`_spline_infomats_3d`. Rows preset by the caller (identity)
    for non-converged fits are left untouched; converged rows are zeroed and
    filled here. CRLB diagonals are invariant to the per-parameter gradient sign,
    so x/y/z use the unsigned spline derivative..

    NOTE (pre-existing): this samples every channel at the same
    ``x_shift``/``y_shift`` and ignores the calibration's per-channel affine,
    unlike the CUDA model, which chain-rules the shared shift through each
    channel's transform. May affect the fit's actual precision."""
    n_channels = coeff.shape[0]
    niz = coeff.shape[1]
    niy = coeff.shape[2]
    nix = coeff.shape[3]
    n_locs = x_shift.shape[0]
    n_params = 3 + 2 * n_channels
    for m in numba.prange(n_locs):
        if not finite[m]:
            continue
        for a_ in range(n_params):
            for b_ in range(n_params):
                bread[m, a_, b_] = 0.0
                meat[m, a_, b_] = 0.0
        g = np.zeros(n_params)
        zc = z_eval[m]
        zi = int(np.floor(zc))
        zi = 0 if zi < 0 else (niz - 1 if zi > niz - 1 else zi)
        fz = zc - zi
        pz0, pz1, pz2, pz3 = 1.0, fz, fz * fz, fz * fz * fz
        dz1, dz2, dz3 = 1.0, 2.0 * fz, 3.0 * fz * fz
        for ch in range(n_channels):
            nc = photons[m, ch]
            bgc = bg[m, ch]
            # per-channel affine + ROI residual, as in _spline_infomats_3d
            a00 = aff[ch, 0]
            a01 = aff[ch, 1]
            a10 = aff[ch, 2]
            a11 = aff[ch, 3]
            sx = a00 * x_shift[m] + a01 * y_shift[m] + res[m, ch, 0]
            sy = a10 * x_shift[m] + a11 * y_shift[m] + res[m, ch, 1]
            for i in range(box):
                xco = i - sx
                xi = int(np.floor(xco))
                xi = 0 if xi < 0 else (nix - 1 if xi > nix - 1 else xi)
                fx = xco - xi
                px0, px1, px2, px3 = 1.0, fx, fx * fx, fx * fx * fx
                dx1, dx2, dx3 = 1.0, 2.0 * fx, 3.0 * fx * fx
                for j in range(box):
                    yco = j - sy
                    yi = int(np.floor(yco))
                    yi = 0 if yi < 0 else (niy - 1 if yi > niy - 1 else yi)
                    fy = yco - yi
                    py0, py1, py2, py3 = 1.0, fy, fy * fy, fy * fy * fy
                    dy1, dy2, dy3 = 1.0, 2.0 * fy, 3.0 * fy * fy
                    phi = gx = gy = gz = 0.0
                    for zp in range(4):
                        pzv = (
                            pz0
                            if zp == 0
                            else (
                                pz1 if zp == 1 else (pz2 if zp == 2 else pz3)
                            )
                        )
                        dzv = (
                            0.0
                            if zp == 0
                            else (
                                dz1 if zp == 1 else (dz2 if zp == 2 else dz3)
                            )
                        )
                        for yp in range(4):
                            pyv = (
                                py0
                                if yp == 0
                                else (
                                    py1
                                    if yp == 1
                                    else (py2 if yp == 2 else py3)
                                )
                            )
                            dyv = (
                                0.0
                                if yp == 0
                                else (
                                    dy1
                                    if yp == 1
                                    else (dy2 if yp == 2 else dy3)
                                )
                            )
                            for xp in range(4):
                                cf = coeff[ch, zi, yi, xi, zp, yp, xp]
                                pxv = (
                                    px0
                                    if xp == 0
                                    else (
                                        px1
                                        if xp == 1
                                        else (px2 if xp == 2 else px3)
                                    )
                                )
                                dxv = (
                                    0.0
                                    if xp == 0
                                    else (
                                        dx1
                                        if xp == 1
                                        else (dx2 if xp == 2 else dx3)
                                    )
                                )
                                phi += cf * pzv * pyv * pxv
                                gx += cf * pzv * pyv * dxv
                                gy += cf * pzv * dyv * pxv
                                gz += cf * dzv * pyv * pxv
                    mu = bgc + nc * phi
                    # Huang et al.'s sCMOS shift: a pixel's readout
                    # variance adds to the model mean, so the Fisher
                    # weight becomes 1/(mu + var) and the least-
                    # squares sandwich meat becomes the true pixel
                    # variance.
                    if use_var:
                        mu += var[m, ch, j, i]
                    if mu < mu_floor:
                        mu = mu_floor
                    if mle:
                        wa = 1.0 / mu
                        wb = 0.0
                    else:
                        wa = 1.0
                        wb = mu
                    for t in range(n_params):
                        g[t] = 0.0
                    g[0] = nc * (a00 * gx + a10 * gy)
                    g[1] = nc * (a01 * gx + a11 * gy)
                    g[2] = nc * gz
                    g[3 + ch] = phi
                    g[3 + n_channels + ch] = 1.0
                    for a_ in range(n_params):
                        ga = g[a_]
                        if ga == 0.0:
                            continue
                        for b_ in range(a_, n_params):
                            v = ga * g[b_]
                            bread[m, a_, b_] += v * wa
                            if not mle:
                                meat[m, a_, b_] += v * wb
        for a_ in range(n_params):
            for b_ in range(a_ + 1, n_params):
                bread[m, b_, a_] = bread[m, a_, b_]
                if not mle:
                    meat[m, b_, a_] = meat[m, a_, b_]


# ---------------------------------------------------------------------------
# CUDA (numba.cuda) spline CRLB
# ---------------------------------------------------------------------------

_SPLINE_CRLB_PINV_RCOND = 1e-15
_SPLINE_CRLB_JACOBI_SWEEPS = 30
_SPLINE_CRLB_JACOBI_TOL = 1e-30

# Largest link-XYZ parameter count. Device-local arrays need a compile-time
# constant shape, so one kernel is compiled at this size for every channel count
# and indexes the leading ``n_params`` rows and columns.
_LINK_XYZ_MAX_P = 3 + 2 * _LINK_XYZ_MAX_CHANNELS

_SPLINE_CRLB_CUDA_THREADS = 128
# Device working set (inputs + outputs) per kernel launch. Chunking bounds
# device memory and paces the progress callback; unlike the host path there is
# no (n, P, P) matrix stack to budget for, so the chunks are large.
_SPLINE_CRLB_CUDA_CHUNK_BYTES = 256 << 20
# Ceiling on one launch, so a display-attached GPU cannot trip a watchdog
# timeout on a single enormous grid.
_SPLINE_CRLB_CUDA_MAX_ROWS = 4_000_000


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------


@cuda.jit(device=True)
def _sym_pinv_device(a, n, v, lam) -> int:
    """Moore-Penrose pseudo-inverse of the symmetric ``a[:n, :n]``, in place.

    Cyclic Jacobi eigendecomposition ``a = V Λ Vᵀ``, then
    ``a⁺ = Σ_{|λ| > rcond·max|λ|} v vᵀ / λ``. ``v`` and ``lam`` are per-thread
    scratch of at least ``(n, n)`` and ``(n,)``. Returns 0 on success, 1 if the
    sweeps did not converge (the caller then hands that localization back to the
    host).

    A plain inverse would be cheaper, but the information matrix is genuinely
    rank deficient whenever the model's parameters are locally collinear - a PSF
    whose z-dependence is a pure rescaling makes the z and amplitude columns
    proportional, and then only the truncating pseudo-inverse gives a finite
    answer. The CPU path uses ``numpy.linalg.pinv``.
    """
    for p in range(n):
        for q in range(n):
            v[p, q] = 1.0 if p == q else 0.0
    converged = False
    for _ in range(_SPLINE_CRLB_JACOBI_SWEEPS):
        off = 0.0
        fro = 0.0
        for p in range(n):
            fro += a[p, p] * a[p, p]
            for q in range(p + 1, n):
                off += a[p, q] * a[p, q]
        fro += 2.0 * off
        if off <= _SPLINE_CRLB_JACOBI_TOL * fro:
            converged = True
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = a[p, q]
                if apq == 0.0:
                    continue
                # Rotation that annihilates a[p, q]; the smaller root of
                # t² + 2θt - 1 = 0 keeps the rotation angle below 45°.
                theta = (a[q, q] - a[p, p]) / (2.0 * apq)
                sgn = 1.0 if theta >= 0.0 else -1.0
                t = sgn / (abs(theta) + math.sqrt(theta * theta + 1.0))
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                for k in range(n):  # a <- a J
                    akp = a[k, p]
                    akq = a[k, q]
                    a[k, p] = c * akp - s * akq
                    a[k, q] = s * akp + c * akq
                for k in range(n):  # a <- Jᵀ a
                    apk = a[p, k]
                    aqk = a[q, k]
                    a[p, k] = c * apk - s * aqk
                    a[q, k] = s * apk + c * aqk
                for k in range(n):  # v <- v J
                    vkp = v[k, p]
                    vkq = v[k, q]
                    v[k, p] = c * vkp - s * vkq
                    v[k, q] = s * vkp + c * vkq
    if not converged:
        return 1
    biggest = 0.0
    for p in range(n):
        lam[p] = a[p, p]
        if abs(lam[p]) > biggest:
            biggest = abs(lam[p])
    cutoff = _SPLINE_CRLB_PINV_RCOND * biggest
    for p in range(n):
        for q in range(p, n):
            acc = 0.0
            for k in range(n):
                if abs(lam[k]) > cutoff:
                    acc += v[p, k] * v[q, k] / lam[k]
            a[p, q] = acc
            a[q, p] = acc
    return 0


@cuda.jit(device=True)
def _crlb_diag_device(ainv, meat, n, mle, out, m) -> None:
    """Write the covariance diagonal of localization ``m`` into ``out``.

    With ``mle`` the covariance is the inverse Fisher matrix itself, so the
    diagonal is read straight off ``ainv``. Otherwise it is the unweighted
    least-squares sandwich ``diag(J⁻¹ M J⁻¹)``, evaluated using the symmetry of
    ``J⁻¹`` so only its ``p``-th row is needed per parameter.
    """
    for p in range(n):
        if mle:
            out[m, p] = ainv[p, p]
        else:
            s = 0.0
            for k in range(n):
                t = 0.0
                for ll in range(n):
                    t += meat[k, ll] * ainv[p, ll]
                s += ainv[p, k] * t
            out[m, p] = s


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


@cuda.jit(cache=True)
def _spline_crlb_3d_kernel(
    coeff,
    aff,
    res,
    box,
    amp,
    x_shift,
    y_shift,
    z_eval,
    offset,
    finite,
    mu_floor,
    mle,
    var,
    use_var,
    crlb,
    status,
) -> None:
    """One thread per localization: covariance diagonal of the 3D cubic-spline
    model, parameter order [x, y, z, amplitude, offset]. CUDA transcription of
    :func:`_spline_infomats_3d` with the solve fused in. ``coeff`` is
    ``(n_channels, niz, niy, nix, 4, 4, 4)``; ``crlb`` is ``(n_locs, 5)`` and
    ``status`` ``(n_locs,)`` (0 ok, 1 not positive definite).

    ``aff`` is ``(n_channels, 4)`` ``[a00, a01, a10, a11]`` and ``res`` is
    ``(n_locs, n_channels, 2)`` sub-pixel ROI offsets, exactly as in
    :func:`_spline_infomats_3d` - each channel is evaluated at the shared shift
    mapped through its own affine, minus its ROI residual, and the x/y
    derivatives pick up the matching ``Aᵀ`` chain rule. Both reduce to the
    single-channel case at identity and zero.
    """
    m = cuda.grid(1)
    if m >= amp.shape[0]:
        return
    status[m] = 0
    if finite[m] == 0:
        # Skipped rows are NaN-masked on the host; write a definite value so no
        # uninitialized device memory is ever copied back.
        for p in range(5):
            crlb[m, p] = 0.0
        return
    n_channels = coeff.shape[0]
    niz = coeff.shape[1]
    niy = coeff.shape[2]
    nix = coeff.shape[3]
    a = amp[m]
    o = offset[m]
    # bread accumulators (f*): Fisher when mle else Gauss-Newton normal J.
    f00 = f01 = f02 = f03 = f04 = 0.0
    f11 = f12 = f13 = f14 = 0.0
    f22 = f23 = f24 = 0.0
    f33 = f34 = 0.0
    f44 = 0.0
    # meat accumulators (s*): least-squares sandwich M = Σ μ g gᵀ (0 if mle).
    s00 = s01 = s02 = s03 = s04 = 0.0
    s11 = s12 = s13 = s14 = 0.0
    s22 = s23 = s24 = 0.0
    s33 = s34 = 0.0
    s44 = 0.0
    # z basis (one slice per localization)
    zc = z_eval[m]
    zi = int(math.floor(zc))
    zi = 0 if zi < 0 else (niz - 1 if zi > niz - 1 else zi)
    fz = zc - zi
    pz0, pz1, pz2, pz3 = 1.0, fz, fz * fz, fz * fz * fz
    dz1, dz2, dz3 = 1.0, 2.0 * fz, 3.0 * fz * fz
    for ch in range(n_channels):
        # This channel sees the shared lateral shift through its own affine
        # and sits on its own sub-pixel ROI offset - exactly as in
        # spline_3d_multichannel.cuh. Hoisted: constant over the box.
        a00 = aff[ch, 0]
        a01 = aff[ch, 1]
        a10 = aff[ch, 2]
        a11 = aff[ch, 3]
        sx = a00 * x_shift[m] + a01 * y_shift[m] + res[m, ch, 0]
        sy = a10 * x_shift[m] + a11 * y_shift[m] + res[m, ch, 1]
        for i in range(box):
            xco = i - sx
            xi = int(math.floor(xco))
            xi = 0 if xi < 0 else (nix - 1 if xi > nix - 1 else xi)
            fx = xco - xi
            px0, px1, px2, px3 = 1.0, fx, fx * fx, fx * fx * fx
            dx1, dx2, dx3 = 1.0, 2.0 * fx, 3.0 * fx * fx
            for j in range(box):
                yco = j - sy
                yi = int(math.floor(yco))
                yi = 0 if yi < 0 else (niy - 1 if yi > niy - 1 else yi)
                fy = yco - yi
                py0, py1, py2, py3 = 1.0, fy, fy * fy, fy * fy * fy
                dy1, dy2, dy3 = 1.0, 2.0 * fy, 3.0 * fy * fy
                phi = gx = gy = gz = 0.0
                for zp in range(4):
                    pzv = (
                        pz0
                        if zp == 0
                        else (pz1 if zp == 1 else (pz2 if zp == 2 else pz3))
                    )
                    dzv = (
                        0.0
                        if zp == 0
                        else (dz1 if zp == 1 else (dz2 if zp == 2 else dz3))
                    )
                    for yp in range(4):
                        pyv = (
                            py0
                            if yp == 0
                            else (
                                py1 if yp == 1 else (py2 if yp == 2 else py3)
                            )
                        )
                        dyv = (
                            0.0
                            if yp == 0
                            else (
                                dy1 if yp == 1 else (dy2 if yp == 2 else dy3)
                            )
                        )
                        for xp in range(4):
                            cf = coeff[ch, zi, yi, xi, zp, yp, xp]
                            pxv = (
                                px0
                                if xp == 0
                                else (
                                    px1
                                    if xp == 1
                                    else (px2 if xp == 2 else px3)
                                )
                            )
                            dxv = (
                                0.0
                                if xp == 0
                                else (
                                    dx1
                                    if xp == 1
                                    else (dx2 if xp == 2 else dx3)
                                )
                            )
                            phi += cf * pzv * pyv * pxv
                            gx += cf * pzv * pyv * dxv
                            gy += cf * pzv * dyv * pxv
                            gz += cf * dzv * pyv * pxv
                mu = o + a * phi
                # Huang et al.'s sCMOS shift: a pixel's readout
                # variance adds to the model mean, so the Fisher
                # weight becomes 1/(mu + var) and the least-
                # squares sandwich meat becomes the true pixel
                # variance.
                if use_var:
                    mu += var[m, ch, j, i]
                if mu < mu_floor:
                    mu = mu_floor
                # bread weight wa (1/μ Fisher, else 1) and meat weight wb
                # (μ for the least-squares sandwich, else unused).
                if mle:
                    wa = 1.0 / mu
                    wb = 0.0
                else:
                    wa = 1.0
                    wb = mu
                # d(mu)/d(param); the CRLB diagonal is sign-invariant per
                # parameter, so native-coordinate vs shift sign is irrelevant.
                d0 = a * (a00 * gx + a10 * gy)
                d1 = a * (a01 * gx + a11 * gy)
                d2 = a * gz
                d3 = phi
                f00 += d0 * d0 * wa
                f01 += d0 * d1 * wa
                f02 += d0 * d2 * wa
                f03 += d0 * d3 * wa
                f04 += d0 * wa
                f11 += d1 * d1 * wa
                f12 += d1 * d2 * wa
                f13 += d1 * d3 * wa
                f14 += d1 * wa
                f22 += d2 * d2 * wa
                f23 += d2 * d3 * wa
                f24 += d2 * wa
                f33 += d3 * d3 * wa
                f34 += d3 * wa
                f44 += wa
                s00 += d0 * d0 * wb
                s01 += d0 * d1 * wb
                s02 += d0 * d2 * wb
                s03 += d0 * d3 * wb
                s04 += d0 * wb
                s11 += d1 * d1 * wb
                s12 += d1 * d2 * wb
                s13 += d1 * d3 * wb
                s14 += d1 * wb
                s22 += d2 * d2 * wb
                s23 += d2 * d3 * wb
                s24 += d2 * wb
                s33 += d3 * d3 * wb
                s34 += d3 * wb
                s44 += wb
    bread = cuda.local.array((5, 5), numba.float64)
    meat = cuda.local.array((5, 5), numba.float64)
    vecs = cuda.local.array((5, 5), numba.float64)
    lam = cuda.local.array(5, numba.float64)
    bread[0, 0] = f00
    bread[0, 1] = bread[1, 0] = f01
    bread[0, 2] = bread[2, 0] = f02
    bread[0, 3] = bread[3, 0] = f03
    bread[0, 4] = bread[4, 0] = f04
    bread[1, 1] = f11
    bread[1, 2] = bread[2, 1] = f12
    bread[1, 3] = bread[3, 1] = f13
    bread[1, 4] = bread[4, 1] = f14
    bread[2, 2] = f22
    bread[2, 3] = bread[3, 2] = f23
    bread[2, 4] = bread[4, 2] = f24
    bread[3, 3] = f33
    bread[3, 4] = bread[4, 3] = f34
    bread[4, 4] = f44
    meat[0, 0] = s00
    meat[0, 1] = meat[1, 0] = s01
    meat[0, 2] = meat[2, 0] = s02
    meat[0, 3] = meat[3, 0] = s03
    meat[0, 4] = meat[4, 0] = s04
    meat[1, 1] = s11
    meat[1, 2] = meat[2, 1] = s12
    meat[1, 3] = meat[3, 1] = s13
    meat[1, 4] = meat[4, 1] = s14
    meat[2, 2] = s22
    meat[2, 3] = meat[3, 2] = s23
    meat[2, 4] = meat[4, 2] = s24
    meat[3, 3] = s33
    meat[3, 4] = meat[4, 3] = s34
    meat[4, 4] = s44
    if _sym_pinv_device(bread, 5, vecs, lam) != 0:
        status[m] = 1
        for p in range(5):
            crlb[m, p] = 0.0
        return
    _crlb_diag_device(bread, meat, 5, mle, crlb, m)


@cuda.jit(cache=True)
def _spline_crlb_2d_kernel(
    coeff,
    box,
    amp,
    x_shift,
    y_shift,
    offset,
    finite,
    mu_floor,
    mle,
    var,
    use_var,
    crlb,
    status,
) -> None:
    """2D analogue of :func:`_spline_crlb_3d_kernel`, transcribing
    :func:`_spline_infomats_2d`. ``coeff`` is
    ``(n_channels, niy, nix, 4, 4)``; parameter order [x, y, amplitude, offset].
    """
    m = cuda.grid(1)
    if m >= amp.shape[0]:
        return
    status[m] = 0
    if finite[m] == 0:
        for p in range(4):
            crlb[m, p] = 0.0
        return
    n_channels = coeff.shape[0]
    niy = coeff.shape[1]
    nix = coeff.shape[2]
    a = amp[m]
    o = offset[m]
    # bread accumulators (f*): Fisher when mle else Gauss-Newton normal J.
    f00 = f01 = f02 = f03 = 0.0
    f11 = f12 = f13 = 0.0
    f22 = f23 = 0.0
    f33 = 0.0
    # meat accumulators (s*): least-squares sandwich M = Σ μ g gᵀ (0 if mle).
    s00 = s01 = s02 = s03 = 0.0
    s11 = s12 = s13 = 0.0
    s22 = s23 = 0.0
    s33 = 0.0
    for ch in range(n_channels):
        for i in range(box):
            xco = i - x_shift[m]
            xi = int(math.floor(xco))
            xi = 0 if xi < 0 else (nix - 1 if xi > nix - 1 else xi)
            fx = xco - xi
            px0, px1, px2, px3 = 1.0, fx, fx * fx, fx * fx * fx
            dx1, dx2, dx3 = 1.0, 2.0 * fx, 3.0 * fx * fx
            for j in range(box):
                yco = j - y_shift[m]
                yi = int(math.floor(yco))
                yi = 0 if yi < 0 else (niy - 1 if yi > niy - 1 else yi)
                fy = yco - yi
                py0, py1, py2, py3 = 1.0, fy, fy * fy, fy * fy * fy
                dy1, dy2, dy3 = 1.0, 2.0 * fy, 3.0 * fy * fy
                phi = gx = gy = 0.0
                for yp in range(4):
                    pyv = (
                        py0
                        if yp == 0
                        else (py1 if yp == 1 else (py2 if yp == 2 else py3))
                    )
                    dyv = (
                        0.0
                        if yp == 0
                        else (dy1 if yp == 1 else (dy2 if yp == 2 else dy3))
                    )
                    for xp in range(4):
                        cf = coeff[ch, yi, xi, yp, xp]
                        pxv = (
                            px0
                            if xp == 0
                            else (
                                px1 if xp == 1 else (px2 if xp == 2 else px3)
                            )
                        )
                        dxv = (
                            0.0
                            if xp == 0
                            else (
                                dx1 if xp == 1 else (dx2 if xp == 2 else dx3)
                            )
                        )
                        phi += cf * pyv * pxv
                        gx += cf * pyv * dxv
                        gy += cf * dyv * pxv
                mu = o + a * phi
                # Huang et al.'s sCMOS shift: a pixel's readout
                # variance adds to the model mean, so the Fisher
                # weight becomes 1/(mu + var) and the least-
                # squares sandwich meat becomes the true pixel
                # variance.
                if use_var:
                    mu += var[m, ch, j, i]
                if mu < mu_floor:
                    mu = mu_floor
                if mle:
                    wa = 1.0 / mu
                    wb = 0.0
                else:
                    wa = 1.0
                    wb = mu
                d0, d1, d2 = a * gx, a * gy, phi
                f00 += d0 * d0 * wa
                f01 += d0 * d1 * wa
                f02 += d0 * d2 * wa
                f03 += d0 * wa
                f11 += d1 * d1 * wa
                f12 += d1 * d2 * wa
                f13 += d1 * wa
                f22 += d2 * d2 * wa
                f23 += d2 * wa
                f33 += wa
                s00 += d0 * d0 * wb
                s01 += d0 * d1 * wb
                s02 += d0 * d2 * wb
                s03 += d0 * wb
                s11 += d1 * d1 * wb
                s12 += d1 * d2 * wb
                s13 += d1 * wb
                s22 += d2 * d2 * wb
                s23 += d2 * wb
                s33 += wb
    bread = cuda.local.array((4, 4), numba.float64)
    meat = cuda.local.array((4, 4), numba.float64)
    vecs = cuda.local.array((4, 4), numba.float64)
    lam = cuda.local.array(4, numba.float64)
    bread[0, 0] = f00
    bread[0, 1] = bread[1, 0] = f01
    bread[0, 2] = bread[2, 0] = f02
    bread[0, 3] = bread[3, 0] = f03
    bread[1, 1] = f11
    bread[1, 2] = bread[2, 1] = f12
    bread[1, 3] = bread[3, 1] = f13
    bread[2, 2] = f22
    bread[2, 3] = bread[3, 2] = f23
    bread[3, 3] = f33
    meat[0, 0] = s00
    meat[0, 1] = meat[1, 0] = s01
    meat[0, 2] = meat[2, 0] = s02
    meat[0, 3] = meat[3, 0] = s03
    meat[1, 1] = s11
    meat[1, 2] = meat[2, 1] = s12
    meat[1, 3] = meat[3, 1] = s13
    meat[2, 2] = s22
    meat[2, 3] = meat[3, 2] = s23
    meat[3, 3] = s33
    if _sym_pinv_device(bread, 4, vecs, lam) != 0:
        status[m] = 1
        for p in range(4):
            crlb[m, p] = 0.0
        return
    _crlb_diag_device(bread, meat, 4, mle, crlb, m)


@cuda.jit(cache=True)
def _spline_crlb_link_xyz_kernel(
    coeff,
    aff,
    res,
    box,
    x_shift,
    y_shift,
    z_eval,
    photons,
    bg,
    finite,
    mu_floor,
    mle,
    var,
    use_var,
    crlb,
    status,
) -> None:
    """One thread per localization: covariance diagonal of the photon-decoupled
    (link-XYZ) 3D cubic-spline model, parameter order
    ``[x, y, z, N_0..N_{c-1}, bg_0..bg_{c-1}]``. CUDA transcription of
    :func:`_spline_infomats_link_xyz_3d` with the solve fused in. ``aff`` and
    ``res`` carry the same per-channel affine and ROI residual as in
    :func:`_spline_crlb_3d_kernel`.

    The CPU kernel accumulates through a dense ``n_params``-long gradient
    vector; here the block sparsity is spelled out instead. Each pixel touches
    only 15 distinct matrix entries - the six shared x/y/z ones plus nine that
    belong to its own channel - so all of them live in registers, and the nine
    channel-local ones are written out once per channel. The summation order is
    the CPU's, so the two agree to rounding.
    """
    m = cuda.grid(1)
    if m >= x_shift.shape[0]:
        return
    n_channels = coeff.shape[0]
    n_params = 3 + 2 * n_channels
    status[m] = 0
    if finite[m] == 0:
        for p in range(n_params):
            crlb[m, p] = 0.0
        return
    niz = coeff.shape[1]
    niy = coeff.shape[2]
    nix = coeff.shape[3]

    bread = cuda.local.array((_LINK_XYZ_MAX_P, _LINK_XYZ_MAX_P), numba.float64)
    meat = cuda.local.array((_LINK_XYZ_MAX_P, _LINK_XYZ_MAX_P), numba.float64)
    vecs = cuda.local.array((_LINK_XYZ_MAX_P, _LINK_XYZ_MAX_P), numba.float64)
    lam = cuda.local.array(_LINK_XYZ_MAX_P, numba.float64)
    # The cross-channel photon/background blocks are structurally zero (a pixel
    # belongs to exactly one channel), so clear both matrices once and then fill
    # only the blocks that are actually touched.
    for p in range(n_params):
        for q in range(n_params):
            bread[p, q] = 0.0
            meat[p, q] = 0.0

    # Shared x/y/z block, accumulated across every channel.
    xx = xy = xz = yy = yz = zz = 0.0
    sxx = sxy = sxz = syy = syz = szz = 0.0
    zc = z_eval[m]
    zi = int(math.floor(zc))
    zi = 0 if zi < 0 else (niz - 1 if zi > niz - 1 else zi)
    fz = zc - zi
    pz0, pz1, pz2, pz3 = 1.0, fz, fz * fz, fz * fz * fz
    dz1, dz2, dz3 = 1.0, 2.0 * fz, 3.0 * fz * fz
    for ch in range(n_channels):
        nc = photons[m, ch]
        bgc = bg[m, ch]
        # This channel's own photon (n) and background (b) entries.
        bxn = byn = bzn = bxb = byb = bzb = bnn = bnb = bbb = 0.0
        sxn = syn = szn = sxb = syb = szb = snn = snb = sbb = 0.0
        # per-channel affine + ROI residual, as in _spline_crlb_3d_kernel
        a00 = aff[ch, 0]
        a01 = aff[ch, 1]
        a10 = aff[ch, 2]
        a11 = aff[ch, 3]
        sx = a00 * x_shift[m] + a01 * y_shift[m] + res[m, ch, 0]
        sy = a10 * x_shift[m] + a11 * y_shift[m] + res[m, ch, 1]
        for i in range(box):
            xco = i - sx
            xi = int(math.floor(xco))
            xi = 0 if xi < 0 else (nix - 1 if xi > nix - 1 else xi)
            fx = xco - xi
            px0, px1, px2, px3 = 1.0, fx, fx * fx, fx * fx * fx
            dx1, dx2, dx3 = 1.0, 2.0 * fx, 3.0 * fx * fx
            for j in range(box):
                yco = j - sy
                yi = int(math.floor(yco))
                yi = 0 if yi < 0 else (niy - 1 if yi > niy - 1 else yi)
                fy = yco - yi
                py0, py1, py2, py3 = 1.0, fy, fy * fy, fy * fy * fy
                dy1, dy2, dy3 = 1.0, 2.0 * fy, 3.0 * fy * fy
                phi = gx = gy = gz = 0.0
                for zp in range(4):
                    pzv = (
                        pz0
                        if zp == 0
                        else (pz1 if zp == 1 else (pz2 if zp == 2 else pz3))
                    )
                    dzv = (
                        0.0
                        if zp == 0
                        else (dz1 if zp == 1 else (dz2 if zp == 2 else dz3))
                    )
                    for yp in range(4):
                        pyv = (
                            py0
                            if yp == 0
                            else (
                                py1 if yp == 1 else (py2 if yp == 2 else py3)
                            )
                        )
                        dyv = (
                            0.0
                            if yp == 0
                            else (
                                dy1 if yp == 1 else (dy2 if yp == 2 else dy3)
                            )
                        )
                        for xp in range(4):
                            cf = coeff[ch, zi, yi, xi, zp, yp, xp]
                            pxv = (
                                px0
                                if xp == 0
                                else (
                                    px1
                                    if xp == 1
                                    else (px2 if xp == 2 else px3)
                                )
                            )
                            dxv = (
                                0.0
                                if xp == 0
                                else (
                                    dx1
                                    if xp == 1
                                    else (dx2 if xp == 2 else dx3)
                                )
                            )
                            phi += cf * pzv * pyv * pxv
                            gx += cf * pzv * pyv * dxv
                            gy += cf * pzv * dyv * pxv
                            gz += cf * dzv * pyv * pxv
                mu = bgc + nc * phi
                # Huang et al.'s sCMOS shift: a pixel's readout
                # variance adds to the model mean, so the Fisher
                # weight becomes 1/(mu + var) and the least-
                # squares sandwich meat becomes the true pixel
                # variance.
                if use_var:
                    mu += var[m, ch, j, i]
                if mu < mu_floor:
                    mu = mu_floor
                if mle:
                    wa = 1.0 / mu
                    wb = 0.0
                else:
                    wa = 1.0
                    wb = mu
                # Gradient columns: x/y/z scale with this channel's photons, the
                # photon column is phi and the background column is 1. x/y also
                # pick up the channel affine's Aᵀ chain rule.
                d0 = nc * (a00 * gx + a10 * gy)
                d1 = nc * (a01 * gx + a11 * gy)
                d2 = nc * gz
                xx += d0 * d0 * wa
                xy += d0 * d1 * wa
                xz += d0 * d2 * wa
                yy += d1 * d1 * wa
                yz += d1 * d2 * wa
                zz += d2 * d2 * wa
                bxn += d0 * phi * wa
                byn += d1 * phi * wa
                bzn += d2 * phi * wa
                bxb += d0 * wa
                byb += d1 * wa
                bzb += d2 * wa
                bnn += phi * phi * wa
                bnb += phi * wa
                bbb += wa
                sxx += d0 * d0 * wb
                sxy += d0 * d1 * wb
                sxz += d0 * d2 * wb
                syy += d1 * d1 * wb
                syz += d1 * d2 * wb
                szz += d2 * d2 * wb
                sxn += d0 * phi * wb
                syn += d1 * phi * wb
                szn += d2 * phi * wb
                sxb += d0 * wb
                syb += d1 * wb
                szb += d2 * wb
                snn += phi * phi * wb
                snb += phi * wb
                sbb += wb
        cn = 3 + ch
        cb = 3 + n_channels + ch
        bread[0, cn] = bread[cn, 0] = bxn
        bread[1, cn] = bread[cn, 1] = byn
        bread[2, cn] = bread[cn, 2] = bzn
        bread[0, cb] = bread[cb, 0] = bxb
        bread[1, cb] = bread[cb, 1] = byb
        bread[2, cb] = bread[cb, 2] = bzb
        bread[cn, cn] = bnn
        bread[cn, cb] = bread[cb, cn] = bnb
        bread[cb, cb] = bbb
        meat[0, cn] = meat[cn, 0] = sxn
        meat[1, cn] = meat[cn, 1] = syn
        meat[2, cn] = meat[cn, 2] = szn
        meat[0, cb] = meat[cb, 0] = sxb
        meat[1, cb] = meat[cb, 1] = syb
        meat[2, cb] = meat[cb, 2] = szb
        meat[cn, cn] = snn
        meat[cn, cb] = meat[cb, cn] = snb
        meat[cb, cb] = sbb
    bread[0, 0] = xx
    bread[0, 1] = bread[1, 0] = xy
    bread[0, 2] = bread[2, 0] = xz
    bread[1, 1] = yy
    bread[1, 2] = bread[2, 1] = yz
    bread[2, 2] = zz
    meat[0, 0] = sxx
    meat[0, 1] = meat[1, 0] = sxy
    meat[0, 2] = meat[2, 0] = sxz
    meat[1, 1] = syy
    meat[1, 2] = meat[2, 1] = syz
    meat[2, 2] = szz
    if _sym_pinv_device(bread, n_params, vecs, lam) != 0:
        status[m] = 1
        for p in range(n_params):
            crlb[m, p] = 0.0
        return
    _crlb_diag_device(bread, meat, n_params, mle, crlb, m)


# ---------------------------------------------------------------------------
# CUDA CRLB host drivers (launch the kernels above)
# ---------------------------------------------------------------------------


def _require_crlb_cuda() -> None:
    if not CUDA_AVAILABLE:
        raise RuntimeError(
            "GPU spline CRLB requested but no CUDA-capable GPU is available."
        )


def _crlb_chunk_rows(bytes_per_row: int) -> int:
    """Localizations per kernel launch for a given per-row device footprint."""
    rows = max(1024, _SPLINE_CRLB_CUDA_CHUNK_BYTES // max(bytes_per_row, 1))
    return int(min(rows, _SPLINE_CRLB_CUDA_MAX_ROWS))


def _spline_crlb_cuda(
    coeff: np.ndarray,
    aff: np.ndarray,
    res: np.ndarray,
    box: int,
    amplitude: lib.FloatArray1D,
    x_shift: lib.FloatArray1D,
    y_shift: lib.FloatArray1D,
    z_eval: lib.FloatArray1D | None,
    offset: lib.FloatArray1D,
    finite: np.ndarray,
    mu_floor: float,
    mle: bool,
    progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    variance: lib.FloatArray4D | None = None,
) -> tuple[lib.FloatArray2D, np.ndarray]:
    """Covariance diagonal of the shared-amplitude spline models on the GPU.

    Array-in / array-out counterpart of :func:`_spline_crlb_cpu`; the
    caller owns the calibration parsing and the NaN masking. ``z_eval`` None
    selects the 2D model, which has no channel geometry and so ignores ``aff``
    and ``res``.

    Returns
    -------
    crlb : lib.FloatArray2D
        ``(n_locs, 4 or 5)`` raw variances.
    failed : np.ndarray
        Boolean mask of localizations whose information matrix the device could
        not diagonalize. Their ``crlb`` rows are meaningless and the caller is
        expected to recompute them on the CPU.
    """
    _require_crlb_cuda()
    is_3d = z_eval is not None
    n_params = 5 if is_3d else 4
    n_locs = len(amplitude)
    crlb = np.empty((n_locs, n_params), dtype=np.float64)
    failed = np.zeros(n_locs, dtype=bool)
    if n_locs == 0:
        return crlb, failed

    amplitude = np.ascontiguousarray(amplitude, dtype=np.float64)
    x_shift = np.ascontiguousarray(x_shift, dtype=np.float64)
    y_shift = np.ascontiguousarray(y_shift, dtype=np.float64)
    offset = np.ascontiguousarray(offset, dtype=np.float64)
    if is_3d:
        z_eval = np.ascontiguousarray(z_eval, dtype=np.float64)
    finite_u8 = np.ascontiguousarray(finite).astype(np.uint8)
    aff = np.ascontiguousarray(aff, dtype=np.float64)
    res = np.ascontiguousarray(res, dtype=np.float64)
    n_channels = coeff.shape[0]

    # The variance patches are the same for every chunk, so upload once. With
    # no calibration this is a four-byte dummy that only exists to keep the
    # kernel's argument types stable.
    use_variance = variance is not None
    d_variance = cuda.to_device(
        np.ascontiguousarray(variance, dtype=np.float32)
        if use_variance
        else np.zeros((1, 1, 1, 1), dtype=np.float32)
    )
    d_coeff = cuda.to_device(np.ascontiguousarray(coeff))
    # constant over the whole run, so uploaded once
    d_aff = cuda.to_device(aff)
    # inputs (amp, x, y, [z], offset, per-channel residual) + outputs (crlb
    # row, status byte)
    n_inputs = 5 if is_3d else 4
    chunk = min(
        n_locs,
        _crlb_chunk_rows(8 * (n_inputs + 2 * n_channels + n_params) + 1),
    )

    use_tqdm = progress_callback == "console"
    do_callback = callable(progress_callback)
    pbar = (
        tqdm(total=n_locs, desc="Computing spline CRLB", unit="locs")
        if use_tqdm
        else None
    )
    for start in range(0, n_locs, chunk):
        stop = min(start + chunk, n_locs)
        n = stop - start
        d_amp = cuda.to_device(amplitude[start:stop])
        d_x = cuda.to_device(x_shift[start:stop])
        d_y = cuda.to_device(y_shift[start:stop])
        d_off = cuda.to_device(offset[start:stop])
        d_finite = cuda.to_device(finite_u8[start:stop])
        d_crlb = cuda.device_array((n, n_params), dtype=np.float64)
        d_status = cuda.device_array(n, dtype=np.uint8)
        blocks = (
            n + _SPLINE_CRLB_CUDA_THREADS - 1
        ) // _SPLINE_CRLB_CUDA_THREADS
        if is_3d:
            d_z = cuda.to_device(z_eval[start:stop])
            d_res = cuda.to_device(res[start:stop])
            _spline_crlb_3d_kernel[blocks, _SPLINE_CRLB_CUDA_THREADS](
                d_coeff,
                d_aff,
                d_res,
                box,
                d_amp,
                d_x,
                d_y,
                d_z,
                d_off,
                d_finite,
                mu_floor,
                mle,
                d_variance,
                use_variance,
                d_crlb,
                d_status,
            )
        else:
            _spline_crlb_2d_kernel[blocks, _SPLINE_CRLB_CUDA_THREADS](
                d_coeff,
                box,
                d_amp,
                d_x,
                d_y,
                d_off,
                d_finite,
                mu_floor,
                mle,
                d_variance,
                use_variance,
                d_crlb,
                d_status,
            )
        crlb[start:stop] = d_crlb.copy_to_host()
        failed[start:stop] = d_status.copy_to_host() != 0
        if use_tqdm:
            pbar.update(n)
        elif do_callback:
            progress_callback(stop)
    if use_tqdm:
        pbar.close()
    return crlb, failed


def _spline_link_xyz_crlb_cuda(
    coeff: np.ndarray,
    aff: np.ndarray,
    res: np.ndarray,
    box: int,
    x_shift: lib.FloatArray1D,
    y_shift: lib.FloatArray1D,
    z_eval: lib.FloatArray1D,
    photons: lib.FloatArray2D,
    bg: lib.FloatArray2D,
    finite: np.ndarray,
    mu_floor: float,
    mle: bool,
    progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    variance: lib.FloatArray4D | None = None,
) -> tuple[lib.FloatArray2D, np.ndarray]:
    """Covariance diagonal of the photon-decoupled (link-XYZ) spline model on
    the GPU. Array-in / array-out counterpart of
    :func:`_spline_link_xyz_crlb_cpu`; see :func:`_spline_crlb_cuda` for the
    return convention and for ``aff`` / ``res``."""
    _require_crlb_cuda()
    n_channels = coeff.shape[0]
    n_params = 3 + 2 * n_channels
    if n_params > _LINK_XYZ_MAX_P:
        raise ValueError(
            f"link-XYZ CRLB on the GPU supports up to "
            f"{(_LINK_XYZ_MAX_P - 3) // 2} channels, got {n_channels}."
        )
    n_locs = len(x_shift)
    crlb = np.empty((n_locs, n_params), dtype=np.float64)
    failed = np.zeros(n_locs, dtype=bool)
    if n_locs == 0:
        return crlb, failed

    x_shift = np.ascontiguousarray(x_shift, dtype=np.float64)
    y_shift = np.ascontiguousarray(y_shift, dtype=np.float64)
    z_eval = np.ascontiguousarray(z_eval, dtype=np.float64)
    photons = np.ascontiguousarray(photons, dtype=np.float64)
    bg = np.ascontiguousarray(bg, dtype=np.float64)
    finite_u8 = np.ascontiguousarray(finite).astype(np.uint8)
    aff = np.ascontiguousarray(aff, dtype=np.float64)
    res = np.ascontiguousarray(res, dtype=np.float64)

    # The variance patches are the same for every chunk, so upload once. With
    # no calibration this is a four-byte dummy that only exists to keep the
    # kernel's argument types stable.
    use_variance = variance is not None
    d_variance = cuda.to_device(
        np.ascontiguousarray(variance, dtype=np.float32)
        if use_variance
        else np.zeros((1, 1, 1, 1), dtype=np.float32)
    )
    d_coeff = cuda.to_device(np.ascontiguousarray(coeff))
    # constant over the whole run, so uploaded once
    d_aff = cuda.to_device(aff)
    # inputs (x, y, z, photons, bg, per-channel residual) + outputs (crlb row,
    # status byte)
    chunk = min(
        n_locs, _crlb_chunk_rows(8 * (3 + 4 * n_channels + n_params) + 1)
    )

    use_tqdm = progress_callback == "console"
    do_callback = callable(progress_callback)
    pbar = (
        tqdm(total=n_locs, desc="Computing spline CRLB", unit="locs")
        if use_tqdm
        else None
    )
    for start in range(0, n_locs, chunk):
        stop = min(start + chunk, n_locs)
        n = stop - start
        d_x = cuda.to_device(x_shift[start:stop])
        d_y = cuda.to_device(y_shift[start:stop])
        d_z = cuda.to_device(z_eval[start:stop])
        d_photons = cuda.to_device(photons[start:stop])
        d_bg = cuda.to_device(bg[start:stop])
        d_finite = cuda.to_device(finite_u8[start:stop])
        d_res = cuda.to_device(res[start:stop])
        d_crlb = cuda.device_array((n, n_params), dtype=np.float64)
        d_status = cuda.device_array(n, dtype=np.uint8)
        blocks = (
            n + _SPLINE_CRLB_CUDA_THREADS - 1
        ) // _SPLINE_CRLB_CUDA_THREADS
        _spline_crlb_link_xyz_kernel[blocks, _SPLINE_CRLB_CUDA_THREADS](
            d_coeff,
            d_aff,
            d_res,
            box,
            d_x,
            d_y,
            d_z,
            d_photons,
            d_bg,
            d_finite,
            mu_floor,
            mle,
            d_variance,
            use_variance,
            d_crlb,
            d_status,
        )
        crlb[start:stop] = d_crlb.copy_to_host()
        failed[start:stop] = d_status.copy_to_host() != 0
        if use_tqdm:
            pbar.update(n)
        elif do_callback:
            progress_callback(stop)
    if use_tqdm:
        pbar.close()
    return crlb, failed


def _spline_link_xyz_crlb_cpu(
    coeff: np.ndarray,
    aff: np.ndarray,
    res: np.ndarray,
    box: int,
    x_shift: lib.FloatArray1D,
    y_shift: lib.FloatArray1D,
    z_eval: lib.FloatArray1D,
    photons: lib.FloatArray2D,
    bg: lib.FloatArray2D,
    finite: np.ndarray,
    mle: bool,
    progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    variance: lib.FloatArray4D | None = None,
) -> lib.FloatArray2D:
    """CPU (numba) parameter variances for the photon-decoupled (link-XYZ)
    model. The numerical core of :func:`_spline_link_xyz_crlb`, split out so it
    can also be run on a subset of rows (the GPU path falls back here for
    localizations the device could not diagonalize). ``aff`` and ``res`` are the
    per-channel affine and ROI residuals the fit used (see
    :func:`_spline_channel_affines` / :func:`_spline_crlb_residuals`); ``res``
    must already be sliced to the same rows as ``x_shift``. Returns the raw
    ``(n_locs, 3 + 2*n_channels)`` covariance diagonal; masking non-finite and
    non-positive entries is the caller's job."""
    n_channels = coeff.shape[0]
    n_params = 3 + 2 * n_channels
    n_locs = len(x_shift)

    use_tqdm = progress_callback == "console"
    do_callback = callable(progress_callback)
    rows_per_chunk = max(
        1024, _SPLINE_CRLB_CHUNK_BYTES // (n_params * n_params * 8)
    )
    chunk = int(min(max(n_locs, 1), rows_per_chunk, 100_000))
    starts = range(0, n_locs, chunk) if n_locs else []
    if use_tqdm:
        starts = tqdm(starts, desc="Computing spline CRLB")

    crlb = np.empty((n_locs, n_params), dtype=np.float64)
    for start in starts:
        stop = min(start + chunk, n_locs)
        sl = slice(start, stop)
        # Rows the kernel skips (non-finite theta) keep the identity, so the
        # batched pinv stays well-defined; the caller NaNs them.
        bread = np.tile(np.eye(n_params), (stop - start, 1, 1))
        meat = np.zeros((stop - start, n_params, n_params))
        _spline_infomats_link_xyz_3d(
            coeff,
            aff,
            res[sl],
            box,
            x_shift[sl],
            y_shift[sl],
            z_eval[sl],
            photons[sl],
            bg[sl],
            finite[sl],
            _SPLINE_CRLB_MU_FLOOR,
            mle,
            _crlb_variance_chunk(variance, sl),
            variance is not None,
            bread,
            meat,
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            bread_inv = np.linalg.pinv(bread)
            cov = bread_inv if mle else bread_inv @ meat @ bread_inv
            crlb[sl] = np.diagonal(cov, axis1=1, axis2=2)
        if do_callback:
            progress_callback(stop)
    return crlb


def _spline_link_xyz_crlb(
    theta: lib.FloatArray2D,
    calibration: dict,
    box: int,
    residuals: np.ndarray | None = None,
    mle: bool = True,
    em: bool = False,
    progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    variance: lib.FloatArray4D | None = None,
) -> lib.FloatArray2D:
    """CRLB / least-squares variances for the photon-decoupled (link-XYZ) fit.

    Same estimator theory as :func:`_spline_crlb`, but for the
    ``(3 + 2*n_channels)``-parameter model
    ``[x, y, z, N_0..N_{c-1}, bg_0..bg_{c-1}]`` (see
    :func:`_spline_infomats_link_xyz_3d`). ``residuals`` are the per-channel
    sub-pixel ROI offsets the fit used, as in :func:`_spline_crlb`, and ``em``
    applies the same EMCCD excess-noise doubling. Returns
    ``(n_locs, 3 + 2*n_channels)`` variances in that parameter order."""
    theta = np.asarray(theta, dtype=np.float64)
    n_locs = len(theta)
    n_channels = _spline_n_channels(calibration)
    variance = _crlb_variance_channel_major(variance, n_channels)

    x_shift = np.ascontiguousarray(theta[:, 0])
    y_shift = np.ascontiguousarray(theta[:, 1])
    # Native z sampling coordinate = -z_shift (see _spline_crlb).
    z_eval = np.ascontiguousarray(-theta[:, 2])
    photons = np.ascontiguousarray(theta[:, 3 : 3 + n_channels])
    bg = np.ascontiguousarray(theta[:, 3 + n_channels : 3 + 2 * n_channels])
    finite = np.isfinite(theta).all(axis=1)
    # same per-channel geometry as the fit (affine + ROI residual), so
    # the reported precision belongs to the model actually fitted
    aff = _spline_channel_affines(calibration, n_channels)
    res = _spline_crlb_residuals(residuals, n_locs, n_channels)

    crlb = None
    if CUDA_AVAILABLE:
        try:
            crlb, failed = _spline_link_xyz_crlb_cuda(
                _spline_coeff_reshaped(
                    calibration, dtype=_spline_crlb_coeff_dtype(calibration)
                ),
                aff,
                res,
                box,
                x_shift,
                y_shift,
                z_eval,
                photons,
                bg,
                finite,
                _SPLINE_CRLB_MU_FLOOR,
                mle,
                variance=variance,
                progress_callback=progress_callback,
            )
        except Exception as exc:
            # Degrade to the CPU (e.g. device out of memory while the fit still
            # holds allocations) rather than failing the whole fit.
            _warn_crlb_gpu_fallback(exc)
            crlb = None
        else:
            if failed.any():
                # The device could not diagonalize these information matrices;
                # redo just those rows through the CPU kernel and pinv.
                crlb[failed] = _spline_link_xyz_crlb_cpu(
                    _spline_coeff_reshaped(calibration),
                    aff,
                    res[failed],
                    box,
                    x_shift[failed],
                    y_shift[failed],
                    z_eval[failed],
                    photons[failed],
                    bg[failed],
                    finite[failed],
                    mle,
                    variance=(None if variance is None else variance[failed]),
                )
    if crlb is None:
        crlb = _spline_link_xyz_crlb_cpu(
            _spline_coeff_reshaped(calibration),
            aff,
            res,
            box,
            x_shift,
            y_shift,
            z_eval,
            photons,
            bg,
            finite,
            mle,
            variance=variance,
            progress_callback=progress_callback,
        )
    if em:
        crlb *= _EM_EXCESS_NOISE_FACTOR
    crlb[~finite] = np.nan
    crlb = np.where(crlb > 0.0, crlb, np.nan)
    return crlb


_crlb_gpu_fallback_warned = False


def _warn_crlb_gpu_fallback(exc: Exception) -> None:
    """Report the first time a GPU CRLB attempt falls back to the CPU.

    Having no CUDA device at all is not an error and is never reported - the
    CPU kernels are simply used. This is for the other case: a device is
    present but the attempt failed (out of memory, driver error), where the
    results are still correct but a silent fallback would hide a broken device
    path indefinitely. Warned once per process so a per-chunk failure cannot
    spam the log."""
    global _crlb_gpu_fallback_warned
    if _crlb_gpu_fallback_warned:
        return
    _crlb_gpu_fallback_warned = True
    warnings.warn(
        f"Spline CRLB on the GPU failed ({exc!r}); falling back to the CPU. "
        "Results are unaffected, only slower.",
        RuntimeWarning,
        stacklevel=3,
    )


def _spline_crlb_cpu(
    coeff: np.ndarray,
    aff: np.ndarray,
    res: np.ndarray,
    box: int,
    amplitude: lib.FloatArray1D,
    x_shift: lib.FloatArray1D,
    y_shift: lib.FloatArray1D,
    z_eval: lib.FloatArray1D | None,
    offset: lib.FloatArray1D,
    finite: np.ndarray,
    mle: bool,
    progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    variance: lib.FloatArray4D | None = None,
) -> lib.FloatArray2D:
    """CPU (numba) parameter variances for the shared-amplitude spline models.

    The numerical core of :func:`_spline_crlb`, split out so it can also be run
    on a subset of rows (the GPU path falls back here for localizations the
    device could not diagonalize). ``z_eval`` None selects the 2D model. ``aff``
    and ``res`` are the per-channel affine and ROI residuals the fit used (see
    :func:`_spline_channel_affines` / :func:`_spline_crlb_residuals`); ``res``
    must already be sliced to the same rows as ``amplitude``, and neither is
    used by the single-channel 2D model. Returns the raw ``(n_locs, P)``
    covariance diagonal; masking non-finite and non-positive entries is the
    caller's job."""
    is_3d = z_eval is not None
    n_params = 5 if is_3d else 4
    n_locs = len(amplitude)

    # Per-localization information matrices (float64). ``bread`` is the Fisher
    # matrix (mle) or the least-squares normal matrix J; non-converged rows stay
    # the identity so the batched pinv is well-defined (the caller NaNs them).
    # ``meat`` M is only filled for the least-squares sandwich (stays 0 for mle).
    bread = np.tile(np.eye(n_params), (max(n_locs, 1), 1, 1))
    meat = np.zeros((max(n_locs, 1), n_params, n_params))

    use_tqdm = progress_callback == "console"
    do_callback = callable(progress_callback)
    # One kernel call spans all localizations; chunk only to report progress.
    chunk = (
        max(1, min(n_locs, 100_000)) if (use_tqdm or do_callback) else n_locs
    )
    starts = range(0, n_locs, chunk) if n_locs else []
    if use_tqdm:
        starts = tqdm(starts, desc="Computing spline CRLB")

    for start in starts:
        stop = min(start + chunk, n_locs)
        sl = slice(start, stop)
        if is_3d:
            _spline_infomats_3d(
                coeff,
                aff,
                res[sl],
                box,
                amplitude[sl],
                x_shift[sl],
                y_shift[sl],
                z_eval[sl],
                offset[sl],
                finite[sl],
                _SPLINE_CRLB_MU_FLOOR,
                mle,
                _crlb_variance_chunk(variance, sl),
                variance is not None,
                bread[sl],
                meat[sl],
            )
        else:
            _spline_infomats_2d(
                coeff,
                box,
                amplitude[sl],
                x_shift[sl],
                y_shift[sl],
                offset[sl],
                finite[sl],
                _SPLINE_CRLB_MU_FLOOR,
                mle,
                _crlb_variance_chunk(variance, sl),
                variance is not None,
                bread[sl],
                meat[sl],
            )
        if do_callback:
            progress_callback(stop)

    with np.errstate(invalid="ignore", divide="ignore"):
        bread_inv = np.linalg.pinv(bread)
        if mle:
            # cov = I⁻¹ (Cramer-Rao bound); bread is the Fisher matrix.
            cov = bread_inv
        else:
            # cov = J⁻¹ M J⁻¹ (unweighted-least-squares sandwich).
            cov = bread_inv @ meat @ bread_inv
        crlb = np.diagonal(cov, axis1=1, axis2=2).copy()
    return crlb[:n_locs]


def _spline_crlb(
    theta: lib.FloatArray2D,
    calibration: dict,
    box: int,
    mle: bool = True,
    em: bool = False,
    progress_callback: (
        Callable[[int], None] | Literal["console"] | None
    ) = None,
    residuals: np.ndarray | None = None,
    variance: lib.FloatArray4D | None = None,
) -> lib.FloatArray2D:
    """Parameter-variance estimates for spline-fitted localizations.

    Runs on a CUDA GPU when one is available and falls back to the CPU kernels
    otherwise; both compute the same quantity, so the choice is invisible.

    Evaluates the estimator covariance at the fitted parameters, using the
    cubic-spline PSF model ``mu = offset + amplitude * Phi`` and its analytic
    spatial derivatives.

    With ``mle`` True the result is the Cramer-Rao lower bound: the diagonal of
    the inverse Poisson Fisher-information matrix ``I = Σ g gᵀ / μ`` (``g =
    ∂μ/∂θ``), which an efficient maximum-likelihood estimator attains. Mirrors
    ``gaussmle._mlefit_sigmaxy_crlb``.

    With ``mle`` False the result is the covariance of the *unweighted*
    least-squares estimator (the ``spline-gpu`` LSE mode), the Huber
    sandwich ``J⁻¹ M J⁻¹`` with normal matrix ``J = Σ g gᵀ`` and, for Poisson
    pixel noise (``σ² = μ``), meat ``M = Σ μ g gᵀ``. This is ≥ the Cramer-Rao
    bound elementwise (least squares is not efficient for Poisson data), so it
    is the honest precision for LSQ fits rather than the optimistic MLE floor.

    Parameters
    ----------
    theta : lib.FloatArray2D
        Fitted parameters, columns ``[amplitude, x_shift, y_shift, offset]``
        (2D) or ``[amplitude, x_shift, y_shift, z_shift, offset]`` (3D and 3D
        multichannel). Photon units (spots are gain-converted before fitting),
        so ``mu`` is an expected photon count and the Poisson noise model
        applies directly.
    calibration : dict
        The spline PSF calibration (see ``picasso.io.load_spline_calibration``).
    box : int
        Fit box side length (camera pixels).
    mle : bool, optional
        If True (default), return the Poisson Cramer-Rao bound (for
        maximum-likelihood fits). If False, return the least-squares sandwich
        covariance (for ``spline-gpu`` least-squares fits).
    em : bool, optional
        Whether the camera is an EMCCD. Its stochastic multiplication doubles
        every pixel's variance on top of the Poisson term, so all variances are
        scaled by 2. Applies to both estimators: the doubling is a property of
        the detector, not of the fit. Default False.
    progress_callback : callable, "console" or None, optional
        Progress over localization chunks. ``"console"`` shows a tqdm bar; a
        callable is invoked with the cumulative number of localizations done.
    residuals : np.ndarray, optional
        Per-localization, per-channel sub-pixel ROI offsets ``(n_locs,
        n_channels, 2)``, as passed to the fit (see
        :func:`picasso.localize.channel_roi_residuals`). Multichannel only; ``None`` (the
        default) means zero, which is the single-channel case. Pass whatever
        the fit used: the covariance is evaluated at ``theta`` under the same
        geometry, and the per-channel affine that goes with it is read from
        ``calibration["channel_transforms"]``.

    Returns
    -------
    crlb : lib.FloatArray2D
        ``(n_locs, n_params)`` array of parameter variances (float64) in native
        parameter order ``[x_shift, y_shift, (z_shift,) amplitude, offset]``
        (pixels, (z-slices,) photons, photons). Non-converged fits and
        numerically singular problems are NaN.
    """
    model = calibration["model"]
    if model == _LINK_XYZ_MODEL:
        # Photon-decoupled model: distinct 7-parameter block structure
        # [x, y, z, N_0..N_{c-1}, bg_0..bg_{c-1}], handled by its own kernel.
        return _spline_link_xyz_crlb(
            theta,
            calibration,
            box,
            residuals=residuals,
            mle=mle,
            em=em,
            progress_callback=progress_callback,
            variance=variance,
        )
    is_3d = model != "spline-2d"

    theta = np.asarray(theta, dtype=np.float64)
    n_locs = len(theta)

    amplitude = np.ascontiguousarray(theta[:, 0])
    x_shift = np.ascontiguousarray(theta[:, 1])
    y_shift = np.ascontiguousarray(theta[:, 2])
    offset = np.ascontiguousarray(theta[:, -1])
    # Native z sampling coordinate = -z_shift (single-frame
    # "position = pixel_index - parameter", pixel_index_z = 0). The kernel
    # clamps the z-interval, so no pre-clamping is needed here.
    z_eval = np.ascontiguousarray(-theta[:, 3]) if is_3d else None
    finite = np.isfinite(theta).all(axis=1)
    # same per-channel geometry as the fit (affine + ROI residual), so
    # the reported precision belongs to the model actually fitted
    n_channels = _spline_n_channels(calibration)
    variance = _crlb_variance_channel_major(variance, n_channels)
    aff = _spline_channel_affines(calibration, n_channels)
    res = _spline_crlb_residuals(residuals, n_locs, n_channels)

    crlb = None
    if CUDA_AVAILABLE:
        try:
            crlb, failed = _spline_crlb_cuda(
                _spline_coeff_reshaped(
                    calibration, dtype=_spline_crlb_coeff_dtype(calibration)
                ),
                aff,
                res,
                box,
                amplitude,
                x_shift,
                y_shift,
                z_eval,
                offset,
                finite,
                _SPLINE_CRLB_MU_FLOOR,
                mle,
                variance=variance,
                progress_callback=progress_callback,
            )
        except Exception as exc:
            # Degrade to the CPU (e.g. device out of memory while the fit still
            # holds allocations) rather than failing the whole fit.
            _warn_crlb_gpu_fallback(exc)
            crlb = None
        else:
            if failed.any():
                # The device could not diagonalize these information matrices;
                # redo just those rows through the CPU kernel and pinv.
                crlb[failed] = _spline_crlb_cpu(
                    _spline_coeff_reshaped(calibration),
                    aff,
                    res[failed],
                    box,
                    amplitude[failed],
                    x_shift[failed],
                    y_shift[failed],
                    None if z_eval is None else z_eval[failed],
                    offset[failed],
                    finite[failed],
                    mle,
                    variance=(None if variance is None else variance[failed]),
                )
    if crlb is None:
        crlb = _spline_crlb_cpu(
            _spline_coeff_reshaped(calibration),
            aff,
            res,
            box,
            amplitude,
            x_shift,
            y_shift,
            z_eval,
            offset,
            finite,
            mle,
            variance=variance,
            progress_callback=progress_callback,
        )
    if em:
        crlb *= _EM_EXCESS_NOISE_FACTOR
    crlb[~finite] = np.nan
    crlb = np.where(crlb > 0.0, crlb, np.nan)
    return crlb
