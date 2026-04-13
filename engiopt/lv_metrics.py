"""Latent-space metrics for evaluating generative models using a trained LVAE.

Provides per-sample metrics computed in the performance-constrained LVAE
latent space: projection residual, dual-LVAE projection gap, Mahalanobis
typicality, novelty-quality Pareto analysis, and volume coverage.

Distribution-level metrics (LV-MMD, LV-DPP) are computed directly via
``engiopt.metrics`` in the evaluation scripts.
"""

from __future__ import annotations

from typing import Any, NamedTuple, TYPE_CHECKING
import warnings

import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import cdist
from scipy.stats import chi2

from engiopt import metrics as _metrics
from engiopt.vanilla_lvae.utils import decode_designs
from engiopt.vanilla_lvae.utils import encode_designs

if TYPE_CHECKING:
    from torch import nn


# ---------------------------------------------------------------------------
# Reference statistics
# ---------------------------------------------------------------------------


class LatentReferenceStats(NamedTuple):
    """Precomputed statistics of the training set in active latent space."""

    mean: npt.NDArray
    """Centroid, shape (D,)."""
    cov: npt.NDArray
    """Covariance matrix, shape (D, D)."""
    cov_inv: npt.NDArray
    """Regularized inverse covariance, shape (D, D)."""
    log_det_cov: float
    """Log-determinant of the covariance matrix."""
    z_train: npt.NDArray
    """Training latent codes, shape (N_train, D). Stored for nearest-neighbor lookups."""
    n_train: int
    """Number of training samples."""
    n_dims: int
    """Number of active latent dimensions (D)."""


def compute_latent_reference_stats(
    z_train: npt.NDArray,
    reg: float = 1e-6,
) -> LatentReferenceStats:
    """Compute training-set latent statistics shared across multiple metrics.

    Args:
        z_train: Training latent codes, shape (N, D), already sliced to active dims.
        reg: Regularization added to covariance diagonal for numerical stability.

    Returns:
        ``LatentReferenceStats`` namedtuple.
    """
    n_train, n_dims = z_train.shape
    mean = z_train.mean(axis=0)
    cov = np.cov(z_train, rowvar=False)
    # np.cov returns a scalar when n_dims == 1
    if cov.ndim == 0:
        cov = cov.reshape(1, 1)
    cov_reg = cov + reg * np.eye(n_dims)
    cov_inv = np.linalg.inv(cov_reg)
    sign, log_det = np.linalg.slogdet(cov_reg)
    log_det_cov = float(log_det) if sign > 0 else float("-inf")
    return LatentReferenceStats(
        mean=mean,
        cov=cov_reg,
        cov_inv=cov_inv,
        log_det_cov=log_det_cov,
        z_train=z_train,
        n_train=n_train,
        n_dims=n_dims,
    )


# ---------------------------------------------------------------------------
# Encode-decode helper
# ---------------------------------------------------------------------------


def _get_device(model: nn.Module) -> object:
    """Infer the device a model lives on from its first parameter."""
    return next(model.parameters()).device


def _encode_decode(
    encoder: nn.Module,
    decoder: nn.Module,
    designs: npt.NDArray,
    device: str | object,
    batch_size: int = 256,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Encode then decode, returning latent codes and reconstructions.

    The encoder runs on ``device`` (typically GPU for speed).  The decoder
    runs on whatever device it already lives on — kept on CPU by default to
    avoid CUDA SIGFPE from spectral-norm deconvolutions.

    Args:
        encoder: Trained LVAE encoder.
        decoder: Trained LVAE decoder.
        designs: Designs of shape (N, H, W) or (N, 1, H, W).
        device: Torch device for the encoder.
        batch_size: Batch size for processing.

    Returns:
        Tuple of (latent_codes (N, D), reconstructed_designs (N, H, W)).
    """
    z = encode_designs(encoder, designs, device, batch_size=batch_size)
    decode_device = _get_device(decoder)
    recon = decode_designs(decoder, z, decode_device, batch_size=batch_size)
    return z, recon


# ---------------------------------------------------------------------------
# Per-sample metrics
# ---------------------------------------------------------------------------


def lv_project_designs(
    encoder: nn.Module,
    decoder: nn.Module,
    designs: npt.NDArray,
    device: str | object,
    batch_size: int = 256,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Project designs onto the learned manifold and measure the distance.

    Per-sample MSE between input designs and their LVAE projection
    (encode→decode).  Higher residual means the generator produced designs
    that required more correction by the LVAE.

    Args:
        encoder: Trained LVAE encoder.
        decoder: Trained LVAE decoder.
        designs: Designs of shape (N, H, W) or (N, 1, H, W).
        device: Torch device.
        batch_size: Batch size for processing.

    Returns:
        Tuple of (per-sample MSE (N,), projected designs (N, H, W)).
    """
    _, recon = _encode_decode(encoder, decoder, designs, device, batch_size)
    orig = designs.squeeze(1) if designs.ndim == 4 else designs  # noqa: PLR2004
    mse = np.mean((orig - recon) ** 2, axis=(1, 2))
    return mse, recon


def lv_dual_projection_gap(
    encoder_perf: nn.Module,
    decoder_perf: nn.Module,
    encoder_recon: nn.Module,
    decoder_recon: nn.Module,
    designs: npt.NDArray,
    device: str | object,
    batch_size: int = 256,
) -> dict[str, Any]:
    """Compare projections from perf-constrained vs recon-only LVAEs.

    The gap between the two projections isolates the *performance-attributable*
    component of the correction.  Large gap → the deviation from the manifold
    was in a performance-relevant direction.  Small gap → cosmetic noise only.

    Args:
        encoder_perf: Perf-constrained LVAE encoder.
        decoder_perf: Perf-constrained LVAE decoder.
        encoder_recon: Recon-only LVAE encoder.
        decoder_recon: Recon-only LVAE decoder.
        designs: Designs of shape (N, H, W) or (N, 1, H, W).
        device: Torch device.
        batch_size: Batch size for processing.

    Returns:
        Dictionary with per-sample arrays and scalar summaries:
        - ``dual_gap``: per-sample MSE between the two projections (N,).
        - ``residual_perf``: per-sample MSE, original vs perf projection (N,).
        - ``residual_recon``: per-sample MSE, original vs recon projection (N,).
        - ``dual_gap_mean``, ``residual_perf_mean``, ``residual_recon_mean``.
    """
    _, proj_perf = _encode_decode(encoder_perf, decoder_perf, designs, device, batch_size)
    _, proj_recon = _encode_decode(encoder_recon, decoder_recon, designs, device, batch_size)

    orig = designs.squeeze(1) if designs.ndim == 4 else designs  # noqa: PLR2004

    dual_gap = np.mean((proj_perf - proj_recon) ** 2, axis=(1, 2))
    residual_perf = np.mean((orig - proj_perf) ** 2, axis=(1, 2))
    residual_recon = np.mean((orig - proj_recon) ** 2, axis=(1, 2))

    return {
        "dual_gap": dual_gap,
        "residual_perf": residual_perf,
        "residual_recon": residual_recon,
        "dual_gap_mean": float(dual_gap.mean()),
        "residual_perf_mean": float(residual_perf.mean()),
        "residual_recon_mean": float(residual_recon.mean()),
    }


# ---------------------------------------------------------------------------
# Tier 1.3: Reconstruction residual summary statistics
# ---------------------------------------------------------------------------


def lv_reconstruction_residual_stats(
    encoder: nn.Module,
    decoder: nn.Module,
    designs: npt.NDArray,
    device: str | object,
    batch_size: int = 256,
) -> dict[str, Any]:
    """Summary statistics of per-sample reconstruction residuals.

    Wraps :func:`lv_project_designs` and aggregates the per-sample MSE
    into mean, median, 90th-percentile, and standard deviation.

    Args:
        encoder: Trained LVAE encoder.
        decoder: Trained LVAE decoder.
        designs: Designs of shape (N, H, W) or (N, 1, H, W).
        device: Torch device.
        batch_size: Batch size for processing.

    Returns:
        Dict with per-sample ``residuals`` (N,) and scalar summaries
        ``residual_mean``, ``residual_median``, ``residual_p90``,
        ``residual_std``.
    """
    mse, _ = lv_project_designs(encoder, decoder, designs, device, batch_size)
    return {
        "residuals": mse,
        "residual_mean": float(np.mean(mse)),
        "residual_median": float(np.median(mse)),
        "residual_p90": float(np.percentile(mse, 90)),
        "residual_std": float(np.std(mse)),
    }


# ---------------------------------------------------------------------------
# Tier 2.1: Mahalanobis typicality score
# ---------------------------------------------------------------------------


def _mahalanobis_distances(
    z: npt.NDArray,
    ref_stats: LatentReferenceStats,
) -> npt.NDArray:
    """Mahalanobis distance of each row of *z* from the reference centroid."""
    diff = z - ref_stats.mean
    # (N, D) @ (D, D) -> (N, D), element-wise * (N, D) -> sum over D
    d_sq = np.sum((diff @ ref_stats.cov_inv) * diff, axis=1)
    # Clamp tiny negatives from numerical noise before sqrt
    return np.sqrt(np.maximum(d_sq, 0.0))


def lv_mahalanobis_typicality(
    z_gen: npt.NDArray,
    ref_stats: LatentReferenceStats,
) -> dict[str, Any]:
    """Mahalanobis distance of generated latent codes from the training centroid.

    Higher distance means the generated design is less typical of the training
    distribution.  The *plausibility rate* is the fraction of samples whose
    squared Mahalanobis distance falls below the chi-squared 95th-percentile
    threshold for ``D`` degrees of freedom.

    Args:
        z_gen: Generated latent codes, shape (N, D), already active-dim-sliced.
        ref_stats: Precomputed training latent statistics.

    Returns:
        Dict with per-sample ``mahal_distances`` (N,) and scalar summaries
        ``mahal_mean``, ``mahal_median``, ``mahal_std``, ``plausibility_rate``,
        ``chi2_threshold``.
    """
    n_samples, n_dims = z_gen.shape
    if n_samples <= n_dims:
        warnings.warn(
            f"n_samples ({n_samples}) <= n_dims ({n_dims}); "
            "Mahalanobis distances may be unreliable due to singular covariance.",
            stacklevel=2,
        )

    distances = _mahalanobis_distances(z_gen, ref_stats)
    threshold = float(chi2.ppf(0.95, df=ref_stats.n_dims))
    plausibility_rate = float(np.mean(distances**2 <= threshold))

    return {
        "mahal_distances": distances,
        "mahal_mean": float(np.mean(distances)),
        "mahal_median": float(np.median(distances)),
        "mahal_std": float(np.std(distances)),
        "plausibility_rate": plausibility_rate,
        "chi2_threshold": threshold,
    }


# ---------------------------------------------------------------------------
# Tier 2.3: Conditional LV-MMD
# ---------------------------------------------------------------------------


def lv_conditional_mmd(
    z_gen: npt.NDArray,
    z_ref: npt.NDArray,
    conditions: npt.NDArray,
    n_bins: int = 5,
    sigma: float | None = None,
) -> dict[str, float]:
    """Conditional MMD in LVAE latent space.

    Thin wrapper around :func:`engiopt.metrics.conditional_mmd` that prefixes
    all returned keys with ``lv_`` for consistent naming in evaluation outputs.

    Args:
        z_gen: Generated latent codes (N, D), active-dim-sliced.
        z_ref: Reference latent codes (N, D), active-dim-sliced.
        conditions: Condition values (N, n_conds).
        n_bins: Number of quantile bins per condition dimension.
        sigma: Bandwidth. None = median heuristic on *z_ref*.

    Returns:
        Dict with ``lv_cond_mmd``, ``lv_cond_mmd_dim{i}``, ``lv_cond_mmd_sigma``.
    """
    raw = _metrics.conditional_mmd(z_gen, z_ref, conditions, n_bins=n_bins, sigma=sigma)
    return {f"lv_{k}": v for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Tier 3.1: Novelty-Quality Pareto front
# ---------------------------------------------------------------------------


def lv_novelty_quality_pareto(
    z_gen: npt.NDArray,
    ref_stats: LatentReferenceStats,
) -> dict[str, Any]:
    """Novelty-quality Pareto analysis with hypervolume indicator.

    For each generated sample:

    * **Quality** = negative Mahalanobis distance (higher = more plausible).
    * **Novelty** = minimum Euclidean distance to any training sample in
      active latent space (higher = more novel).

    The 2-D hypervolume indicator summarises the Pareto front.

    Args:
        z_gen: Generated latent codes (N, D), active-dim-sliced.
        ref_stats: Precomputed training latent statistics.

    Returns:
        Dict with per-sample ``novelty`` (N,), ``quality`` (N,),
        scalar ``hypervolume``, and ``pareto_front_indices``.
    """
    from pymoo.indicators.hv import HV

    # Negate Mahalanobis so higher values mean more plausible
    mahal = _mahalanobis_distances(z_gen, ref_stats)
    quality = -mahal

    # Novelty = min L2 distance to any training point
    dists_to_train = cdist(z_gen, ref_stats.z_train, metric="euclidean")
    novelty = dists_to_train.min(axis=1)

    # Pareto front: both objectives to maximise.  pymoo minimises, so negate.
    points = np.column_stack([-novelty, -quality])  # shape (N, 2), minimise both

    # Reference point: worst in both objectives (slightly beyond observed worst)
    ref_point = np.array([0.0, np.max(mahal) + 1.0])  # negated: 0 novelty, worst quality

    hv_indicator = HV(ref_point=ref_point)
    hypervolume = float(hv_indicator(points))

    # Extract Pareto-optimal indices via simple 2-D dominance check
    pareto_mask = np.ones(len(points), dtype=bool)
    for i, p in enumerate(points):
        if pareto_mask[i]:
            # A point j dominates i if j <= i in all objectives and j < i in at least one
            pareto_mask[i] = not np.any(
                np.all(points <= p, axis=1) & np.any(points < p, axis=1) & (np.arange(len(points)) != i)
            )

    return {
        "novelty": novelty,
        "quality": quality,
        "hypervolume": hypervolume,
        "pareto_front_indices": np.where(pareto_mask)[0],
    }


# ---------------------------------------------------------------------------
# Tier 3.2: Latent volume coverage ratio
# ---------------------------------------------------------------------------


def lv_volume_coverage_ratio(
    z_gen: npt.NDArray,
    ref_stats: LatentReferenceStats,
    reg: float = 1e-6,
) -> dict[str, float]:
    """Log-det ratio of generated vs training covariance in active latent space.

    Measures how much of the training latent volume the generated set covers.

    * Near 0 → similar spread.
    * Negative → generated is more concentrated.
    * Positive → generated is more spread out.

    Args:
        z_gen: Generated latent codes (N, D), active-dim-sliced.
        ref_stats: Precomputed training latent statistics.
        reg: Regularization for generated covariance.

    Returns:
        Dict with ``volume_ratio``, ``log_det_gen``, ``log_det_train``.
    """
    n_dims = z_gen.shape[1]
    cov_gen = np.cov(z_gen, rowvar=False)
    if cov_gen.ndim == 0:
        cov_gen = cov_gen.reshape(1, 1)
    cov_gen_reg = cov_gen + reg * np.eye(n_dims)
    sign, log_det_gen = np.linalg.slogdet(cov_gen_reg)
    log_det_gen_val = float(log_det_gen) if sign > 0 else float("-inf")

    return {
        "volume_ratio": log_det_gen_val - ref_stats.log_det_cov,
        "log_det_gen": log_det_gen_val,
        "log_det_train": ref_stats.log_det_cov,
    }
