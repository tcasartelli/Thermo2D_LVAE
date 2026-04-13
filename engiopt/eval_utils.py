"""Shared evaluation utilities for all generative model eval scripts.

Extracts the ~290 lines of identical code that was copy-pasted across
evaluate_cgan_cnn_2d.py, evaluate_gan_cnn_2d.py, evaluate_diffusion_2d_cond.py,
and evaluate_vqgan.py.
"""

from __future__ import annotations

import dataclasses
import itertools
import os
from typing import Any

from engibench.utils.all_problems import BUILTIN_PROBLEMS
import numpy as np
import pandas as pd
import torch as th

from engiopt import metrics
from engiopt.dataset_sample_conditions import sample_conditions

# ---------------------------------------------------------------------------
# Shared Args base class
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class BaseEvaluationArgs:
    """Common command-line arguments shared by all evaluation scripts."""

    problem_id: str = "beams2d"
    """Problem identifier."""
    seed: int = 1
    """Random seed to run."""
    wandb_project: str = "engiopt"
    """Wandb project name."""
    wandb_entity: str | None = None
    """Wandb entity name."""
    n_samples: int = 50
    """Number of generated samples per seed."""
    sigma: float | None = None
    """Kernel bandwidth for MMD and DPP metrics. If None, uses median heuristic from reference data."""
    output_csv: str = ""
    """Output CSV path template; may include {problem_id}. Override in subclass."""

    # LV metrics (optional) — provide seed and at least one threshold list to enable
    lvae_seed: int | None = None
    """LVAE seed. If provided along with thresholds, computes LV-MMD and LV-DPP."""
    lvae_rec_thresholds: str = ""
    """Comma-separated LVAE reconstruction NMSE thresholds (e.g. '0.0005,0.001,0.005')."""
    lvae_perf_thresholds: str = ""
    """Comma-separated LVAE performance NMSE thresholds (e.g. '0.001,1000')."""

    # LV metric suite (optional) — requires lvae_seed + thresholds above
    compute_lv_suite: bool = False
    """Compute full LV metric suite (projection residual, dual gap). Requires decoder."""

    # WandB logging
    log_to_wandb: bool = False
    """Log evaluation metrics to the original WandB training run."""


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------


class RunRetrievalError(ValueError):
    """Raised when a WandB run cannot be retrieved from an artifact."""

    def __init__(self) -> None:
        super().__init__("Failed to retrieve the run")


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------


def parse_thresholds(args: BaseEvaluationArgs) -> tuple[list[float], list[float]]:
    """Parse comma-separated threshold strings into lists of floats."""
    rec = [float(x) for x in args.lvae_rec_thresholds.split(",") if x.strip()] if args.lvae_rec_thresholds else []
    perf = [float(x) for x in args.lvae_perf_thresholds.split(",") if x.strip()] if args.lvae_perf_thresholds else []
    return rec, perf


def setup_evaluation(
    args: BaseEvaluationArgs,
) -> tuple[Any, th.device, np.random.Generator, th.Tensor, Any, np.ndarray, np.ndarray]:
    """Shared setup: problem, device, RNG, condition sampling.

    Returns:
        (problem, device, rng, conditions_tensor, sampled_conditions, sampled_designs_np, selected_indices)
    """
    seed = args.seed
    problem = BUILTIN_PROBLEMS[args.problem_id]()
    problem.reset(seed=seed)

    th.manual_seed(seed)
    rng = np.random.default_rng(seed)
    th.backends.cudnn.deterministic = True

    if th.backends.mps.is_available():
        device = th.device("mps")
    elif th.cuda.is_available():
        device = th.device("cuda")
    else:
        device = th.device("cpu")

    conditions_tensor, sampled_conditions, sampled_designs_np, selected_indices = sample_conditions(
        problem=problem, n_samples=args.n_samples, device=device, seed=seed
    )

    return problem, device, rng, conditions_tensor, sampled_conditions, sampled_designs_np, selected_indices


# ---------------------------------------------------------------------------
# LVAE metric loop
# ---------------------------------------------------------------------------


def compute_lvae_metrics(  # noqa: PLR0913, PLR0915
    *,
    args: BaseEvaluationArgs,
    gen_designs_np: np.ndarray,
    sampled_designs_np: np.ndarray,
    problem: Any,
    device: th.device,
    rec_thresh: float,
    perf_thresh: float,
    suffix: str,
    metrics_dict: dict[str, Any],
    per_sample_data: dict[str, np.ndarray],
) -> None:
    """One iteration of the LVAE loop.

    Computes LV-MMD/DPP, reference stats, Mahalanobis typicality,
    novelty-quality Pareto, volume coverage, optional suite, and PCA baseline.
    Mutates *metrics_dict* and *per_sample_data* in place.
    """
    from sklearn.decomposition import PCA

    from engiopt.lv_metrics import compute_latent_reference_stats
    from engiopt.lv_metrics import lv_mahalanobis_typicality
    from engiopt.lv_metrics import lv_novelty_quality_pareto
    from engiopt.lv_metrics import lv_volume_coverage_ratio
    from engiopt.vanilla_lvae.utils import encode_designs
    from engiopt.vanilla_lvae.utils import get_active_mask
    from engiopt.vanilla_lvae.utils import load_lvae_encoder

    encoder, _lvae_config = load_lvae_encoder(
        problem_id=args.problem_id,
        seed=args.lvae_seed,
        rec_threshold=rec_thresh,
        perf_threshold=perf_thresh,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        device=device,
    )

    # Encode designs to latent space and slice to active (unpruned) dims
    z_gen = encode_designs(encoder, gen_designs_np, device)
    z_data = encode_designs(encoder, sampled_designs_np, device)
    active = get_active_mask(encoder)
    n_active = int(active.sum())
    z_gen = z_gen[:, active]
    z_data = z_data[:, active]

    lv_sigma = metrics.compute_median_sigma(z_data)
    lv_mmd_val = metrics.mmd(z_gen, z_data, sigma=lv_sigma)
    lv_dpp_val = metrics.dpp_diversity(z_gen, sigma=lv_sigma)

    metrics_dict[f"lv_mmd{suffix}"] = lv_mmd_val
    metrics_dict[f"lv_dpp{suffix}"] = lv_dpp_val
    metrics_dict[f"lv_sigma{suffix}"] = lv_sigma
    metrics_dict[f"lvae_n_active_dims{suffix}"] = n_active

    # Build reference stats from full training set (not just sampled test designs)
    train_designs_np = np.array(problem.dataset["train"]["optimal_design"])
    z_train = encode_designs(encoder, train_designs_np, device)[:, active]
    ref_stats = compute_latent_reference_stats(z_train)
    mahal = lv_mahalanobis_typicality(z_gen, ref_stats)
    metrics_dict[f"lv_mahal_mean{suffix}"] = mahal["mahal_mean"]
    metrics_dict[f"lv_mahal_median{suffix}"] = mahal["mahal_median"]
    metrics_dict[f"lv_plausibility_rate{suffix}"] = mahal["plausibility_rate"]

    nq = lv_novelty_quality_pareto(z_gen, ref_stats)
    metrics_dict[f"lv_nq_hypervolume{suffix}"] = nq["hypervolume"]

    vol = lv_volume_coverage_ratio(z_gen, ref_stats)
    metrics_dict[f"lv_volume_ratio{suffix}"] = vol["volume_ratio"]

    # Store per-sample latent metric arrays for NPZ
    per_sample_data[f"lv_mahal_distances{suffix}"] = mahal["mahal_distances"]
    per_sample_data[f"lv_novelty{suffix}"] = nq["novelty"]
    per_sample_data[f"lv_quality{suffix}"] = nq["quality"]

    print(
        f"  Mahal: {mahal['mahal_mean']:.3f}, Plaus: {mahal['plausibility_rate']:.2%}, "
        f"HV: {nq['hypervolume']:.4f}, VolRatio: {vol['volume_ratio']:.3f}"
    )

    # LV metric suite: projection residual + dual gap
    if args.compute_lv_suite:
        try:
            from engiopt.lv_metrics import lv_dual_projection_gap
            from engiopt.lv_metrics import lv_reconstruction_residual_stats
            from engiopt.vanilla_lvae.utils import load_lvae_encoder_decoder

            print("  [diag] Loading decoder...", flush=True)
            _, decoder, _ = load_lvae_encoder_decoder(
                problem_id=args.problem_id,
                seed=args.lvae_seed,
                rec_threshold=rec_thresh,
                perf_threshold=perf_thresh,
                wandb_project=args.wandb_project,
                wandb_entity=args.wandb_entity,
                device=device,
                design_shape=problem.design_space.shape,
            )
            print(f"  [diag] Decoder loaded on {next(decoder.parameters()).device}", flush=True)
            recon_stats = lv_reconstruction_residual_stats(encoder, decoder, gen_designs_np, device)
            metrics_dict[f"lv_proj_residual_mean{suffix}"] = recon_stats["residual_mean"]
            metrics_dict[f"lv_residual_median{suffix}"] = recon_stats["residual_median"]
            metrics_dict[f"lv_residual_p90{suffix}"] = recon_stats["residual_p90"]
            metrics_dict[f"lv_residual_std{suffix}"] = recon_stats["residual_std"]
            per_sample_data[f"lv_residuals{suffix}"] = recon_stats["residuals"]
            print(f"  LV-ProjRes: {recon_stats['residual_mean']:.6f}")

            # Dual-LVAE projection gap: compare perf vs recon-only projections
            recon_only_thresh = 1000.0
            if perf_thresh != recon_only_thresh:
                enc_ro, dec_ro, _ = load_lvae_encoder_decoder(
                    problem_id=args.problem_id,
                    seed=args.lvae_seed,
                    rec_threshold=rec_thresh,
                    perf_threshold=recon_only_thresh,
                    wandb_project=args.wandb_project,
                    wandb_entity=args.wandb_entity,
                    device=device,
                    design_shape=problem.design_space.shape,
                )
                dual = lv_dual_projection_gap(encoder, decoder, enc_ro, dec_ro, gen_designs_np, device)
                metrics_dict[f"lv_dual_gap_mean{suffix}"] = dual["dual_gap_mean"]
                metrics_dict[f"lv_residual_perf_mean{suffix}"] = dual["residual_perf_mean"]
                metrics_dict[f"lv_residual_recon_mean{suffix}"] = dual["residual_recon_mean"]
                per_sample_data[f"lv_dual_gap{suffix}"] = dual["dual_gap"]
                per_sample_data[f"lv_residual_perf{suffix}"] = dual["residual_perf"]
                per_sample_data[f"lv_residual_recon{suffix}"] = dual["residual_recon"]
                print(f"  LV-DualGap: {dual['dual_gap_mean']:.6f}")
        except Exception as e:  # noqa: BLE001
            print(f"  LV suite failed: {e}")

    # PCA baseline: project to same dimensionality as LVAE active dims
    n_pca = min(n_active, args.n_samples - 1)
    if n_pca > 0:
        flat_gen = gen_designs_np.reshape(gen_designs_np.shape[0], -1)
        flat_data = sampled_designs_np.reshape(sampled_designs_np.shape[0], -1)
        pca = PCA(n_components=n_pca)
        pca.fit(flat_data)
        pca_gen = pca.transform(flat_gen)
        pca_data = pca.transform(flat_data)

        pca_sigma = metrics.compute_median_sigma(pca_data)
        pca_mmd_val = metrics.mmd(pca_gen, pca_data, sigma=pca_sigma)
        pca_dpp_val = metrics.dpp_diversity(pca_gen, sigma=pca_sigma)

        metrics_dict[f"pca_mmd{suffix}"] = pca_mmd_val
        metrics_dict[f"pca_dpp{suffix}"] = pca_dpp_val
        metrics_dict[f"pca_sigma{suffix}"] = pca_sigma
        print(f"  PCA({n_pca}): MMD={pca_mmd_val:.6f}, DPP={pca_dpp_val:.6e}")

    print(f"  LV-MMD: {lv_mmd_val:.6f}, LV-DPP: {lv_dpp_val:.6e}, Active: {n_active}")


# ---------------------------------------------------------------------------
# LVAE loop driver
# ---------------------------------------------------------------------------


def run_lvae_loop(  # noqa: PLR0913
    *,
    args: BaseEvaluationArgs,
    rec_thresholds: list[float],
    perf_thresholds: list[float],
    gen_designs_np: np.ndarray,
    sampled_designs_np: np.ndarray,
    problem: Any,
    device: th.device,
    metrics_dict: dict[str, Any],
    per_sample_data: dict[str, np.ndarray],
    run: Any | None,
) -> None:
    """Drive the full LVAE threshold loop, including WandB logging."""
    metrics_dict["lvae_seed"] = args.lvae_seed

    # Log base metrics to WandB immediately before LVAE loop (timeout-safe)
    if args.log_to_wandb and run is not None:
        log_base_metrics_to_wandb(run, metrics_dict)

    for rec_thresh, perf_thresh in itertools.product(rec_thresholds, perf_thresholds):
        suffix = f"_rec{rec_thresh}_perf{perf_thresh}"
        print(f"Loading LVAE (seed={args.lvae_seed}, rec={rec_thresh}, perf={perf_thresh})...")
        try:
            compute_lvae_metrics(
                args=args,
                gen_designs_np=gen_designs_np,
                sampled_designs_np=sampled_designs_np,
                problem=problem,
                device=device,
                rec_thresh=rec_thresh,
                perf_thresh=perf_thresh,
                suffix=suffix,
                metrics_dict=metrics_dict,
                per_sample_data=per_sample_data,
            )

            # Log this combo immediately so a timeout can't erase it
            if args.log_to_wandb and run is not None:
                log_lvae_combo_to_wandb(run, metrics_dict, suffix, args.lvae_seed, args.compute_lv_suite)

        except Exception as e:  # noqa: BLE001
            print(f"  Failed for rec={rec_thresh}, perf={perf_thresh}: {e}")


# ---------------------------------------------------------------------------
# WandB logging
# ---------------------------------------------------------------------------

_BASE_METRIC_KEYS = [
    "mmd",
    "dpp",
    "iog",
    "cog",
    "fog",
    "iog_median",
    "cog_median",
    "fog_median",
    "iog_iqr",
    "cog_iqr",
    "fog_iqr",
    "iog_var",
    "cog_var",
    "fog_var",
    "viol",
    "mmd_sigma",
]


def log_base_metrics_to_wandb(run: Any, metrics_dict: dict[str, Any]) -> None:
    """Log simulation-level metrics (IOG/COG/FOG/MMD/DPP) to WandB summary."""
    for key in _BASE_METRIC_KEYS:
        run.summary[f"eval/{key}"] = metrics_dict.get(key)
    run.summary.update()
    print("  Logged base metrics to WandB.")


def log_lvae_combo_to_wandb(
    run: Any,
    metrics_dict: dict[str, Any],
    suffix: str,
    lvae_seed: int | None,
    compute_lv_suite: bool,  # noqa: FBT001
) -> None:
    """Log one LVAE (rec, perf) combo's metrics to WandB summary."""
    # Extract rec/perf from suffix to build prefix
    # suffix is like "_rec0.001_perf1000"
    rec_str = suffix.split("_rec")[1].split("_perf")[0]
    perf_str = suffix.split("_perf")[1]
    prefix = f"eval/lv_rec{rec_str}_perf{perf_str}_lvae{lvae_seed}"

    run.summary[f"{prefix}/lv_mmd"] = metrics_dict.get(f"lv_mmd{suffix}")
    run.summary[f"{prefix}/lv_dpp"] = metrics_dict.get(f"lv_dpp{suffix}")
    run.summary[f"{prefix}/lv_sigma"] = metrics_dict.get(f"lv_sigma{suffix}")
    run.summary[f"{prefix}/n_active_dims"] = metrics_dict.get(f"lvae_n_active_dims{suffix}")
    run.summary[f"{prefix}/lv_mahal_mean"] = metrics_dict.get(f"lv_mahal_mean{suffix}")
    run.summary[f"{prefix}/lv_plausibility_rate"] = metrics_dict.get(f"lv_plausibility_rate{suffix}")
    run.summary[f"{prefix}/lv_nq_hypervolume"] = metrics_dict.get(f"lv_nq_hypervolume{suffix}")
    run.summary[f"{prefix}/lv_volume_ratio"] = metrics_dict.get(f"lv_volume_ratio{suffix}")
    run.summary[f"{prefix}/pca_mmd"] = metrics_dict.get(f"pca_mmd{suffix}")
    run.summary[f"{prefix}/pca_dpp"] = metrics_dict.get(f"pca_dpp{suffix}")
    run.summary[f"{prefix}/pca_sigma"] = metrics_dict.get(f"pca_sigma{suffix}")
    if compute_lv_suite:
        run.summary[f"{prefix}/lv_proj_residual_mean"] = metrics_dict.get(f"lv_proj_residual_mean{suffix}")
        run.summary[f"{prefix}/lv_residual_median"] = metrics_dict.get(f"lv_residual_median{suffix}")
        run.summary[f"{prefix}/lv_residual_p90"] = metrics_dict.get(f"lv_residual_p90{suffix}")
        run.summary[f"{prefix}/lv_dual_gap_mean"] = metrics_dict.get(f"lv_dual_gap_mean{suffix}")
        run.summary[f"{prefix}/lv_residual_perf_mean"] = metrics_dict.get(f"lv_residual_perf_mean{suffix}")
        run.summary[f"{prefix}/lv_residual_recon_mean"] = metrics_dict.get(f"lv_residual_recon_mean{suffix}")
    run.summary.update()


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------


def save_per_sample_npz(  # noqa: PLR0913
    metrics_dict: dict[str, Any],
    per_sample_data: dict[str, np.ndarray],
    sampled_conditions: Any,
    gen_designs_np: np.ndarray,
    sampled_designs_np: np.ndarray,
    output_csv: str,
    problem_id: str,
    seed: int,
) -> None:
    """Merge simulation per-sample lists + LVAE per-sample arrays and save to NPZ."""
    per_sample_keys = ["iog_list", "cog_list", "fog_list", "viol_list"]
    per_sample_data.update({k: np.array(metrics_dict[k]) for k in per_sample_keys if k in metrics_dict})
    # Include condition values and generated designs for condition-stratified analysis
    scalar_cols = [c for c in sampled_conditions.column_names if np.asarray(sampled_conditions[0][c]).ndim == 0]
    cond_array = np.column_stack([np.array(sampled_conditions[c]) for c in scalar_cols])
    per_sample_data["conditions"] = cond_array
    per_sample_data["condition_names"] = np.array(scalar_cols)
    per_sample_data["gen_designs"] = gen_designs_np
    per_sample_data["ref_designs"] = sampled_designs_np
    if per_sample_data:
        npz_path = output_csv.format(problem_id=problem_id).replace(".csv", f"_seed{seed}_per_sample.npz")
        np.savez(npz_path, **per_sample_data)
        print(f"  Per-sample data saved to {npz_path}")


def save_metrics_csv(
    metrics_dict: dict[str, Any],
    output_csv: str,
    problem_id: str,
    seed: int,
) -> None:
    """Filter to scalars, append one row to CSV."""
    csv_dict = {k: v for k, v in metrics_dict.items() if not isinstance(v, (list, np.ndarray))}
    metrics_df = pd.DataFrame([csv_dict])
    out_path = output_csv.format(problem_id=problem_id)
    write_header = not os.path.exists(out_path)
    metrics_df.to_csv(out_path, mode="a", header=write_header, index=False)
    print(f"Seed {seed} done; appended to {out_path}")
