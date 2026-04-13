"""Batch evaluation of HP tuning sweep runs.

For each finished CGAN-CNN-2D run in a WandB HP tuning sweep:
1. Load the generator from the run's artifacts
2. Generate designs from shared sampled conditions
3. Compute cheap metrics: pixel-MMD, LV-MMD (perf), LV-MMD (no-perf)
4. Optionally compute expensive metrics: COG, IOG, FOG via simulation
5. Save results to CSV and optionally log back to WandB
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any

from engibench.utils.all_problems import BUILTIN_PROBLEMS
import numpy as np
import pandas as pd
import torch as th
import tyro

from engiopt import metrics
from engiopt.dataset_sample_conditions import sample_conditions
from engiopt.transforms import get_scalar_condition_keys
from engiopt.vanilla_lvae.utils import encode_designs
from engiopt.vanilla_lvae.utils import get_active_mask
from engiopt.vanilla_lvae.utils import load_lvae_encoder
import wandb

# Best LVAE config per problem: {problem: {"perf": (rec, perf), "no_perf": (rec, perf_off)}}
LVAE_CONFIGS: dict[str, dict[str, tuple[float, float]]] = {
    "beams2d": {"perf": (0.001, 0.001), "no_perf": (0.001, 1000.0)},
    "photonics2d": {"perf": (0.1, 0.005), "no_perf": (0.1, 1000.0)},
    "heatconduction2d": {"perf": (0.005, 0.005), "no_perf": (0.005, 1000.0)},
}


@dataclasses.dataclass
class Args:
    """Arguments for batch HP sweep evaluation."""

    problem_id: str = "beams2d"
    """Problem identifier."""
    sweep_project: str = "lv-mmd-hp-tuning"
    """WandB project containing the HP sweep runs."""
    wandb_entity: str | None = None
    """WandB entity (org/user). None for default."""
    lvae_project: str = "engiopt"
    """WandB project where pre-trained LVAE models are stored."""
    n_samples: int = 50
    """Number of designs to generate per sweep run."""
    lvae_seed: int = 1
    """Seed of the pre-trained LVAE encoder to use."""
    compute_cog: bool = False
    """If True, run expensive COG/IOG/FOG evaluation (requires simulation)."""
    output_csv: str = "hp_sweep_{problem_id}_results.csv"
    """Output CSV path template."""
    log_to_wandb: bool = False
    """Log evaluation metrics back to each sweep run's WandB summary."""
    sweep_id: str | None = None
    """Optional: evaluate only runs from this specific WandB sweep ID."""
    skip_evaluated: bool = True
    """Skip runs that already have eval/pixel_mmd in their summary."""


def load_generator_from_run(
    run: wandb.apis.public.Run,
    problem,
    device: th.device,
) -> th.nn.Module | None:
    """Load a CGAN-CNN-2D generator from a sweep run's logged artifacts.

    Args:
        run: WandB run object from the sweep.
        problem: EngiBench problem instance.
        device: Device to load the model onto.

    Returns:
        Loaded Generator model, or None if no artifact found.
    """
    from engiopt.cgan_cnn_2d.cgan_cnn_2d import Generator

    for artifact in run.logged_artifacts():
        if "generator" in artifact.name:
            artifact_dir = artifact.download()
            ckpt = th.load(
                os.path.join(artifact_dir, "generator.pth"),
                map_location=device,
            )
            model = Generator(
                latent_dim=run.config["latent_dim"],
                n_conds=len(get_scalar_condition_keys(problem, problem.dataset["test"])),
                design_shape=problem.design_space.shape,
            )
            model.load_state_dict(ckpt["generator"])
            model.eval().to(device)
            return model
    return None


def generate_designs(
    model: th.nn.Module,
    latent_dim: int,
    conditions_tensor: th.Tensor,
    n_samples: int,
    design_shape: tuple,
    device: th.device,
) -> np.ndarray:
    """Generate designs using a CGAN-CNN-2D generator.

    Args:
        model: Loaded Generator model.
        latent_dim: Latent dimension of the generator.
        conditions_tensor: Conditions tensor of shape (n_samples, n_conds).
        n_samples: Number of designs to generate.
        design_shape: Shape of a single design (H, W).
        device: Device for generation.

    Returns:
        Generated designs as numpy array, shape (n_samples, H, W), clipped to [1e-3, 1].
    """
    with th.no_grad():
        z = th.randn((n_samples, latent_dim, 1, 1), device=device)
        cond = conditions_tensor.unsqueeze(-1).unsqueeze(-1)
        gen_designs = model(z, cond)
    gen_designs_np = gen_designs.detach().cpu().numpy()
    gen_designs_np = gen_designs_np.reshape(n_samples, *design_shape)
    return np.clip(gen_designs_np, 1e-3, 1.0)


if __name__ == "__main__":
    args = tyro.cli(Args)

    # Setup
    problem = BUILTIN_PROBLEMS[args.problem_id]()
    problem.reset(seed=1)

    th.manual_seed(1)
    np.random.seed(1)  # noqa: NPY002
    th.backends.cudnn.deterministic = True

    if th.backends.mps.is_available():
        device = th.device("mps")
    elif th.cuda.is_available():
        device = th.device("cuda")
    else:
        device = th.device("cpu")

    print(f"Problem: {args.problem_id}, Device: {device}")

    # ── Sample conditions ONCE (shared across all sweep runs) ──
    conditions_tensor, sampled_conditions, sampled_designs_np, selected_indices = sample_conditions(
        problem=problem, n_samples=args.n_samples, device=device, seed=1,
    )

    test_ds = problem.dataset["test"]
    objective_values = np.array(test_ds[problem.objectives_keys[0]])[selected_indices]
    scalar_cols = [c for c in sampled_conditions.column_names if np.asarray(sampled_conditions[0][c]).ndim == 0]
    cond_array = np.column_stack([np.array(sampled_conditions[c]) for c in scalar_cols])
    obj_array = objective_values.reshape(-1, 1)

    # Pixel-space reference sigma (computed once from dataset designs)
    flat_ds = sampled_designs_np.reshape(sampled_designs_np.shape[0], -1)
    pixel_sigma = metrics.compute_median_sigma(flat_ds)
    print(f"Pixel sigma (median heuristic): {pixel_sigma:.6f}")

    # ── Load LVAE encoders (perf + no_perf) ──
    lvae_cfgs = LVAE_CONFIGS.get(args.problem_id, LVAE_CONFIGS["beams2d"])
    encoders: dict[str, tuple[th.nn.Module, np.ndarray, float, int]] = {}

    for label, (rec_t, perf_t) in lvae_cfgs.items():
        try:
            encoder, lvae_config = load_lvae_encoder(
                problem_id=args.problem_id,
                seed=args.lvae_seed,
                rec_threshold=rec_t,
                perf_threshold=perf_t,
                wandb_project=args.lvae_project,
                wandb_entity=args.wandb_entity,
                device=device,
            )
            z_data = encode_designs(encoder, sampled_designs_np, device)
            active = get_active_mask(encoder)
            n_active = int(active.sum())
            z_data = z_data[:, active]
            lv_sigma = metrics.compute_median_sigma(z_data)
            encoders[label] = (encoder, active, z_data, lv_sigma, n_active)
            print(f"LVAE ({label}): rec={rec_t}, perf={perf_t}, active_dims={n_active}, sigma={lv_sigma:.6f}")
        except Exception as e:  # noqa: BLE001
            print(f"Failed to load LVAE ({label}): {e}")

    # ── Query WandB for sweep runs ──
    api = wandb.Api()
    project_path = f"{args.wandb_entity}/{args.sweep_project}" if args.wandb_entity else args.sweep_project

    filters: dict[str, Any] = {
        "config.problem_id": args.problem_id,
        "state": "finished",
    }
    if args.sweep_id:
        filters["sweep"] = args.sweep_id

    runs = list(api.runs(project_path, filters=filters))
    print(f"\nFound {len(runs)} finished runs for {args.problem_id}")

    # ── Evaluate each run ──
    results: list[dict[str, Any]] = []

    for i, run in enumerate(runs):
        run_name = run.name or run.id
        print(f"\n[{i + 1}/{len(runs)}] Run: {run_name} (id={run.id})")

        # Skip already-evaluated runs
        if args.skip_evaluated and run.summary.get("eval/pixel_mmd") is not None:
            print("  Already evaluated, skipping.")
            continue

        # Load generator from this run's artifacts
        model = load_generator_from_run(run, problem, device)
        if model is None:
            print("  No generator artifact found, skipping.")
            continue

        # Generate designs
        gen_designs_np = generate_designs(
            model,
            run.config["latent_dim"],
            conditions_tensor,
            args.n_samples,
            problem.design_space.shape,
            device,
        )
        print(f"  Generated {args.n_samples} designs, shape: {gen_designs_np.shape}")

        # Result row: run config + metrics
        row: dict[str, Any] = {
            "run_id": run.id,
            "run_name": run_name,
            "problem_id": args.problem_id,
            "latent_dim": run.config.get("latent_dim"),
            "lr_gen": run.config.get("lr_gen"),
            "lr_disc": run.config.get("lr_disc"),
            "batch_size": run.config.get("batch_size"),
            "n_epochs": run.config.get("n_epochs"),
            "n_samples": args.n_samples,
        }

        # ── Pixel-space metrics (cheap) ──
        flat_gen = gen_designs_np.reshape(gen_designs_np.shape[0], -1)
        pixel_mmd = metrics.mmd(flat_gen, flat_ds, sigma=pixel_sigma)
        pixel_dpp = metrics.dpp_diversity(flat_gen, sigma=pixel_sigma)
        pixel_cond_mmd = metrics.conditional_mmd(flat_gen, flat_ds, cond_array, sigma=pixel_sigma)
        pixel_cond_perf_mmd = metrics.conditional_mmd(flat_gen, flat_ds, obj_array, sigma=pixel_sigma)

        row["pixel_mmd"] = pixel_mmd
        row["pixel_dpp"] = pixel_dpp
        row["pixel_sigma"] = pixel_sigma
        row["pixel_cond_mmd"] = pixel_cond_mmd["cond_mmd"]
        row["pixel_cond_perf_mmd"] = pixel_cond_perf_mmd["cond_mmd"]
        print(f"  Pixel-MMD: {pixel_mmd:.6f}, Pixel-DPP: {pixel_dpp:.6e}")

        # ── LV-space metrics (cheap) ──
        for label, (encoder, active, z_data, lv_sigma, n_active) in encoders.items():
            z_gen = encode_designs(encoder, gen_designs_np, device)[:, active]
            lv_mmd = metrics.mmd(z_gen, z_data, sigma=lv_sigma)
            lv_dpp = metrics.dpp_diversity(z_gen, sigma=lv_sigma)
            lv_cond_mmd = metrics.conditional_mmd(z_gen, z_data, cond_array, sigma=lv_sigma)
            lv_cond_perf_mmd = metrics.conditional_mmd(z_gen, z_data, obj_array, sigma=lv_sigma)

            row[f"lv_mmd_{label}"] = lv_mmd
            row[f"lv_dpp_{label}"] = lv_dpp
            row[f"lv_sigma_{label}"] = lv_sigma
            row[f"lv_n_active_{label}"] = n_active
            row[f"lv_cond_mmd_{label}"] = lv_cond_mmd["cond_mmd"]
            row[f"lv_cond_perf_mmd_{label}"] = lv_cond_perf_mmd["cond_mmd"]
            print(f"  LV-MMD ({label}): {lv_mmd:.6f}, LV-DPP: {lv_dpp:.6e}")

        # ── COG/IOG/FOG metrics (expensive, optional) ──
        if args.compute_cog:
            print("  Computing COG (expensive)...")
            cog_metrics = metrics.metrics(
                problem,
                gen_designs_np,
                sampled_designs_np,
                sampled_conditions,
                sigma=pixel_sigma,
                objective_values=objective_values,
            )
            row["iog"] = cog_metrics["iog"]
            row["cog"] = cog_metrics["cog"]
            row["fog"] = cog_metrics["fog"]
            row["iog_var"] = cog_metrics.get("iog_var", float("nan"))
            row["cog_var"] = cog_metrics.get("cog_var", float("nan"))
            row["fog_var"] = cog_metrics.get("fog_var", float("nan"))
            row["viol"] = cog_metrics.get("viol", float("nan"))
            print(f"  IOG: {cog_metrics['iog']:.4f}, COG: {cog_metrics['cog']:.4f}, FOG: {cog_metrics['fog']:.4f}")

        results.append(row)

        # ── Log to WandB ──
        if args.log_to_wandb:
            run.summary["eval/pixel_mmd"] = pixel_mmd
            run.summary["eval/pixel_dpp"] = pixel_dpp
            run.summary["eval/pixel_sigma"] = pixel_sigma
            run.summary["eval/pixel_cond_mmd"] = pixel_cond_mmd["cond_mmd"]
            run.summary["eval/pixel_cond_perf_mmd"] = pixel_cond_perf_mmd["cond_mmd"]
            for label in encoders:
                run.summary[f"eval/lv_mmd_{label}"] = row.get(f"lv_mmd_{label}")
                run.summary[f"eval/lv_dpp_{label}"] = row.get(f"lv_dpp_{label}")
                run.summary[f"eval/lv_sigma_{label}"] = row.get(f"lv_sigma_{label}")
                run.summary[f"eval/lv_n_active_{label}"] = row.get(f"lv_n_active_{label}")
                run.summary[f"eval/lv_cond_mmd_{label}"] = row.get(f"lv_cond_mmd_{label}")
                run.summary[f"eval/lv_cond_perf_mmd_{label}"] = row.get(f"lv_cond_perf_mmd_{label}")
            if args.compute_cog:
                run.summary["eval/iog"] = row.get("iog")
                run.summary["eval/cog"] = row.get("cog")
                run.summary["eval/fog"] = row.get("fog")
                run.summary["eval/iog_var"] = row.get("iog_var")
                run.summary["eval/cog_var"] = row.get("cog_var")
                run.summary["eval/fog_var"] = row.get("fog_var")
                run.summary["eval/viol"] = row.get("viol")
            run.summary.update()
            print(f"  Logged metrics to WandB run: {run_name}")

    # ── Save results to CSV ──
    if results:
        df = pd.DataFrame(results)
        out_path = args.output_csv.format(problem_id=args.problem_id)
        df.to_csv(out_path, index=False)
        print(f"\nResults saved to {out_path} ({len(results)} runs)")
    else:
        print("\nNo results to save.")
