"""Lightweight LV metrics evaluation - generates designs once, loops over LVAE buckets.

This script is designed to be efficient by:
1. Loading the generative model and generating designs ONCE
2. Looping over multiple LVAE configurations to compute LV-MMD/LV-DPP
3. Skipping expensive simulation-based metrics (IOG/COG/FOG)
"""

from __future__ import annotations

import dataclasses
import os
from typing import Literal

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

# LVAE threshold combinations per problem — must match evaluate_*.py and the notebook.
# Format: (rec_threshold, perf_threshold); perf=1000.0 is the recon-only ablation.
LVAE_BUCKETS: dict[str, list[tuple[float, float]]] = {
    "beams2d": [
        (0.0005, 0.001), (0.0005, 1000.0),
        (0.001,  0.001), (0.001,  1000.0),
        (0.0025, 0.001), (0.0025, 1000.0),
    ],
    "heatconduction2d": [
        (0.001, 0.001), (0.001, 1000.0),
        (0.005, 0.001), (0.005, 1000.0),
        (0.01,  0.001), (0.01,  1000.0),
    ],
    "thermoelastic2d": [
        (0.001, 0.001), (0.001, 1000.0),
        (0.005, 0.001), (0.005, 1000.0),
        (0.01,  0.001), (0.01,  1000.0),
    ],
    "photonics2d": [
        (0.1,  0.005), (0.1,  1000.0),
        (0.15, 0.005), (0.15, 1000.0),
        (0.2,  0.005), (0.2,  1000.0),
    ],
}


@dataclasses.dataclass
class Args:
    """Command-line arguments for LV-only evaluation."""

    problem_id: str = "beams2d"
    """Problem identifier."""
    gen_model: Literal["cgan_cnn_2d", "gan_cnn_2d", "diffusion_2d_cond", "vqgan"] = "cgan_cnn_2d"
    """Generative model type."""
    gen_seed: int = 1
    """Generative model seed."""
    n_samples: int = 50
    """Number of samples to generate."""
    lvae_seeds: tuple[int, ...] = (1, 2, 3)
    """LVAE seeds to evaluate."""
    wandb_project: str = "engiopt"
    """WandB project name."""
    wandb_entity: str | None = None
    """WandB entity name."""
    log_to_wandb: bool = False
    """Log metrics to the original WandB training run."""
    output_csv: str = "lv_only_{problem_id}_{gen_model}.csv"
    """Output CSV path template."""


def load_generator_and_generate(
    args: Args,
    problem,
    conditions_tensor: th.Tensor,
    device: th.device,
) -> tuple[np.ndarray, wandb.apis.public.Run | None]:
    """Load generator from WandB and generate designs.

    Returns:
        Tuple of (generated designs as numpy array, WandB run object).
    """
    api = wandb.Api()

    if args.gen_model == "cgan_cnn_2d":
        from engiopt.cgan_cnn_2d.cgan_cnn_2d import Generator

        if args.wandb_entity:
            artifact_path = f"{args.wandb_entity}/{args.wandb_project}/{args.problem_id}_cgan_cnn_2d_generator:seed_{args.gen_seed}"
        else:
            artifact_path = f"{args.wandb_project}/{args.problem_id}_cgan_cnn_2d_generator:seed_{args.gen_seed}"

        artifact = api.artifact(artifact_path, type="model")
        run = artifact.logged_by()
        artifact_dir = artifact.download()

        ckpt = th.load(os.path.join(artifact_dir, "generator.pth"), map_location=device)
        model = Generator(
            latent_dim=run.config["latent_dim"],
            n_conds=len(get_scalar_condition_keys(problem, problem.dataset["test"])),
            design_shape=problem.design_space.shape,
        )
        model.load_state_dict(ckpt["generator"])
        model.eval().to(device)

        z = th.randn((args.n_samples, run.config["latent_dim"], 1, 1), device=device)
        cond = conditions_tensor.unsqueeze(-1).unsqueeze(-1)
        gen_designs = model(z, cond)

    elif args.gen_model == "gan_cnn_2d":
        from engiopt.gan_cnn_2d.gan_cnn_2d import Generator

        if args.wandb_entity:
            artifact_path = f"{args.wandb_entity}/{args.wandb_project}/{args.problem_id}_gan_cnn_2d_generator:seed_{args.gen_seed}"
        else:
            artifact_path = f"{args.wandb_project}/{args.problem_id}_gan_cnn_2d_generator:seed_{args.gen_seed}"

        artifact = api.artifact(artifact_path, type="model")
        run = artifact.logged_by()
        artifact_dir = artifact.download()

        ckpt = th.load(os.path.join(artifact_dir, "generator.pth"), map_location=device)
        model = Generator(latent_dim=run.config["latent_dim"], design_shape=problem.design_space.shape)
        model.load_state_dict(ckpt["generator"])
        model.eval().to(device)

        z = th.randn((args.n_samples, run.config["latent_dim"], 1, 1), device=device)
        gen_designs = model(z)

    elif args.gen_model == "diffusion_2d_cond":
        from diffusers import UNet2DConditionModel

        from engiopt.diffusion_2d_cond.diffusion_2d_cond import beta_schedule
        from engiopt.diffusion_2d_cond.diffusion_2d_cond import DiffusionSampler

        if args.wandb_entity:
            artifact_path = f"{args.wandb_entity}/{args.wandb_project}/{args.problem_id}_diffusion_2d_cond_model:seed_{args.gen_seed}"
        else:
            artifact_path = f"{args.wandb_project}/{args.problem_id}_diffusion_2d_cond_model:seed_{args.gen_seed}"

        artifact = api.artifact(artifact_path, type="model")
        run = artifact.logged_by()
        artifact_dir = artifact.download()

        ckpt = th.load(os.path.join(artifact_dir, "model.pth"), map_location=device)
        model = UNet2DConditionModel(
            sample_size=problem.design_space.shape,
            in_channels=1,
            out_channels=1,
            cross_attention_dim=64,
            block_out_channels=(32, 64, 128, 256),
            down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
            layers_per_block=run.config["layers_per_block"],
            transformer_layers_per_block=1,
            encoder_hid_dim=len(get_scalar_condition_keys(problem, problem.dataset["test"])),
            only_cross_attention=True,
        ).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        options = {
            "cosine": run.config["noise_schedule"] == "cosine",
            "exp_biasing": run.config["noise_schedule"] == "exp",
            "exp_bias_factor": 1,
        }
        betas = beta_schedule(run.config["num_timesteps"], 1e-4, 0.02, 1.0, options)
        sampler = DiffusionSampler(run.config["num_timesteps"], betas)

        cond = conditions_tensor.unsqueeze(1)
        gen_designs = th.randn((args.n_samples, 1, *problem.design_space.shape), device=device)
        for i in reversed(range(run.config["num_timesteps"])):
            t = th.full((args.n_samples,), i, device=device, dtype=th.long)
            gen_designs = sampler.sample_timestep(model, gen_designs, t, cond)
        gen_designs = gen_designs.squeeze(1)

    elif args.gen_model == "vqgan":
        from engiopt.transforms import drop_constant
        from engiopt.transforms import normalize
        from engiopt.transforms import resize_to
        from engiopt.vqgan.vqgan import VQGAN
        from engiopt.vqgan.vqgan import VQGANTransformer

        if args.wandb_entity:
            artifact_path_0 = f"{args.wandb_entity}/{args.wandb_project}/{args.problem_id}_vqgan_cvqgan:seed_{args.gen_seed}"
            artifact_path_1 = f"{args.wandb_entity}/{args.wandb_project}/{args.problem_id}_vqgan_vqgan:seed_{args.gen_seed}"
            artifact_path_2 = f"{args.wandb_entity}/{args.wandb_project}/{args.problem_id}_vqgan_transformer:seed_{args.gen_seed}"
        else:
            artifact_path_0 = f"{args.wandb_project}/{args.problem_id}_vqgan_cvqgan:seed_{args.gen_seed}"
            artifact_path_1 = f"{args.wandb_project}/{args.problem_id}_vqgan_vqgan:seed_{args.gen_seed}"
            artifact_path_2 = f"{args.wandb_project}/{args.problem_id}_vqgan_transformer:seed_{args.gen_seed}"

        artifact_0 = api.artifact(artifact_path_0, type="model")
        artifact_1 = api.artifact(artifact_path_1, type="model")
        artifact_2 = api.artifact(artifact_path_2, type="model")
        run = artifact_2.logged_by()

        ckpt_0 = th.load(os.path.join(artifact_0.download(), "cvqgan.pth"), map_location=device, weights_only=False)
        ckpt_1 = th.load(os.path.join(artifact_1.download(), "vqgan.pth"), map_location=device, weights_only=False)
        ckpt_2 = th.load(os.path.join(artifact_2.download(), "transformer.pth"), map_location=device, weights_only=False)

        vqgan = VQGAN(
            device=device,
            is_c=False,
            encoder_channels=run.config["encoder_channels"],
            encoder_start_resolution=run.config["image_size"],
            encoder_attn_resolutions=run.config["encoder_attn_resolutions"],
            encoder_num_res_blocks=run.config["encoder_num_res_blocks"],
            decoder_channels=run.config["decoder_channels"],
            decoder_start_resolution=run.config["latent_size"],
            decoder_attn_resolutions=run.config["decoder_attn_resolutions"],
            decoder_num_res_blocks=run.config["decoder_num_res_blocks"],
            image_channels=run.config["image_channels"],
            latent_dim=run.config["latent_dim"],
            num_codebook_vectors=run.config["num_codebook_vectors"],
        )
        vqgan.load_state_dict(ckpt_1["vqgan"])
        vqgan.eval().to(device)

        cvqgan = VQGAN(
            device=device,
            is_c=True,
            cond_feature_map_dim=run.config["cond_feature_map_dim"],
            cond_dim=run.config["cond_dim"],
            cond_hidden_dim=run.config["cond_hidden_dim"],
            cond_latent_dim=run.config["cond_latent_dim"],
            cond_codebook_vectors=run.config["cond_codebook_vectors"],
        )
        cvqgan.load_state_dict(ckpt_0["cvqgan"])
        cvqgan.eval().to(device)

        transformer = VQGANTransformer(
            conditional=run.config["conditional"],
            vqgan=vqgan,
            cvqgan=cvqgan,
            image_size=run.config["image_size"],
            decoder_channels=run.config["decoder_channels"],
            cond_feature_map_dim=run.config["cond_feature_map_dim"],
            num_codebook_vectors=run.config["num_codebook_vectors"],
            n_layer=run.config["n_layer"],
            n_head=run.config["n_head"],
            n_embd=run.config["n_embd"],
            dropout=run.config["dropout"],
        )
        transformer.load_state_dict(ckpt_2["transformer"])
        transformer.eval().to(device)

        # Process conditions for VQGAN
        # Note: This is a simplified version - may need adjustment based on training config
        if run.config["conditional"]:
            c = transformer.encode_to_z(x=conditions_tensor, is_c=True)[1]
        else:
            c = th.ones(args.n_samples, 1, dtype=th.int64, device=device) * transformer.sos_token

        latent_designs = transformer.sample(
            x=th.empty(args.n_samples, 0, dtype=th.int64, device=device),
            c=c,
            steps=(run.config["latent_size"] ** 2),
        )
        gen_designs = resize_to(transformer.z_to_image(latent_designs), problem.design_space.shape[0], problem.design_space.shape[1])

    else:
        raise ValueError(f"Unknown gen_model: {args.gen_model}")

    gen_designs_np = gen_designs.detach().cpu().numpy()
    gen_designs_np = gen_designs_np.reshape(args.n_samples, *problem.design_space.shape)
    gen_designs_np = np.clip(gen_designs_np, 1e-3, 1.0)

    return gen_designs_np, run


if __name__ == "__main__":
    args = tyro.cli(Args)

    # Setup
    problem = BUILTIN_PROBLEMS[args.problem_id]()
    problem.reset(seed=args.gen_seed)

    th.manual_seed(args.gen_seed)
    np.random.seed(args.gen_seed)
    th.backends.cudnn.deterministic = True

    if th.backends.mps.is_available():
        device = th.device("mps")
    elif th.cuda.is_available():
        device = th.device("cuda")
    else:
        device = th.device("cpu")

    print(f"Problem: {args.problem_id}, Model: {args.gen_model}, Seed: {args.gen_seed}")
    print(f"Device: {device}")

    # Sample conditions and dataset designs
    conditions_tensor, sampled_conditions, sampled_designs_np, selected_indices = sample_conditions(
        problem=problem, n_samples=args.n_samples, device=device, seed=args.gen_seed
    )

    # Extract objective values for performance-conditioned MMD
    test_ds = problem.dataset["test"]
    objective_values = np.array(test_ds[problem.objectives_keys[0]])[selected_indices]

    # Precompute condition arrays for conditional MMD
    scalar_cols = [c for c in sampled_conditions.column_names if np.asarray(sampled_conditions[0][c]).ndim == 0]
    cond_array = np.column_stack([np.array(sampled_conditions[c]) for c in scalar_cols])
    obj_array = objective_values.reshape(-1, 1)

    # Generate designs ONCE
    print(f"Generating {args.n_samples} designs...")
    gen_designs_np, gen_run = load_generator_and_generate(args, problem, conditions_tensor, device)
    print(f"  Generated designs shape: {gen_designs_np.shape}")

    # Get LVAE buckets for this problem
    buckets = LVAE_BUCKETS.get(args.problem_id, LVAE_BUCKETS["beams2d"])

    # Loop over LVAE configurations
    results = []
    for lvae_seed in args.lvae_seeds:
        for rec_thresh, perf_thresh in buckets:
            print(f"  LVAE: seed={lvae_seed}, rec={rec_thresh}, perf={perf_thresh}")

            try:
                from sklearn.decomposition import PCA

                encoder, lvae_config = load_lvae_encoder(
                    problem_id=args.problem_id,
                    seed=lvae_seed,
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
                lv_cond_mmd = metrics.conditional_mmd(z_gen, z_data, cond_array, sigma=lv_sigma)
                lv_cond_perf_mmd = metrics.conditional_mmd(z_gen, z_data, obj_array, sigma=lv_sigma)

                print(f"    LV-MMD: {lv_mmd_val:.6f}, LV-DPP: {lv_dpp_val:.6e}, Active: {n_active}")

                # PCA baseline at same dimensionality as LVAE active dims
                pca_mmd_val = pca_dpp_val = pca_sigma = pca_cond_mmd_val = pca_cond_perf_mmd_val = float("nan")
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
                    pca_cond_mmd_val = metrics.conditional_mmd(pca_gen, pca_data, cond_array, sigma=pca_sigma)["cond_mmd"]
                    pca_cond_perf_mmd_val = metrics.conditional_mmd(pca_gen, pca_data, obj_array, sigma=pca_sigma)["cond_mmd"]
                    print(f"    PCA({n_pca}): MMD={pca_mmd_val:.6f}")

                row = {
                    "problem_id": args.problem_id,
                    "gen_model": args.gen_model,
                    "gen_seed": args.gen_seed,
                    "lvae_seed": lvae_seed,
                    "lvae_rec_threshold": rec_thresh,
                    "lvae_perf_threshold": perf_thresh,
                    "lv_mmd": lv_mmd_val,
                    "lv_dpp": lv_dpp_val,
                    "lv_sigma": lv_sigma,
                    "lvae_n_active_dims": n_active,
                    "lv_cond_mmd": lv_cond_mmd["cond_mmd"],
                    "lv_cond_perf_mmd": lv_cond_perf_mmd["cond_mmd"],
                    "pca_mmd": pca_mmd_val,
                    "pca_dpp": pca_dpp_val,
                    "pca_sigma": pca_sigma,
                    "pca_cond_mmd": pca_cond_mmd_val,
                    "pca_cond_perf_mmd": pca_cond_perf_mmd_val,
                    "n_samples": args.n_samples,
                }
                results.append(row)

                # Log to WandB immediately so partial results survive a timeout
                if args.log_to_wandb and gen_run is not None:
                    prefix = f"eval/lv_rec{rec_thresh}_perf{perf_thresh}_lvae{lvae_seed}"
                    gen_run.summary[f"{prefix}/lv_mmd"] = lv_mmd_val
                    gen_run.summary[f"{prefix}/lv_dpp"] = lv_dpp_val
                    gen_run.summary[f"{prefix}/lv_sigma"] = lv_sigma
                    gen_run.summary[f"{prefix}/n_active_dims"] = n_active
                    gen_run.summary[f"{prefix}/lv_cond_mmd"] = lv_cond_mmd["cond_mmd"]
                    gen_run.summary[f"{prefix}/lv_cond_perf_mmd"] = lv_cond_perf_mmd["cond_mmd"]
                    gen_run.summary[f"{prefix}/pca_mmd"] = pca_mmd_val
                    gen_run.summary[f"{prefix}/pca_dpp"] = pca_dpp_val
                    gen_run.summary[f"{prefix}/pca_sigma"] = pca_sigma
                    gen_run.summary[f"{prefix}/pca_cond_mmd"] = pca_cond_mmd_val
                    gen_run.summary[f"{prefix}/pca_cond_perf_mmd"] = pca_cond_perf_mmd_val
                    gen_run.summary.update()

            except Exception as e:
                print(f"    Failed: {e}")

    # Save results to CSV
    if results:
        df = pd.DataFrame(results)
        out_path = args.output_csv.format(problem_id=args.problem_id, gen_model=args.gen_model)
        write_header = not os.path.exists(out_path)
        df.to_csv(out_path, mode="a", header=write_header, index=False)
        print(f"\nResults appended to {out_path}")
    else:
        print("\nNo results to save.")
