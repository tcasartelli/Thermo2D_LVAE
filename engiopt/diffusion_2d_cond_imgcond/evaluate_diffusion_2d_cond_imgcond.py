"""Evaluation for the Diffusion 2D conditional model with image conditions.

Image conditions are concatenated as extra UNet input channels (standard approach).
Scalar conditions go through cross-attention via encoder_hidden_states.
"""

from __future__ import annotations

import dataclasses
import os

from diffusers import UNet2DConditionModel
import numpy as np
import torch as th
import tyro

from engiopt import metrics
from engiopt.diffusion_2d_cond_imgcond.diffusion_2d_cond_imgcond import beta_schedule
from engiopt.diffusion_2d_cond_imgcond.diffusion_2d_cond_imgcond import DiffusionSampler
from engiopt.eval_utils import BaseEvaluationArgs
from engiopt.eval_utils import log_base_metrics_to_wandb
from engiopt.eval_utils import parse_thresholds
from engiopt.eval_utils import run_lvae_loop
from engiopt.eval_utils import RunRetrievalError
from engiopt.eval_utils import save_metrics_csv
from engiopt.eval_utils import save_per_sample_npz
from engiopt.eval_utils import setup_evaluation
from engiopt.transforms import get_image_condition_keys
from engiopt.transforms import get_scalar_condition_keys
import wandb


@dataclasses.dataclass
class Args(BaseEvaluationArgs):
    """Command-line arguments for a single-seed Diffusion 2D Conditional (image-conditioned) evaluation."""

    output_csv: str = "diffusion_2d_cond_imgcond_{problem_id}_metrics.csv"
    """Output CSV path template; may include {problem_id}."""


if __name__ == "__main__":
    args = tyro.cli(Args)
    rec_thresholds, perf_thresholds = parse_thresholds(args)
    seed = args.seed

    problem, device, rng, conditions_tensor, sampled_conditions, sampled_designs_np, selected_indices = setup_evaluation(
        args
    )

    # Image conditions
    img_cond_keys = get_image_condition_keys(problem, problem.dataset["test"])
    n_img_conds = len(img_cond_keys)
    has_img_conds = n_img_conds > 0
    img_conditions_tensor: th.Tensor | None = None
    if has_img_conds:
        img_tensors = []
        for key in img_cond_keys:
            arr = np.array(sampled_conditions[key])
            t = th.tensor(arr, dtype=th.float32, device=device)
            _ndim_no_channel = 3
            if t.ndim == _ndim_no_channel:
                t = t.unsqueeze(1)
            img_tensors.append(t)
        img_conditions_tensor = th.cat(img_tensors, dim=1)

    ### Set Up Diffusion Model ###
    if args.wandb_entity is not None:
        artifact_path = (
            f"{args.wandb_entity}/{args.wandb_project}/{args.problem_id}_diffusion_2d_cond_imgcond_model:seed_{seed}"
        )
    else:
        artifact_path = f"{args.wandb_project}/{args.problem_id}_diffusion_2d_cond_imgcond_model:seed_{seed}"

    api = wandb.Api()
    artifact = api.artifact(artifact_path, type="model")

    run = artifact.logged_by()
    if run is None or not hasattr(run, "config"):
        raise RunRetrievalError

    artifact_dir = artifact.download()
    ckpt_path = os.path.join(artifact_dir, "model.pth")
    ckpt = th.load(ckpt_path, map_location=device)

    # UNet: in_channels = 1 + n_img_conds (image conditions are extra input channels)
    # encoder_hid_dim = n_scalar_conds (only scalars go through cross-attention)
    n_scalar_conds = len(get_scalar_condition_keys(problem, problem.dataset["test"]))
    unet_in_channels = 1 + n_img_conds

    model = UNet2DConditionModel(
        sample_size=problem.design_space.shape,
        in_channels=unet_in_channels,
        out_channels=1,
        cross_attention_dim=64,
        block_out_channels=(32, 64, 128, 256),
        down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        layers_per_block=run.config["layers_per_block"],
        transformer_layers_per_block=1,
        encoder_hid_dim=n_scalar_conds,
        only_cross_attention=True,
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()

    options = {
        "cosine": run.config["noise_schedule"] == "cosine",
        "exp_biasing": run.config["noise_schedule"] == "exp",
        "exp_bias_factor": 1,
    }
    betas = beta_schedule(t=run.config["num_timesteps"], start=1e-4, end=0.02, scale=1.0, options=options)
    ddm_sampler = DiffusionSampler(run.config["num_timesteps"], betas)

    # Scalar conditions for cross-attention: (n_samples, 1, n_scalar_conds)
    encoder_hidden_states = conditions_tensor.unsqueeze(1)

    # Resize image conditions to design spatial dims (for UNet input channel concat)
    design_shape: tuple = problem.design_space.shape
    resized_img_conds: th.Tensor | None = None
    if has_img_conds and img_conditions_tensor is not None:
        resized_img_conds = th.nn.functional.interpolate(
            img_conditions_tensor,
            size=(design_shape[0], design_shape[1]),
            mode="bilinear",
            align_corners=False,
        )

    # Generate designs via iterative denoising
    gen_designs = th.randn((args.n_samples, 1, *design_shape), device=device)
    assert run.config["num_timesteps"] is not None
    for i in reversed(range(run.config["num_timesteps"])):
        t = th.full((args.n_samples,), i, device=device, dtype=th.long)
        gen_designs = ddm_sampler.sample_timestep(model, gen_designs, t, encoder_hidden_states, img_conds=resized_img_conds)

    gen_designs = gen_designs.squeeze(1)
    gen_designs_np = gen_designs.detach().cpu().numpy().reshape(args.n_samples, *problem.design_space.shape)
    gen_designs_np = np.clip(gen_designs_np, 1e-3, 1.0)

    ### Shared: metrics, LVAE loop, save ###
    metrics_dict = metrics.metrics(problem, gen_designs_np, sampled_designs_np, sampled_conditions, sigma=args.sigma)
    metrics_dict.update(
        {"seed": seed, "problem_id": args.problem_id, "model_id": "diffusion_2d_cond_imgcond", "n_samples": args.n_samples}
    )
    per_sample_data: dict[str, np.ndarray] = {}

    if args.lvae_seed is not None and rec_thresholds and perf_thresholds:
        run_lvae_loop(
            args=args,
            rec_thresholds=rec_thresholds,
            perf_thresholds=perf_thresholds,
            gen_designs_np=gen_designs_np,
            sampled_designs_np=sampled_designs_np,
            problem=problem,
            device=device,
            metrics_dict=metrics_dict,
            per_sample_data=per_sample_data,
            run=run,
        )
    elif args.log_to_wandb and run is not None:
        log_base_metrics_to_wandb(run, metrics_dict)

    save_per_sample_npz(
        metrics_dict,
        per_sample_data,
        sampled_conditions,
        gen_designs_np,
        sampled_designs_np,
        args.output_csv,
        args.problem_id,
        seed,
    )
    save_metrics_csv(metrics_dict, args.output_csv, args.problem_id, seed)
