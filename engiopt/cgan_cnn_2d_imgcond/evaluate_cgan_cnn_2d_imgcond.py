"""Evaluation for the CGAN 2D w/ CNN with image condition support."""

from __future__ import annotations

import dataclasses
import os

import numpy as np
import torch as th
import tyro

from engiopt import metrics
from engiopt.cgan_cnn_2d_imgcond.cgan_cnn_2d_imgcond import Generator
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
    """Command-line arguments for a single-seed cGAN CNN 2D (image-conditioned) evaluation."""

    run_id: str | None = None
    """WandB run ID to evaluate (e.g. from a sweep). If provided, overrides seed-based artifact lookup."""
    output_csv: str = "cgan_cnn_2d_imgcond_{problem_id}_metrics.csv"
    """Output CSV path template; may include {problem_id}."""


if __name__ == "__main__":
    args = tyro.cli(Args)
    rec_thresholds, perf_thresholds = parse_thresholds(args)
    seed = args.seed

    problem, device, rng, conditions_tensor, sampled_conditions, sampled_designs_np, selected_indices = setup_evaluation(
        args
    )

    # Reshape scalar conditions for CNN: (B, n_scalar_conds) -> (B, n_scalar_conds, 1, 1)
    conditions_tensor = conditions_tensor.unsqueeze(-1).unsqueeze(-1)

    # Image conditions
    img_cond_keys = get_image_condition_keys(problem, problem.dataset["test"])
    n_img_conds = len(img_cond_keys)
    img_conditions_tensor: th.Tensor | None = None
    if n_img_conds > 0:
        img_tensors = []
        for key in img_cond_keys:
            arr = np.array(sampled_conditions[key])
            t = th.tensor(arr, dtype=th.float32, device=device)
            _ndim_no_channel = 3
            if t.ndim == _ndim_no_channel:  # (N, H, W) -> (N, 1, H, W)
                t = t.unsqueeze(1)
            img_tensors.append(t)
        img_conditions_tensor = th.cat(img_tensors, dim=1)  # (N, n_img_conds, H, W)

    ### Set Up Generator ###
    alias = f"run_{args.run_id}" if args.run_id is not None else f"seed_{seed}"
    if args.wandb_entity is not None:
        artifact_path = f"{args.wandb_entity}/{args.wandb_project}/{args.problem_id}_cgan_cnn_2d_imgcond_generator:{alias}"
    else:
        artifact_path = f"{args.wandb_project}/{args.problem_id}_cgan_cnn_2d_imgcond_generator:{alias}"

    api = wandb.Api()
    artifact = api.artifact(artifact_path, type="model")

    run = artifact.logged_by()
    if run is None or not hasattr(run, "config"):
        raise RunRetrievalError
    artifact_dir = artifact.download()

    ckpt_path = os.path.join(artifact_dir, "generator.pth")
    ckpt = th.load(ckpt_path, map_location=th.device(device))
    model = Generator(
        latent_dim=run.config["latent_dim"],
        n_scalar_conds=len(get_scalar_condition_keys(problem, problem.dataset["test"])),
        n_img_conds=n_img_conds,
        design_shape=problem.design_space.shape,
    )
    model.load_state_dict(ckpt["generator"])
    model.eval()
    model.to(device)

    z = th.randn((args.n_samples, run.config["latent_dim"], 1, 1), device=device, dtype=th.float)
    gen_designs = model(z, conditions_tensor, img_conditions_tensor)
    gen_designs_np = gen_designs.detach().cpu().numpy().reshape(args.n_samples, *problem.design_space.shape)
    gen_designs_np = np.clip(gen_designs_np, 1e-3, 1)

    ### Shared: metrics, LVAE loop, save ###
    metrics_dict = metrics.metrics(problem, gen_designs_np, sampled_designs_np, sampled_conditions, sigma=args.sigma)
    metrics_dict.update(
        {"seed": seed, "problem_id": args.problem_id, "model_id": "cgan_cnn_2d_imgcond", "n_samples": args.n_samples}
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
