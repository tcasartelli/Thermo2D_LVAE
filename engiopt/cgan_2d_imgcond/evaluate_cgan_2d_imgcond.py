"""Evaluation for the CGAN 2D with image conditions."""

from __future__ import annotations

import dataclasses
import os

from engibench.utils.all_problems import BUILTIN_PROBLEMS
import numpy as np
import pandas as pd
import torch as th
import tyro

from engiopt import metrics
from engiopt.cgan_2d_imgcond.cgan_2d_imgcond import Generator
from engiopt.dataset_sample_conditions import sample_conditions
from engiopt.transforms import get_image_condition_keys
from engiopt.transforms import get_scalar_condition_keys
import wandb


@dataclasses.dataclass
class Args:
    """Command-line arguments for a single-seed CGAN 2D image-conditioned evaluation."""

    problem_id: str = "beams2d"
    """Problem identifier (e.g. beams2d)."""
    seed: int = 1
    """Random seed to run."""
    wandb_project: str = "engiopt"
    """Wandb project name."""
    wandb_entity: str | None = None
    """Wandb entity name (if any)."""
    n_samples: int = 50
    """Number of generated samples per seed."""
    sigma: float = 10.0
    """Kernel bandwidth for MMD and DPP metrics."""
    output_csv: str = "cgan_2d_imgcond_{problem_id}_metrics.csv"
    """Output CSV path template; may include {problem_id}."""


if __name__ == "__main__":
    args = tyro.cli(Args)

    seed = args.seed
    problem = BUILTIN_PROBLEMS[args.problem_id]()
    problem.reset(seed=seed)

    # Reproducibility
    th.manual_seed(seed)
    rng = np.random.default_rng(seed)
    th.backends.cudnn.deterministic = True

    # Select device
    if th.backends.mps.is_available():
        device = th.device("mps")
    elif th.cuda.is_available():
        device = th.device("cuda")
    else:
        device = th.device("cpu")

    ### Set up testing conditions ###
    conditions_tensor, sampled_conditions, sampled_designs_np, _ = sample_conditions(
        problem=problem,
        n_samples=args.n_samples,
        device=device,
        seed=seed,
    )

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
            if t.ndim == _ndim_no_channel:
                t = t.unsqueeze(1)
            img_tensors.append(t)
        img_conditions_tensor = th.cat(img_tensors, dim=1)

    ### Set Up Generator ###
    if args.wandb_entity is not None:
        artifact_path = f"{args.wandb_entity}/{args.wandb_project}/{args.problem_id}_cgan_2d_imgcond_generator:seed_{seed}"
    else:
        artifact_path = f"{args.wandb_project}/{args.problem_id}_cgan_2d_imgcond_generator:seed_{seed}"

    api = wandb.Api()
    artifact = api.artifact(artifact_path, type="model")

    class RunRetrievalError(ValueError):
        def __init__(self):
            super().__init__("Failed to retrieve the run")

    run = artifact.logged_by()
    if run is None or not hasattr(run, "config"):
        raise RunRetrievalError

    artifact_dir = artifact.download()
    ckpt_path = os.path.join(artifact_dir, "generator.pth")
    ckpt = th.load(ckpt_path, map_location=device)

    model = Generator(
        latent_dim=run.config["latent_dim"],
        n_scalar_conds=len(get_scalar_condition_keys(problem, problem.dataset["test"])),
        n_img_conds=n_img_conds,
        design_shape=problem.design_space.shape,
        img_resize_dim=run.config.get("img_resize_dim", 16),
    ).to(device)
    model.load_state_dict(ckpt["generator"])
    model.eval()

    # Sample noise and generate designs
    z = th.randn((args.n_samples, run.config["latent_dim"]), device=device)
    gen_designs = model(z, conditions_tensor, img_conditions_tensor)
    gen_designs_np = gen_designs.detach().cpu().numpy()
    gen_designs_np = np.clip(gen_designs_np, 1e-3, 1.0)

    # Compute metrics
    metrics_dict = metrics.metrics(
        problem,
        gen_designs_np,
        sampled_designs_np,
        sampled_conditions,
        sigma=args.sigma,
    )

    # Add metadata to metrics
    metrics_dict.update(
        {
            "seed": seed,
            "problem_id": args.problem_id,
            "model_id": "cgan_2d_imgcond",
            "n_samples": args.n_samples,
            "sigma": args.sigma,
        }
    )

    # Append result row to CSV
    metrics_df = pd.DataFrame([metrics_dict])
    out_path = args.output_csv.format(problem_id=args.problem_id)
    write_header = not os.path.exists(out_path)
    metrics_df.to_csv(out_path, mode="a", header=write_header, index=False)

    print(f"Seed {seed} done; appended to {out_path}")
