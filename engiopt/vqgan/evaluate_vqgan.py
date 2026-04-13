"""Evaluation for the VQGAN."""

from __future__ import annotations

import dataclasses
import gc
import os

import numpy as np
import torch as th
import tyro

from engiopt import metrics
from engiopt.eval_utils import BaseEvaluationArgs
from engiopt.eval_utils import log_base_metrics_to_wandb
from engiopt.eval_utils import parse_thresholds
from engiopt.eval_utils import run_lvae_loop
from engiopt.eval_utils import RunRetrievalError
from engiopt.eval_utils import save_metrics_csv
from engiopt.eval_utils import save_per_sample_npz
from engiopt.eval_utils import setup_evaluation
from engiopt.transforms import drop_constant
from engiopt.transforms import normalize
from engiopt.transforms import resize_to
from engiopt.vqgan.vqgan import VQGAN
from engiopt.vqgan.vqgan import VQGANTransformer
import wandb


@dataclasses.dataclass
class Args(BaseEvaluationArgs):
    """Command-line arguments for a single-seed VQGAN 2D evaluation."""

    output_csv: str = "vqgan_{problem_id}_metrics.csv"
    """Output CSV path template; may include {problem_id}."""


if __name__ == "__main__":
    args = tyro.cli(Args)
    rec_thresholds, perf_thresholds = parse_thresholds(args)
    seed = args.seed

    problem, device, rng, conditions_tensor, sampled_conditions, sampled_designs_np, selected_indices = setup_evaluation(
        args
    )

    ### Set Up Transformer ###
    if args.wandb_entity is not None:
        artifact_path_0 = f"{args.wandb_entity}/{args.wandb_project}/{args.problem_id}_vqgan_cvqgan:seed_{seed}"
        artifact_path_1 = f"{args.wandb_entity}/{args.wandb_project}/{args.problem_id}_vqgan_vqgan:seed_{seed}"
        artifact_path_2 = f"{args.wandb_entity}/{args.wandb_project}/{args.problem_id}_vqgan_transformer:seed_{seed}"
    else:
        artifact_path_0 = f"{args.wandb_project}/{args.problem_id}_vqgan_cvqgan:seed_{seed}"
        artifact_path_1 = f"{args.wandb_project}/{args.problem_id}_vqgan_vqgan:seed_{seed}"
        artifact_path_2 = f"{args.wandb_project}/{args.problem_id}_vqgan_transformer:seed_{seed}"

    api = wandb.Api()
    artifact_0 = api.artifact(artifact_path_0, type="model")
    artifact_1 = api.artifact(artifact_path_1, type="model")
    artifact_2 = api.artifact(artifact_path_2, type="model")

    run = artifact_2.logged_by()
    if run is None or not hasattr(run, "config"):
        raise RunRetrievalError
    artifact_dir_0 = artifact_0.download()
    artifact_dir_1 = artifact_1.download()
    artifact_dir_2 = artifact_2.download()

    ckpt_path_0 = os.path.join(artifact_dir_0, "cvqgan.pth")
    ckpt_path_1 = os.path.join(artifact_dir_1, "vqgan.pth")
    ckpt_path_2 = os.path.join(artifact_dir_2, "transformer.pth")
    ckpt_0 = th.load(ckpt_path_0, map_location=th.device(device), weights_only=False)
    ckpt_1 = th.load(ckpt_path_1, map_location=th.device(device), weights_only=False)
    ckpt_2 = th.load(ckpt_path_2, map_location=th.device(device), weights_only=False)

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
    vqgan.eval()
    vqgan.to(device)

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
    cvqgan.eval()
    cvqgan.to(device)

    model = VQGANTransformer(
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
    model.load_state_dict(ckpt_2["transformer"])
    model.eval()
    model.to(device)

    ### Set up testing conditions ###
    # Clean up conditions based on model training settings and convert back to tensor
    sampled_conditions_new = sampled_conditions.select(range(len(sampled_conditions)))
    conditions = sampled_conditions_new.column_names

    if run.config["drop_constant_conditions"]:
        sampled_conditions_new, conditions = drop_constant(sampled_conditions_new, sampled_conditions_new.column_names)

    if run.config["normalize_conditions"]:
        sampled_conditions_new, mean, std = normalize(sampled_conditions_new, conditions)

    conditions_tensor = th.stack([th.as_tensor(sampled_conditions_new[c][:]).float() for c in conditions], dim=1).to(device)

    if run.config["conditional"]:
        c = model.encode_to_z(x=conditions_tensor, is_c=True)[1]
    else:
        c = th.ones(args.n_samples, 1, dtype=th.int64, device=device) * model.sos_token

    # Generate a batch of designs
    latent_designs = model.sample(
        x=th.empty(args.n_samples, 0, dtype=th.int64, device=device), c=c, steps=(run.config["latent_size"] ** 2)
    )
    gen_designs = resize_to(
        data=model.z_to_image(latent_designs), h=problem.design_space.shape[0], w=problem.design_space.shape[1]
    )
    gen_designs_np = gen_designs.detach().cpu().numpy().reshape(args.n_samples, *problem.design_space.shape)
    gen_designs_np = np.clip(gen_designs_np, 1e-3, 1)

    ### Shared: metrics, LVAE loop, save ###
    metrics_dict = metrics.metrics(problem, gen_designs_np, sampled_designs_np, sampled_conditions, sigma=args.sigma)
    metrics_dict.update({"seed": seed, "problem_id": args.problem_id, "model_id": "vqgan", "n_samples": args.n_samples})
    per_sample_data: dict[str, np.ndarray] = {}

    # Free VQGAN memory before LVAE encoding to avoid OOM
    del model, vqgan, cvqgan
    gc.collect()
    if th.cuda.is_available():
        th.cuda.empty_cache()

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
