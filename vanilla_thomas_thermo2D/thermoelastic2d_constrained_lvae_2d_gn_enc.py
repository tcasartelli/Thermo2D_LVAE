"""Constrained LVAE for thermoelastic2D designs with plummet-based dynamic pruning.

Adapted from constrained_vanilla_lvae_2d.py — uses NMSE threshold constraint
instead of fixed volume weight. problem_id and resize_dimensions set for
thermoelastic2d dataset (64x64 designs).

Encoder variant: SNEncoder2D_GN — spectral norm + GroupNorm instead of BatchNorm.
GroupNorm removes the train/val statistics mismatch introduced by BatchNorm's
running_mean/running_var, which behave differently in train vs eval mode.

For more information on LVAE, see: https://arxiv.org/abs/2404.17773
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import os
import random
import time

from engibench.utils.all_problems import BUILTIN_PROBLEMS
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import tqdm
import tyro

from engiopt.vanilla_lvae.aes import ConstrainedLeastVolumeAE_DP
from engiopt.vanilla_lvae.components_sn_encoder import SNEncoder2D_GN as Encoder2D
from engiopt.vanilla_lvae.components import TrueSNDecoder2D
import wandb


@dataclass
class Args:
    """Command-line arguments for constrained vanilla LVAE training on thermoelastic2d."""

    # Problem and tracking
    problem_id: str = "thermoelastic2d"
    """Problem ID — thermoelastic2d has 64x64 designs."""
    algo: str = os.path.basename(__file__)[: -len(".py")]
    """Algorithm name for tracking purposes."""
    track: bool = True
    """Whether to track with Weights & Biases."""
    wandb_project: str = "spectral_norm_encoder"
    """WandB project name."""
    wandb_entity: str | None = None
    """WandB entity name. If None, uses the default entity."""
    wandb_run_name: str | None = None
    """WandB run name. If None, auto-generated from algo+seed+timestamp."""
    seed: int = 1
    """Random seed for reproducibility."""
    save_model: bool = True
    """Whether to save the model after training."""
    sample_interval: int = 500
    """Interval for sampling designs during training."""

    # Training parameters
    n_epochs: int = 10000
    """Number of training epochs."""
    batch_size: int = 128
    """Batch size for training."""
    lr: float = 1e-4
    """Learning rate for the optimizer."""

    # LVAE-specific
    latent_dim: int = 100
    """Dimensionality of the latent space (overestimate)."""
    # Constraint parameters
    nmse_threshold: float = 0.05
    """NMSE ceiling. Training aims to stay at or below this threshold."""
    constraint_mode: str = "one_sided"
    """Constraint mode: 'one_sided' (rec or vol), 'gated' (rec + vol), 'gradient_balanced'."""
    w_vol: float = 1.0
    """Volume loss weight (gated mode only)."""
    ema_beta: float = 0.9
    """EMA smoothing for loss tracking (gradient_balanced mode)."""

    # Pruning parameters
    pruning_epoch: int = 500
    """Epoch to start pruning dimensions."""
    pruning_threshold: float = 0.05
    """Threshold for pruning (ratio for plummet, percentile for lognorm)."""
    pruning_strategy: str = "plummet"
    """Pruning strategy to use: 'plummet' or 'lognorm'."""
    alpha: float = 0.0
    """(lognorm only) Blending factor between reference and current distribution."""

    # Architecture
    resize_dimensions: tuple[int, int] = (100, 100)
    """Resize input to this before encoding. Matches thermoelastic2d native resolution."""
    decoder_lipschitz_scale: float = 1.0
    """Lipschitz bound for spectrally normalized decoder."""
    gn_groups: int = 8
    """Number of groups for GroupNorm in the encoder (must divide each channel count)."""

    # Output dirs (override for Euler scratch)
    images_dir: str = os.path.join(os.environ.get("SCRATCH", "."), "thermoelastic2d_constrained_lvae_gn_enc", "images")
    """Directory to save visualisation images."""
    checkpoint_dir: str = os.path.join(os.environ.get("SCRATCH", "."), "thermoelastic2d_constrained_lvae_gn_enc", "checkpoints")
    """Directory to save model checkpoints."""

    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 0.001
    early_stopping_start_epoch: int = 0
    weight_filter: float | None = None


if __name__ == "__main__":
    args = tyro.cli(Args)

    # ---- Problem setup ----
    problem = BUILTIN_PROBLEMS[args.problem_id]()
    problem.reset(seed=args.seed)
    design_shape = problem.design_space.shape  # (64, 64)

    # ---- Output directories (on $SCRATCH) ----
    os.makedirs(args.images_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ---- W&B logging ----
    run_name = args.wandb_run_name or f"{args.problem_id}__{args.algo}__{args.seed}__{int(time.time())}"
    if args.track:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            save_code=True,
            name=run_name,
        )

    # ---- Seeding ----
    th.manual_seed(args.seed)
    th.cuda.manual_seed_all(args.seed)
    np.random.default_rng(args.seed)
    random.seed(args.seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    g = th.Generator().manual_seed(args.seed)

    # ---- Device ----
    if th.backends.mps.is_available():
        device = th.device("mps")
    elif th.cuda.is_available():
        device = th.device("cuda")
    else:
        device = th.device("cpu")

    # ---- Build encoder & decoder ----
    enc = Encoder2D(args.latent_dim, design_shape, args.resize_dimensions, gn_groups=args.gn_groups)
    dec = TrueSNDecoder2D(args.latent_dim, design_shape, lipschitz_scale=args.decoder_lipschitz_scale)

    # ---- Constrained LVAE ----
    lvae = ConstrainedLeastVolumeAE_DP(
        encoder=enc,
        decoder=dec,
        optimizer=Adam(list(enc.parameters()) + list(dec.parameters()), lr=args.lr),
        latent_dim=args.latent_dim,
        nmse_threshold=args.nmse_threshold,
        pruning_epoch=args.pruning_epoch,
        pruning_threshold=args.pruning_threshold,
        pruning_strategy=args.pruning_strategy,
        alpha=args.alpha,
    ).to(device)

    print(f"\n{'=' * 60}")
    print("Constrained LVAE — thermoelastic2d [GN encoder]")
    print(f"Design shape     : {design_shape}")
    print(f"Resize to        : {args.resize_dimensions}")
    print(f"Latent dim       : {args.latent_dim}")
    print(f"Lipschitz        : {args.decoder_lipschitz_scale}")
    print(f"NMSE threshold   : {args.nmse_threshold}")
    print(f"Constraint mode  : {args.constraint_mode}")
    print(f"GN groups        : {args.gn_groups}")
    print(f"Pruning from epoch {args.pruning_epoch} ({args.pruning_strategy}, thr={args.pruning_threshold})")
    print(f"Images → {args.images_dir}")
    print(f"Checkpoints → {args.checkpoint_dir}")
    print(f"{'=' * 60}\n")

    # ---- DataLoader ----
    hf = problem.dataset.with_format("torch")
    train_ds = hf["train"]
    val_ds = hf["val"]

    if args.weight_filter is not None:
        mask_tr = np.array(train_ds["weight"][:]) == args.weight_filter
        mask_va = np.array(val_ds["weight"][:])   == args.weight_filter
        x_train = train_ds["optimal_design"][:][mask_tr].unsqueeze(1)
        x_val   = val_ds["optimal_design"][:][mask_va].unsqueeze(1)
        print(f"Filtered to weight={args.weight_filter}: {len(x_train)} train, {len(x_val)} val")
    else:
        x_train = train_ds["optimal_design"][:].unsqueeze(1)
        x_val   = val_ds["optimal_design"][:].unsqueeze(1)

    lvae.set_data_variance(x_train)
    print(f"Data variance    : {lvae.data_var:.6f}")

    loader     = DataLoader(TensorDataset(x_train), batch_size=args.batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(TensorDataset(x_val),   batch_size=args.batch_size, shuffle=False)

    # ---- Early stopping state ----
    best_val_nmse = float("inf")
    epochs_no_improve = 0

    # ---- Training loop ----
    for epoch in range(args.n_epochs):
        lvae.epoch_hook(epoch=epoch)

        bar = tqdm.tqdm(loader, desc=f"Epoch {epoch}")
        for i, batch in enumerate(bar):
            x_batch = batch[0].to(device)
            lvae.optim.zero_grad()

            loss = lvae.loss(x_batch)
            loss.backward()
            lvae.optim.step()

            bar.set_postfix({
                "rec": f"{lvae.rec_loss:.4f}",
                "vol": f"{lvae.vol_loss:.4f}",
                "nmse": f"{lvae.nmse:.4f}",
                "active": int(lvae.vol_active),
                "dim": lvae.dim,
            })

            if args.track:
                batches_done = epoch * len(bar) + i
                wandb.log({
                    "rec_loss"    : lvae.rec_loss,
                    "vol_loss"    : lvae.vol_loss,
                    "total_loss"  : loss.item(),
                    "nmse"        : lvae.nmse,
                    "nmse_threshold": args.nmse_threshold,
                    "vol_active"  : int(lvae.vol_active),
                    "active_dims" : lvae.dim,
                    "epoch"       : epoch,
                    "constraint_mode": args.constraint_mode,
                })

        # ---- Validation ----
        with th.no_grad():
            lvae.eval()
            val_rec = val_vol = val_nmse = 0.0
            n = 0
            for batch_v in val_loader:
                x_v = batch_v[0].to(device)
                _ = lvae.loss(x_v)
                bsz = x_v.size(0)
                val_rec  += lvae.rec_loss * bsz
                val_vol  += lvae.vol_loss * bsz
                val_nmse += lvae.nmse * bsz
                n += bsz
            val_rec  /= n
            val_vol  /= n
            val_nmse /= n

        # ---- Early stopping ----
        if args.early_stopping and epoch >= args.early_stopping_start_epoch:
            if val_nmse < best_val_nmse - args.min_delta:
                best_val_nmse = val_nmse
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

        lvae.epoch_report(epoch=epoch, callbacks=[], batch=None, loss=loss, pbar=None)

        if args.track:
            wandb.log({
                "epoch"       : epoch,
                "val_rec_loss": val_rec,
                "val_vol_loss": val_vol,
                "val_nmse"    : val_nmse,
            }, commit=True)

        th.cuda.empty_cache()
        lvae.train()

        # ---- Checkpoint ----
        if args.save_model and epoch == args.n_epochs - 1:
            ckpt_path = os.path.join(args.checkpoint_dir, "thermoelastic2d_constrained_lvae_gn_enc.pth")
            th.save({
                "epoch"    : epoch,
                "encoder"  : lvae.encoder.state_dict(),
                "decoder"  : lvae.decoder.state_dict(),
                "optimizer": lvae.optim.state_dict(),
                "args"     : vars(args),
            }, ckpt_path)
            if args.track:
                artifact = wandb.Artifact(f"{args.problem_id}_{args.algo}", type="model")
                artifact.add_file(ckpt_path)
                wandb.log_artifact(artifact, aliases=[f"seed_{args.seed}"])

    if args.track:
        wandb.finish()
