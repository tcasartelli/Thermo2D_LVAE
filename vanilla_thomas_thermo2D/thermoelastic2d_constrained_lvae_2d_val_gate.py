"""Constrained LVAE for thermoelastic2D — GN encoder + validation-gated volume loss.

Volume pressure and pruning activate only when BOTH train_nmse AND val_nmse are
below the NMSE threshold. This prevents the model from compressing its latent
space when it has not yet learned to generalize, which is the main cause of
the train/val NMSE gap in standard constrained LVAE.

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
    """Command-line arguments for constrained LVAE with val-gate training on thermoelastic2d."""

    # Problem and tracking
    problem_id: str = "thermoelastic2d"
    algo: str = os.path.basename(__file__)[: -len(".py")]
    track: bool = True
    wandb_project: str = "spectral_norm_encoder"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    seed: int = 1
    save_model: bool = True
    sample_interval: int = 500

    # Training parameters
    n_epochs: int = 10000
    batch_size: int = 128
    lr: float = 1e-4

    # LVAE-specific
    latent_dim: int = 100
    nmse_threshold: float = 0.05
    constraint_mode: str = "one_sided"
    w_vol: float = 1.0
    ema_beta: float = 0.9

    # Pruning parameters
    pruning_epoch: int = 500
    pruning_threshold: float = 0.05
    pruning_strategy: str = "plummet"
    alpha: float = 0.0

    # Architecture
    resize_dimensions: tuple[int, int] = (100, 100)
    decoder_lipschitz_scale: float = 1.0
    gn_groups: int = 8

    # Output dirs
    images_dir: str = os.path.join(os.environ.get("SCRATCH", "."), "thermoelastic2d_constrained_lvae_val_gate", "images")
    checkpoint_dir: str = os.path.join(os.environ.get("SCRATCH", "."), "thermoelastic2d_constrained_lvae_val_gate", "checkpoints")

    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 0.001
    early_stopping_start_epoch: int = 0
    weight_filter: float | None = None


if __name__ == "__main__":
    args = tyro.cli(Args)

    problem = BUILTIN_PROBLEMS[args.problem_id]()
    problem.reset(seed=args.seed)
    design_shape = problem.design_space.shape

    os.makedirs(args.images_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    run_name = args.wandb_run_name or f"{args.problem_id}__{args.algo}__{args.seed}__{int(time.time())}"
    if args.track:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            save_code=True,
            name=run_name,
        )

    th.manual_seed(args.seed)
    th.cuda.manual_seed_all(args.seed)
    np.random.default_rng(args.seed)
    random.seed(args.seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    g = th.Generator().manual_seed(args.seed)

    if th.backends.mps.is_available():
        device = th.device("mps")
    elif th.cuda.is_available():
        device = th.device("cuda")
    else:
        device = th.device("cpu")

    enc = Encoder2D(args.latent_dim, design_shape, args.resize_dimensions, gn_groups=args.gn_groups)
    dec = TrueSNDecoder2D(args.latent_dim, design_shape, lipschitz_scale=args.decoder_lipschitz_scale)

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
        val_gate=True,  # volume + pruning gated on val_nmse
    ).to(device)

    print(f"\n{'=' * 60}")
    print("Constrained LVAE — thermoelastic2d [GN encoder + val-gate]")
    print(f"Design shape     : {design_shape}")
    print(f"Resize to        : {args.resize_dimensions}")
    print(f"Latent dim       : {args.latent_dim}")
    print(f"NMSE threshold   : {args.nmse_threshold}")
    print(f"GN groups        : {args.gn_groups}")
    print(f"Val-gate         : True")
    print(f"Pruning from epoch {args.pruning_epoch} ({args.pruning_strategy}, thr={args.pruning_threshold})")
    print(f"Images → {args.images_dir}")
    print(f"Checkpoints → {args.checkpoint_dir}")
    print(f"{'=' * 60}\n")

    hf = problem.dataset.with_format("torch")
    train_ds = hf["train"]
    val_ds = hf["val"]

    if args.weight_filter is not None:
        mask_tr = np.array(train_ds["weight"][:]) == args.weight_filter
        mask_va = np.array(val_ds["weight"][:])   == args.weight_filter
        x_train = train_ds["optimal_design"][:][mask_tr].unsqueeze(1)
        x_val   = val_ds["optimal_design"][:][mask_va].unsqueeze(1)
    else:
        x_train = train_ds["optimal_design"][:].unsqueeze(1)
        x_val   = val_ds["optimal_design"][:].unsqueeze(1)

    lvae.set_data_variance(x_train)
    print(f"Data variance    : {lvae.data_var:.6f}")

    loader     = DataLoader(TensorDataset(x_train), batch_size=args.batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(TensorDataset(x_val),   batch_size=args.batch_size, shuffle=False)

    best_val_nmse = float("inf")
    epochs_no_improve = 0

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
                "rec" : f"{lvae.rec_loss:.4f}",
                "vol" : f"{lvae.vol_loss:.4f}",
                "nmse": f"{lvae.nmse:.4f}",
                "vol_on": int(lvae.vol_active),
                "val_ok": int(lvae._val_nmse_ok),
                "dim" : lvae.dim,
            })

            if args.track:
                wandb.log({
                    "rec_loss"       : lvae.rec_loss,
                    "vol_loss"       : lvae.vol_loss,
                    "total_loss"     : loss.item(),
                    "nmse"           : lvae.nmse,
                    "nmse_threshold" : args.nmse_threshold,
                    "vol_active"     : int(lvae.vol_active),
                    "val_nmse_ok"    : int(lvae._val_nmse_ok),
                    "active_dims"    : lvae.dim,
                    "epoch"          : epoch,
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

        # Update val gate for next epoch
        lvae.set_val_nmse(val_nmse)

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

        if args.save_model and epoch == args.n_epochs - 1:
            ckpt_path = os.path.join(args.checkpoint_dir, "thermoelastic2d_constrained_lvae_val_gate.pth")
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
