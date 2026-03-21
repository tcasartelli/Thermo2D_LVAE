"""Vanilla LVAE for thermoelastic2D designs with plummet-based dynamic pruning.

Adapted from vanilla_lvae_2d.py — only problem_id and resize_dimensions changed
to match the thermoelastic2d dataset (64x64 designs).

For more information on LVAE, see: https://arxiv.org/abs/2404.17773
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
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

from engiopt.vanilla_lvae.aes import LeastVolumeAE_DynamicPruning
from engiopt.vanilla_lvae.components import Encoder2D
from engiopt.vanilla_lvae.components import TrueSNDecoder2D
import wandb


@dataclass
class Args:
    """Command-line arguments for vanilla LVAE training on thermoelastic2d."""

    # Problem and tracking
    problem_id: str = "thermoelastic2d"
    """Problem ID — thermoelastic2d has 64x64 designs."""
    algo: str = os.path.basename(__file__)[: -len(".py")]
    """Algorithm name for tracking purposes."""
    track: bool = True
    """Whether to track with Weights & Biases."""
    wandb_project: str = "engiopt"
    """WandB project name."""
    wandb_entity: str | None = None
    """WandB entity name. If None, uses the default entity."""
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
    w_reconstruction: float = 1.0
    """Weight for reconstruction loss."""
    w_volume: float = 0.01
    """Weight for volume loss."""

    # Pruning parameters
    pruning_epoch: int = 500
    """Epoch to start pruning dimensions."""
    pruning_threshold: float = 0.05
    """Threshold for pruning (ratio for plummet, percentile for lognorm)."""
    pruning_strategy: str = "plummet"
    """Pruning strategy to use: 'plummet' or 'lognorm'."""
    alpha: float = 0.0
    """(lognorm only) Blending factor between reference and current distribution."""

    # Volume weight warmup
    volume_warmup_epochs: int = 0
    """Epochs to polynomially ramp volume weight from 0 to w_volume. 0 disables warmup."""
    volume_warmup_degree: float = 2.0
    """Polynomial degree for volume weight warmup (1.0=linear, 2.0=quadratic)."""

    # Architecture
    # thermoelastic2d designs are 64x64 — no need to upscale to 100x100
    resize_dimensions: tuple[int, int] = (64, 64)
    """Resize input to this before encoding. Matches thermoelastic2d native resolution."""
    decoder_lipschitz_scale: float = 1.0
    """Lipschitz bound for spectrally normalized decoder."""

    # Output dirs (override for Euler scratch)
    images_dir: str = os.path.join(os.environ.get("SCRATCH", "."), "thermoelastic2d_lvae", "images")
    """Directory to save visualisation images."""
    checkpoint_dir: str = os.path.join(os.environ.get("SCRATCH", "."), "thermoelastic2d_lvae", "checkpoints")
    """Directory to save model checkpoints."""


def volume_weight_schedule(epoch: int, w_rec: float, w_vol: float, warmup_epochs: int, degree: float) -> th.Tensor:
    """Polynomial ramp on volume weight."""
    if warmup_epochs <= 0:
        return th.tensor([w_rec, w_vol], dtype=th.float)
    t = min(epoch / warmup_epochs, 1.0)
    return th.tensor([w_rec, w_vol * (t**degree)], dtype=th.float)


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
    run_name = f"{args.problem_id}__{args.algo}__{args.seed}__{int(time.time())}"
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

    # ---- Build encoder & decoder (vanilla, no custom arch) ----
    enc = Encoder2D(args.latent_dim, design_shape, args.resize_dimensions)
    dec = TrueSNDecoder2D(args.latent_dim, design_shape, lipschitz_scale=args.decoder_lipschitz_scale)

    weights = partial(
        volume_weight_schedule,
        w_rec=args.w_reconstruction,
        w_vol=args.w_volume,
        warmup_epochs=args.volume_warmup_epochs,
        degree=args.volume_warmup_degree,
    )

    # ---- LVAE (vanilla, from engiopt) ----
    lvae = LeastVolumeAE_DynamicPruning(
        encoder=enc,
        decoder=dec,
        optimizer=Adam(list(enc.parameters()) + list(dec.parameters()), lr=args.lr),
        latent_dim=args.latent_dim,
        weights=weights,
        pruning_epoch=args.pruning_epoch,
        pruning_threshold=args.pruning_threshold,
        pruning_strategy=args.pruning_strategy,
        alpha=args.alpha,
    ).to(device)

    print(f"\n{'=' * 60}")
    print("Vanilla LVAE — thermoelastic2d")
    print(f"Design shape : {design_shape}")
    print(f"Resize to    : {args.resize_dimensions}")
    print(f"Latent dim   : {args.latent_dim}")
    print(f"Lipschitz    : {args.decoder_lipschitz_scale}")
    print(f"Pruning from epoch {args.pruning_epoch} ({args.pruning_strategy}, thr={args.pruning_threshold})")
    print(f"Images → {args.images_dir}")
    print(f"Checkpoints → {args.checkpoint_dir}")
    print(f"{'=' * 60}\n")

    # ---- DataLoader ----
    hf = problem.dataset.with_format("torch")
    train_ds = hf["train"]
    val_ds = hf["val"]

    x_train = train_ds["optimal_design"][:].unsqueeze(1)  # (N, 1, 64, 64)
    x_val   = val_ds["optimal_design"][:].unsqueeze(1)

    loader     = DataLoader(TensorDataset(x_train), batch_size=args.batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(TensorDataset(x_val),   batch_size=args.batch_size, shuffle=False)

    # ---- Training loop ----
    for epoch in range(args.n_epochs):
        lvae.epoch_hook(epoch=epoch)

        bar = tqdm.tqdm(loader, desc=f"Epoch {epoch}")
        for i, batch in enumerate(bar):
            x_batch = batch[0].to(device)
            lvae.optim.zero_grad()

            losses = lvae.loss(x_batch)
            loss   = (losses * lvae.w).sum()
            loss.backward()
            lvae.optim.step()

            bar.set_postfix({
                "rec": f"{losses[0].item():.4f}",
                "vol": f"{losses[1].item():.4f}",
                "dim": lvae.dim,
            })

            if args.track:
                batches_done = epoch * len(bar) + i
                wandb.log({
                    "rec_loss"   : losses[0].item(),
                    "vol_loss"   : losses[1].item(),
                    "total_loss" : loss.item(),
                    "active_dims": lvae.dim,
                    "epoch"      : epoch,
                    "w_volume"   : lvae.w[1].item(),
                })

                # ---- Visualisation ----
                if batches_done % args.sample_interval == 0:
                    with th.no_grad():
                        xs       = x_train.to(device)
                        z        = lvae.encode(xs)
                        z_std, idx = th.sort(z.std(0), descending=True)
                        z_mean   = z.mean(0)
                        n_active = (z_std > 0).sum().item()

                        x_ints = []
                        for alpha in [0, 0.25, 0.5, 0.75, 1]:
                            z_ = (1 - alpha) * z[:25] + alpha * th.roll(z, -1, 0)[:25]
                            x_ints.append(lvae.decode(z_).cpu().numpy())

                        z_rand = z_mean.unsqueeze(0).repeat([25, 1])
                        z_rand[:, idx[:n_active]] += z_std[:n_active] * th.randn_like(z_rand[:, idx[:n_active]])
                        x_rand   = lvae.decode(z_rand).cpu().numpy()
                        z_std_cpu = z_std.cpu().numpy()
                        xs_cpu   = xs.cpu().numpy()

                    # Plot 1: latent std
                    plt.figure(figsize=(12, 6))
                    plt.subplot(211)
                    plt.bar(np.arange(len(z_std_cpu)), z_std_cpu)
                    plt.yscale("log")
                    plt.xlabel("Latent dimension index")
                    plt.ylabel("Standard deviation")
                    plt.title(f"Active dims = {n_active}")
                    plt.subplot(212)
                    plt.bar(np.arange(n_active), z_std_cpu[:n_active])
                    plt.yscale("log")
                    plt.xlabel("Latent dimension index")
                    plt.ylabel("Standard deviation")
                    dim_path = os.path.join(args.images_dir, f"dim_{batches_done}.png")
                    plt.savefig(dim_path); plt.close()

                    # Plot 2: interpolations
                    fig, axs = plt.subplots(25, 6, figsize=(12, 25))
                    for i_row, j in product(range(25), range(5)):
                        axs[i_row, j + 1].imshow(x_ints[j][i_row].reshape(design_shape))
                        axs[i_row, j + 1].axis("off")
                        axs[i_row, j + 1].set_aspect("equal")
                    for ax, alpha in zip(axs[0, 1:], [0, 0.25, 0.5, 0.75, 1]):
                        ax.set_title(rf"$\alpha$ = {alpha}")
                    for i_row in range(25):
                        axs[i_row, 0].imshow(xs_cpu[i_row].reshape(design_shape))
                        axs[i_row, 0].axis("off")
                        axs[i_row, 0].set_aspect("equal")
                    axs[0, 0].set_title("groundtruth")
                    fig.tight_layout()
                    interp_path = os.path.join(args.images_dir, f"interp_{batches_done}.png")
                    plt.savefig(interp_path); plt.close()

                    # Plot 3: random samples
                    fig, axs = plt.subplots(5, 5, figsize=(15, 7.5))
                    for k, (i_row, j) in enumerate(product(range(5), range(5))):
                        axs[i_row, j].imshow(x_rand[k].reshape(design_shape))
                        axs[i_row, j].axis("off")
                        axs[i_row, j].set_aspect("equal")
                    fig.tight_layout()
                    plt.suptitle("Gaussian random designs from latent space")
                    norm_path = os.path.join(args.images_dir, f"norm_{batches_done}.png")
                    plt.savefig(norm_path); plt.close()

                    wandb.log({
                        "dim_plot"   : wandb.Image(dim_path),
                        "interp_plot": wandb.Image(interp_path),
                        "norm_plot"  : wandb.Image(norm_path),
                    })

        # ---- Validation ----
        with th.no_grad():
            lvae.eval()
            val_rec = val_vol = 0.0
            n = 0
            for batch_v in val_loader:
                x_v = batch_v[0].to(device)
                vlosses = lvae.loss(x_v)
                bsz = x_v.size(0)
                val_rec += vlosses[0].item() * bsz
                val_vol += vlosses[1].item() * bsz
                n += bsz
            val_rec /= n
            val_vol /= n

        lvae.epoch_report(epoch=epoch, callbacks=[], batch=None, loss=losses, pbar=None)

        if args.track:
            wandb.log({"epoch": epoch, "val_rec": val_rec, "val_vol_loss": val_vol}, commit=True)

        th.cuda.empty_cache()
        lvae.train()

        # ---- Checkpoint ----
        if args.save_model and epoch == args.n_epochs - 1:
            ckpt_path = os.path.join(args.checkpoint_dir, "thermoelastic2d_vanilla_lvae.pth")
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
