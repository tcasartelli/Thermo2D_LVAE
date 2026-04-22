"""Constrained Performance-LVAE for thermoelastic2D — GN encoder + latent noise injection.

Extends the GN encoder variant by adding Gaussian noise to the latent code
before decoding during training (latent_noise_sigma). The decoder is forced to
be robust to small perturbations in z, which reduces overfitting and closes the
train/val NMSE gap.

Clean z is still used for pruning statistics, volume loss, and performance prediction.

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
from sklearn.preprocessing import RobustScaler
import torch as th
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import tqdm
import tyro

from engiopt.transforms import get_performance_target
from engiopt.transforms import get_scalar_condition_keys
from engiopt.vanilla_lvae.aes import ConstrainedPerfLeastVolumeAE_DP
from engiopt.vanilla_lvae.components_sn_encoder import SNEncoder2D_GN as Encoder2D
from engiopt.vanilla_lvae.components import SNMLPPredictor
from engiopt.vanilla_lvae.components import TrueSNDecoder2D
import wandb


@dataclass
class Args:
    """Command-line arguments for constrained PLVAE with latent noise training on thermoelastic2d."""

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
    perf_dim: int = -1
    """Number of latent dimensions dedicated to performance prediction. -1 = all."""

    # Constraint parameters
    nmse_threshold_rec: float = 0.05
    """NMSE threshold for reconstruction. Default: 0.05 (R2 = 95%)."""
    nmse_threshold_perf: float = 0.05
    """NMSE threshold for performance prediction. Default: 0.05 (R2 = 95%)."""

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
    """Resize input to this before encoding. Must be 100x100 for encoder compatibility."""
    predictor_hidden_dims: tuple[int, ...] = (256, 128)
    """Hidden dimensions for the MLP predictor."""
    conditional_predictor: bool = False
    """Whether to include conditions in performance prediction."""
    decoder_lipschitz_scale: float = 1.0
    """Lipschitz bound for spectrally normalized decoder."""
    predictor_lipschitz_scale: float = 1.0
    """Lipschitz bound for spectrally normalized MLP predictor."""
    gn_groups: int = 8
    """Number of groups for GroupNorm in the encoder (must divide each channel count)."""
    latent_noise_sigma: float = 0.1
    """Std of Gaussian noise added to z before decoding during training. 0 disables noise."""

    # Output dirs (override for Euler scratch)
    images_dir: str = os.path.join(os.environ.get("SCRATCH", "."), "thermoelastic2d_constrained_plvae_latent_noise", "images")
    """Directory to save visualisation images."""
    checkpoint_dir: str = os.path.join(os.environ.get("SCRATCH", "."), "thermoelastic2d_constrained_plvae_latent_noise", "checkpoints")
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
    scalar_cond_keys = get_scalar_condition_keys(problem, problem.dataset["train"])
    n_conds = len(scalar_cond_keys)

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

    # ---- Build encoder and decoder ----
    enc = Encoder2D(args.latent_dim, design_shape, args.resize_dimensions, gn_groups=args.gn_groups)
    dec = TrueSNDecoder2D(args.latent_dim, design_shape, lipschitz_scale=args.decoder_lipschitz_scale)

    # ---- Determine perf_dim ----
    perf_dim = args.latent_dim if args.perf_dim == -1 else args.perf_dim
    n_perf = 1  # single performance objective

    # ---- Build MLP predictor ----
    predictor_input_dim = perf_dim + (n_conds if args.conditional_predictor else 0)
    predictor = SNMLPPredictor(
        input_dim=predictor_input_dim,
        output_dim=n_perf,
        hidden_dims=args.predictor_hidden_dims,
        lipschitz_scale=args.predictor_lipschitz_scale,
    )

    # ---- Constrained Performance LVAE with latent noise ----
    plvae = ConstrainedPerfLeastVolumeAE_DP(
        encoder=enc,
        decoder=dec,
        predictor=predictor,
        optimizer=Adam(
            list(enc.parameters()) + list(dec.parameters()) + list(predictor.parameters()),
            lr=args.lr,
        ),
        latent_dim=args.latent_dim,
        perf_dim=perf_dim,
        nmse_threshold_rec=args.nmse_threshold_rec,
        nmse_threshold_perf=args.nmse_threshold_perf,
        pruning_epoch=args.pruning_epoch,
        pruning_threshold=args.pruning_threshold,
        pruning_strategy=args.pruning_strategy,
        alpha=args.alpha,
        latent_noise_sigma=args.latent_noise_sigma,
    ).to(device)

    print(f"\n{'=' * 60}")
    print("Constrained Performance-LVAE — thermoelastic2d [GN encoder + latent noise]")
    print(f"Design shape          : {design_shape}")
    print(f"Resize to             : {args.resize_dimensions}")
    print(f"Latent dim            : {args.latent_dim}")
    print(f"Perf dim              : {perf_dim}")
    print(f"Predictor mode        : {'Conditional' if args.conditional_predictor else 'Unconditional'}")
    print(f"Lipschitz (decoder)   : {args.decoder_lipschitz_scale}")
    print(f"Lipschitz (predictor) : {args.predictor_lipschitz_scale}")
    print(f"NMSE threshold (rec)  : {args.nmse_threshold_rec}")
    print(f"NMSE threshold (perf) : {args.nmse_threshold_perf}")
    print(f"GN groups             : {args.gn_groups}")
    print(f"Latent noise σ        : {args.latent_noise_sigma}")
    print(f"Pruning from epoch    : {args.pruning_epoch} ({args.pruning_strategy}, thr={args.pruning_threshold})")
    print(f"Images → {args.images_dir}")
    print(f"Checkpoints → {args.checkpoint_dir}")
    print(f"{'=' * 60}\n")

    # ---- DataLoader ----
    hf = problem.dataset.with_format("torch")
    train_ds = hf["train"]
    val_ds = hf["val"]

    x_train = train_ds["optimal_design"][:].unsqueeze(1)  # (N, 1, 64, 64)
    c_train = th.stack([train_ds[key][:] for key in scalar_cond_keys], dim=-1)
    p_train = get_performance_target(problem, train_ds)

    x_val = val_ds["optimal_design"][:].unsqueeze(1)
    c_val = th.stack([val_ds[key][:] for key in scalar_cond_keys], dim=-1)
    p_val = get_performance_target(problem, val_ds)

    p_scaler = RobustScaler()
    p_train_scaled = th.from_numpy(p_scaler.fit_transform(p_train.numpy())).to(p_train.dtype)
    p_val_scaled = th.from_numpy(p_scaler.transform(p_val.numpy())).to(p_val.dtype)
    p_scaler1 = RobustScaler()
    _ = p_scaler1.fit_transform(p_train[:, :1].numpy())

    if args.conditional_predictor:
        c_scaler = RobustScaler()
        c_train_scaled = th.from_numpy(c_scaler.fit_transform(c_train.numpy())).to(c_train.dtype)
        c_val_scaled = th.from_numpy(c_scaler.transform(c_val.numpy())).to(c_val.dtype)
    else:
        c_train_scaled = th.zeros(len(x_train), 0)
        c_val_scaled = th.zeros(len(x_val), 0)

    plvae.set_data_variance(x_train)
    plvae.set_perf_variance(p_train_scaled)

    print(f"Data variance (designs) : {plvae.data_var:.6f}")
    print(f"Perf variance (scaled)  : {plvae.perf_var:.6f}")

    loader = DataLoader(
        TensorDataset(x_train, c_train_scaled, p_train_scaled),
        batch_size=args.batch_size,
        shuffle=True,
        generator=g,
    )
    val_loader = DataLoader(
        TensorDataset(x_val, c_val_scaled, p_val_scaled),
        batch_size=args.batch_size,
        shuffle=False,
    )

    # ---- Early stopping state ----
    best_val_nmse = float("inf")
    epochs_no_improve = 0

    # ---- Training loop ----
    for epoch in range(args.n_epochs):
        plvae.epoch_hook(epoch=epoch)

        bar = tqdm.tqdm(loader, desc=f"Epoch {epoch}")
        for i, batch in enumerate(bar):
            x_batch = batch[0].to(device)
            c_batch = batch[1].to(device)
            p_batch = batch[2].to(device)

            plvae.optim.zero_grad()

            loss = plvae.loss((x_batch, c_batch, p_batch))
            loss.backward()
            plvae.optim.step()

            bar.set_postfix({
                "rec"   : f"{plvae.rec_loss:.4f}",
                "perf"  : f"{plvae.perf_loss:.4f}",
                "vol"   : f"{plvae.vol_loss:.4f}",
                "nmse_r": f"{plvae.nmse_rec:.4f}",
                "nmse_p": f"{plvae.nmse_perf:.4f}",
                "vol_on": int(plvae.vol_active),
                "dim"   : plvae.dim,
            })

            if args.track:
                batches_done = epoch * len(bar) + i
                wandb.log({
                    "rec_loss"           : plvae.rec_loss,
                    "perf_loss"          : plvae.perf_loss,
                    "vol_loss"           : plvae.vol_loss,
                    "total_loss"         : loss.item(),
                    "nmse_rec"           : plvae.nmse_rec,
                    "nmse_perf"          : plvae.nmse_perf,
                    "nmse_threshold_rec" : args.nmse_threshold_rec,
                    "nmse_threshold_perf": args.nmse_threshold_perf,
                    "vol_active"         : int(plvae.vol_active),
                    "rec_violated"       : int(plvae.nmse_rec > args.nmse_threshold_rec),
                    "perf_violated"      : int(plvae.nmse_perf > args.nmse_threshold_perf),
                    "active_dims"        : plvae.dim,
                    "epoch"              : epoch,
                    "latent_noise_sigma" : args.latent_noise_sigma,
                })

                if batches_done % args.sample_interval == 0:
                    with th.no_grad():
                        xs = x_train.to(device)
                        z = plvae.encode(xs)
                        z_std, idx = th.sort(z.std(0), descending=True)
                        z_mean = z.mean(0)
                        n_active = (z_std > 0).sum().item()

                        x_ints = []
                        for alpha in [0, 0.25, 0.5, 0.75, 1]:
                            z_ = (1 - alpha) * z[:25] + alpha * th.roll(z, -1, 0)[:25]
                            x_ints.append(plvae.decode(z_).cpu().numpy())

                        z_rand = z_mean.unsqueeze(0).repeat([25, 1])
                        z_rand[:, idx[:n_active]] += z_std[:n_active] * th.randn_like(z_rand[:, idx[:n_active]])
                        x_rand = plvae.decode(z_rand).cpu().numpy()

                        pz_train = z[:, :perf_dim]
                        p_pred_scaled = plvae.predictor(pz_train)
                        p_actual = p_scaler1.inverse_transform(
                            p_train_scaled[:, :1].cpu().numpy()
                        ).flatten()
                        p_predicted = p_scaler1.inverse_transform(
                            p_pred_scaled.cpu().numpy()[:, :1][:, :1]
                        ).flatten()

                        z_std_cpu = z_std.cpu().numpy()
                        xs_cpu = xs.cpu().numpy()

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
                    dim_path = os.path.join(args.images_dir, f"dim_{batches_done}.png")
                    plt.savefig(dim_path); plt.close()

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

                    fig, axs = plt.subplots(5, 5, figsize=(15, 7.5))
                    for k, (i_row, j) in enumerate(product(range(5), range(5))):
                        axs[i_row, j].imshow(x_rand[k].reshape(design_shape))
                        axs[i_row, j].axis("off")
                        axs[i_row, j].set_aspect("equal")
                    fig.tight_layout()
                    plt.suptitle("Gaussian random designs from latent space")
                    norm_path = os.path.join(args.images_dir, f"norm_{batches_done}.png")
                    plt.savefig(norm_path); plt.close()

                    plt.figure(figsize=(8, 8))
                    plt.scatter(p_actual, p_predicted, alpha=0.5, s=20)
                    min_val = min(p_actual.min(), p_predicted.min())
                    max_val = max(p_actual.max(), p_predicted.max())
                    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="1:1 line")
                    plt.xlabel("Actual Performance")
                    plt.ylabel("Predicted Performance")
                    mse_value = np.mean((p_actual - p_predicted) ** 2)
                    plt.title(f"MSE: {mse_value:.4e}")
                    plt.grid(visible=True, alpha=0.3)
                    plt.legend()
                    plt.axis("equal")
                    plt.tight_layout()
                    perf_path = os.path.join(args.images_dir, f"perf_pred_{batches_done}.png")
                    plt.savefig(perf_path); plt.close()

                    wandb.log({
                        "dim_plot"           : wandb.Image(dim_path),
                        "interp_plot"        : wandb.Image(interp_path),
                        "norm_plot"          : wandb.Image(norm_path),
                        "perf_pred_vs_actual": wandb.Image(perf_path),
                    })

        # ---- Validation ----
        with th.no_grad():
            plvae.eval()
            val_rec = val_perf = val_vol = 0.0
            val_nmse_rec = val_nmse_perf = 0.0
            n = 0
            for batch_v in val_loader:
                x_v = batch_v[0].to(device)
                c_v = batch_v[1].to(device)
                p_v = batch_v[2].to(device)
                _ = plvae.loss((x_v, c_v, p_v))
                bsz = x_v.size(0)
                val_rec      += plvae.rec_loss * bsz
                val_perf     += plvae.perf_loss * bsz
                val_vol      += plvae.vol_loss * bsz
                val_nmse_rec += plvae.nmse_rec * bsz
                val_nmse_perf += plvae.nmse_perf * bsz
                n += bsz
            val_rec      /= n
            val_perf     /= n
            val_vol      /= n
            val_nmse_rec /= n
            val_nmse_perf /= n

        # ---- Early stopping ----
        if args.early_stopping and epoch >= args.early_stopping_start_epoch:
            if val_nmse_rec < best_val_nmse - args.min_delta:
                best_val_nmse = val_nmse_rec
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

        plvae.epoch_report(epoch=epoch, callbacks=[], batch=None, loss=loss, pbar=None)

        if args.track:
            wandb.log({
                "epoch"        : epoch,
                "val_rec"      : val_rec,
                "val_perf"     : val_perf,
                "val_vol"      : val_vol,
                "val_nmse_rec" : val_nmse_rec,
                "val_nmse_perf": val_nmse_perf,
            }, commit=True)

        th.cuda.empty_cache()
        plvae.train()

        # ---- Checkpoint ----
        if args.save_model and epoch == args.n_epochs - 1:
            ckpt_path = os.path.join(args.checkpoint_dir, "thermoelastic2d_constrained_plvae_latent_noise.pth")
            th.save({
                "epoch"    : epoch,
                "encoder"  : plvae.encoder.state_dict(),
                "decoder"  : plvae.decoder.state_dict(),
                "predictor": plvae.predictor.state_dict(),
                "optimizer": plvae.optim.state_dict(),
                "pruning_mask"    : plvae._p.cpu(),
                "pruning_frozen_z": plvae._z.cpu(),
                "args"     : vars(args),
            }, ckpt_path)
            if args.track:
                artifact = wandb.Artifact(f"{args.problem_id}_{args.algo}", type="model")
                artifact.add_file(ckpt_path)
                wandb.log_artifact(
                    artifact,
                    aliases=[f"seed_{args.seed}_rec{args.nmse_threshold_rec}_perf{args.nmse_threshold_perf}"],
                )

    if args.track:
        wandb.finish()
