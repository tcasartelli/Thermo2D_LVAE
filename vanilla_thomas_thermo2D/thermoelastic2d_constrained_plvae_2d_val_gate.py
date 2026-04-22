"""Constrained PLVAE for thermoelastic2D — GN encoder + validation-gated volume loss.

Volume pressure and pruning activate only when BOTH train_nmse AND val_nmse are
below the NMSE threshold (for both rec and perf). This prevents the model from
compressing its latent space before it has generalized.

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
    """Command-line arguments for constrained PLVAE with val-gate training on thermoelastic2d."""

    problem_id: str = "thermoelastic2d"
    algo: str = os.path.basename(__file__)[: -len(".py")]
    track: bool = True
    wandb_project: str = "spectral_norm_encoder"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    seed: int = 1
    save_model: bool = True
    sample_interval: int = 500

    n_epochs: int = 10000
    batch_size: int = 128
    lr: float = 1e-4

    latent_dim: int = 100
    perf_dim: int = -1
    nmse_threshold_rec: float = 0.05
    nmse_threshold_perf: float = 0.05

    pruning_epoch: int = 500
    pruning_threshold: float = 0.05
    pruning_strategy: str = "plummet"
    alpha: float = 0.0

    resize_dimensions: tuple[int, int] = (100, 100)
    predictor_hidden_dims: tuple[int, ...] = (256, 128)
    conditional_predictor: bool = False
    decoder_lipschitz_scale: float = 1.0
    predictor_lipschitz_scale: float = 1.0
    gn_groups: int = 8

    images_dir: str = os.path.join(os.environ.get("SCRATCH", "."), "thermoelastic2d_constrained_plvae_val_gate", "images")
    checkpoint_dir: str = os.path.join(os.environ.get("SCRATCH", "."), "thermoelastic2d_constrained_plvae_val_gate", "checkpoints")

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
    scalar_cond_keys = get_scalar_condition_keys(problem, problem.dataset["train"])
    n_conds = len(scalar_cond_keys)

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

    perf_dim = args.latent_dim if args.perf_dim == -1 else args.perf_dim
    n_perf = 1

    predictor_input_dim = perf_dim + (n_conds if args.conditional_predictor else 0)
    predictor = SNMLPPredictor(
        input_dim=predictor_input_dim,
        output_dim=n_perf,
        hidden_dims=args.predictor_hidden_dims,
        lipschitz_scale=args.predictor_lipschitz_scale,
    )

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
        val_gate=True,  # volume + pruning gated on val_nmse_rec AND val_nmse_perf
    ).to(device)

    print(f"\n{'=' * 60}")
    print("Constrained PLVAE — thermoelastic2d [GN encoder + val-gate]")
    print(f"Design shape          : {design_shape}")
    print(f"Latent dim / perf dim : {args.latent_dim} / {perf_dim}")
    print(f"NMSE threshold (rec)  : {args.nmse_threshold_rec}")
    print(f"NMSE threshold (perf) : {args.nmse_threshold_perf}")
    print(f"GN groups             : {args.gn_groups}")
    print(f"Val-gate              : True")
    print(f"Pruning from epoch    : {args.pruning_epoch} ({args.pruning_strategy}, thr={args.pruning_threshold})")
    print(f"{'=' * 60}\n")

    hf = problem.dataset.with_format("torch")
    train_ds = hf["train"]
    val_ds = hf["val"]

    x_train = train_ds["optimal_design"][:].unsqueeze(1)
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

    loader = DataLoader(
        TensorDataset(x_train, c_train_scaled, p_train_scaled),
        batch_size=args.batch_size, shuffle=True, generator=g,
    )
    val_loader = DataLoader(
        TensorDataset(x_val, c_val_scaled, p_val_scaled),
        batch_size=args.batch_size, shuffle=False,
    )

    best_val_nmse = float("inf")
    epochs_no_improve = 0

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
                "nmse_r": f"{plvae.nmse_rec:.4f}",
                "nmse_p": f"{plvae.nmse_perf:.4f}",
                "vol_on": int(plvae.vol_active),
                "val_ok": int(plvae._val_nmse_ok),
                "dim"   : plvae.dim,
            })

            if args.track:
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
                    "val_nmse_ok"        : int(plvae._val_nmse_ok),
                    "active_dims"        : plvae.dim,
                    "epoch"              : epoch,
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

        # Update val gate for next epoch
        plvae.set_val_nmse(val_nmse_rec, val_nmse_perf)

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

        if args.save_model and epoch == args.n_epochs - 1:
            ckpt_path = os.path.join(args.checkpoint_dir, "thermoelastic2d_constrained_plvae_val_gate.pth")
            th.save({
                "epoch"    : epoch,
                "encoder"  : plvae.encoder.state_dict(),
                "decoder"  : plvae.decoder.state_dict(),
                "predictor": plvae.predictor.state_dict(),
                "optimizer": plvae.optim.state_dict(),
                "args"     : vars(args),
            }, ckpt_path)
            if args.track:
                artifact = wandb.Artifact(f"{args.problem_id}_{args.algo}", type="model")
                artifact.add_file(ckpt_path)
                wandb.log_artifact(artifact, aliases=[f"seed_{args.seed}"])

    if args.track:
        wandb.finish()
