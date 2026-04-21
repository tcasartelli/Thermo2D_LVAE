"""Constrained LVAE for thermoelastic2D designs with plummet-based dynamic pruning.

Adapted from constrained_vanilla_lvae_2d.py — uses NMSE threshold constraint
instead of fixed volume weight. problem_id and resize_dimensions set for
thermoelastic2d dataset (64x64 designs).

For more information on LVAE, see: https://arxiv.org/abs/2404.17773
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import os
import random
import time
import numpy as np 

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
from engiopt.vanilla_lvae.components import Encoder2D
from engiopt.vanilla_lvae.components import TrueSNDecoder2D
import wandb


@dataclass
class Args: # ARgs è un dataclass che contiene tutti i parametri del training con i loro valori di default
    """Command-line arguments for constrained vanilla LVAE training on thermoelastic2d."""
    # aggiunta sul ssperimento dei weight
    # In Args aggiungi:
    weight_filter: float | None = None
    """If set, train only on samples with this weight value (e.g. 0.0 or 1.0). None = use all."""
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
    batch_size: int = 128 #numero di campioni che vengono processati alla volta, batch è in tensore [128,1,64,64] ossia 128 design con channel 1 a simensione 64x64
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
    # thermoelastic2d designs are 64x64 — no need to upscale to 100x100
    resize_dimensions: tuple[int, int] = (100, 100)
    """Resize input to this before encoding. Matches thermoelastic2d native resolution."""
    decoder_lipschitz_scale: float = 1.0
    """Lipschitz bound for spectrally normalized decoder."""

    # Output dirs (override for Euler scratch)
    images_dir: str = os.path.join(os.environ.get("SCRATCH", "."), "thermoelastic2d_constrained_lvae", "images")
    """Directory to save visualisation images."""
    checkpoint_dir: str = os.path.join(os.environ.get("SCRATCH", "."), "thermoelastic2d_constrained_lvae", "checkpoints")
    """Directory to save model checkpoints."""
    #early stopping per monitorare se val_nmse sta migliorando opure no
    early_stopping: bool = True # ilmeccanismo di early stopping si attiva solo se early_stopping è True, e serve per fermare il training se dopo un certo numero di epoche (patience) non c'è miglioranmento nella mterica di validazione (in questo caso, la NMSE).
    patience: int = 10 # quanto epoche di NON-MIGLIORAMENTO  tolleri prima di fermarti
    min_delta: float = 0.001 #val_nmse < best_val_nmse - min_delta, min_delta è la soglia minima di miglioranemnto reale 
    early_stopping_start_epoch: int = 0 # da quale epoca iniziare a controllare
    weight_filter: float | None = None #
    





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
    run_name = f"nmse_threshold_{args.nmse_threshold}_thermoelastic_2d"
    if args.track:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            save_code=True,
            name=run_name,
        )

    # ---- Seeding ----  #serve per la riproducibilità: se faccio il training due volte con lo stesso seme ottengo gli stessi risultati
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
    enc = Encoder2D(args.latent_dim, design_shape, args.resize_dimensions)  # (dimensione latente 100, input immagine 64x64, resize a 100x100)
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
    print("Constrained LVAE — thermoelastic2d")
    print(f"Design shape     : {design_shape}")
    print(f"Resize to        : {args.resize_dimensions}")
    print(f"Latent dim       : {args.latent_dim}")
    print(f"Lipschitz        : {args.decoder_lipschitz_scale}")
    print(f"NMSE threshold   : {args.nmse_threshold}")
    print(f"Constraint mode  : {args.constraint_mode}")
    print(f"Pruning from epoch {args.pruning_epoch} ({args.pruning_strategy}, thr={args.pruning_threshold})")
    print(f"Images → {args.images_dir}")
    print(f"Checkpoints → {args.checkpoint_dir}")
    print(f"{'=' * 60}\n")

    # ---- DataLoader ----
    hf = problem.dataset.with_format("torch") # with_format("torch") converte auromaticamte tutti i numpy array in tensori di Pytorch, il risultato viene salvato in hf 
    train_ds = hf["train"] # train_ds e val_ds sono dei tensori che contengono rispettivamente i dati di addestramento e di validazione, in particolare contengono un tensore chiamato "optimal_design" che rappresenta i design ottimali per il problema thermoelastic2d, e un tensore chiamato "weight" che rappresenta i pesi associati a ciascun design (ad esempio, 0.0 o 1.0)
    val_ds = hf["val"] # train, test e val sono i tre slplit del dataset su hugging face

    if args.weight_filter is not None: # la maschera è un modo di filtare, dammi solo i weighr che fanno true
        mask_tr = np.array(train_ds["weight"][:]) == args.weight_filter
        mask_va = np.array(val_ds["weight"][:])   == args.weight_filter
        x_train = train_ds["optimal_design"][:][mask_tr].unsqueeze(1) # [mask_tr]  è una maschera booleana, tiene solo le righe  True, unsqueeze(1) aggiunge la dimensione del canale 1, da (N, 64,64] a [batch, canali, H, W])
        x_val   = val_ds["optimal_design"][:][mask_va].unsqueeze(1)
        print(f"Filtered to weight={args.weight_filter}: {len(x_train)} train, {len(x_val)} val")
    else:
        x_train = train_ds["optimal_design"][:].unsqueeze(1)
        x_val   = val_ds["optimal_design"][:].unsqueeze(1)
        
    # Set data variance for NMSE computation NMSE = MSE / varianza_dei_dati, Nomrlized rende il valore confronatbile indipendentemente dalla scala dei dati, ottieni un valore tra 0 e 1
    lvae.set_data_variance(x_train) # set_data_variance prende tutti i valori di pixel dei design in x_train e calcola quanto varinao in media 
    print(f"Data variance    : {lvae.data_var:.6f}") # print 6 valori floating del fata variance

    # NMSE viene confrontata co la soglia nmse_treshold
    # se NMSE > soglia allora riscotrsuisce male,  duqnue va a ottimizzare solo la reconstruction loss e le sdimesnioni aumentano
    # il NMSE scende sotto il valore di soglia, la ricostruzionde diventa ok, allora inizzia a ottmizzare la comporessione 

    loader     = DataLoader(TensorDataset(x_train), batch_size=args.batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(TensorDataset(x_val),   batch_size=args.batch_size, shuffle=False)

    # Dataloader è il meccanismo che dhe divide il dataset in mini-batch e li fornische uno alla volta durante il training
    # TensorDataset è una class di PyTorch che avvolge uno o pi?u tensori un un goggeto datset
    # shuffle=True rimescola i dati ad ogni epoch per evitare che il modello imoari l'orinde, chiaramente il shuffle=False durante il validation per ché l'ordine non importa e vuoi risultati riproducibili
    # loader è per il training mentre val_loader è per la validation 

    # ---- Early stopping state ----
    best_val_nmse = float("inf")
    epochs_no_improve = 0

    # ---- Training loop ----
    for epoch in range(args.n_epochs):
        lvae.epoch_hook(epoch=epoch) #

        bar = tqdm.tqdm(loader, desc=f"Epoch {epoch}") #bar Epoch 42: 100%|████████████| 78/78 [00:12<00:00, rec=0.0234, nmse=0.0412]
        for i, batch in enumerate(bar): # iterazione sui mini-batch, bacth è una tupla di forma (batch_size, 1, 64, 64) 
            x_batch = batch[0].to(device) #to(device) sposta il tensore x_bacth sulla GPU se disponibile, altrimenti rimane sulla CPIU
            lvae.optim.zero_grad() #azzera i gradienti dell'ottimizazione 

            loss = lvae.loss(x_batch) # significa che calcola la loss totatle (reconstruction + volume) e la salva in lvae.rec_loss, lvae.vol_loss e lvae.nmse
            loss.backward() # calcola i gradienti nuovi
            lvae.optim.step() # aggiorni i pesi del modello in base ai gradienti clacolati

            bar.set_postfix({
                "rec": f"{lvae.rec_loss:.4f}",
                "vol": f"{lvae.vol_loss:.4f}",
                "nmse": f"{lvae.nmse:.4f}",
                "active": int(lvae.vol_active),
                "dim": lvae.dim,
            })

            if args.track:
                batches_done = epoch * len(bar) + i # len(bar) corrisponde al numero di mini-bacth in un epoch  , ossia lne(bar) = 10'000 samples / 120 bacthes = 79 passagi per vedere tutto il
                wandb.log({
                    "rec_loss"    : lvae.rec_loss,
                    "vol_loss"    : lvae.vol_loss,
                    "total_loss"  : loss.item(),
                    "nmse"        : lvae.nmse,
                    "nmse_threshold": args.nmse_threshold,
                    "vol_active"  : int(lvae.vol_active),
                    #"balance_factor": lvae.balance_factor,
                    "active_dims" : lvae.dim,
                    "epoch"       : epoch,
                    "constraint_mode": args.constraint_mode,
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

                    log_dict = {}
                    for key, path in [("dim_plot", dim_path), ("interp_plot", interp_path), ("norm_plot", norm_path)]:
                        try:
                            log_dict[key] = wandb.Image(path)
                        except Exception:
                            pass
                    wandb.log(log_dict)

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
            ckpt_path = os.path.join(args.checkpoint_dir, "thermoelastic2d_constrained_lvae.pth")
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
