"""Conditional GAN for 2D problems with support for image conditions.

This is based on cgan_2d.py but adds support for image conditions (e.g. thermoelastic2d's
65x65 boundary matrices) alongside scalar conditions. When no image conditions are present,
behavior is identical to the original cgan_2d.

Image conditions are handled by resizing to a fixed spatial resolution and flattening,
then concatenating with other inputs to the MLP layers.

The original code is largely based on the excellent PyTorch GAN repo:
https://github.com/eriklindernoren/PyTorch-GAN
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import random
import time

from engibench.utils.all_problems import BUILTIN_PROBLEMS
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from torch import nn
import tqdm
import tyro

from engiopt.transforms import get_image_condition_keys
from engiopt.transforms import get_image_condition_shape
from engiopt.transforms import get_scalar_condition_keys
import wandb


@dataclass
class Args:
    """Command-line arguments."""

    problem_id: str = "beams2d"
    """Problem identifier."""
    algo: str = os.path.basename(__file__)[: -len(".py")]
    """The name of this algorithm."""

    # Tracking
    track: bool = True
    """Track the experiment with wandb."""
    wandb_project: str = "engiopt"
    """Wandb project name."""
    wandb_entity: str | None = None
    """Wandb entity name."""
    seed: int = 1
    """Random seed."""
    save_model: bool = False
    """Saves the model to disk."""

    # Algorithm specific
    n_epochs: int = 200
    """number of epochs of training"""
    batch_size: int = 64
    """size of the batches"""
    lr_gen: float = 0.0001
    """learning rate for the generator"""
    lr_disc: float = 0.0004
    """learning rate for the discriminator"""
    b1: float = 0.5
    """decay of first order momentum of gradient"""
    b2: float = 0.999
    """decay of first order momentum of gradient"""
    n_cpu: int = 8
    """number of cpu threads to use during batch generation"""
    latent_dim: int = 100
    """dimensionality of the latent space"""
    sample_interval: int = 400
    """interval between image samples"""
    img_resize_dim: int = 16
    """Spatial resolution to resize image conditions before flattening."""


class Generator(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        n_scalar_conds: int,
        n_img_conds: int,
        design_shape: tuple,
        img_resize_dim: int = 16,
    ):
        super().__init__()
        self.design_shape = design_shape
        self.n_img_conds = n_img_conds
        self.img_resize_dim = img_resize_dim

        input_dim = latent_dim + n_scalar_conds + n_img_conds * img_resize_dim * img_resize_dim

        def block(in_feat: int, out_feat: int, *, normalize: bool = True) -> list[nn.Module]:
            layers: list[nn.Module] = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(input_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(design_shape))),
            nn.Tanh(),
        )

    def forward(
        self,
        z: th.Tensor,
        scalar_conds: th.Tensor,
        img_conds: th.Tensor | None = None,
    ) -> th.Tensor:
        """Forward pass for the generator.

        Args:
            z: Latent space input tensor (B, latent_dim).
            scalar_conds: Scalar condition tensor (B, n_scalar).
            img_conds: Optional image condition tensor (B, n_img, H, W).

        Returns:
            Generated design tensor.
        """
        if img_conds is not None:
            img_resized = th.nn.functional.interpolate(
                img_conds,
                size=(self.img_resize_dim, self.img_resize_dim),
                mode="bilinear",
                align_corners=False,
            )
            img_flat = img_resized.flatten(1)
            gen_input = th.cat([z, scalar_conds, img_flat], dim=-1)
        else:
            gen_input = th.cat([z, scalar_conds], dim=-1)
        design = self.model(gen_input)
        return design.view(design.size(0), *self.design_shape)


class Discriminator(nn.Module):
    def __init__(
        self,
        design_shape: tuple,
        n_scalar_conds: int,
        n_img_conds: int,
        img_resize_dim: int = 16,
    ):
        super().__init__()
        self.n_img_conds = n_img_conds
        self.img_resize_dim = img_resize_dim

        input_dim = int(np.prod(design_shape)) + n_scalar_conds + n_img_conds * img_resize_dim * img_resize_dim

        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        design: th.Tensor,
        scalar_conds: th.Tensor,
        img_conds: th.Tensor | None = None,
    ) -> th.Tensor:
        """Forward pass for the discriminator.

        Args:
            design: Input design tensor.
            scalar_conds: Scalar condition tensor (B, n_scalar).
            img_conds: Optional image condition tensor (B, n_img, H, W).

        Returns:
            Validity score for the input design.
        """
        design_flat = design.view(design.size(0), -1)
        if img_conds is not None:
            img_resized = th.nn.functional.interpolate(
                img_conds,
                size=(self.img_resize_dim, self.img_resize_dim),
                mode="bilinear",
                align_corners=False,
            )
            img_flat = img_resized.flatten(1)
            d_in = th.cat([design_flat, scalar_conds, img_flat], dim=-1)
        else:
            d_in = th.cat([design_flat, scalar_conds], dim=-1)
        return self.model(d_in)


if __name__ == "__main__":
    args = tyro.cli(Args)

    problem = BUILTIN_PROBLEMS[args.problem_id]()
    problem.reset(seed=args.seed)

    design_shape = problem.design_space.shape
    conditions = problem.conditions
    scalar_cond_keys = get_scalar_condition_keys(problem, problem.dataset["train"])
    n_scalar_conds = len(scalar_cond_keys)

    img_cond_keys = get_image_condition_keys(problem, problem.dataset["train"])
    n_img_conds = len(img_cond_keys)
    img_cond_shape: tuple[int, ...] | None = None
    if n_img_conds > 0:
        img_cond_shape = get_image_condition_shape(problem.dataset["train"], img_cond_keys)

    # Logging
    algo = os.path.basename(__file__)[: -len(".py")]
    run_name = f"{args.problem_id}__{algo}__{args.seed}__{int(time.time())}"
    if args.track:
        config = vars(args)
        config["n_img_conds"] = n_img_conds
        config["img_cond_shape"] = img_cond_shape
        config["img_resize_dim"] = args.img_resize_dim
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=config, save_code=True, name=run_name)

    # Seeding
    th.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    th.backends.cudnn.deterministic = True

    os.makedirs("images", exist_ok=True)

    if th.backends.mps.is_available():
        device = th.device("mps")
    elif th.cuda.is_available():
        device = th.device("cuda")
    else:
        device = th.device("cpu")

    # Loss function
    adversarial_loss = th.nn.BCELoss()

    # Initialize generator and discriminator
    generator = Generator(args.latent_dim, n_scalar_conds, n_img_conds, design_shape, img_resize_dim=args.img_resize_dim)
    discriminator = Discriminator(design_shape, n_scalar_conds, n_img_conds, img_resize_dim=args.img_resize_dim)

    generator.to(device)
    discriminator.to(device)
    adversarial_loss.to(device)

    # Configure data loader
    training_ds = problem.dataset.with_format("torch", device=device)["train"]

    # Build dataset tensors: designs, then scalar conds, then optionally image conds
    ds_tensors: list[th.Tensor] = [
        training_ds["optimal_design"][:].flatten(1),
        *[training_ds[key][:] for key in scalar_cond_keys],
    ]
    stacked_img_conds: th.Tensor | None = None
    if n_img_conds > 0:
        # Stack image conditions into a single tensor: (N, n_img_conds, H, W)
        img_cond_tensors = [training_ds[key][:].float() for key in img_cond_keys]
        stacked_img_conds = th.stack(img_cond_tensors, dim=1)
        ds_tensors.append(stacked_img_conds)

    training_ds = th.utils.data.TensorDataset(*ds_tensors)
    dataloader = th.utils.data.DataLoader(
        training_ds,
        batch_size=args.batch_size,
        shuffle=True,
    )

    # Optimizers
    optimizer_generator = th.optim.Adam(generator.parameters(), lr=args.lr_gen, betas=(args.b1, args.b2))
    optimizer_discriminator = th.optim.Adam(discriminator.parameters(), lr=args.lr_disc, betas=(args.b1, args.b2))

    @th.no_grad()
    def sample_designs(n_designs: int) -> tuple[th.Tensor, th.Tensor]:
        """Samples n_designs from the generator."""
        # Sample noise
        z = th.randn((n_designs, args.latent_dim), device=device, dtype=th.float)

        linspaces = [
            th.linspace(scalar_conds_all[:, i].min(), scalar_conds_all[:, i].max(), n_designs, device=device)
            for i in range(scalar_conds_all.shape[1])
        ]

        desired_conds = th.stack(linspaces, dim=1)

        # For sampling, use first n_designs image conditions from training set if available
        sample_img_conds: th.Tensor | None = None
        if n_img_conds > 0 and stacked_img_conds is not None:
            sample_img_conds = stacked_img_conds[:n_designs].to(device)

        gen_imgs = generator(z, desired_conds, sample_img_conds)
        return desired_conds, gen_imgs

    # ----------
    #  Training
    # ----------
    for epoch in tqdm.trange(args.n_epochs):
        for i, data in enumerate(dataloader):
            # Unpack: designs, then n_scalar_conds scalar tensors, then optionally image conds
            designs = data[0]

            # Stack scalar conditions
            scalar_conds_batch = th.stack(data[1 : 1 + n_scalar_conds], dim=1)

            # Image conditions (last element if present)
            img_conds_batch: th.Tensor | None = None
            if n_img_conds > 0:
                img_conds_batch = data[1 + n_scalar_conds]

            # Keep a reference for sample_designs on first batch
            if epoch == 0 and i == 0:
                scalar_conds_all = scalar_conds_batch

            # Adversarial ground truths
            valid = th.ones((designs.size(0), 1), requires_grad=False, device=device)
            fake = th.zeros((designs.size(0), 1), requires_grad=False, device=device)

            # -----------------
            #  Train Generator
            # min log(1 - D(G(z))) <==> max log(D(G(z)))
            # -----------------
            optimizer_generator.zero_grad()

            # Sample noise as generator input
            z = th.randn((designs.size(0), args.latent_dim), device=device, dtype=th.float)

            # Generate a batch of images
            gen_designs = generator(z, scalar_conds_batch, img_conds_batch)

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_designs, scalar_conds_batch, img_conds_batch), valid)

            g_loss.backward()
            optimizer_generator.step()

            # ---------------------
            #  Train Discriminator
            # max log(D(real)) + log(1 - D(G(z)))
            # ---------------------
            optimizer_discriminator.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(designs, scalar_conds_batch, img_conds_batch), valid)
            fake_loss = adversarial_loss(discriminator(gen_designs.detach(), scalar_conds_batch, img_conds_batch), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_discriminator.step()

            # ----------
            #  Logging
            # ----------
            if args.track:
                batches_done = epoch * len(dataloader) + i
                wandb.log(
                    {
                        "d_loss": d_loss.item(),
                        "g_loss": g_loss.item(),
                        "epoch": epoch,
                        "batch": batches_done,
                    }
                )
                print(
                    f"[Epoch {epoch}/{args.n_epochs}] [Batch {i}/{len(dataloader)}] "
                    f"[D loss: {d_loss.item()}] [G loss: {g_loss.item()}]"
                )

                # This saves a grid image of 25 generated designs every sample_interval
                if batches_done % args.sample_interval == 0:
                    # Extract 25 designs
                    desired_conds, sample_imgs = sample_designs(25)
                    fig, axes = plt.subplots(5, 5, figsize=(12, 12))

                    # Flatten axes for easy indexing
                    axes = axes.flatten()

                    # Plot each tensor as a scatter plot
                    for j, tensor in enumerate(sample_imgs):
                        img = tensor.cpu().numpy().reshape(design_shape[0], design_shape[1])
                        dc = desired_conds[j].cpu()
                        axes[j].imshow(img)
                        title = [(scalar_cond_keys[k], f"{dc[k]:.2f}") for k in range(n_scalar_conds)]
                        title_string = "\n ".join(f"{condition}: {value}" for condition, value in title)
                        axes[j].title.set_text(title_string)
                        axes[j].set_xticks([])
                        axes[j].set_yticks([])

                    plt.tight_layout()
                    img_fname = f"images/{batches_done}.png"
                    plt.savefig(img_fname)
                    plt.close()
                    wandb.log({"designs": wandb.Image(img_fname)})

                # --------------
                #  Save models
                # --------------
                if args.save_model and epoch == args.n_epochs - 1 and i == len(dataloader) - 1:
                    ckpt_gen = {
                        "epoch": epoch,
                        "batches_done": batches_done,
                        "generator": generator.state_dict(),
                        "optimizer_generator": optimizer_generator.state_dict(),
                        "loss": g_loss.item(),
                        "n_img_conds": n_img_conds,
                        "img_cond_shape": img_cond_shape,
                        "img_resize_dim": args.img_resize_dim,
                    }
                    ckpt_disc = {
                        "epoch": epoch,
                        "batches_done": batches_done,
                        "discriminator": discriminator.state_dict(),
                        "optimizer_discriminator": optimizer_discriminator.state_dict(),
                        "loss": d_loss.item(),
                        "n_img_conds": n_img_conds,
                        "img_cond_shape": img_cond_shape,
                        "img_resize_dim": args.img_resize_dim,
                    }

                    th.save(ckpt_gen, "generator.pth")
                    th.save(ckpt_disc, "discriminator.pth")
                    if args.track:
                        artifact_gen = wandb.Artifact(f"{args.problem_id}_cgan_2d_imgcond_generator", type="model")
                        artifact_gen.add_file("generator.pth")
                        artifact_disc = wandb.Artifact(f"{args.problem_id}_cgan_2d_imgcond_discriminator", type="model")
                        artifact_disc.add_file("discriminator.pth")

                        wandb.log_artifact(artifact_gen, aliases=[f"seed_{args.seed}"])
                        wandb.log_artifact(artifact_disc, aliases=[f"seed_{args.seed}"])

    wandb.finish()
