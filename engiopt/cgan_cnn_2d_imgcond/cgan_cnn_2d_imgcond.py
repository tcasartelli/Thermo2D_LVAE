"""Conditional DCGAN with CNN for 2D problems, with support for image conditions.

Based on cgan_cnn_2d.py. Uses spatial concatenation so that problems with
array-valued conditions (e.g. thermoelastic2d's 65x65 boundary matrices) can be
handled alongside scalar conditions. When no image conditions exist the model
behaves identically to the original cgan_cnn_2d.
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
from torchvision import transforms
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
    batch_size: int = 32
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
    latent_dim: int = 32
    """dimensionality of the latent space"""
    sample_interval: int = 1000
    """interval between image samples"""


class Generator(nn.Module):
    """Conditional GAN generator that outputs 100 x 100 images.

    Supports both scalar and image conditions. Scalar conditions are processed
    through a learned path to 7x7 feature maps. Image conditions are spatially
    resized to 7x7 and concatenated at the first upsampling stage.

    Args:
        latent_dim: Dimensionality of the noise (latent) vector.
        n_scalar_conds: Number of scalar conditional features.
        n_img_conds: Number of image condition channels.
        design_shape: Spatial shape of the output design.
        num_filters: Number of filters in each upsampling stage.
        out_channels: Number of output channels in the final image.
    """

    def __init__(  # noqa: PLR0913
        self,
        latent_dim: int,
        n_scalar_conds: int,
        n_img_conds: int = 0,
        design_shape: tuple[int, int] = (100, 100),
        num_filters: list[int] = [256, 128, 64, 32],  # noqa: B006
        out_channels: int = 1,
    ):
        super().__init__()
        self.design_shape = design_shape
        self.n_img_conds = n_img_conds

        # Path for noise z
        self.z_path = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, num_filters[0] // 2, kernel_size=7, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_filters[0] // 2),
            nn.ReLU(inplace=True),
        )
        # Path for scalar conditions only
        self.c_path = nn.Sequential(
            nn.ConvTranspose2d(n_scalar_conds, num_filters[0] // 2, kernel_size=7, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_filters[0] // 2),
            nn.ReLU(inplace=True),
        )

        # After concat: z_feat + c_feat + img_conds = num_filters[0] + n_img_conds channels
        # When n_img_conds=0 this is just num_filters[0], identical to original.
        up_in_channels = num_filters[0] + n_img_conds

        # 4 upsampling layers: 7x7 -> 13x13 -> 25x25 -> 50x50 -> 100x100
        self.up_blocks = nn.Sequential(
            # 7x7 -> 13x13 (kernel=3, stride=2, pad=1)
            nn.ConvTranspose2d(up_in_channels, num_filters[1], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_filters[1]),
            nn.ReLU(inplace=True),
            # 13x13 -> 25x25 (kernel=3, stride=2, pad=1)
            nn.ConvTranspose2d(num_filters[1], num_filters[2], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_filters[2]),
            nn.ReLU(inplace=True),
            # 25x25 -> 50x50 (kernel=4, stride=2, pad=1)
            nn.ConvTranspose2d(num_filters[2], num_filters[3], kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_filters[3]),
            nn.ReLU(inplace=True),
            # 50x50 -> 100x100 (kernel=4, stride=2, pad=1)
            nn.ConvTranspose2d(num_filters[3], out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: th.Tensor, scalar_conds: th.Tensor, img_conds: th.Tensor | None = None) -> th.Tensor:
        """Forward pass for the Generator.

        Args:
            z: (B, z_dim, 1, 1) noise vector.
            scalar_conds: (B, n_scalar_conds, 1, 1) scalar conditions.
            img_conds: (B, n_img_conds, H, W) image conditions, or None.

        Returns:
            (B, out_channels, design_H, design_W) generated design.
        """
        # Run noise & scalar conditions through separate stems
        z_feat = self.z_path(z)  # -> (B, num_filters[0]//2, 7, 7)
        c_feat = self.c_path(scalar_conds)  # -> (B, num_filters[0]//2, 7, 7)

        # Concat along channel dim, optionally including spatially-resized image conditions
        if self.n_img_conds > 0 and img_conds is not None:
            resized_img = th.nn.functional.interpolate(img_conds, size=(7, 7), mode="bilinear", align_corners=False)
            x = th.cat([z_feat, c_feat, resized_img], dim=1)  # (B, num_filters[0] + n_img_conds, 7, 7)
        else:
            x = th.cat([z_feat, c_feat], dim=1)  # (B, num_filters[0], 7, 7)

        # Upsample through the main blocks
        out = self.up_blocks(x)  # -> (B, out_channels, 100, 100)

        # Resize Image
        return transforms.Resize((self.design_shape[0], self.design_shape[1]))(out)


class Discriminator(nn.Module):
    """Conditional GAN discriminator that takes 100 x 100 images.

    Supports both scalar and image conditions via spatial concatenation (pix2pix style).
    Scalar conditions are expanded to 100x100 and image conditions are resized to 100x100,
    then concatenated along the channel dimension.

    Args:
        n_scalar_conds: Number of scalar conditional channels.
        n_img_conds: Number of image condition channels.
        in_channels: Number of channels in real images.
        num_filters: Number of filters in each downsampling stage.
        out_channels: Typically 1 for final real/fake score (sigmoid).
    """

    def __init__(
        self,
        n_scalar_conds: int,
        n_img_conds: int = 0,
        in_channels: int = 1,
        num_filters: list[int] = [32, 64, 128, 256],  # noqa: B006
        out_channels: int = 1,
    ):
        super().__init__()
        self.n_img_conds = n_img_conds

        # Total condition channels: scalar channels expanded to 100x100 + image channels resized to 100x100
        total_cond_channels = n_scalar_conds + n_img_conds

        # Path for real image
        self.img_path = nn.Sequential(
            nn.Conv2d(in_channels, num_filters[0] // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Path for condition
        self.cond_path = nn.Sequential(
            nn.Conv2d(total_cond_channels, num_filters[0] // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Combined path (50x50 -> 25x25 -> 13x13 -> 7x7)
        self.down_blocks = nn.Sequential(
            # 50->25 (kernel=4, stride=2, pad=1)
            nn.Conv2d(num_filters[0], num_filters[1], kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_filters[1]),
            nn.LeakyReLU(0.2, inplace=True),
            # 25->13 (kernel=3, stride=2, pad=1)
            nn.Conv2d(num_filters[1], num_filters[2], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_filters[2]),
            nn.LeakyReLU(0.2, inplace=True),
            # 13->7 (kernel=3, stride=2, pad=1)
            nn.Conv2d(num_filters[2], num_filters[3], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_filters[3]),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Final 7x7 -> 1x1
        self.final_conv = nn.Sequential(
            nn.Conv2d(num_filters[3], out_channels, kernel_size=7, stride=1, padding=0, bias=False), nn.Sigmoid()
        )

    def forward(self, x: th.Tensor, scalar_conds: th.Tensor, img_conds: th.Tensor | None = None) -> th.Tensor:
        """Forward pass for the Discriminator.

        Args:
            x: (B, in_channels, 100, 100) real or generated image.
            scalar_conds: (B, n_scalar_conds, 1, 1) scalar conditions.
            img_conds: (B, n_img_conds, H, W) image conditions, or None.

        Returns:
            (B, out_channels, 1, 1) real/fake score.
        """
        # Resize Image
        x = transforms.Resize((100, 100))(x)

        # Build the full spatial condition tensor (B, total_cond_channels, 100, 100)
        expanded_scalar = scalar_conds.expand(-1, -1, 100, 100)  # (B, n_scalar_conds, 100, 100)

        if self.n_img_conds > 0 and img_conds is not None:
            resized_img = th.nn.functional.interpolate(img_conds, size=(100, 100), mode="bilinear", align_corners=False)
            c = th.cat([expanded_scalar, resized_img], dim=1)  # (B, n_scalar_conds + n_img_conds, 100, 100)
        else:
            c = expanded_scalar

        # Separate stem for image and condition
        x_feat = self.img_path(x)  # (B, num_filters[0]//2, 50, 50)
        c_feat = self.cond_path(c)  # (B, num_filters[0]//2, 50, 50)

        # Concat along channel dimension
        h = th.cat([x_feat, c_feat], dim=1)  # (B, num_filters[0], 50, 50)

        # Downsample
        h = self.down_blocks(h)  # -> (B, num_filters[3], 7, 7)

        # Final conv => (B, out_channels, 1, 1)
        return self.final_conv(h)


if __name__ == "__main__":
    args = tyro.cli(Args)

    problem = BUILTIN_PROBLEMS[args.problem_id]()
    problem.reset(seed=args.seed)

    design_shape = problem.design_space.shape
    conditions = problem.conditions
    scalar_cond_keys = get_scalar_condition_keys(problem, problem.dataset["train"])
    n_scalar_conds = len(scalar_cond_keys)

    # Detect image conditions
    img_cond_keys = get_image_condition_keys(problem, problem.dataset["train"])
    n_img_conds = len(img_cond_keys)
    img_cond_shape: tuple[int, ...] | None = None
    if n_img_conds > 0:
        img_cond_shape = get_image_condition_shape(problem.dataset["train"], img_cond_keys)
        print(f"Found {n_img_conds} image condition(s): {img_cond_keys}, shape={img_cond_shape}")
    else:
        print("No image conditions found; behaving like standard cgan_cnn_2d.")

    # Logging
    run_name = f"{args.problem_id}__{args.algo}__{args.seed}__{int(time.time())}"
    if args.track:
        config = vars(args)
        config["n_img_conds"] = n_img_conds
        config["img_cond_shape"] = img_cond_shape
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
    generator = Generator(
        latent_dim=args.latent_dim,
        n_scalar_conds=n_scalar_conds,
        n_img_conds=n_img_conds,
        design_shape=design_shape,
    )
    discriminator = Discriminator(
        n_scalar_conds=n_scalar_conds,
        n_img_conds=n_img_conds,
    )

    generator.to(device)
    discriminator.to(device)
    adversarial_loss.to(device)

    # Configure data loader
    training_ds = problem.dataset.with_format("torch", device=device)["train"]

    # Build dataset tensors: designs, scalar conds, and (optionally) image conds
    ds_tensors: list[th.Tensor] = [
        training_ds["optimal_design"][:].flatten(1),
        *[training_ds[key][:] for key in scalar_cond_keys],
    ]
    if n_img_conds > 0:
        # Stack all image conditions into a single tensor: (N, n_img_conds, H, W)
        img_cond_tensors = [training_ds[key][:].float() for key in img_cond_keys]
        # Each tensor is (N, H, W); stack along a new channel dim
        stacked_img_conds = th.stack(img_cond_tensors, dim=1)  # (N, n_img_conds, H, W)
        ds_tensors.append(stacked_img_conds)

    training_dataset = th.utils.data.TensorDataset(*ds_tensors)
    dataloader = th.utils.data.DataLoader(
        training_dataset,
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
        z = th.randn((n_designs, args.latent_dim, 1, 1), device=device, dtype=th.float)

        linspaces = [
            th.linspace(scalar_conds_all[:, i].min(), scalar_conds_all[:, i].max(), n_designs, device=device)
            for i in range(scalar_conds_all.shape[1])
        ]

        desired_conds = th.stack(linspaces, dim=1)
        desired_conds_reshaped = desired_conds.reshape(-1, n_scalar_conds, 1, 1)

        # For image conditions, use zeros for visualization (neutral boundary condition)
        sample_img_conds: th.Tensor | None = None
        if n_img_conds > 0 and img_cond_shape is not None:
            sample_img_conds = th.zeros(
                (n_designs, n_img_conds, *img_cond_shape),
                device=device,
                dtype=th.float,
            )

        gen_imgs = generator(z, desired_conds_reshaped, sample_img_conds)
        return desired_conds, gen_imgs

    # ----------
    #  Training
    # ----------
    for epoch in tqdm.trange(args.n_epochs):
        for i, data in enumerate(dataloader):
            # Unpack data: designs, scalar_cond_1, ..., scalar_cond_N, [img_conds]
            designs = data[0]

            # Scalar conditions: indices 1 through n_scalar_conds
            scalar_conds_batch = th.stack(data[1 : 1 + n_scalar_conds], dim=1).reshape(-1, n_scalar_conds, 1, 1)

            # Image conditions (last tensor in data, if present)
            img_conds_batch: th.Tensor | None = None
            if n_img_conds > 0:
                img_conds_batch = data[1 + n_scalar_conds]  # (B, n_img_conds, H, W)

            # Keep a reference for sample_designs
            scalar_conds_all = scalar_conds_batch.reshape(-1, n_scalar_conds)

            # Adversarial ground truths
            valid = th.ones((designs.size(0), 1), requires_grad=False, device=device)
            fake = th.zeros((designs.size(0), 1), requires_grad=False, device=device)

            # -----------------
            #  Train Generator
            # min log(1 - D(G(z))) <==> max log(D(G(z)))
            # -----------------
            optimizer_generator.zero_grad()

            # Sample noise as generator input
            z = th.randn((designs.size(0), args.latent_dim, 1, 1), device=device, dtype=th.float)

            # Generate a batch of images
            gen_designs = generator(z, scalar_conds_batch, img_conds_batch)

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_designs, scalar_conds_batch, img_conds_batch)[:, 0, 0], valid)

            g_loss.backward()
            optimizer_generator.step()

            # ---------------------
            #  Train Discriminator
            # max log(D(real)) + log(1 - D(G(z)))
            # ---------------------
            optimizer_discriminator.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(
                discriminator(
                    designs.reshape(-1, 1, design_shape[0], design_shape[1]),
                    scalar_conds_batch,
                    img_conds_batch,
                )[:, 0, 0],
                valid,
            )
            fake_loss = adversarial_loss(
                discriminator(gen_designs.detach(), scalar_conds_batch, img_conds_batch)[:, 0, 0], fake
            )
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
                    # Keep scalar_conds_all accessible for sample_designs
                    scalar_conds_all = scalar_conds_batch.reshape(-1, n_scalar_conds)
                    # Extract 25 designs
                    desired_conds, sample_designs_out = sample_designs(25)
                    fig, axes = plt.subplots(5, 5, figsize=(12, 12))

                    # Flatten axes for easy indexing
                    axes = axes.flatten()

                    # Plot each tensor as a scatter plot
                    for j, tensor in enumerate(sample_designs_out):
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
                    }
                    ckpt_disc = {
                        "epoch": epoch,
                        "batches_done": batches_done,
                        "discriminator": discriminator.state_dict(),
                        "optimizer_discriminator": optimizer_discriminator.state_dict(),
                        "loss": d_loss.item(),
                        "n_img_conds": n_img_conds,
                        "img_cond_shape": img_cond_shape,
                    }

                    th.save(ckpt_gen, "generator.pth")
                    th.save(ckpt_disc, "discriminator.pth")
                    if args.track:
                        artifact_gen = wandb.Artifact(f"{args.problem_id}_{args.algo}_generator", type="model")
                        artifact_gen.add_file("generator.pth")
                        artifact_disc = wandb.Artifact(f"{args.problem_id}_{args.algo}_discriminator", type="model")
                        artifact_disc.add_file("discriminator.pth")

                        wandb.log_artifact(artifact_gen, aliases=[f"seed_{args.seed}", f"run_{wandb.run.id}"])
                        wandb.log_artifact(artifact_disc, aliases=[f"seed_{args.seed}", f"run_{wandb.run.id}"])

    wandb.finish()
