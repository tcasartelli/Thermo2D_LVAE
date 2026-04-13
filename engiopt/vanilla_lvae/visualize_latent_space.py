"""Visualize LVAE latent space structure and interpretability.

Provides:
1. Gradient sensitivity maps: How each latent dimension relates to pixel regions
2. Latent traversals: Visual effect of varying individual dimensions
3. Dimension importance: Bar charts of variance/importance per dimension
4. Latent projections: t-SNE/UMAP colored by performance
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Literal

from engibench.utils.all_problems import BUILTIN_PROBLEMS
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import torch as th
import tyro

from engiopt.transforms import get_performance_target
from engiopt.vanilla_lvae.utils import encode_designs
from engiopt.vanilla_lvae.utils import get_active_mask
from engiopt.vanilla_lvae.utils import load_full_lvae


@dataclass
class Args:
    """Command-line arguments for latent space visualization."""

    # Problem and model
    problem_id: str = "beams2d"
    """Problem identifier."""

    seed: int = 1
    """Seed used for LVAE training."""

    rec_threshold: float = 0.001
    """Reconstruction NMSE threshold."""

    perf_threshold: float = 0.001
    """Performance NMSE threshold."""

    # WandB
    wandb_project: str = "engiopt"
    """WandB project name."""

    wandb_entity: str | None = None
    """WandB entity name."""

    # Visualization options
    n_samples: int = 500
    """Number of samples for projections/statistics."""

    n_traversal_steps: int = 7
    """Number of steps in latent traversals."""

    n_top_dims: int = 10
    """Number of top dimensions to visualize in traversals."""

    traversal_range: float = 2.0
    """Standard deviations to traverse (+/-)."""

    projection: Literal["tsne", "umap"] = "tsne"
    """Projection method for latent visualization."""

    # Output
    output_dir: str = "visualizations/{problem_id}_lvae"
    """Output directory for saved figures."""


def compute_gradient_sensitivity(
    encoder: th.nn.Module,
    design: th.Tensor,
    device: th.device,
) -> th.Tensor:
    """Compute gradient sensitivity map: d(z_i)/d(pixels) for each dimension.

    Args:
        encoder: LVAE encoder.
        design: Single design tensor of shape (1, 1, H, W).
        device: Computation device.

    Returns:
        Sensitivity maps of shape (latent_dim, H, W).
    """
    encoder.eval()
    design = design.to(device).clone().requires_grad_(True)

    # Forward pass
    z = encoder(design)  # (1, latent_dim)
    latent_dim = z.shape[1]

    sensitivities = []
    for i in range(latent_dim):
        # Clear gradients
        if design.grad is not None:
            design.grad.zero_()

        # Backward pass for each latent dimension
        z[0, i].backward(retain_graph=True)

        # Get gradient magnitude
        grad = design.grad.detach().clone()  # (1, 1, H, W)
        sensitivities.append(grad.abs().squeeze().cpu())

    return th.stack(sensitivities, dim=0)  # (latent_dim, H, W)


def plot_gradient_sensitivity(
    encoder: th.nn.Module,
    designs: th.Tensor,
    n_dims: int,
    sorted_indices: np.ndarray,
    importance: np.ndarray,
    output_path: str,
    device: th.device,
) -> None:
    """Plot gradient sensitivity maps for top dimensions.

    Args:
        encoder: LVAE encoder.
        designs: Batch of designs (N, 1, H, W).
        n_dims: Number of top dimensions to plot.
        sorted_indices: Indices of dimensions sorted by importance.
        importance: Variance per dimension.
        output_path: Path to save figure.
        device: Computation device.
    """
    # Average sensitivity over multiple designs
    n_avg = min(10, len(designs))
    sensitivities_sum = None

    print(f"  Computing gradient sensitivity over {n_avg} designs...")
    for i in range(n_avg):
        sens = compute_gradient_sensitivity(encoder, designs[i : i + 1], device)
        if sensitivities_sum is None:
            sensitivities_sum = sens.numpy()
        else:
            sensitivities_sum += sens.numpy()

    assert sensitivities_sum is not None
    sensitivities_avg = sensitivities_sum / n_avg  # (latent_dim, H, W)

    # Determine grid layout
    n_cols = min(5, n_dims)
    n_rows = (n_dims + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    axes_flat = axes.flatten()

    for idx, dim_idx in enumerate(sorted_indices[:n_dims]):
        ax = axes_flat[idx]
        im = ax.imshow(sensitivities_avg[dim_idx], cmap="hot")
        ax.set_title(f"Dim {dim_idx}\n(var={importance[dim_idx]:.2e})", fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)

    # Hide unused axes
    for idx in range(n_dims, len(axes_flat)):
        axes_flat[idx].axis("off")

    plt.suptitle("Gradient Sensitivity Maps: d|z_i|/d|pixels|", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_latent_traversals(
    encoder: th.nn.Module,
    decoder: th.nn.Module,
    reference_design: th.Tensor,
    n_steps: int,
    n_dims: int,
    sorted_indices: np.ndarray,
    importance: np.ndarray,
    traversal_range: float,
    output_path: str,
    device: th.device,
) -> None:
    """Plot latent traversals for top dimensions.

    For each important dimension, traverse from -range*std to +range*std
    while keeping other dimensions fixed at the reference encoding.

    Args:
        encoder: LVAE encoder.
        decoder: LVAE decoder.
        reference_design: Reference design (1, 1, H, W).
        n_steps: Number of steps per traversal.
        n_dims: Number of dimensions to traverse.
        sorted_indices: Indices sorted by importance (descending).
        importance: Variance per dimension.
        traversal_range: Range in standard deviations.
        output_path: Path to save figure.
        device: Computation device.
    """
    encoder.eval()
    decoder.eval()

    with th.no_grad():
        z_ref = encoder(reference_design.to(device))  # (1, latent_dim)

    stds = np.sqrt(importance + 1e-12)

    fig, axes = plt.subplots(n_dims, n_steps, figsize=(2 * n_steps, 2 * n_dims))
    if n_dims == 1:
        axes = axes.reshape(1, -1)

    alphas = np.linspace(-traversal_range, traversal_range, n_steps)

    for row_idx, dim_idx in enumerate(sorted_indices[:n_dims]):
        dim_std = stds[dim_idx]

        for col_idx, alpha in enumerate(alphas):
            z_mod = z_ref.clone()
            z_mod[0, dim_idx] += alpha * dim_std

            with th.no_grad():
                x_mod = decoder(z_mod)  # (1, 1, H, W)

            ax = axes[row_idx, col_idx]
            ax.imshow(x_mod[0, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            ax.axis("off")

            if row_idx == 0:
                ax.set_title(f"{alpha:+.1f}σ", fontsize=9)

        # Add row label
        axes[row_idx, 0].set_ylabel(f"Dim {dim_idx}", fontsize=8, rotation=0, labelpad=35, va="center")

    plt.suptitle("Latent Traversals (rows = dimensions, cols = σ offsets)", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_importance_barchart(
    importance: np.ndarray,
    n_active: int,
    output_path: str,
) -> None:
    """Plot dimension importance as bar chart.

    Args:
        importance: Variance per dimension.
        n_active: Number of active dimensions.
        output_path: Path to save figure.
    """
    sorted_idx = np.argsort(importance)[::-1]
    sorted_importance = importance[sorted_idx]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # All dimensions (log scale)
    ax1.bar(range(len(importance)), sorted_importance, color="steelblue")
    ax1.set_yscale("log")
    ax1.set_xlabel("Dimension (sorted by variance)")
    ax1.set_ylabel("Variance (log scale)")
    ax1.set_title(f"All {len(importance)} Latent Dimensions")
    ax1.axvline(x=n_active - 0.5, color="r", linestyle="--", label=f"Active dims: {n_active}")
    ax1.legend()

    # Active dimensions only (linear scale)
    ax2.bar(range(n_active), sorted_importance[:n_active], color="steelblue")
    ax2.set_xlabel("Dimension (sorted by variance)")
    ax2.set_ylabel("Variance")
    ax2.set_title(f"Top {n_active} Active Dimensions")

    # Add cumulative explained variance
    cumsum = np.cumsum(sorted_importance[:n_active]) / sorted_importance.sum()
    ax2_twin = ax2.twinx()
    ax2_twin.plot(range(n_active), cumsum, "r-", linewidth=2)
    ax2_twin.set_ylabel("Cumulative Explained Variance", color="r")
    ax2_twin.tick_params(axis="y", labelcolor="r")
    ax2_twin.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_latent_projection(
    encoder: th.nn.Module,
    designs: th.Tensor,
    performance: np.ndarray,
    projection: str,
    output_path: str,
    device: th.device,
) -> None:
    """Plot t-SNE/UMAP of latent space colored by performance.

    Args:
        encoder: LVAE encoder.
        designs: Designs tensor (N, 1, H, W).
        performance: Performance values (N,).
        projection: "tsne" or "umap".
        output_path: Path to save figure.
        device: Computation device.
    """
    encoder.eval()

    # Encode all designs and slice to active (unpruned) dims
    print(f"  Encoding {len(designs)} designs...")
    z_all = encode_designs(encoder, designs.numpy(), device)
    active = get_active_mask(encoder)
    z_all = z_all[:, active]

    # Project
    print(f"  Computing {projection} projection ({z_all.shape[1]} active dims)...")
    if projection == "tsne":
        projector = TSNE(n_components=2, random_state=42, perplexity=min(30, len(z_all) - 1))
        z_2d = projector.fit_transform(z_all)
    elif projection == "umap":
        try:
            import umap

            projector = umap.UMAP(n_components=2, random_state=42)
            z_2d = projector.fit_transform(z_all)
        except ImportError:
            print("  UMAP not installed, falling back to t-SNE")
            projector = TSNE(n_components=2, random_state=42, perplexity=min(30, len(z_all) - 1))
            z_2d = projector.fit_transform(z_all)
    else:
        raise ValueError(f"Unknown projection: {projection}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        z_2d[:, 0],
        z_2d[:, 1],
        c=performance,
        cmap="viridis",
        alpha=0.7,
        s=20,
    )
    ax.set_xlabel(f"{projection.upper()} 1")
    ax.set_ylabel(f"{projection.upper()} 2")
    ax.set_title(f"Latent Space Projection ({projection.upper()}) Colored by Performance")

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Performance")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_sample_reconstructions(
    encoder: th.nn.Module,
    decoder: th.nn.Module,
    designs: th.Tensor,
    n_samples: int,
    output_path: str,
    device: th.device,
) -> None:
    """Plot original designs and their reconstructions.

    Args:
        encoder: LVAE encoder.
        decoder: LVAE decoder.
        designs: Designs tensor (N, 1, H, W).
        n_samples: Number of samples to show.
        output_path: Path to save figure.
        device: Computation device.
    """
    encoder.eval()
    decoder.eval()

    n_show = min(n_samples, len(designs))

    with th.no_grad():
        batch = designs[:n_show].to(device)
        z = encoder(batch)
        recon = decoder(z)

    fig, axes = plt.subplots(2, n_show, figsize=(2 * n_show, 4))

    for i in range(n_show):
        # Original
        axes[0, i].imshow(batch[i, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_ylabel("Original", fontsize=10)

        # Reconstruction
        axes[1, i].imshow(recon[i, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_ylabel("Recon", fontsize=10)

    plt.suptitle("Original vs Reconstruction", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


if __name__ == "__main__":
    args = tyro.cli(Args)

    # Set up
    problem = BUILTIN_PROBLEMS[args.problem_id]()
    problem.reset(seed=args.seed)

    th.manual_seed(args.seed)
    np.random.seed(args.seed)

    if th.backends.mps.is_available():
        device = th.device("mps")
    elif th.cuda.is_available():
        device = th.device("cuda")
    else:
        device = th.device("cpu")

    print(f"Using device: {device}")
    print(f"Problem: {args.problem_id}")
    print(f"LVAE: seed={args.seed}, rec={args.rec_threshold}, perf={args.perf_threshold}")

    # Create output directory
    output_dir = args.output_dir.format(problem_id=args.problem_id)
    os.makedirs(output_dir, exist_ok=True)

    # Load LVAE
    print("\nLoading LVAE from WandB...")
    encoder, decoder, predictor, config = load_full_lvae(
        args.problem_id,
        args.seed,
        args.rec_threshold,
        args.perf_threshold,
        args.wandb_project,
        args.wandb_entity,
        device,
    )
    print(f"  Latent dim: {config.latent_dim}, Perf dim: {config.perf_dim}")

    # Load dataset
    print("\nLoading dataset...")
    hf = problem.dataset.with_format("torch")
    train_ds = hf["train"]

    # Sample subset for visualization
    n_total = len(train_ds["optimal_design"])
    indices = np.random.choice(n_total, min(args.n_samples, n_total), replace=False)

    designs = train_ds["optimal_design"][indices].unsqueeze(1).float()  # (N, 1, H, W)

    # Build scalar performance target (handles multi-objective via weighted sum)
    all_perf = get_performance_target(problem, hf["train"])  # (N, 1)
    performance = all_perf[indices].squeeze(-1).numpy()

    print(f"  Loaded {len(designs)} designs")

    # Compute importance (per-dimension variance in latent space)
    print("\nComputing latent importance...")
    z_all = encode_designs(encoder, designs.numpy(), device)
    active = get_active_mask(encoder)
    importance = np.var(z_all, axis=0)
    n_active = int(active.sum())
    total_var = importance[active].sum()
    normalized_importance = importance / total_var if total_var > 0 else np.ones_like(importance) / len(importance)
    sorted_indices = np.argsort(importance)[::-1]

    print(f"  Active dimensions: {n_active} / {len(importance)}")

    # 1. Importance bar chart
    print("\n1. Plotting importance bar chart...")
    plot_importance_barchart(
        importance,
        n_active,
        os.path.join(output_dir, f"importance_barchart_rec{args.rec_threshold}_perf{args.perf_threshold}.png"),
    )

    # 2. Sample reconstructions
    print("\n2. Plotting sample reconstructions...")
    plot_sample_reconstructions(
        encoder,
        decoder,
        designs,
        n_samples=8,
        output_path=os.path.join(output_dir, f"reconstructions_rec{args.rec_threshold}_perf{args.perf_threshold}.png"),
        device=device,
    )

    # 3. Latent traversals
    print("\n3. Plotting latent traversals...")
    ref_idx = np.random.randint(len(designs))
    n_dims_to_traverse = min(args.n_top_dims, n_active, 8)  # Cap at 8 for reasonable figure size
    plot_latent_traversals(
        encoder,
        decoder,
        designs[ref_idx : ref_idx + 1],
        args.n_traversal_steps,
        n_dims_to_traverse,
        sorted_indices,
        importance,
        args.traversal_range,
        os.path.join(output_dir, f"latent_traversals_rec{args.rec_threshold}_perf{args.perf_threshold}.png"),
        device,
    )

    # 4. Gradient sensitivity maps
    print("\n4. Computing gradient sensitivity maps...")
    n_dims_to_show = min(args.n_top_dims, n_active, 10)  # Cap at 10 for reasonable figure size
    plot_gradient_sensitivity(
        encoder,
        designs[:10],
        n_dims_to_show,
        sorted_indices,
        importance,
        os.path.join(output_dir, f"gradient_sensitivity_rec{args.rec_threshold}_perf{args.perf_threshold}.png"),
        device,
    )

    # 5. Latent projection
    print(f"\n5. Computing {args.projection} projection...")
    plot_latent_projection(
        encoder,
        designs,
        performance,
        args.projection,
        os.path.join(
            output_dir, f"latent_projection_{args.projection}_rec{args.rec_threshold}_perf{args.perf_threshold}.png"
        ),
        device,
    )

    print(f"\nVisualization complete! Files saved to: {output_dir}")
