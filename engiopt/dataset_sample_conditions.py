"""Contains function for sampling conditions from the dataset.

Also formats for use in problem.optimize and problem.simulate
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch as th

if TYPE_CHECKING:
    from datasets import Dataset
    from engibench.core import Problem

from engiopt.transforms import get_image_condition_keys
from engiopt.transforms import get_scalar_condition_keys


def sample_conditions(
    problem: Problem, n_samples: int, device: th.device, seed: int
) -> tuple[th.Tensor, Dataset, np.ndarray, np.ndarray]:
    """Samples conditions and designs from the dataset and prepares tensors for the generator.

    Only scalar (non-image), non-constant conditions are included in the returned tensor.
    The returned Dataset contains ALL available conditions for use as ``config`` in
    ``problem.optimize()`` / ``problem.simulate()``.

    Args:
        problem: The problem containing the dataset with conditions and designs.
        n_samples: Number of samples to draw.
        device: The device (e.g., 'cpu', 'mps', 'cuda') to place the tensors on.
        seed: Random seed for reproducibility.

    Returns:
        conditions_tensor: Scalar conditions as ``(n_samples, n_scalar_conds)`` tensor.
        sampled_conditions: A Hugging Face Dataset with all available conditions (for config).
        sampled_designs_np: A NumPy array of sampled optimal designs.
        selected_indices: The indices of the sampled conditions and designs.
    """
    ### Set up testing conditions ###
    rng = np.random.default_rng(seed)

    # Extract conditions — keep all dataset-available keys for config passing
    dataset = problem.dataset["test"]
    available_keys = [k for k in problem.conditions_keys if k in dataset.column_names]
    conditions_ds = dataset.select_columns(available_keys)

    # Sample conditions and test_ds designs at random indices
    selected_indices = rng.choice(len(dataset), n_samples, replace=True)
    sampled_conditions = conditions_ds.select(selected_indices)
    sampled_designs_np = np.array(dataset["optimal_design"])[selected_indices]

    # Create tensor from scalar-only conditions (for model input)
    scalar_keys = get_scalar_condition_keys(problem, dataset)
    if scalar_keys:
        conditions_tensor = th.stack(
            [th.tensor(np.array(sampled_conditions[key]), dtype=th.float32, device=device) for key in scalar_keys],
            dim=1,
        )
    else:
        conditions_tensor = th.zeros(n_samples, 0, dtype=th.float32, device=device)

    return conditions_tensor, sampled_conditions, sampled_designs_np, selected_indices


def sample_conditions_with_images(
    problem: Problem, n_samples: int, device: th.device, seed: int
) -> tuple[th.Tensor, th.Tensor | None, Dataset, np.ndarray, np.ndarray]:
    """Like :func:`sample_conditions` but also returns image condition tensors.

    Args:
        problem: The problem containing the dataset with conditions and designs.
        n_samples: Number of samples to draw.
        device: The device to place the tensors on.
        seed: Random seed for reproducibility.

    Returns:
        scalar_conditions: Scalar conditions as ``(n_samples, n_scalar_conds)`` tensor.
        img_conditions: Image conditions as ``(n_samples, n_img_conds, H, W)`` tensor,
            or ``None`` if there are no image conditions.
        sampled_conditions: A Hugging Face Dataset with all available conditions.
        sampled_designs_np: A NumPy array of sampled optimal designs.
        selected_indices: The indices of the sampled conditions and designs.
    """
    rng = np.random.default_rng(seed)

    dataset = problem.dataset["test"]
    available_keys = [k for k in problem.conditions_keys if k in dataset.column_names]
    conditions_ds = dataset.select_columns(available_keys)

    selected_indices = rng.choice(len(dataset), n_samples, replace=True)
    sampled_conditions = conditions_ds.select(selected_indices)
    sampled_designs_np = np.array(dataset["optimal_design"])[selected_indices]

    # Scalar conditions
    scalar_keys = get_scalar_condition_keys(problem, dataset)
    if scalar_keys:
        scalar_tensor = th.stack(
            [th.tensor(np.array(sampled_conditions[key]), dtype=th.float32, device=device) for key in scalar_keys],
            dim=1,
        )
    else:
        scalar_tensor = th.zeros(n_samples, 0, dtype=th.float32, device=device)

    # Image conditions
    img_keys = get_image_condition_keys(problem, dataset)
    if img_keys:
        img_tensors = []
        for key in img_keys:
            arr = np.array(sampled_conditions[key])
            t = th.tensor(arr, dtype=th.float32, device=device)
            _ndim_no_channel = 3
            if t.ndim == _ndim_no_channel:  # (N, H, W) — add channel dim
                t = t.unsqueeze(1)
            img_tensors.append(t)
        img_tensor: th.Tensor | None = th.cat(img_tensors, dim=1)  # (N, n_img_conds, H, W)
    else:
        img_tensor = None

    return scalar_tensor, img_tensor, sampled_conditions, sampled_designs_np, selected_indices
