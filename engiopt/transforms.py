"""Transformations for the data."""

from collections.abc import Callable

from datasets import Dataset
from engibench.core import Problem
from gymnasium import spaces
import numpy as np
import torch as th
import torch.nn.functional as f


def flatten_dict_factory(problem: Problem, device: th.device) -> Callable:
    """Factory function to create a flatten_dict function."""

    def flatten_dict(x):
        """Convert each design in the batch to a flattened tensor."""
        flattened = []
        for design in x:
            # Move to CPU for numpy conversion, then back to device
            design_cpu = {k: v.cpu().numpy() if isinstance(v, th.Tensor) else v for k, v in design.items()}
            flattened_array = spaces.flatten(problem.design_space, design_cpu)
            flattened.append(th.tensor(flattened_array, device=device))
        return th.stack(flattened)

    return flatten_dict


def resize_to(data: th.Tensor, h: int, w: int, mode: str = "bicubic") -> th.Tensor:
    """Resize 2D data back to any desired (h, w). Data should be a Tensor in the format (B, C, H, W)."""
    low_dim = 3
    if data.ndim == low_dim:
        data = data.unsqueeze(1)  # (B, 1, H, W)
    return f.interpolate(data, size=(h, w), mode=mode)


def get_scalar_condition_keys(problem: Problem, dataset: Dataset, *, drop_constants: bool = False) -> list[str]:
    """Return condition keys that are scalar, present in dataset, and (optionally) non-constant.

    Filters ``problem.conditions_keys`` to only those that:
    1. Exist as columns in *dataset*.
    2. Are scalar-valued (``ndim == 0``).
    3. (When *drop_constants* is True) Have non-zero standard deviation.

    Args:
        problem: An EngiBench problem instance.
        dataset: A HuggingFace Dataset split (e.g. ``problem.dataset["train"]``).
        drop_constants: If True, drop columns whose std is 0 across the dataset.

    Returns:
        A list of condition key names suitable for use as model inputs.
    """
    scalar_keys: list[str] = []
    for key in problem.conditions_keys:
        if key not in dataset.column_names:
            continue
        if np.asarray(dataset[0][key]).ndim > 0:
            continue  # skip image / array conditions
        scalar_keys.append(key)

    if drop_constants and scalar_keys:
        conds = th.stack([th.as_tensor(dataset[c][:]).float() for c in scalar_keys], dim=1)
        std = conds.std(dim=0)
        dropped = [c for i, c in enumerate(scalar_keys) if std[i] == 0]
        if dropped:
            print(f"Info: Dropping constant scalar conditions (std=0): {dropped}")
        scalar_keys = [c for i, c in enumerate(scalar_keys) if std[i] > 0]

    return scalar_keys


def get_image_condition_keys(problem: Problem, dataset: Dataset) -> list[str]:
    """Return condition keys that are image/array-valued and present in the dataset.

    This is the complement of :func:`get_scalar_condition_keys` — it returns
    keys whose first sample has ``ndim > 0`` (e.g. 65x65 boundary matrices
    in thermoelastic2d).

    Args:
        problem: An EngiBench problem instance.
        dataset: A HuggingFace Dataset split (e.g. ``problem.dataset["train"]``).

    Returns:
        A list of condition key names that are array-valued.
    """
    img_keys: list[str] = []
    for key in problem.conditions_keys:
        if key not in dataset.column_names:
            continue
        if np.asarray(dataset[0][key]).ndim > 0:
            img_keys.append(key)
    return img_keys


def get_image_condition_shape(dataset: Dataset, img_keys: list[str]) -> tuple[int, ...]:
    """Return the spatial shape of the first image condition.

    Assumes all image conditions share the same spatial dimensions.

    Args:
        dataset: A HuggingFace Dataset split.
        img_keys: Image condition key names (from :func:`get_image_condition_keys`).

    Returns:
        Shape tuple, e.g. ``(65, 65)``.
    """
    return tuple(np.asarray(dataset[0][img_keys[0]]).shape)


def rasterize_index_conditions(
    dataset: Dataset,
    keys: list[str],
    grid_shape: tuple[int, int],
) -> th.Tensor:
    """Convert sparse node-index condition arrays into dense binary masks.

    Many EngiBench problems store boundary conditions as variable-length arrays
    of flat node indices (e.g. ``fixed_elements = [23, 24, 25, ...]``) on the
    FE mesh.  For a design grid of shape ``(H, W)``, the node grid is typically
    ``(H+1, W+1)`` (corner junctions of pixels).  Pass the **node grid shape**
    as ``grid_shape`` so that indices are unravelled correctly.

    The resulting masks can be fed to a ``ConditionEncoder2D`` which will resize
    them to a standard resolution (e.g. 100×100) regardless of the input size.

    Args:
        dataset: A HuggingFace Dataset split.
        keys: Condition key names containing flat node indices.
        grid_shape: ``(H, W)`` of the node grid to rasterise onto.  For FE
            problems this is usually ``(design_H + 1, design_W + 1)``.

    Returns:
        Dense binary masks of shape ``(N, len(keys), H, W)`` as a float tensor.
    """
    n_samples = len(dataset[keys[0]])
    h, w = grid_shape
    n_cols = w  # number of columns used to unravel flat indices
    masks = th.zeros(n_samples, len(keys), h, w)
    for ch, key in enumerate(keys):
        for i in range(n_samples):
            indices = th.as_tensor(np.asarray(dataset[i][key])).long()
            if indices.numel() == 0:
                continue
            rows = indices // n_cols
            cols = indices % n_cols
            valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
            masks[i, ch, rows[valid], cols[valid]] = 1.0
    return masks


def get_performance_target(problem: Problem, dataset: Dataset) -> th.Tensor:
    """Build a performance target for each sample.

    For single-objective problems this returns ``dataset[obj_key]`` as ``(N, 1)``.
    For multi-objective problems this returns the vector of all objectives as
    ``(N, n_objs)``, so the predictor learns to reconstruct each objective
    independently.

    Returns:
        Tensor of shape ``(N, 1)`` for single-objective or ``(N, n_objs)`` for
        multi-objective problems.
    """
    obj_keys = [name for name, _ in problem.objectives]
    n_objs = len(obj_keys)
    if n_objs == 1:
        return th.as_tensor(dataset[obj_keys[0]][:]).float().unsqueeze(-1)

    # Multi-objective: return full vector
    obj_tensors = [th.as_tensor(dataset[k][:]).float() for k in obj_keys]
    print(f"Multi-objective performance target: {obj_keys}")
    return th.stack(obj_tensors, dim=-1)


def normalize(ds: Dataset, condition_names: list[str]) -> tuple[Dataset, th.Tensor, th.Tensor]:
    """Normalize specified condition columns with global mean/std."""
    # stack condition columns into a single tensor (N, C) on CPU
    conds = th.stack([th.as_tensor(ds[c][:]).float() for c in condition_names], dim=1)
    mean = conds.mean(dim=0)
    std = conds.std(dim=0).clamp(min=1e-8)

    # normalize each condition column (HF expects numpy back)
    ds = ds.map(
        lambda batch: {
            c: ((th.as_tensor(batch[c][:]).float() - mean[i]) / std[i]).numpy() for i, c in enumerate(condition_names)
        },
        batched=True,
    )

    return ds, mean, std


def drop_constant(ds: Dataset, condition_names: list[str]) -> tuple[Dataset, list[str]]:
    """Drop constant condition columns (std=0) from dataset."""
    conds = th.stack([th.as_tensor(ds[c][:]).float() for c in condition_names], dim=1)
    std = conds.std(dim=0)

    kept = [c for i, c in enumerate(condition_names) if std[i] > 0]
    dropped = [c for i, c in enumerate(condition_names) if std[i] == 0]

    if dropped:
        print(f"Warning: Dropping constant condition columns (std=0): {dropped}")

    # remove dropped columns from dataset
    ds = ds.remove_columns(dropped)

    return ds, kept
