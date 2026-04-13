"""This module provides metrics for evaluating generative model designs.

Maximum Mean Discrepancy (MMD), Determinantal Point Process (DPP) diversity,
and optimality gap calculations.
"""

from __future__ import annotations

import multiprocessing
import os
import sys
import traceback
from typing import Any, TYPE_CHECKING

from gymnasium import spaces
import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import cdist

if TYPE_CHECKING:
    from datasets import Dataset
    from engibench import OptiStep
    from engibench.core import Problem


def _numpy_config(config: dict[str, Any] | None) -> dict[str, Any] | None:
    """Convert list-valued condition entries to numpy arrays.

    HuggingFace datasets return image-shaped conditions (e.g. 65x65 boundary
    matrices in thermoelastic2d) as nested Python lists.  EngiBench's
    ``optimize`` / ``simulate`` expect numpy arrays, so this helper converts
    them in-place.
    """
    if config is None:
        return None
    return {k: np.asarray(v) if isinstance(v, list) else v for k, v in config.items()}


def _scalar_obj(obj_values: Any, weights: npt.NDArray | None = None) -> float:
    """Reduce a (possibly multi-objective) result to a single scalar.

    For single-objective problems ``obj_values`` is already a scalar or a
    length-1 array.  For multi-objective problems (e.g. thermoelastic2d
    which returns ``[structural, thermal, volume_fraction]``) we compute a
    weighted sum when *weights* is provided, otherwise fall back to the
    first objective.

    Args:
        obj_values: Scalar or array of objective values.
        weights: Optional weight vector (same length as obj_values).  When
            given, the scalar is ``weights @ obj_values``.
    """
    arr = np.asarray(obj_values, dtype=float)
    if arr.ndim == 0:
        return float(arr)
    if arr.size == 1:
        return float(arr.flat[0])
    if weights is not None:
        return float(np.dot(weights[:len(arr)], arr))
    return float(arr.flat[0])


if sys.platform != "win32":  #  only set fork on non-Windows
    multiprocessing.set_start_method("fork", force=True)
else:
    multiprocessing.set_start_method("spawn", force=True)


def mmd(x: np.ndarray, y: np.ndarray, sigma: float | None = 1.0) -> float:
    """Compute the Maximum Mean Discrepancy (MMD) between two sets of samples.

    Args:
        x: Array of shape (n, ...) for generated designs (or latent codes).
        y: Array of shape (m, ...) for dataset/reference designs (or latent codes).
        sigma: Bandwidth parameter for the Gaussian kernel. If None, uses median heuristic
            on the reference data y only (so sigma is consistent across different generators).

    Returns:
        float: The MMD value.
    """
    x_flat = x.reshape(x.shape[0], -1)
    y_flat = y.reshape(y.shape[0], -1)

    if sigma is None:
        sigma = compute_median_sigma(y_flat)

    k_xx = np.exp(-cdist(x_flat, x_flat, "sqeuclidean") / (2 * sigma**2))
    k_yy = np.exp(-cdist(y_flat, y_flat, "sqeuclidean") / (2 * sigma**2))
    k_xy = np.exp(-cdist(x_flat, y_flat, "sqeuclidean") / (2 * sigma**2))

    return float(k_xx.mean() + k_yy.mean() - 2 * k_xy.mean())


def dpp_diversity(x: np.ndarray, sigma: float | None = None) -> float:
    """Compute the Determinantal Point Process (DPP) diversity for a set of samples.

    Uses the log-determinant for numerical stability (raw determinant underflows
    for even moderate n when kernel entries are close to 1).

    Note: sigma defaults to None (median heuristic on x itself), NOT the reference
    data sigma. DPP measures within-set diversity, so the bandwidth should be
    calibrated to the pairwise distances of the samples being evaluated.

    Args:
        x: Array of shape (n, ...) for generated designs (or latent codes).
        sigma: Bandwidth parameter for the Gaussian kernel. If None, uses median
            heuristic on x (recommended for DPP).

    Returns:
        float: The log-determinant of the DPP kernel matrix (higher = more diverse).
    """
    x_flat = x.reshape(x.shape[0], -1)

    if sigma is None:
        sigma = compute_median_sigma(x_flat)

    pairwise_sq_dists = cdist(x_flat, x_flat, "sqeuclidean")
    similarity_matrix = np.exp(-pairwise_sq_dists / (2 * sigma**2))

    # Regularize the matrix slightly to avoid numerical issues
    reg_matrix = similarity_matrix + 1e-6 * np.eye(x.shape[0])

    try:
        sign, logdet = np.linalg.slogdet(reg_matrix)
        if sign <= 0:
            return float("-inf")
        return float(logdet)
    except np.linalg.LinAlgError:
        return float("-inf")


def compute_median_sigma(x: np.ndarray, y: np.ndarray | None = None) -> float:
    """Compute sigma using median heuristic on pairwise distances.

    Args:
        x: First set of samples of shape (n, d).
        y: Second set of samples (optional, uses x if None).

    Returns:
        Computed sigma value.
    """
    x = x.reshape(x.shape[0], -1)
    if y is not None:
        y = y.reshape(y.shape[0], -1)

    n_sample = min(500, len(x))
    rng = np.random.default_rng(42)
    idx_x = rng.choice(len(x), n_sample, replace=len(x) < n_sample)

    if y is not None:
        idx_y = rng.choice(len(y), n_sample, replace=len(y) < n_sample)
        dists = cdist(x[idx_x], y[idx_y], "sqeuclidean")
    else:
        dists = cdist(x[idx_x], x[idx_x], "sqeuclidean")
        # Use upper triangle (exclude diagonal)
        dists = dists[np.triu_indices_from(dists, k=1)]

    sigma = np.sqrt(np.median(dists) / 2) if len(dists) > 0 else 1.0
    return max(sigma, 1e-6)


def conditional_mmd(
    x: np.ndarray,
    y: np.ndarray,
    conditions: np.ndarray,
    n_bins: int = 5,
    sigma: float | None = None,
) -> dict[str, float]:
    """Compute MMD per condition bin, then average.

    For each condition dimension, samples are grouped into quantile-based bins.
    MMD is computed within each bin between the generated and dataset samples
    that share that bin. The per-bin MMDs are averaged (weighted by bin size)
    to produce a single conditional MMD value per condition dimension, plus
    an overall average across all condition dimensions.

    Args:
        x: Generated designs/latent codes, shape (n, d).
        y: Dataset designs/latent codes, shape (n, d). Must be same length as x
            (paired with the same sampled conditions).
        conditions: Condition values, shape (n, n_conds). Each column is one
            condition dimension.
        n_bins: Number of quantile bins per condition dimension. Default 5.
        sigma: Bandwidth for Gaussian kernel. If None, uses median heuristic
            on the full y set (shared across bins for consistency).

    Returns:
        Dictionary with keys:
            - "cond_mmd": Overall conditional MMD (average across condition dims).
            - "cond_mmd_{name}": Per-condition-dimension conditional MMD, where
              name is "dim0", "dim1", etc.
            - "cond_mmd_sigma": The sigma used.
    """
    x_flat = x.reshape(x.shape[0], -1)
    y_flat = y.reshape(y.shape[0], -1)
    conditions = np.atleast_2d(conditions)
    if conditions.shape[0] != x_flat.shape[0]:
        conditions = conditions.T
    n_samples, n_conds = conditions.shape

    # Resolve sigma once from full reference set
    if sigma is None:
        sigma = compute_median_sigma(y_flat)

    per_cond_mmds: list[float] = []
    result: dict[str, float] = {"cond_mmd_sigma": sigma}

    for c in range(n_conds):
        cond_vals = conditions[:, c]
        # Quantile-based bin edges
        bin_edges = np.quantile(cond_vals, np.linspace(0, 1, n_bins + 1))
        bin_indices = np.digitize(cond_vals, bin_edges[1:-1])  # 0..n_bins-1

        bin_mmds: list[float] = []
        bin_sizes: list[int] = []
        for b in range(n_bins):
            mask = bin_indices == b
            n_in_bin = int(mask.sum())
            if n_in_bin < 2:  # noqa: PLR2004
                continue
            bin_mmds.append(mmd(x_flat[mask], y_flat[mask], sigma=sigma))
            bin_sizes.append(n_in_bin)

        if bin_sizes:
            weights = np.array(bin_sizes, dtype=float)
            weights /= weights.sum()
            cond_mmd_val = float(np.average(bin_mmds, weights=weights))
        else:
            cond_mmd_val = float("nan")

        result[f"cond_mmd_dim{c}"] = cond_mmd_val
        per_cond_mmds.append(cond_mmd_val)

    valid = [v for v in per_cond_mmds if not np.isnan(v)]
    result["cond_mmd"] = float(np.mean(valid)) if valid else float("nan")
    return result


def optimality_gap(
    opt_history: list[OptiStep], baseline: float, weights: npt.NDArray | None = None
) -> list[float]:
    """Compute the optimality gap of an optimization history.

    Args:
        opt_history (list[OptiStep]): The optimization history.
        baseline (float): The baseline value to compare against.
        weights: Optional objective weights for multi-objective problems.

    Returns:
        list[float]: The optimality gap at each step in opt_history.
    """
    return [_scalar_obj(opt.obj_values, weights) - baseline for opt in opt_history]


def simulate_failure_ratio(  # noqa: C901
    problem: Problem,
    gen_designs: npt.NDArray,
    sampled_conditions: Dataset | None = None,
) -> float:
    """Compute the failure ratio of generated designs. This is designed for the airfoil problem.

    Args:
        problem: The optimization problem to evaluate.
        gen_designs (np.ndarray): Array of shape (n_samples, l, w) for generative model designs.
        sampled_conditions (Dataset): Dataset of sampled conditions for optimization. If None, no conditions are used.

    Returns:
        float: The failure ratio of generated designs.
    """
    failure_count = 0
    for idx, design in enumerate(gen_designs):
        if isinstance(problem.design_space, spaces.Dict):
            # Need to unflatten the design to be used for optimization or simulation
            unflattened_design = spaces.unflatten(problem.design_space, design)
            unflattened_design["angle_of_attack"] = unflattened_design["angle_of_attack"][0]
        else:
            unflattened_design = design

        def worker(idx, config, return_queue):
            try:
                problem.reset(seed=42)  # Reset the problem state before simulating
                objs = problem.simulate(unflattened_design, config=config, mpicores=10)  # noqa: B023
                if np.isnan(objs[0]) or np.isnan(objs[1]):
                    print(f"Simulation returned NaN values for design {idx}")
                    raise Exception("Simulation returned NaN values")  # noqa: TRY002, TRY301
                return_queue.put(("ok", objs))
            except Exception:  # noqa: BLE001
                return_queue.put(("error", traceback.format_exc()))

        # Attempt to simulate the design
        def run_with_timeout(idx, timeout=30):
            config = sampled_conditions[idx] if sampled_conditions is not None else None
            q = multiprocessing.Queue()
            p = multiprocessing.Process(target=worker, args=(idx, config, q))
            p.start()
            p.join(timeout)

            if p.is_alive():
                p.terminate()  # force-kill the child process
                p.join()
                os.system("docker stop machaero")  # noqa: S605
                raise RuntimeError(f"Simulation timed out for design {idx}")

            if not q.empty():
                status, payload = q.get()
                if status == "ok":
                    print(f"Simulation successful for design {idx}: {payload}")
                else:
                    raise RuntimeError(f"Simulation error for design {idx}:\n{payload}")
            else:
                raise RuntimeError("Simulation process ended without reporting back")

        try:
            run_with_timeout(idx, timeout=30)
        except RuntimeError as e:
            failure_count += 1
            print(e)

    return failure_count / len(gen_designs)  # Return the failure ratio


def metrics(  # noqa: PLR0915
    problem: Problem,
    gen_designs: npt.NDArray,
    dataset_designs: npt.NDArray,
    sampled_conditions: Dataset | None = None,
    sigma: float | None = None,
) -> dict[str, Any]:
    """Compute various metrics for evaluating generative model designs.

    Args:
        problem: The optimization problem to evaluate.
        gen_designs (np.ndarray): Array of shape (n_samples, l, w) for generative model designs (potentially flattened for dict spaces).
        dataset_designs (np.ndarray): Array of shape (n_samples, l, w) for dataset designs (these are not flattened).
        sampled_conditions (Dataset): Dataset of sampled conditions for optimization. If None, no conditions are used.
        sigma: Bandwidth parameter for the Gaussian kernel (in mmd and dpp calculation).
            If None, uses median heuristic on the reference (dataset) designs.

    Returns:
        dict[str, Any]: A dictionary containing the computed metrics:
            - "iog": Average Initial Optimality Gap (float).
            - "cog": Average Cumulative Optimality Gap (float).
            - "fog": Average Final Optimality Gap (float).
            - "iog_median": Median Initial Optimality Gap (float).
            - "cog_median": Median Cumulative Optimality Gap (float).
            - "fog_median": Median Final Optimality Gap (float).
            - "iog_iqr": IQR of Initial Optimality Gap (float).
            - "cog_iqr": IQR of Cumulative Optimality Gap (float).
            - "fog_iqr": IQR of Final Optimality Gap (float).
            - "iog_var": Variance of Initial Optimality Gap (float).
            - "cog_var": Variance of Cumulative Optimality Gap (float).
            - "fog_var": Variance of Final Optimality Gap (float).
            - "iog_list": Per-sample Initial Optimality Gaps (list[float]).
            - "cog_list": Per-sample Cumulative Optimality Gaps (list[float]).
            - "fog_list": Per-sample Final Optimality Gaps (list[float]).
            - "viol_list": Per-sample constraint violations (list[bool]).
            - "mmd": Maximum Mean Discrepancy (float).
            - "dpp": Determinantal Point Process diversity (float).
            - "mmd_sigma": The actual sigma used for MMD/DPP (float).
    """
    n_samples = len(gen_designs)

    # Build per-sample objective weights for multi-objective problems.
    # For thermoelastic2d "weight" trades off structural vs thermal compliance.
    n_objs = len(problem.objectives_keys)

    cog_list = []
    iog_list = []
    fog_list = []
    viol_list = []
    for i in range(n_samples):
        conditions = _numpy_config(sampled_conditions[i]) if sampled_conditions is not None else None
        if isinstance(problem.design_space, spaces.Dict):
            # Need to unflatten the design to be used for optimization or simulation
            unflattened_design = spaces.unflatten(problem.design_space, gen_designs[i])
        else:
            unflattened_design = gen_designs[i]

        # Per-sample objective weights (only relevant for multi-objective problems)
        obj_weights: npt.NDArray | None = None
        if n_objs > 1 and conditions is not None and "weight" in conditions:
            w = float(conditions["weight"])
            # [structural, thermal, volume_fraction] -> weight on first two, 0 on rest
            obj_weights = np.zeros(n_objs)
            obj_weights[0] = w
            obj_weights[1] = 1.0 - w

        problem.reset(seed=42)  # Reset the problem state before optimization/simulation
        _, opt_history = problem.optimize(unflattened_design, config=conditions)

        problem.reset(seed=42)  # Reset again before simulating the reference optimum
        reference_optimum = _scalar_obj(problem.simulate(dataset_designs[i], config=conditions), obj_weights)
        opt_history_gaps = optimality_gap(opt_history, reference_optimum, obj_weights)

        problem.reset(seed=42)  # Reset again before simulating the optimized design
        iog_list.append(
            _scalar_obj(problem.simulate(unflattened_design, config=conditions), obj_weights) - reference_optimum
        )
        cog_list.append(np.sum(opt_history_gaps))
        fog_list.append(opt_history_gaps[-1])

        # Check if conditions dict has 'volfrac' or 'volume' key and compare with design mean
        if conditions:
            tol = 0.01  # Tolerance for equality constraint deviation
            target_vol = conditions.get("volfrac") or conditions.get("volume")
            if target_vol is not None:
                viol = np.abs(np.mean(unflattened_design) - target_vol) >= tol
                viol_list.append(viol)

    # Compute aggregated statistics for IOG, COG, FOG
    iog_arr = np.array(iog_list)
    cog_arr = np.array(cog_list)
    fog_arr = np.array(fog_list)

    average_iog: float = float(np.mean(iog_arr))
    average_cog: float = float(np.mean(cog_arr))
    average_fog: float = float(np.mean(fog_arr))
    average_viol: float = float(np.mean(viol_list))

    median_iog: float = float(np.median(iog_arr))
    median_cog: float = float(np.median(cog_arr))
    median_fog: float = float(np.median(fog_arr))

    iqr_iog: float = float(np.subtract(*np.percentile(iog_arr, [75, 25])))
    iqr_cog: float = float(np.subtract(*np.percentile(cog_arr, [75, 25])))
    iqr_fog: float = float(np.subtract(*np.percentile(fog_arr, [75, 25])))

    var_iog: float = float(np.var(iog_arr))
    var_cog: float = float(np.var(cog_arr))
    var_fog: float = float(np.var(fog_arr))

    # Compute the Maximum Mean Discrepancy (MMD) between generated and dataset designs
    # We compute the MMD on the flattened designs
    flattened_ds_designs: list[npt.NDArray] = []
    for design in dataset_designs:
        if isinstance(problem.design_space, spaces.Dict):
            flattened = spaces.flatten(problem.design_space, design)
            flattened_ds_designs.append(np.array(flattened))
        else:
            flattened_ds_designs.append(design)
    flattened_ds_designs_array: npt.NDArray = np.array(flattened_ds_designs)

    # Resolve sigma: if None, compute median heuristic from reference (dataset) designs
    if sigma is None:
        sigma = compute_median_sigma(flattened_ds_designs_array)

    mmd_value: float = mmd(gen_designs, flattened_ds_designs_array, sigma=sigma)

    # Compute the Determinantal Point Process (DPP) diversity for generated designs
    # Use the same reference sigma as MMD so DPP is comparable across models
    dpp_value: float = dpp_diversity(gen_designs, sigma=sigma)

    result: dict[str, Any] = {
        "iog": average_iog,
        "cog": average_cog,
        "fog": average_fog,
        "iog_median": median_iog,
        "cog_median": median_cog,
        "fog_median": median_fog,
        "iog_iqr": iqr_iog,
        "cog_iqr": iqr_cog,
        "fog_iqr": iqr_fog,
        "iog_var": var_iog,
        "cog_var": var_cog,
        "fog_var": var_fog,
        "iog_list": iog_list,
        "cog_list": cog_list,
        "fog_list": fog_list,
        "viol_list": viol_list,
        "mmd": mmd_value,
        "dpp": dpp_value,
        "mmd_sigma": sigma,
        "viol": average_viol,
    }

    return result
