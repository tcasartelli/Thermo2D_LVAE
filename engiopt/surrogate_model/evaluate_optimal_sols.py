"""Evaluate the optimal designs found by the surrogate model on the simulator."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Literal

from engibench.problems.power_electronics import PowerElectronics
from hyppo.ksample import MMD
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from tqdm import tqdm
import tyro

pio.kaleido.scope.mathjax = None
DVARS = [f"x{i}" for i in range(10)]

# constant terms for simulation
CONST_DESIGN_TERMS = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0])


@dataclass
class Args:
    results_dir: str
    """Directory to save the results, e.g. "results_power_electronics__run_pe_optimization__0__1746086123"."""
    seed: int
    """Random seed for the simulation, must match the seed used to train the models."""
    pareto_path: str = "pareto_both.csv"
    """Path to the Pareto CSV file inside the results directory, e.g. "pareto_both.csv"."""
    pareto_with_sim_path: str = "pareto_both_with_sim.csv"
    """Path to the Pareto with simulation results CSV file inside the results directory, e.g. "pareto_both_with_sim.csv"."""
    mode: Literal["simulate", "plot"] = "simulate"


def simulate_objectives(i, x_row) -> tuple[float, float, float]:
    """Simulate each design (with error-catching) and track failures.

    Return:
        - (|DcGain-0.25|, Voltage_Ripple, violated_ratio) on success.
        - (nan, nan, violated_ratio) on failure.
    """
    problem = PowerElectronics()
    problem.reset(seed=args.seed)

    design = np.hstack([x_row.to_numpy(), CONST_DESIGN_TERMS], dtype=problem.design_space.dtype)
    constraints_violation = problem.check_constraints(design, {})
    if len(constraints_violation) > 0:
        violated_ratio = len(constraints_violation) / constraints_violation.n_constraints
    else:
        violated_ratio = 0.0
    try:
        sim = problem.simulate(design)
        if constraints_violation:
            print(f"Design {i} violated {len(constraints_violation)} of {constraints_violation.n_constraints} constraints.")
        return np.abs(sim[0] - 0.25), sim[1], violated_ratio
    except Exception:  # noqa: BLE001
        print(
            f"Design {i} violated {len(constraints_violation)} of {constraints_violation.n_constraints} constraints, failed to simulate."
        )
        return np.nan, np.nan, violated_ratio


def simulate_all(args: Args):
    """Simulate all designs in the Pareto front and save the results."""
    # Load your surrogate-computed Pareto front (has x0…x9, f0, f1)
    df_front = pd.read_csv(os.path.join(args.results_dir, args.pareto_path))
    df_X = df_front[DVARS]  # noqa: N806

    f0_sims, f1_sims, violated_ratios = [], [], []
    # use tqdm to show progress
    for i, row in tqdm(df_X.iterrows(), total=len(df_X)):
        f0, f1, violated_ratio = simulate_objectives(i, row)
        f0_sims.append(f0)
        f1_sims.append(f1)
        violated_ratios.append(violated_ratio)

    df_front["f0_sim"] = f0_sims
    df_front["f1_sim"] = f1_sims
    df_front["violated_ratio"] = violated_ratios
    # Identify failed rows
    failed_mask = df_front[["f0_sim", "f1_sim"]].isna().any(axis=1)

    # Save the failed designs for inspection
    df_front.loc[failed_mask, [*DVARS, "f0", "f1", "violated_ratio"]].to_csv(
        f"{args.results_dir}/pareto_failures.csv", index=False
    )
    # Save enriched Pareto front (with sim results and failure flags)
    df_front.to_csv(f"{args.results_dir}/{args.pareto_with_sim_path}", index=False)


def plot_pareto_front(args: Args):
    """Plot the Pareto front and compare it to the simulated Pareto front."""
    # Load results and filter out failed designs
    df_front = pd.read_csv(os.path.join(args.results_dir, args.pareto_with_sim_path))
    failed_mask = df_front[["f0_sim", "f1_sim"]].isna().any(axis=1)
    df_valid = df_front.loc[~failed_mask].reset_index(drop=True)

    # Prepare data for plotting
    df_pred = (
        df_valid.rename(columns={"f0": "r", "f1": "abs_g"}).assign(kind="Surrogate").loc[:, [*DVARS, "r", "abs_g", "kind"]]
    )
    df_sim = (
        df_valid.rename(columns={"f0_sim": "abs_g", "f1_sim": "r"})
        .assign(kind="Simulated")
        .loc[:, [*DVARS, "r", "abs_g", "kind"]]
    )
    df_both = pd.concat([df_pred, df_sim], ignore_index=True)

    # Failed rows
    failed_mask = df_front["violated_ratio"] > 0.0
    n_failed = failed_mask.sum()
    failed_ratio = n_failed / len(df_front)

    # Violated rows
    violated_mask = df_front["violated_ratio"] > 0.0
    n_violated = violated_mask.sum()
    violated_ratio = n_violated / len(df_front)

    blue = "#97cfff"
    red = "#e93561"
    fig = px.scatter(
        df_both,
        x="r",
        y="abs_g",
        color="kind",
        color_discrete_map={"Surrogate": blue, "Simulated": red},
        hover_data=DVARS,
        title="",
        labels={"r": "|DcGain - 0.25|", "abs_g": "Voltage ripple"},
    )

    if violated_ratio > 0.0 or failed_ratio > 0.0:
        fig.add_annotation(
            text=f"Simulation failure ratio: {failed_ratio:.1%} ({n_failed} simulations failed).\n Violated ratio: {violated_ratio:.1%} ({n_violated} designs violated at least one constraint).",
            xref="paper",
            yref="paper",
            x=0.5,
            y=1.05,
            showarrow=False,
        )

    fig.update_traces(marker={"size": 9, "opacity": 0.8})
    fig.update_layout(
        legend={"x": 0.75, "y": 0.98, "title": "Method"},
        hovermode="closest",
        font={"size": 18},
        xaxis={"title": {"font": {"size": 18}}},
        yaxis={"title": {"font": {"size": 18}}},
        legend_title={"font": {"size": 18}},
        legend_font={"size": 18},
    )
    fig.write_image(os.path.join(args.results_dir, f"pareto_front_{args.seed}.pdf"), engine="kaleido")

    # Two-sample MMD² test on valid designs only
    X = df_pred[["r", "abs_g"]].to_numpy()  # noqa: N806
    Y = df_sim[["r", "abs_g"]].to_numpy()  # noqa: N806

    mmd = MMD(compute_kernel="rbf", bias=False)
    stat, p_val = mmd.test(X, Y, reps=1000, auto=False)

    print(f"\nMMD² = {stat:.4e}, permutation p-value = {p_val:.3f}")
    if p_val < 0.05:  # noqa: PLR2004
        print("→ Clouds differ significantly (reject H₀ at α=0.05)")  # noqa: RUF001
    else:
        print("→ No significant difference detected (fail to reject H₀)")


if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.mode == "simulate":
        simulate_all(args)
    elif args.mode == "plot":
        plot_pareto_front(args)
    else:
        raise ValueError(f"Invalid mode: {args.mode}, must be one of 'simulate' or 'plot'.")
