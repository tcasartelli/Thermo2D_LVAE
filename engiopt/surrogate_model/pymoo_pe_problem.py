"""This module defines the PymooPowerElecProblem class for multi-objective optimization using pymoo."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from engibench.problems.power_electronics.v0 import PowerElectronics
import numpy as np
import pandas as pd
from pymoo.core.problem import ElementwiseProblem

if TYPE_CHECKING:
    import torch

    from engiopt.surrogate_model.model_pipeline import ModelPipeline


class PymooPowerElecProblem(ElementwiseProblem):
    def __init__(self, pipeline_r: ModelPipeline, pipeline_g: ModelPipeline, device: torch.device | None = None) -> None:
        """Initialize the PymooPowerElecProblem.

        Args:
            pipeline_r: A ModelPipeline instance for predicting r.
            pipeline_g: A ModelPipeline instance for predicting g.
            device: The torch device to use for predictions. Defaults to None (will use CPU).
        """
        # Define the design space:
        # C1, C2, C3, C4, C5, C6, L1, L2, L3, T1
        problem = PowerElectronics()
        xl = np.array(problem.design_space.low[:10])
        xu = np.array(problem.design_space.high[:10])
        # Two objectives:
        #  f1 = predicted r (minimize)
        #  f2 = |predicted g - 0.25| (minimize)
        super().__init__(n_var=10, n_obj=2, n_ieq_constr=0, xl=xl, xu=xu)
        self.pipeline_r = pipeline_r
        self.pipeline_g = pipeline_g
        self.device = device

    def _evaluate(self, x: np.ndarray, out: dict[str, Any], *_args: object, **_kwargs: object) -> None:
        """Evaluate a candidate solution.

        Args:
            x: A numpy array containing one candidate solution.
            out: A dict to store the objective values.
            _args: Unused positional arguments.
            _kwargs: Unused keyword arguments.
        """
        # Format the input data as expected by the pipeline models
        # initial_design should be a list of [C1, C2, C3, C4, C5, C6, L1, L2, L3]
        initial_design = [x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9]]

        # Build a one-row DataFrame with the expected format
        df = pd.DataFrame({"initial_design": [initial_design]})

        # Get predictions using your trained pipelines.
        # (Assuming your pipelines' predict method returns a numpy array.)
        r_pred = self.pipeline_r.predict(df, batch_size=np.min([len(df), 1024]), device=self.device)
        g_pred = self.pipeline_g.predict(df, batch_size=np.min([len(df), 1024]), device=self.device)

        # Objectives:
        f1 = r_pred
        f2 = abs(g_pred - 0.25)

        out["F"] = np.column_stack([f1, f2])
