# NOTE: Decorators & Types
from dataclasses import dataclass
from typing import Literal

# NOTE: Engine
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

# NOTE: Cross-Files
from romapy.results import resultROMA

"""
    DESC: core file for romapy classes, ROMA is the baseline.
"""

@dataclass
class ROMA:
    center: Literal["fixed", "standard"] = "fixed"
    robust: bool = True
    z_max: float = 3.0
    n_permutations: int = 1000
    random_state: int | None = None

    def fit(self, expression: pd.DataFrame, gene_sets: dict[str, list[str]], min_genes: int = 5) -> resultROMA:
        # TODO: this function does the main calculation, in the end


        raise NotImplementedError("<func> fit")

    def _compute_module(self, submatrix: pd.DataFrame, global_center: pd.Series | None) -> dict[str, pd.Series | float]:
        # TODO: core PCA calc, offer fixed (global mean) or standard (row mean)
        

        raise NotImplementedError("<func> _compute_module")

    def _orient_pc1(self, pc1_scores: pd.Series, submatrix: pd.DataFrame) -> pd.Series:
        # TODO: orients the sign of pca based on gene avg correlation
        

        raise NotImplementedError("<func> _orient_pc1")

    def _trim_outliers(self, submatrix: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        # TODO: trimming outliers before computation
        

        raise NotImplementedError("<func> _trim_outliers")

    def _null_distribution(self, expression: pd.DataFrame, module_size: int, global_center: pd.Series | None) -> np.ndarray:
        # TODO: pval checking gene sets with randomised inputs, if sig better proceed
        

        raise NotImplementedError("<func> _null_distribution")