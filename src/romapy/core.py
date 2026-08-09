from dataclasses import dataclass, field
import pandas as pd
from typing import Literal
from romapy.results import resultROMA
from sklearn.decomposition import PCA

# NOTE: Core file for romapy classes, ROMA is the baseline.

@dataclass
class ROMA:
    center: Literal["fixed", "standard"] = "fixed"
    robust: bool = True
    z_max: float = 3.0
    n_permutations: int = 1000
    random_state: int | None = None

    def fit(self, expression, gene_sets, min_genes=5) -> resultROMA:
        ...  # orchestrates everything below, loops over gene_sets

    def _compute_module(self, submatrix: pd.DataFrame, global_center: Literal["fixed", "standard"] = "fixed") -> dict:
        ... # type

    def _orient_pc1(self, pc1_scores, submatrix) -> pc1_scores:
        ...  # sign correction, what we just discussed

    def _trim_outliers(self, submatrix) -> (submatrix, dropped_names):
        ...  # leave-one-out robust trimming

    def _null_distribution(self, expression, module_size, global_center) -> array:
        ...  # permutation testing