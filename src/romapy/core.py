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
        if self.center == "standard":
            row_means = submatrix.mean(axis=1)
            X = submatrix.sub(row_means, axis="index")
        else:  # "fixed"
            X = submatrix.sub(global_center, axis="columns")

        pca = PCA(n_components=2, random_state=self.random_state)
        pca.fit(X.T)                          # samples as rows, genes as columns
        scores = pca.transform(X.T)[:, 0]     # PC1 projection only
        l1, l2 = pca.explained_variance_ratio_

        return {
            "scores": pd.Series(scores, index=submatrix.columns),
            "l1": float(l1),
            "l2": float(l2),
            "gene_weights": pd.Series(pca.components_[0], index=submatrix.index),
        }

    def _orient_pc1(self, pc1_scores: pd.Series, submatrix: pd.DataFrame) -> pd.Series:
        mean_expr = submatrix.mean(axis=0)
        correlation = np.corrcoef(pc1_scores, mean_expr)[0, 1]

        if correlation < 0:
            return -pc1_scores
        return pc1_scores

    def _trim_outliers(self, submatrix: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        samples = submatrix.columns.tolist()

        leave_one_out_l1 = []
        for sample in samples:
            reduced = submatrix.drop(columns=sample)
            row_means = reduced.mean(axis=1)
            X = reduced.sub(row_means, axis="index")
            pca = PCA(n_components=1, random_state=self.random_state)
            pca.fit(X.T)
            leave_one_out_l1.append(pca.explained_variance_ratio_[0])

        leave_one_out_l1 = np.array(leave_one_out_l1)
        std = leave_one_out_l1.std()
        if std == 0:
            return submatrix, []

        z_scores = (leave_one_out_l1 - leave_one_out_l1.mean()) / std
        dropped = [samples[i] for i in range(len(samples)) if abs(z_scores[i]) > self.z_max]
        kept = [s for s in samples if s not in dropped]

        return submatrix[kept], dropped

    def _null_distribution(self, expression: pd.DataFrame, module_size: int, global_center: pd.Series | None) -> np.ndarray:
        # TODO: pval checking gene sets with randomised inputs, if sig better proceed
        

        raise NotImplementedError("<func> _null_distribution")