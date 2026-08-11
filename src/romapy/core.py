# NOTE: Decorators & Types
from dataclasses import dataclass
from typing import Literal

# NOTE: Engine
import pandas as pd
import numpy as np

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
        global_center = expression.mean(axis=0) if self.center == "fixed" else None

        scores_dict, l1_dict, l1_pval_dict, coord_pval_dict = {}, {}, {}, {}
        gene_weights_dict, dropped_dict = {}, {}

        for name, genes in gene_sets.items():
            genes_present = [g for g in genes if g in expression.index]
            if len(genes_present) < min_genes:
                continue

            submatrix = expression.loc[genes_present]

            dropped = []
            if self.robust:
                submatrix, dropped = self._trim_outliers(submatrix)

            result = self._compute_module(submatrix, global_center)
            oriented_scores = self._orient_pc1(result["scores"], submatrix)

            null = self._null_distribution(expression, len(genes_present), global_center)
            real_l1 = result["l1"]
            real_gap = result["l1"] - result["l2"]

            l1_pval = (null[:, 0] >= real_l1).sum() / self.n_permutations
            coord_pval = (null[:, 1] >= real_gap).sum() / self.n_permutations

            scores_dict[name] = oriented_scores
            l1_dict[name] = real_l1
            l1_pval_dict[name] = l1_pval
            coord_pval_dict[name] = coord_pval
            gene_weights_dict[name] = result["gene_weights"]
            dropped_dict[name] = dropped

        return resultROMA(
            scores=pd.DataFrame(scores_dict).T,   # modules x samples
            l1=pd.Series(l1_dict),
            l1_pval=pd.Series(l1_pval_dict),
            coord_pval=pd.Series(coord_pval_dict),
            gene_weights=gene_weights_dict,
            dropped_samples=dropped_dict,
        )

    def _compute_module(self, submatrix: pd.DataFrame, global_center: pd.Series | None) -> dict[str, pd.Series | float]:
        if self.center == "fixed" and global_center is None:
            raise ValueError("global_center is required when center='fixed'")

        if self.center == "standard":
            row_means = submatrix.mean(axis=1)
            X = submatrix.sub(row_means, axis="index")
        else:  # "fixed"
            aligned_center = global_center.loc[submatrix.columns]
            X = submatrix.sub(aligned_center, axis="columns")
            
        scores, var = self._pca_fast(X.T, 2)
        l1, l2 = var

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
            scores, var = self._pca_fast(X.T, 2)
            l1, l2 = var
            leave_one_out_l1.append(l1)

        leave_one_out_l1 = np.array(leave_one_out_l1)
        std = leave_one_out_l1.std()
        if std == 0:
            return submatrix, []

        z_scores = (leave_one_out_l1 - leave_one_out_l1.mean()) / std
        dropped = [samples[i] for i in range(len(samples)) if abs(z_scores[i]) > self.z_max]
        kept = [s for s in samples if s not in dropped]

        return submatrix[kept], dropped

    def _null_distribution(self, expression: pd.DataFrame, module_size: int, global_center: pd.Series | None) -> np.ndarray:
        rng = np.random.default_rng(self.random_state)
        all_genes = expression.index.to_numpy()

        null_l1 = np.empty(self.n_permutations)
        null_gap = np.empty(self.n_permutations)

        for i in range(self.n_permutations):
            random_genes = rng.choice(all_genes, size=module_size, replace=False)
            random_submatrix = expression.loc[random_genes]
            result = self._compute_module(random_submatrix, global_center)
            null_l1[i] = result["l1"]
            null_gap[i] = result["l1"] - result["l2"]

        return np.column_stack([null_l1, null_gap])

    def _pca_fast(X_centered: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, np.ndarray]:
        """
        X_centered: samples x genes, already centered.
        Returns (scores, explained_variance_ratio) for the first n_components.
        """
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        explained_variance = (S ** 2) / (X_centered.shape[0] - 1)
        explained_variance_ratio = explained_variance / explained_variance.sum()
        scores = U[:, :n_components] * S[:n_components]
        return scores[:, :n_components], explained_variance_ratio[:n_components]

