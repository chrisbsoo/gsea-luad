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
        null_cache: dict[int, np.ndarray] = {}   # <-- new

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

            size = len(genes_present)
            if size not in null_cache:
                null_cache[size] = self._null_distribution(expression, size, global_center)
            null = null_cache[size]

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
            scores=pd.DataFrame(scores_dict).T,
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
            
        scores, var = self._pca_fast(X.T.to_numpy(), 2)
        l1, l2 = var
        scores = scores[:, 0]

        return {
            "scores": pd.Series(scores, index=submatrix.columns),
            "l1": float(l1),
            "l2": float(l2),
            "gene_weights": pd.Series(l1, index=submatrix.index),
        }

    def _orient_pc1(self, pc1_scores: pd.Series, submatrix: pd.DataFrame) -> pd.Series:
        mean_expr = submatrix.mean(axis=0)
        correlation = np.corrcoef(pc1_scores, mean_expr)[0, 1]

        if correlation < 0:
            return -pc1_scores
        return pc1_scores

    def _trim_outliers(self, submatrix: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        samples = submatrix.columns.tolist()
        n_samples = len(samples)
        X = submatrix.to_numpy()  # genes x samples

        # build all n_samples leave-one-out folds as one 3D batch:
        # batch[i] = X with column i removed
        batch = np.stack([
            np.delete(X, i, axis=1) for i in range(n_samples)
        ])  # shape (n_samples, n_genes, n_samples - 1)

        row_means = batch.mean(axis=2, keepdims=True)
        batch_centered = batch - row_means

        gram_batch = np.einsum("bik,bjk->bij", batch_centered, batch_centered)
        eigval1, _ = self._batched_power_iteration(gram_batch, random_state=self.random_state)

        trace = np.einsum("bii->b", gram_batch)
        trace[trace == 0] = 1.0
        leave_one_out_l1 = eigval1 / trace

        std = leave_one_out_l1.std()
        if std == 0:
            return submatrix, []

        z_scores = (leave_one_out_l1 - leave_one_out_l1.mean()) / std
        dropped = [samples[i] for i in range(n_samples) if abs(z_scores[i]) > self.z_max]
        kept = [s for s in samples if s not in dropped]

        return submatrix[kept], dropped

    def _null_distribution(self, expression: pd.DataFrame, module_size: int, global_center: pd.Series | None) -> np.ndarray:
        if self.center == "fixed" and global_center is None:
            raise ValueError("global_center is required when center='fixed'")

        rng = np.random.default_rng(self.random_state)
        expr_array = expression.to_numpy()
        n_genes = expr_array.shape[0]

        random_indices = np.array([
            rng.choice(n_genes, size=module_size, replace=False)
            for _ in range(self.n_permutations)
        ])  # shape (n_permutations, module_size)

        batch = expr_array[random_indices]  # shape (n_permutations, module_size, n_samples)

        if self.center == "standard":
            row_means = batch.mean(axis=2, keepdims=True)
            batch_centered = batch - row_means
        else:  # "fixed"
            gc = global_center.to_numpy()
            batch_centered = batch - gc[None, None, :]

        gram_batch = np.einsum("bik,bjk->bij", batch_centered, batch_centered)
        eigval1, eigval2 = self._batched_power_iteration(gram_batch, random_state=self.random_state)

        trace = np.einsum("bii->b", gram_batch)
        trace[trace == 0] = 1.0  # guard degenerate all-zero modules

        l1 = eigval1 / trace
        l2 = eigval2 / trace
        return np.column_stack([l1, l1 - l2])

    def _pca_fast(self, X_centered: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, np.ndarray]:

        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        explained_variance = (S ** 2) / (X_centered.shape[0] - 1)
        explained_variance_ratio = explained_variance / explained_variance.sum()
        scores = U[:, :n_components] * S[:n_components]
        return scores[:, :n_components], explained_variance_ratio[:n_components]

    def _batched_power_iteration(self, 
        gram_batch: np.ndarray, n_iterations: int = 50, random_state: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:

        rng = np.random.default_rng(random_state)
        n_batch, k, _ = gram_batch.shape

        def _top_eigenvalue(G: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            v = rng.normal(size=(n_batch, k))
            v /= np.linalg.norm(v, axis=1, keepdims=True)
            for _ in range(n_iterations):
                v = np.einsum("bij,bj->bi", G, v)
                norms = np.linalg.norm(v, axis=1, keepdims=True)
                norms[norms == 0] = 1.0  # guard degenerate all-zero rows
                v /= norms
            eigval = np.einsum("bi,bij,bj->b", v, G, v)  # Rayleigh quotient
            return eigval, v

        eigval1, v1 = _top_eigenvalue(gram_batch)

        # deflate: remove the found direction, then repeat to get the second eigenvalue
        deflation = eigval1[:, None, None] * np.einsum("bi,bj->bij", v1, v1)
        eigval2, _ = _top_eigenvalue(gram_batch - deflation)

        return eigval1, eigval2


