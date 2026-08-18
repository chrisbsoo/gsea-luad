"""
pipeline/run_roma.py

Applies romapy to the cleaned TCGA-LUAD data:
- intersects samples present in both clinical and expression data
- runs ROMA against a gene set collection (e.g. MSigDB Hallmark, .gmt format)
- tests whether pathway activity scores associate with tumor stage (ANOVA)
"""

import argparse

import pandas as pd
from scipy.stats import f_oneway
from statsmodels.stats.multitest import multipletests  # add this import

from romapy import ROMA, load_gmt


def load_cleaned(clinical_path: str, expression_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    clinical = pd.read_csv(clinical_path, index_col=0)
    expression = pd.read_csv(expression_path, index_col=0)
    return clinical, expression


def intersect_samples(clinical: pd.DataFrame, expression: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    common = clinical.index.intersection(expression.columns)

    n_clinical_only = len(clinical.index.difference(common))
    n_expression_only = len(expression.columns.difference(common))
    if n_clinical_only or n_expression_only:
        print(
            f"[run_roma] dropping {n_clinical_only} clinical-only and "
            f"{n_expression_only} expression-only samples; {len(common)} samples in common"
        )

    return clinical.loc[common], expression[common]


def test_stage_association(results, clinical: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    sig_modules = results.significant(alpha=alpha)
    rows = []

    for module in sig_modules.index:
        scores = results.scores.loc[module]
        merged = pd.DataFrame({"score": scores, "stage": clinical["C_STAGE"]}).dropna()

        groups = [g["score"].values for _, g in merged.groupby("stage") if len(g) >= 2]
        if len(groups) < 2:
            continue

        f_stat, p_value = f_oneway(*groups)
        rows.append({"module": module, "f_stat": f_stat, "stage_anova_pval": p_value})

    result_df = pd.DataFrame(rows).set_index("module").sort_values("stage_anova_pval")

    # multiple-testing correction: we ran one ANOVA per module, so correct
    # across all of them together, not each in isolation
    if len(result_df) > 0:
        _, pvals_adj, _, _ = multipletests(result_df["stage_anova_pval"], method="fdr_bh")
        result_df["stage_anova_pval_adj"] = pvals_adj

    return result_df


def main(clinical_path: str, expression_path: str, gene_sets_path: str, output_dir: str) -> None:
    clinical, expression = load_cleaned(clinical_path, expression_path)
    clinical, expression = intersect_samples(clinical, expression)

    gene_sets = load_gmt(gene_sets_path)
    print(f"[run_roma] loaded {len(gene_sets)} gene sets")

    roma = ROMA(center="fixed", robust=True, n_permutations=1000, random_state=0)
    results = roma.fit(expression, gene_sets)
    print(results)

    sig = results.significant(alpha=0.05)
    print(f"[run_roma] {len(sig)} significant modules")
    sig.to_csv(f"{output_dir}/significant_modules.csv")

    stage_assoc = test_stage_association(results, clinical)
    stage_assoc.to_csv(f"{output_dir}/stage_association.csv")
    print(f"[run_roma] {len(stage_assoc)} modules tested for stage association")
    print(stage_assoc.head(10))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clinical", required=True)
    parser.add_argument("--expression", required=True)
    parser.add_argument("--gene-sets", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    main(args.clinical, args.expression, args.gene_sets, args.output_dir)