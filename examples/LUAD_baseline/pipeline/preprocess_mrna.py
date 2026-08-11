"""
pipeline/preprocess_mrna.py

Cleans TCGA-LUAD mRNA expression data:
- pivots to genes x samples
- checks skew/kurtosis
- applies a Yeo-Johnson power transform to normalize
- caps outliers per-gene using IQR bounds
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df = df.drop(columns=["Entrez_Gene_Id"])
    df = df[df["Hugo_Symbol"].notna() & (df["Hugo_Symbol"] != "")]  # drop unnamed genes
    df = df.set_index("Hugo_Symbol")
    df = df.groupby(df.index).mean()  # collapse duplicate gene symbols, if any
    return df


def _report_skew_kurtosis(df: pd.DataFrame) -> None:
    skew = df.skew(axis=1)
    kurt = df.kurt(axis=1)
    print(f"[preprocess_mrna] skew: mean={skew.mean():.2f}, max={skew.abs().max():.2f}")
    print(f"[preprocess_mrna] kurtosis: mean={kurt.mean():.2f}, max={kurt.abs().max():.2f}")


def _yeo_johnson_normalize(df: pd.DataFrame) -> pd.DataFrame:
    transformer = PowerTransformer(method="yeo-johnson", standardize=True)
    # PowerTransformer expects samples as rows, genes as columns - transpose,
    # fit_transform, transpose back
    transformed = transformer.fit_transform(df.T)
    return pd.DataFrame(transformed.T, index=df.index, columns=df.columns)


def _cap_outliers(df: pd.DataFrame, iqr_multiplier: float = 1.5) -> pd.DataFrame:
    df_capped = df.copy()
    for gene in df_capped.index:
        Q1 = df_capped.loc[gene].quantile(0.25)
        Q3 = df_capped.loc[gene].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - iqr_multiplier * IQR
        upper_bound = Q3 + iqr_multiplier * IQR

        row = df_capped.loc[gene]
        df_capped.loc[gene, row < lower_bound] = lower_bound
        df_capped.loc[gene, row > upper_bound] = upper_bound
    return df_capped


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(axis=0, how="any")  # drop genes with any missing sample values

    _report_skew_kurtosis(df)
    df = _yeo_johnson_normalize(df)
    df = _cap_outliers(df)

    n_nan = df.isna().sum().sum()
    if n_nan > 0:
        raise ValueError(f"cleaned expression matrix still contains {n_nan} NaN values")

    try:
        df.to_csv("./data/processed/cleaned_mrna.csv", index=True)
    except:
        raise ReferenceError("Please make sure run_pipeline.ipynb is in root dir of LUAD_baseline")

    return df


def main(input_path: str, output_path: str) -> None:
    df = load_raw(input_path)
    df = clean(df)
    df.to_csv(output_path)
    print(f"[preprocess_mrna] wrote {df.shape[0]} genes x {df.shape[1]} samples to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(args.input, args.output)