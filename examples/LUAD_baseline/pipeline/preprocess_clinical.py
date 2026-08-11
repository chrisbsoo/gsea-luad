"""
pipeline/preprocess_clinical.py

Cleans TCGA-LUAD clinical data:
- selects relevant columns
- imputes missing AJCC stage codes via RandomForestClassifier (gated by score)
- derives a simplified C_STAGE label from the overall AJCC stage string
"""

import argparse

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

STAGE_COL = "Neoplasm Disease Stage American Joint Committee on Cancer Code"
T_COL = "American Joint Committee on Cancer Tumor Stage Code"
N_COL = "Neoplasm Disease Lymph Node Stage American Joint Committee on Cancer Code"
M_COL = "American Joint Committee on Cancer Metastasis Stage Code"

TARGET_COLS = [
    "Sample ID",
    "Diagnosis Age",
    "Sex",
    STAGE_COL,
    T_COL,
    N_COL,
    M_COL,
    "Aneuploidy Score",
    "Mutation Count",
    "Fraction Genome Altered",
]


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", na_values=["NA"])
    return df


def _impute_stage_column(df: pd.DataFrame, target_col: str, feature_cols: list[str], min_f1: float = 0.6) -> pd.Series:
    """
    Impute missing values in `target_col` using a RandomForestClassifier
    trained on `feature_cols`. Only applies the imputation if the model's
    held-out F1 score clears `min_f1` — otherwise leaves missing values as
    NaN rather than trusting an unreliable model's guesses.
    """
    known = df.dropna(subset=[target_col] + feature_cols)
    unknown = df[df[target_col].isna()]

    if unknown.empty or known.empty:
        return df[target_col]

    X_known = pd.get_dummies(known[feature_cols])
    y_known = known[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X_known, y_known, test_size=0.2, random_state=0, stratify=y_known
    )

    clf = RandomForestClassifier(n_estimators=200, random_state=0)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    f1 = f1_score(y_test, preds, average="weighted")

    result = df[target_col].copy()
    if f1 < min_f1:
        print(f"[preprocess_clinical] skipping imputation for {target_col}: F1={f1:.2f} < {min_f1}")
        return result

    clf.fit(X_known, y_known)
    X_unknown = pd.get_dummies(unknown[feature_cols]).reindex(columns=X_known.columns, fill_value=0)
    imputed = clf.predict(X_unknown)
    result.loc[unknown.index] = imputed

    print(f"[preprocess_clinical] imputed {len(unknown)} values for {target_col}, F1={f1:.2f}")
    return result


def _derive_c_stage(stage: str) -> str | float:
    if pd.isna(stage):
        return float("nan")
    stage = stage.upper().replace("STAGE ", "")
    for prefix in ("IV", "IIIB", "IIIA", "III", "IIB", "IIA", "II", "IB", "IA", "I"):
        if stage.startswith(prefix):
            return prefix
    return float("nan")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df_output = df[TARGET_COLS].copy()

    feature_cols = ["Diagnosis Age", "Sex", "Aneuploidy Score", "Mutation Count"]
    df_output[T_COL] = _impute_stage_column(df_output, T_COL, feature_cols)
    df_output[N_COL] = _impute_stage_column(df_output, N_COL, feature_cols)
    df_output[M_COL] = _impute_stage_column(df_output, M_COL, feature_cols)

    df_output["C_STAGE"] = df_output[STAGE_COL].apply(_derive_c_stage)

    n_before = len(df_output)
    df_output = df_output.dropna(subset=["C_STAGE"])
    n_dropped = n_before - len(df_output)
    print(f"[preprocess_clinical] dropped {n_dropped} of {n_before} samples with no usable stage label")

    df_output = df_output.set_index("Sample ID")
    return df_output


def main(input_path: str, output_path: str) -> None:
    df = load_raw(input_path)
    df = clean(df)
    df.to_csv(output_path)
    print(f"[preprocess_clinical] wrote {len(df)} rows to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(args.input, args.output)