# LUAD Baseline: romapy applied to TCGA lung adenocarcinoma

An end-to-end application of `romapy` to real cancer genomics data: does Hallmark pathway
activity, as scored by ROMA, associate with tumor stage in lung adenocarcinoma (LUAD)?

## Data

TCGA-LUAD PanCancer Atlas cohort (`cbioportal.org`), ~510 patients:
- `data/raw/gsea_clinical.tsv` — clinical annotation (stage, age, sex, mutation burden, etc.)
- `data/raw/gsea_mrna.txt` — mRNA expression, ~7,400 genes x 510 samples
- Gene sets: MSigDB Hallmark collection (50 gene sets), `.gmt` format — see
  [MSigDB](https://www.gsea-msigdb.org/gsea/msigdb) (free registration required; not redistributed
  here per MSigDB license terms — see `pipeline/README.md` for how to obtain it)

## Romapy Baseline Pipeline 

    pipeline/
    ├── preprocess_clinical.py   # RF-imputed AJCC staging, derives C_STAGE label
    ├── preprocess_mrna.py       # Yeo-Johnson normalization, IQR outlier capping
    └── run_roma.py              # intersects samples, runs romapy, tests stage association

Each script is independently runnable and testable (`load_raw` / `clean` / `main`); see
`run_pipeline.ipynb` for a walkthrough that imports and runs all three in sequence with
inline visualization.

**Clinical preprocessing**: AJCC T/N/M stage codes with missing values imputed via
`RandomForestClassifier`, gated by a minimum F1 score so unreliable imputations are left as
missing rather than guessed. Samples with no usable derived stage label are dropped (explicit,
logged count — see script output).

**Expression preprocessing**: genes with any missing sample values dropped; remaining matrix
normalized with a Yeo-Johnson power transform; per-gene outliers capped at 1.5x IQR.

**ROMA**: fixed-center, robust PCA-based module scoring (`center="fixed", robust=True`) across
all 50 Hallmark gene sets, with permutation-based significance testing
(`n_permutations=1000`). See the main [romapy README](../../README.md) for what this actually
computes and why it's more than plain PCA.

## Result

Of the modules ROMA found significantly coordinated (`l1_pval < 0.05`), 9 also showed a
statistically significant association between module activity and tumor stage (one-way ANOVA,
Benjamini-Hochberg FDR corrected):

| Module | F-stat | p (adjusted) |
|---|---|---|
| HALLMARK_MYC_TARGETS_V1 | 6.55 | 5.3e-06 |
| HALLMARK_MTORC1_SIGNALING | 5.61 | 3.9e-05 |
| HALLMARK_UV_RESPONSE_UP | 4.88 | 2.0e-04 |
| HALLMARK_UNFOLDED_PROTEIN_RESPONSE | 4.47 | 4.8e-04 |
| HALLMARK_SPERMATOGENESIS | 4.14 | 7.5e-04 |
| HALLMARK_G2M_CHECKPOINT | 4.14 | 7.5e-04 |
| HALLMARK_E2F_TARGETS | 4.11 | 7.5e-04 |
| HALLMARK_MITOTIC_SPINDLE | 3.74 | 1.8e-03 |
| HALLMARK_MYC_TARGETS_V2 | 3.08 | 9.8e-03 |

Full table with unadjusted p-values in `data/results/stage_association.csv`. Note
`HALLMARK_ALLOGRAFT_REJECTION` was significant in early smoke-testing on a 2-module subset but
does not survive FDR correction on the full 50-module run (`p_adj = 0.17`) — a useful reminder
that small-subset previews can be misleading.

**Interpretation**: the top hits are dominated by proliferation and cell-cycle pathways
(`MYC_TARGETS_V1/V2`, `E2F_TARGETS`, `G2M_CHECKPOINT`, `MITOTIC_SPINDLE`) — consistent with
increased proliferative activity being a well-established feature of cancer progression, rather
than an arbitrary or spurious set of hits. `HALLMARK_MYC_TARGETS_V1` activity increases
monotonically with stage (I -> IV):

![MYC Targets V1 activity by stage](data/results/myc_targets_v1_by_stage.png)

## Caveats

- Stage groups collapsed to I/II/III/IV (from finer AJCC substages) for the ANOVA to keep group
  sizes reasonable; some substages had very few samples individually.
- This is an association, not a causal or mechanistic claim — consistent with, not proof of,
  known cancer biology.
- `robust=True` trimming can occasionally flag a naturally-noisy sample as an outlier at small
  module sizes; see the main package's test suite for the false-positive-rate characterization
  of this behavior.

## Reproducing Datasets

```bash
pip install -e "../../[dev]"
jupyter notebook run_pipeline.ipynb
```

Alternatively, run the jupyter notebook cell-by-cell

## Reproducing Results

```bash
jupyter notebook baseline_analysis.ipynb
```

Alternatively, run the jupyter notebook cell-by-cell
