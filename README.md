# romapy

An independent Python implementation of **ROMA** (Representation and Quantification of Module
Activity) — a method for scoring the activity of a gene set (pathway, transcription-factor target
set, etc.) in individual samples from expression data, with statistical significance testing
built in. Check the PyPI details here: https://pypi.org/project/romapy/

## Why this exists

ROMA was originally published as a **Java** tool:

> Martignetti L, Calzone L, Bonnet E, Barillot E, Zinovyev A. (2016) "ROMA: Representation and
> Quantification of Module Activity from Target Expression Data." *Frontiers in Genetics* 7:18.
> https://doi.org/10.3389/fgene.2016.00018
> Original source: https://github.com/sysbio-curie/Roma

A later **R** reimplementation, **rROMA**, was applied to cystic fibrosis and other datasets:

> Institut Curie, *rROMA: Representation and quantification of module activity from omics data*,
> https://github.com/sysbio-curie/rROMA

As of writing, no Python implementation existed. `romapy` fills that gap. **This project is
independent and is not affiliated with, endorsed by, or reviewed by the original authors.** If
you use ROMA in published work, please cite the original paper above, not this package.

## What ROMA actually does (vs. plain PCA)

Naively taking PC1 of a gene set's expression submatrix is *not* ROMA — any random set of genes
will explain some variance via PC1, so a bare PC1 score has no statistical grounding. ROMA adds,
on top of that:

- **Fixed-center PCA** — PC1 is computed through the center of the *global* expression
  distribution, not the module's own centroid, so it captures both overdispersed and "shifted"
  gene sets
- **Sign orientation** — resolving PCA's inherent mirror-symmetry so scores are interpretable and
  comparable across modules
- **Robust trimming** — leave-one-out outlier removal before computing PC1
- **Permutation-based significance testing** — an empirical null distribution built from random
  gene sets of matching size, giving each module a p-value for being "overdispersed" and a
  separate one for being "coordinated" (rather than just reporting a PC1 score with no indication
  of whether it's distinguishable from noise)

## Status

Baseline linear ROMA is fully implemented and tested: fixed/standard-center PCA, sign
orientation, robust leave-one-out outlier trimming, and permutation-based significance testing
(null distribution + spectral-gap coordination test). All permutation-heavy computation
(significance testing, robust trimming) uses batched power iteration rather than per-sample SVD,
validated against an exact-SVD reference — see `tests/` for correctness tests comparing the two.

Next: kernel PCA and principal-curve extensions for non-linear module activity scoring — see the
original paper's own stated future work.

## Validation example

`examples/LUAD_baseline/` applies `romapy` end-to-end to real TCGA-LUAD (lung adenocarcinoma)
clinical and expression data (~510 patients), testing MSigDB Hallmark pathway activity against
tumor stage. Finds 9 pathways (dominated by proliferation/cell-cycle gene sets — MYC targets,
E2F targets, G2M checkpoint) significantly associated with stage after FDR correction —
consistent with established cancer biology. See
[`examples/LUAD_baseline/README.md`](examples/LUAD_baseline/README.md) for the full pipeline,
results table, and caveats.

## Install

```bash
pip install -e .          # from a local clone, for now
```

## Quickstart

```python
from romapy import ROMA, load_gmt
import pandas as pd

expression = pd.read_csv("expression.csv", index_col=0)   # genes x samples
gene_sets = load_gmt("h.all.v2024.1.Hs.symbols.gmt")       # any MSigDB collection — Hallmark, KEGG, etc.

roma = ROMA(center="fixed", robust=True, n_permutations=1000, random_state=0)
results = roma.fit(expression, gene_sets)

results.scores                     # modules x samples activity scores
results.significant(alpha=0.05)    # modules passing the L1 significance test
```


## For Users / Researchers

```bash
pip install romapy
```

## For Contributors / Developers

```bash
pip install -e ".[dev]"
pytest -v
ruff check .
```

CI runs the full test suite plus linting across Python 3.10–3.13 on every push and PR to `main`.

## License

MIT — see `LICENSE`. This is an independent reimplementation of a published method; see
attribution above and `CITATION.cff`.
