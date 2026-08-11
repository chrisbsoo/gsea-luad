import numpy as np
import pandas as pd
import pytest

from romapy.core import ROMA


@pytest.fixture
def random_expression():
    rng = np.random.default_rng(0)
    genes = [f"GENE_{i}" for i in range(200)]
    samples = [f"sample_{i}" for i in range(40)]
    return pd.DataFrame(rng.normal(size=(len(genes), len(samples))), index=genes, columns=samples)

@pytest.mark.xfail(reason="fit() not yet implemented", strict=True)
def test_l1_pvalue_is_valid_probability(random_expression):
    gene_sets = {"RANDOM_MODULE": list(random_expression.index[:10])}
    results = ROMA(n_permutations=200, random_state=0).fit(random_expression, gene_sets)
    p = results.l1_pval["RANDOM_MODULE"]
    assert 0.0 <= p <= 1.0


@pytest.mark.xfail(reason="fit() not yet implemented", strict=True)
def test_pure_noise_module_is_rarely_significant(random_expression):
    """Sanity check on false-positive rate: a module of pure noise genes
    shouldn't be flagged significant far more often than alpha allows."""
    n_trials = 50
    n_false_positives = 0
    for i in range(n_trials):
        rng = np.random.default_rng(i)
        genes = rng.choice(random_expression.index, size=10, replace=False).tolist()
        results = ROMA(n_permutations=200, random_state=i).fit(random_expression, {"MODULE": genes})
        if results.l1_pval["MODULE"] < 0.05:
            n_false_positives += 1
    assert n_false_positives / n_trials < 0.15


def test_null_distribution_shape(random_expression):
    roma = ROMA(n_permutations=500, random_state=0, center="standard")
    null = roma._null_distribution(random_expression, module_size=10, global_center=None)
    assert null.shape == (500, 2)


def test_null_distribution_values_are_valid(random_expression):
    roma = ROMA(n_permutations=200, random_state=0, center="standard")
    null = roma._null_distribution(random_expression, module_size=10, global_center=None)
    l1_values = null[:, 0]
    assert (l1_values >= 0).all() and (l1_values <= 1).all()


def test_null_distribution_reproducible_with_seed(random_expression):
    roma_a = ROMA(n_permutations=100, random_state=42, center="standard")
    roma_b = ROMA(n_permutations=100, random_state=42, center="standard")
    null_a = roma_a._null_distribution(random_expression, module_size=10, global_center=None)
    null_b = roma_b._null_distribution(random_expression, module_size=10, global_center=None)
    assert np.allclose(null_a, null_b)


def test_null_distribution_depends_on_module_size(random_expression):
    roma = ROMA(n_permutations=300, random_state=0, center="standard")
    null_small = roma._null_distribution(random_expression, module_size=5, global_center=None)
    null_large = roma._null_distribution(random_expression, module_size=50, global_center=None)
    assert null_small[:, 0].mean() > null_large[:, 0].mean()
