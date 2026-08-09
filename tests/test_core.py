import numpy as np
import pandas as pd
import pytest

from romapy.core import ROMA

@pytest.fixture
def coordinated_expression():
    rng = np.random.default_rng(42)
    n_samples = 30
    factor = rng.normal(size=n_samples)

    coord_data = {
        f"COORD_{i}": factor * rng.normal(1.0, 0.15) + rng.normal(scale=0.2, size=n_samples)
        for i in range(5)
    }
    noise_data = {
        f"NOISE_{i}": rng.normal(size=n_samples)
        for i in range(15)
    }

    samples = [f"sample_{i}" for i in range(n_samples)]
    expression = pd.DataFrame({**coord_data, **noise_data}, index=samples).T
    expression.columns = samples

    return expression, factor


def test_compute_module_separates_coordinated_from_noise(coordinated_expression):
    expression, factor = coordinated_expression
    coord_submatrix = expression.loc[[f"COORD_{i}" for i in range(5)]]
    noise_submatrix = expression.loc[[f"NOISE_{i}" for i in range(5)]]

    roma = ROMA(center="standard")
    result_coord = roma._compute_module(coord_submatrix, global_center=None)
    result_noise = roma._compute_module(noise_submatrix, global_center=None)

    assert result_coord["l1"] > result_noise["l1"]


def test_compute_module_recovers_known_signal(coordinated_expression):
    expression, factor = coordinated_expression
    coord_submatrix = expression.loc[[f"COORD_{i}" for i in range(5)]]

    roma = ROMA(center="standard")
    result = roma._compute_module(coord_submatrix, global_center=None)

    correlation = abs(np.corrcoef(result["scores"], factor)[0, 1])
    assert correlation > 0.9


def test_compute_module_returns_expected_shape(coordinated_expression):
    expression, _ = coordinated_expression
    coord_submatrix = expression.loc[[f"COORD_{i}" for i in range(5)]]

    roma = ROMA(center="standard")
    result = roma._compute_module(coord_submatrix, global_center=None)

    assert isinstance(result["scores"], pd.Series)
    assert list(result["scores"].index) == list(coord_submatrix.columns)
    assert list(result["gene_weights"].index) == list(coord_submatrix.index)
    assert 0.0 <= result["l1"] <= 1.0