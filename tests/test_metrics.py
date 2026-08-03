import numpy as np
import pytest
from src.evaluation.metrics import compute_mae, compute_rmse, compute_mape, evaluate
import pandas as pd


def test_mae_perfect():
    a = np.array([1.0, 2.0, 3.0])
    assert compute_mae(a, a) == pytest.approx(0.0)


def test_rmse_known():
    a = np.array([0.0, 0.0, 0.0])
    p = np.array([1.0, 1.0, 1.0])
    assert compute_rmse(a, p) == pytest.approx(1.0)


def test_mape_known():
    a = np.array([100.0, 200.0])
    p = np.array([110.0, 190.0])
    assert compute_mape(a, p) == pytest.approx(7.5)


def test_evaluate_returns_all_keys():
    s = pd.Series([1.0, 2.0, 3.0])
    metrics = evaluate(s, s)
    for key in ["MAE", "RMSE", "MAPE", "sMAPE"]:
        assert key in metrics
