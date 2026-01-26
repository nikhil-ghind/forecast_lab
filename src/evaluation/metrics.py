import numpy as np
import pandas as pd
from typing import Dict


def compute_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = ~np.isnan(actual) & ~np.isnan(predicted)
    return float(np.mean(np.abs(actual[mask] - predicted[mask])))


def compute_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = ~np.isnan(actual) & ~np.isnan(predicted)
    return float(np.sqrt(np.mean((actual[mask] - predicted[mask]) ** 2)))


def compute_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = (~np.isnan(actual) & ~np.isnan(predicted)) & (actual != 0)
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def compute_smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = ~np.isnan(actual) & ~np.isnan(predicted)
    denom = (np.abs(actual[mask]) + np.abs(predicted[mask])) / 2 + 1e-9
    return float(np.mean(np.abs(actual[mask] - predicted[mask]) / denom) * 100)


def evaluate(actual: pd.Series, predicted: pd.Series) -> Dict[str, float]:
    a = actual.values
    p = predicted.values
    return {
        "MAE":   compute_mae(a, p),
        "RMSE":  compute_rmse(a, p),
        "MAPE":  compute_mape(a, p),
        "sMAPE": compute_smape(a, p),
    }
