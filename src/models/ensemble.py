from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List, Dict


class EnsembleForecaster:
    """Weighted average ensemble over multiple forecasters."""

    def __init__(self, forecasters: Dict[str, object], weights: Dict[str, float] = None):
        self.forecasters = forecasters
        total = sum(weights.values()) if weights else len(forecasters)
        self.weights = {k: (weights[k] / total if weights else 1.0 / total)
                        for k in forecasters}

    def fit(self, train: pd.DataFrame) -> "EnsembleForecaster":
        for name, model in self.forecasters.items():
            model.fit(train)
        return self

    def predict(self, periods: int, freq: str = "D") -> pd.DataFrame:
        forecasts = {}
        for name, model in self.forecasters.items():
            fc = model.predict(periods, freq=freq) if hasattr(model, "predict") else model.predict(periods)
            forecasts[name] = fc.set_index("ds")["yhat"]

        df = pd.DataFrame(forecasts)
        ensemble_yhat = sum(df[name] * w for name, w in self.weights.items())
        return pd.DataFrame({"ds": df.index, "yhat": ensemble_yhat.values}).reset_index(drop=True)

    def update_weights_from_validation(self, val: pd.DataFrame):
        """Inverse-MAE weighting based on validation performance."""
        from src.evaluation.metrics import compute_mae
        maes = {}
        for name, model in self.forecasters.items():
            preds = model.predict(len(val))
            preds = preds.set_index("ds")["yhat"].reindex(val["ds"].values)
            maes[name] = compute_mae(val["y"].values, preds.values)

        inv = {k: 1.0 / (v + 1e-9) for k, v in maes.items()}
        total = sum(inv.values())
        self.weights = {k: v / total for k, v in inv.items()}
        return self.weights
