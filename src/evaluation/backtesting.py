from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Callable, List, Dict
from src.evaluation.metrics import evaluate


def walk_forward_validation(df: pd.DataFrame, model_factory: Callable,
                             initial_train: int = 365, horizon: int = 30,
                             step: int = 30) -> pd.DataFrame:
    """
    Walk-forward (expanding window) backtest.
    model_factory() must return an object with .fit(df) and .predict(n) methods.
    """
    results = []
    n = len(df)
    start = initial_train

    while start + horizon <= n:
        train = df.iloc[:start]
        test  = df.iloc[start: start + horizon]

        model = model_factory()
        model.fit(train)
        forecast = model.predict(horizon)

        actual    = test["y"].values
        predicted = forecast["yhat"].values[:len(actual)]

        metrics = evaluate(pd.Series(actual), pd.Series(predicted))
        metrics["window_start"] = str(train["ds"].iloc[-1].date())
        metrics["window_end"]   = str(test["ds"].iloc[-1].date())
        results.append(metrics)
        start += step

    return pd.DataFrame(results)


def summarise_backtest(results_df: pd.DataFrame) -> Dict[str, float]:
    return {col: float(results_df[col].mean())
            for col in ["MAE", "RMSE", "MAPE", "sMAPE"]
            if col in results_df.columns}
