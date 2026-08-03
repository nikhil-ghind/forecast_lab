from __future__ import annotations
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import pmdarima as pm
import mlflow
from typing import Optional, Tuple


def check_stationarity(series: pd.Series, significance: float = 0.05) -> bool:
    result = adfuller(series.dropna())
    return result[1] < significance


class ARIMAForecaster:
    def __init__(self, order: Tuple[int, int, int] = None,
                 seasonal_order: Tuple[int, int, int, int] = None,
                 auto: bool = True):
        self.order = order
        self.seasonal_order = seasonal_order or (0, 0, 0, 0)
        self.auto = auto
        self.model_ = None
        self.result_ = None

    def fit(self, train: pd.DataFrame, freq: str = "D") -> "ARIMAForecaster":
        y = train.set_index("ds")["y"]

        if self.auto or self.order is None:
            auto_model = pm.auto_arima(
                y, seasonal=True, m=7 if freq == "D" else 12,
                stepwise=True, suppress_warnings=True, error_action="ignore",
                information_criterion="aic", max_p=3, max_q=3, max_P=2, max_Q=2,
            )
            self.order = auto_model.order
            self.seasonal_order = auto_model.seasonal_order

        self.model_ = SARIMAX(y, order=self.order, seasonal_order=self.seasonal_order,
                               enforce_stationarity=False, enforce_invertibility=False)
        self.result_ = self.model_.fit(disp=False)
        return self

    def predict(self, periods: int) -> pd.DataFrame:
        forecast = self.result_.get_forecast(steps=periods)
        idx = pd.date_range(
            self.result_.model.data.dates[-1] + pd.Timedelta("1D"), periods=periods
        )
        summary = forecast.summary_frame(alpha=0.05)
        return pd.DataFrame({
            "ds": idx,
            "yhat": summary["mean"].values,
            "yhat_lower": summary["mean_ci_lower"].values,
            "yhat_upper": summary["mean_ci_upper"].values,
        })

    def log_to_mlflow(self, metrics: dict = None, run_name: str = "arima"):
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("order", self.order)
            mlflow.log_param("seasonal_order", self.seasonal_order)
            mlflow.log_param("aic", self.result_.aic)
            mlflow.log_param("bic", self.result_.bic)
            if metrics:
                mlflow.log_metrics(metrics)
