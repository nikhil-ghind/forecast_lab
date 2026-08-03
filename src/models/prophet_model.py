from __future__ import annotations
import pandas as pd
import numpy as np
from prophet import Prophet
from typing import Optional
import mlflow
import mlflow.prophet


class ProphetForecaster:
    def __init__(self, seasonality_mode: str = "multiplicative",
                 changepoint_prior_scale: float = 0.05,
                 seasonality_prior_scale: float = 10.0,
                 yearly_seasonality: bool = True,
                 weekly_seasonality: bool = True,
                 daily_seasonality: bool = False,
                 uncertainty_samples: int = 1000):
        self.params = dict(
            seasonality_mode=seasonality_mode,
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            uncertainty_samples=uncertainty_samples,
        )
        self.model: Optional[Prophet] = None

    def fit(self, train: pd.DataFrame, extra_regressors: list[str] = None) -> "ProphetForecaster":
        self.model = Prophet(**self.params)
        if extra_regressors:
            for reg in extra_regressors:
                self.model.add_regressor(reg)
        self.model.fit(train)
        return self

    def predict(self, periods: int, freq: str = "D",
                future_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if future_df is None:
            future_df = self.model.make_future_dataframe(periods=periods, freq=freq)
        forecast = self.model.predict(future_df)
        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]

    def cross_validate(self, df: pd.DataFrame, initial: str = "365 days",
                       period: str = "30 days", horizon: str = "90 days") -> pd.DataFrame:
        from prophet.diagnostics import cross_validation, performance_metrics
        self.fit(df)
        cv_results = cross_validation(self.model, initial=initial,
                                      period=period, horizon=horizon)
        return performance_metrics(cv_results)

    def log_to_mlflow(self, run_name: str = "prophet"):
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(self.params)
            mlflow.prophet.log_model(self.model, "prophet_model")
