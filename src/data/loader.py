import os
import pandas as pd
import numpy as np
from typing import Optional


def load_csv(path: str, date_col: str = "ds", value_col: str = "y",
             freq: str = "D") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[date_col])
    df = df.rename(columns={date_col: "ds", value_col: "y"})
    df = df.sort_values("ds").reset_index(drop=True)
    df = df.set_index("ds").asfreq(freq).reset_index()
    df["y"] = df["y"].interpolate(method="time")
    return df[["ds", "y"]]


def load_from_databricks(table: str, spark=None) -> pd.DataFrame:
    if spark is None:
        raise RuntimeError("Spark session required for Databricks loading")
    sdf = spark.table(table)
    df = sdf.toPandas()
    df["ds"] = pd.to_datetime(df["ds"])
    return df.sort_values("ds").reset_index(drop=True)


def train_test_split(df: pd.DataFrame, test_periods: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    return df.iloc[:-test_periods].copy(), df.iloc[-test_periods:].copy()


def generate_synthetic(n: int = 730, freq: str = "D", seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq=freq)
    trend = np.linspace(100, 200, n)
    seasonal = 20 * np.sin(2 * np.pi * np.arange(n) / 365.25)
    noise = rng.normal(0, 5, n)
    y = trend + seasonal + noise
    return pd.DataFrame({"ds": dates, "y": y})
