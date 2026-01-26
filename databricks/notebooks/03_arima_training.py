# Databricks notebook source
# MAGIC %md ## 03 — Auto-ARIMA Training on Databricks

# COMMAND ----------
import mlflow
import pandas as pd
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX

mlflow.set_experiment("/forecast_lab/arima")

df = spark.table("forecast_lab.clean_series").toPandas()
df["ds"] = pd.to_datetime(df["ds"])
y = df.set_index("ds")["y"]
train_y = y[y.index < y.index.max() - pd.Timedelta("90D")]

with mlflow.start_run(run_name="auto_arima_databricks"):
    auto = pm.auto_arima(train_y, seasonal=True, m=7, stepwise=True,
                         suppress_warnings=True, error_action="ignore",
                         information_criterion="aic", max_p=3, max_q=3)
    mlflow.log_param("order", auto.order)
    mlflow.log_param("seasonal_order", auto.seasonal_order)
    mlflow.log_metric("aic", auto.aic())
    mlflow.log_metric("bic", auto.bic())

    fc, ci = auto.predict(90, return_conf_int=True)
    idx = pd.date_range(train_y.index[-1] + pd.Timedelta("1D"), periods=90)
    forecast_df = pd.DataFrame({
        "ds": idx, "yhat": fc,
        "yhat_lower": ci[:, 0], "yhat_upper": ci[:, 1]
    })

spark.createDataFrame(forecast_df) \
     .write.mode("overwrite").saveAsTable("forecast_lab.arima_forecast")
print("Forecast saved to forecast_lab.arima_forecast")
