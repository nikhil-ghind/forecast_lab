# Databricks notebook source
# MAGIC %md ## 02 — Prophet Training on Databricks
# MAGIC Train Prophet model on clean series, log to MLflow, write forecasts to Delta.

# COMMAND ----------
import mlflow
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

mlflow.set_experiment("/forecast_lab/prophet")

df = spark.table("forecast_lab.clean_series").toPandas()
df["ds"] = pd.to_datetime(df["ds"])

train = df[df["ds"] < df["ds"].max() - pd.Timedelta("90D")]

with mlflow.start_run(run_name="prophet_databricks"):
    m = Prophet(seasonality_mode="multiplicative", changepoint_prior_scale=0.05)
    m.fit(train)

    future = m.make_future_dataframe(periods=90)
    forecast = m.predict(future)

    cv = cross_validation(m, initial="365 days", period="30 days", horizon="90 days")
    perf = performance_metrics(cv)
    mlflow.log_metric("cv_mae",  perf["mae"].mean())
    mlflow.log_metric("cv_rmse", perf["rmse"].mean())
    mlflow.log_metric("cv_mape", perf["mape"].mean())
    mlflow.prophet.log_model(m, "prophet_model")

spark.createDataFrame(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]) \
     .write.mode("overwrite").saveAsTable("forecast_lab.prophet_forecast")
print("Forecast saved to forecast_lab.prophet_forecast")

# COMMAND ----------
display(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(90))
