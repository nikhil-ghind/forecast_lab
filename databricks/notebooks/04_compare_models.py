# Databricks notebook source
# MAGIC %md ## 04 — Compare Prophet vs ARIMA vs Ensemble

# COMMAND ----------
import pandas as pd
from pyspark.sql import functions as F

actual   = spark.table("forecast_lab.clean_series").toPandas()
prophet  = spark.table("forecast_lab.prophet_forecast").toPandas()
arima    = spark.table("forecast_lab.arima_forecast").toPandas()

test_start = actual["ds"].max() - pd.Timedelta("89D")
actual_test = actual[actual["ds"] >= test_start]

def mae(a, p): return float((a - p).abs().mean())
def rmse(a, p): return float(((a - p) ** 2).mean() ** 0.5)
def mape(a, p): return float(((a - p).abs() / a.abs()).mean() * 100)

merged_p = actual_test.merge(prophet[["ds", "yhat"]], on="ds")
merged_a = actual_test.merge(arima[["ds",  "yhat"]], on="ds")

rows = [
    {"model": "Prophet", "MAE": mae(merged_p["y"], merged_p["yhat"]),
     "RMSE": rmse(merged_p["y"], merged_p["yhat"]),
     "MAPE": mape(merged_p["y"], merged_p["yhat"])},
    {"model": "ARIMA",   "MAE": mae(merged_a["y"], merged_a["yhat"]),
     "RMSE": rmse(merged_a["y"], merged_a["yhat"]),
     "MAPE": mape(merged_a["y"], merged_a["yhat"])},
]
results = pd.DataFrame(rows).sort_values("MAPE")
print(results.to_string(index=False))
spark.createDataFrame(results).write.mode("overwrite").saveAsTable("forecast_lab.model_comparison")

# COMMAND ----------
display(spark.table("forecast_lab.model_comparison"))
