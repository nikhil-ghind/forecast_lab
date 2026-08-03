# Databricks notebook source
# MAGIC %md ## 01 — Data Preparation
# MAGIC Load raw time-series from Delta Lake, clean and write back as `forecast_lab.clean_series`.

# COMMAND ----------
from pyspark.sql import functions as F
from pyspark.sql.window import Window

raw = spark.table("forecast_lab.raw_series")
raw = raw.withColumn("ds", F.to_timestamp("ds")) \
         .withColumn("y", F.col("y").cast("double"))

# Fill short gaps via linear interpolation using pandas UDF
import pandas as pd
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import DoubleType

@pandas_udf(DoubleType())
def interpolate(s: pd.Series) -> pd.Series:
    return s.interpolate(method="linear", limit_direction="both")

raw = raw.withColumn("y", interpolate("y"))

raw.write.mode("overwrite").saveAsTable("forecast_lab.clean_series")
print("Written forecast_lab.clean_series:", raw.count(), "rows")

# COMMAND ----------
display(raw.limit(100))
