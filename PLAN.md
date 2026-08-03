# Forecast Lab

## Project Overview

A data analytics platform that uses time-series forecasting (Prophet and ARIMA) to predict sales trends across business segments, combined with a business headroom sizing model that quantifies addressable opportunity per segment. Data is ingested from SQL sources on Databricks, with automated pipelines and visualization dashboards built with Seaborn/Matplotlib.

**Key Goals:**
- Accurate time-series sales forecasting using Prophet and ARIMA models
- Segment-level forecasting with comparison across business lines
- Business headroom/sizing model quantifying total addressable market (TAM) and gap analysis
- Automated SQL data ingestion on Databricks
- Interactive visualization dashboards for stakeholder consumption
- Reproducible pipeline with automated model selection and evaluation

## Tech Stack

- Python 3.11+
- Prophet 1.1+ (Facebook/Meta time-series forecasting)
- statsmodels 0.14+ (ARIMA/SARIMAX)
- Databricks (Spark, Delta Lake, SQL warehouse)
- SQL (Databricks SQL for data ingestion)
- pandas 2.x, numpy
- Seaborn 0.13+, Matplotlib 3.8+
- scikit-learn (metrics, cross-validation)
- pytest for testing
- Databricks Notebooks / Jobs for orchestration

## Architecture Overview

```
[SQL Data Sources]
       |
       v
[Databricks SQL Ingestion]
       |
       v
[Delta Lake Tables]
       |
       v
[Data Preprocessing Pipeline]
       |
       +-----> [Prophet Forecasting] ----+
       |                                 |
       +-----> [ARIMA Forecasting]  ----+---> [Model Comparison & Selection]
                                              |
                                              v
                                     [Forecast Output Tables]
                                              |
                                              v
                                     [Business Sizing Model]
                                              |
                                              v
                                     [Visualization Dashboards]
```

---

## Phase 1: Data Ingestion and Preprocessing

**Goal:** Set up automated SQL data ingestion on Databricks and build the preprocessing pipeline.

### Tasks

1. Create `src/ingestion/sql_queries.py`:
   - Dictionary of parameterized SQL queries for each data source:
     - `SALES_HISTORY_QUERY`: pull historical sales data (date, segment, product_category, region, revenue, units_sold, channel).
     - `MARKET_DATA_QUERY`: pull market-level data (segment, total_market_size, market_growth_rate, market_share).
     - `SEGMENT_METADATA_QUERY`: pull segment definitions and hierarchies.
   - Each query accepts date range parameters.

2. Create `src/ingestion/data_loader.py`:
   - Class `DataLoader`:
     - Method: `load_sales_data(start_date, end_date) -> pd.DataFrame`: executes SQL via `spark.sql()` or Databricks SQL connector, returns pandas DataFrame.
     - Method: `load_market_data() -> pd.DataFrame`.
     - Method: `load_segment_metadata() -> pd.DataFrame`.
     - Handles connection errors, retries, logging.

3. Create `src/preprocessing/data_cleaner.py`:
   - Function: `clean_sales_data(df: pd.DataFrame) -> pd.DataFrame`:
     - Remove duplicates.
     - Handle missing values: forward-fill for small gaps (<3 days), flag larger gaps.
     - Detect and handle outliers using IQR method (cap at 1.5*IQR).
     - Ensure consistent date frequency (daily/weekly/monthly based on config).
     - Convert date columns to datetime.
     - Validate: no negative revenue, no future dates.

4. Create `src/preprocessing/feature_engineering.py`:
   - Function: `add_temporal_features(df: pd.DataFrame) -> pd.DataFrame`:
     - Add columns: day_of_week, month, quarter, year, is_holiday (US holidays via `holidays` package), week_of_year.
   - Function: `aggregate_to_frequency(df, freq='W') -> pd.DataFrame`:
     - Resample data to weekly or monthly granularity per segment.
   - Function: `create_prophet_format(df, segment) -> pd.DataFrame`:
     - Returns DataFrame with columns `ds` (date) and `y` (target metric) for a given segment.

5. Create `config/pipeline_config.yaml`:
   - Settings: date_range (start, end), forecast_horizon (weeks), granularity (W/M), target_metric (revenue/units_sold), segments (list or "all"), model_selection_metric (mape/rmse/mae).

6. Create `tests/test_data_cleaner.py`:
   - Test duplicate removal.
   - Test outlier capping.
   - Test missing value handling.
   - Test date validation.

---

## Phase 2: Prophet Forecasting Model

**Goal:** Implement segment-level sales forecasting using Facebook Prophet.

### Tasks

1. Create `src/models/prophet_forecaster.py`:
   - Class `ProphetForecaster`:
     - Constructor: accepts config dict (seasonality_mode, changepoint_prior_scale, holidays_country).
     - Method: `fit(df: pd.DataFrame)`: fits Prophet model on df with columns (ds, y). Adds US holidays. Configures yearly, weekly seasonality. Adds custom regressors if provided.
     - Method: `predict(periods: int) -> pd.DataFrame`: generates future dataframe, returns forecast with yhat, yhat_lower, yhat_upper columns.
     - Method: `get_components() -> dict`: returns trend, seasonality components for visualization.
     - Method: `cross_validate(horizon: str, period: str, initial: str) -> pd.DataFrame`: runs Prophet cross-validation, returns performance metrics.

2. Create `src/models/prophet_tuner.py`:
   - Function: `tune_prophet(df, param_grid, metric='mape') -> dict`:
     - Param grid: changepoint_prior_scale [0.01, 0.05, 0.1, 0.5], seasonality_prior_scale [0.1, 1, 10], seasonality_mode ['additive', 'multiplicative'].
     - Uses Prophet cross-validation for each param combo.
     - Returns best params and performance metrics.

3. Create `src/pipeline/forecast_runner.py`:
   - Function: `run_prophet_forecast(sales_df, segments, config) -> dict[str, pd.DataFrame]`:
     - For each segment: filter data, create Prophet format, optionally tune, fit, predict.
     - Returns dict mapping segment name to forecast DataFrame.
     - Logs per-segment metrics (MAPE, RMSE, MAE).

4. Create `tests/test_prophet_forecaster.py`:
   - Test fit/predict on synthetic sinusoidal data.
   - Test cross-validation returns valid metrics.
   - Test forecast horizon matches requested periods.

---

## Phase 3: ARIMA Forecasting Model

**Goal:** Implement ARIMA/SARIMAX forecasting as an alternative model for comparison.

### Tasks

1. Create `src/models/arima_forecaster.py`:
   - Class `ARIMAForecaster`:
     - Constructor: accepts order (p,d,q), seasonal_order (P,D,Q,s).
     - Method: `fit(series: pd.Series)`: fits SARIMAX model using statsmodels.
     - Method: `predict(steps: int) -> pd.DataFrame`: returns forecast with predicted values and confidence intervals.
     - Method: `get_diagnostics() -> dict`: returns AIC, BIC, Ljung-Box p-value, residual normality test.

2. Create `src/models/arima_tuner.py`:
   - Function: `auto_arima(series, max_p=5, max_d=2, max_q=5, seasonal=True, m=52) -> tuple`:
     - Implements grid search or stepwise selection over (p,d,q)(P,D,Q,s) orders.
     - Uses AIC for model selection.
     - Returns best order, seasonal_order, and fitted model.
   - Function: `check_stationarity(series) -> dict`:
     - ADF test (augmented Dickey-Fuller).
     - KPSS test.
     - Returns test statistics, p-values, and recommendation (differencing needed).

3. Create `src/pipeline/arima_runner.py`:
   - Function: `run_arima_forecast(sales_df, segments, config) -> dict[str, pd.DataFrame]`:
     - For each segment: extract series, check stationarity, auto-select order, fit, predict.
     - Returns dict mapping segment to forecast DataFrame.

4. Create `tests/test_arima_forecaster.py`:
   - Test fit/predict on synthetic AR(1) data.
   - Test stationarity check on stationary vs non-stationary series.
   - Test auto_arima selects reasonable order.

---

## Phase 4: Model Comparison and Selection

**Goal:** Compare Prophet and ARIMA forecasts, select the best model per segment.

### Tasks

1. Create `src/evaluation/metrics.py`:
   - Function: `compute_metrics(actual, predicted) -> dict`:
     - Returns: MAPE, RMSE, MAE, SMAPE, R-squared.
   - Function: `compute_coverage(actual, lower, upper) -> float`:
     - Percentage of actual values within prediction interval.

2. Create `src/evaluation/model_comparator.py`:
   - Class `ModelComparator`:
     - Method: `compare(actuals, prophet_forecast, arima_forecast) -> pd.DataFrame`:
       - Computes metrics for both models on holdout period.
       - Returns comparison DataFrame with columns: segment, model, mape, rmse, mae, coverage.
     - Method: `select_best(comparison_df, metric='mape') -> dict[str, str]`:
       - Returns dict mapping segment -> best model name.

3. Create `src/pipeline/model_selection_pipeline.py`:
   - Function: `run_model_selection(sales_df, segments, config) -> dict`:
     - Split data into train (e.g., all but last 12 weeks) and holdout.
     - Run Prophet and ARIMA on train for each segment.
     - Predict on holdout period.
     - Compare models, select best per segment.
     - Refit best model on full data and generate final forecast.
     - Returns: { segment: { model, forecast_df, metrics } }.

4. Create `tests/test_model_comparator.py`:
   - Test with known forecasts where one model is clearly better.
   - Test select_best returns correct model.

---

## Phase 5: Business Headroom Sizing Model

**Goal:** Quantify addressable opportunity (headroom) per segment by comparing forecasted sales against total market size.

### Tasks

1. Create `src/sizing/headroom_model.py`:
   - Class `HeadroomModel`:
     - Method: `compute_headroom(forecast_df, market_data_df) -> pd.DataFrame`:
       - For each segment: current_revenue (last period actual), forecasted_revenue (forecast at horizon), total_market_size (from market data), current_share = current_revenue / total_market_size, forecasted_share = forecasted_revenue / total_market_size, headroom = total_market_size - forecasted_revenue, headroom_pct = headroom / total_market_size.
       - Returns DataFrame with all sizing metrics per segment.
     - Method: `compute_growth_scenarios(headroom_df, scenarios: dict) -> pd.DataFrame`:
       - Scenarios: {"conservative": 0.05, "moderate": 0.10, "aggressive": 0.20} (share gain percentages).
       - For each scenario: incremental_revenue = total_market * share_gain.
       - Returns DataFrame with segment, scenario, incremental_revenue, new_share.

2. Create `src/sizing/segment_prioritizer.py`:
   - Function: `prioritize_segments(headroom_df) -> pd.DataFrame`:
     - Score segments by: headroom_size (40% weight), market_growth_rate (30%), current_share (inverse, 30%).
     - Rank segments by composite score.
     - Returns DataFrame sorted by priority.

3. Create `tests/test_headroom_model.py`:
   - Test headroom computation with known values.
   - Test scenario calculation.
   - Test segment prioritization ordering.

---

## Phase 6: Visualization Dashboards

**Goal:** Build Seaborn/Matplotlib visualizations for forecasts, model comparison, and business sizing.

### Tasks

1. Create `src/visualization/forecast_plots.py`:
   - Function: `plot_forecast(actual_df, forecast_df, segment, model_name, output_path)`:
     - Line plot: actual (solid blue), forecast (dashed orange), confidence interval (shaded).
     - Title, axis labels, legend.
     - Save to output_path as PNG (300 DPI).
   - Function: `plot_forecast_comparison(actual_df, prophet_forecast, arima_forecast, segment, output_path)`:
     - Overlay both model forecasts on same plot.
     - Highlight holdout period.
   - Function: `plot_components(prophet_model, output_path)`:
     - Prophet component decomposition: trend, yearly, weekly seasonality.

2. Create `src/visualization/sizing_plots.py`:
   - Function: `plot_headroom_waterfall(headroom_df, output_path)`:
     - Horizontal bar chart: current revenue, headroom, total market per segment.
   - Function: `plot_market_share_comparison(headroom_df, output_path)`:
     - Grouped bar chart: current share vs forecasted share per segment.
   - Function: `plot_scenario_analysis(scenarios_df, output_path)`:
     - Stacked bar chart showing incremental revenue by scenario per segment.
   - Function: `plot_segment_priority_matrix(priority_df, output_path)`:
     - Scatter plot: x = market_growth_rate, y = headroom_size, size = total_market, color = priority_score.

3. Create `src/visualization/dashboard.py`:
   - Function: `generate_dashboard(results: dict, output_dir: str)`:
     - Calls all plot functions.
     - Generates a multi-page PDF or directory of PNGs.
     - Creates summary table as CSV.

4. Create `src/visualization/style.py`:
   - Set Seaborn theme: `sns.set_theme(style="whitegrid", palette="deep")`.
   - Define color constants for consistent branding.
   - Set matplotlib rcParams for font size, figure size defaults.

---

## Phase 7: Databricks Pipeline Orchestration

**Goal:** Automate the full pipeline on Databricks with scheduled jobs.

### Tasks

1. Create `notebooks/01_data_ingestion.py`:
   - Databricks notebook that runs DataLoader to pull latest data.
   - Writes cleaned data to Delta Lake table: `analytics.sales_history_clean`.
   - Logs row counts and date ranges.

2. Create `notebooks/02_forecasting.py`:
   - Loads data from Delta Lake.
   - Runs model selection pipeline.
   - Writes forecasts to Delta Lake: `analytics.sales_forecasts`.
   - Writes model metrics to: `analytics.model_metrics`.

3. Create `notebooks/03_business_sizing.py`:
   - Loads forecasts and market data.
   - Runs headroom model and scenario analysis.
   - Writes results to: `analytics.headroom_results`, `analytics.scenario_analysis`.

4. Create `notebooks/04_visualization.py`:
   - Generates all plots.
   - Saves to DBFS or cloud storage.
   - Displays inline in notebook for interactive review.

5. Create `databricks/job_config.json`:
   - Databricks job definition:
     - Task chain: 01 -> 02 -> 03 -> 04.
     - Schedule: weekly (Monday 6 AM).
     - Cluster config: single-node, ML runtime.
     - Email notifications on failure.

6. Create `src/utils/logger.py`:
   - Configured Python logger with both console and file handlers.
   - Structured format: timestamp, level, module, message.

7. Create `requirements.txt`:
   - All Python dependencies with pinned versions.
   - Prophet, statsmodels, pandas, numpy, seaborn, matplotlib, scikit-learn, pyyaml, holidays, pytest.
