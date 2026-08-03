# Forecast Lab

Time-series forecasting pipeline combining Prophet and Auto-ARIMA with inverse-MAE ensemble weighting, walk-forward backtesting, MLflow experiment tracking, and end-to-end Databricks notebooks.

## Overview

- **Prophet** — additive/multiplicative seasonality, changepoint detection, uncertainty intervals, Prophet cross-validation diagnostics
- **Auto-ARIMA** — `pmdarima.auto_arima` stepwise search over SARIMA(p,d,q)(P,D,Q,m) with AIC selection
- **Ensemble** — inverse-MAE weighted average; weights updated from validation split
- **Walk-forward backtesting** — expanding-window evaluation with configurable initial/period/horizon
- **Databricks pipeline** — 4 notebooks: data prep → Prophet training → ARIMA training → model comparison, all writing to Delta Lake tables
- **MLflow tracking** — params, metrics (MAE, RMSE, MAPE, sMAPE), model artifacts, cross-validation results

## Tech Stack

Python 3.11 · Prophet · statsmodels · pmdarima · MLflow · Databricks (PySpark + Delta Lake) · pandas · pytest

## Quickstart

```bash
pip install -r requirements.txt

# Run full pipeline locally (uses synthetic data by default)
python -m src.pipeline.run_pipeline --config configs/config.yaml

# Run tests
pytest tests/
```

## Databricks

Upload and run notebooks in order:
1. `databricks/notebooks/01_data_prep.py` — clean raw Delta table
2. `databricks/notebooks/02_prophet_training.py` — train + cross-validate Prophet
3. `databricks/notebooks/03_arima_training.py` — auto-ARIMA fit
4. `databricks/notebooks/04_compare_models.py` — side-by-side metrics table

Configure the experiment path `/forecast_lab/*` in your Databricks MLflow workspace.

## Architecture

```
load_csv / Databricks Delta
    → train_test_split
    → ProphetForecaster.fit()   ┐
    → ARIMAForecaster.fit()     ├── EnsembleForecaster (inverse-MAE weights)
    → evaluate() → MAE/RMSE/MAPE/sMAPE
    → walk_forward_validation() → backtest CSV
    → MLflow: params + metrics + model artifacts
```

## Evaluation

Forecasting metrics are implemented in `src/evaluation/metrics.py` and consumed from `src/evaluation/backtesting.py`.

| Metric | Where | How to compute |
|--------|-------|----------------|
| MAE | `metrics.py` | `mae(y_true, y_pred)` — mean absolute error on the holdout split |
| RMSE | `metrics.py` | `rmse(y_true, y_pred)` |
| MAPE | `metrics.py` | `mape(y_true, y_pred)` — % error; undefined when `y_true=0` is filtered |
| sMAPE | `metrics.py` | `smape(y_true, y_pred)` — symmetric variant, bounded |
| Prediction-interval coverage | `backtesting.py` | Fraction of `y_true` falling inside Prophet's `[yhat_lower, yhat_upper]` over the walk-forward fold; nominal level configured per fit |

Standard splits:

- **Hold-out**: `train_test_split` in `src/data/loader.py` (configurable horizon).
- **Walk-forward**: `walk_forward_validation()` runs an expanding window with `initial`, `period`, `horizon` from `configs/config.yaml` and writes per-fold metrics to a backtest CSV.

To run the full evaluation locally:

```bash
python -m src.pipeline.run_pipeline --config configs/config.yaml
# Per-fold metrics: artifacts/backtest.csv
# Aggregated metrics + model artifacts: MLflow run under experiment "forecast_lab"
mlflow ui   # inspect MAE/RMSE/MAPE/sMAPE side by side for Prophet, ARIMA, ensemble
```

On Databricks, notebook `04_compare_models.py` produces the side-by-side metric table per model written to a Delta table.

## Project Structure

```
forecast_lab/
├── src/
│   ├── data/           # loader.py (CSV, Databricks, synthetic)
│   ├── models/         # prophet_model.py, arima_model.py, ensemble.py
│   ├── evaluation/     # metrics.py, backtesting.py
│   └── pipeline/       # run_pipeline.py
├── databricks/
│   └── notebooks/      # 01–04 PySpark notebooks
├── configs/            # config.yaml
└── tests/              # pytest suite
```
