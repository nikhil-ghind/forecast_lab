import argparse
import os
import yaml
import mlflow
import pandas as pd

from src.data.loader import load_csv, generate_synthetic, train_test_split
from src.models.prophet_model import ProphetForecaster
from src.models.arima_model import ARIMAForecaster
from src.models.ensemble import EnsembleForecaster
from src.evaluation.metrics import evaluate
from src.evaluation.backtesting import walk_forward_validation, summarise_backtest


def main(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if cfg.get("use_synthetic"):
        df = generate_synthetic(n=cfg.get("synthetic_days", 730))
    else:
        df = load_csv(cfg["data_path"], cfg.get("date_col", "ds"), cfg.get("value_col", "y"))

    train, test = train_test_split(df, test_periods=cfg.get("test_periods", 90))
    os.makedirs(cfg.get("output_dir", "outputs"), exist_ok=True)

    mlflow.set_experiment(cfg.get("experiment_name", "forecast_lab"))

    prophet = ProphetForecaster(
        seasonality_mode=cfg.get("prophet_seasonality_mode", "multiplicative"),
        changepoint_prior_scale=cfg.get("changepoint_prior_scale", 0.05),
    )
    arima = ARIMAForecaster(auto=True)

    with mlflow.start_run(run_name="ensemble_pipeline"):
        mlflow.log_params(cfg)

        ensemble = EnsembleForecaster(
            forecasters={"prophet": prophet, "arima": arima},
        )
        ensemble.fit(train)
        weights = ensemble.update_weights_from_validation(test)
        mlflow.log_params({f"weight_{k}": v for k, v in weights.items()})

        forecast = ensemble.predict(len(test))
        metrics = evaluate(test["y"], forecast["yhat"])
        mlflow.log_metrics(metrics)

        print(f"\n=== Ensemble Metrics ===")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        if cfg.get("run_backtest", True):
            bt = walk_forward_validation(
                df,
                model_factory=lambda: ProphetForecaster(),
                initial_train=cfg.get("backtest_initial", 365),
                horizon=cfg.get("test_periods", 30),
                step=cfg.get("backtest_step", 30),
            )
            summary = summarise_backtest(bt)
            mlflow.log_metrics({f"backtest_{k}": v for k, v in summary.items()})
            bt.to_csv(f"{cfg.get('output_dir', 'outputs')}/backtest_results.csv", index=False)
            print(f"\n=== Backtest Summary ===")
            for k, v in summary.items():
                print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    main(parser.parse_args())
