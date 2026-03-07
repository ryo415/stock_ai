"""Backtest utilities for model-driven portfolio simulation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from stock_ai.modeling.train import build_training_pipeline, get_numeric_feature_columns, prepare_feature_matrix
from stock_ai.utils import ConfigError, load_config
from stock_ai.utils.io import ensure_directory, project_path, timestamp_for_filename, write_json_file


@dataclass(frozen=True)
class BacktestResult:
    """Result of a backtest run."""

    report_output_path: Path
    trades_output_path: Path
    equity_output_path: Path
    metadata_path: Path
    dataset_input_path: Path
    model_input_path: Path
    rebalance_count: int
    trade_count: int


@dataclass(frozen=True)
class WalkForwardResult:
    """Result of a walk-forward evaluation run."""

    report_output_path: Path
    trades_output_path: Path
    equity_output_path: Path
    metadata_path: Path
    dataset_input_path: Path
    rebalance_count: int
    trade_count: int
    trained_window_count: int


def _latest_file(directory: Path, pattern: str) -> Path:
    files = sorted(directory.glob(pattern))
    if not files:
        raise ConfigError(f"No files found in {directory} matching {pattern}")
    return files[-1]


def _resolve_input(input_path: str | None, default_dir: str, pattern: str) -> Path:
    if input_path:
        path = Path(input_path)
        if not path.is_absolute():
            path = project_path(path)
        if not path.exists():
            raise ConfigError(f"Input file not found: {path}")
        return path
    return _latest_file(project_path(default_dir), pattern)


def _load_market_benchmark_returns(
    benchmark_name: str,
    holding_period_business_days: int,
) -> dict[pd.Timestamp, float]:
    market_path = _latest_file(project_path("data/interim/market"), "market_normalized_*.csv")
    market = pd.read_csv(market_path)
    market["date"] = pd.to_datetime(market["date"], errors="coerce")
    market = market[market["series_name"] == benchmark_name].copy()
    if market.empty:
        raise ConfigError(f"Benchmark series not found in market data: {benchmark_name}")

    market.sort_values("date", inplace=True)
    market["close"] = pd.to_numeric(market["close"], errors="coerce")
    market["benchmark_exit_close"] = market["close"].shift(-holding_period_business_days)
    market["benchmark_forward_return"] = market["benchmark_exit_close"] / market["close"] - 1.0
    market = market.dropna(subset=["date", "benchmark_forward_return"])
    return {
        pd.Timestamp(row["date"]): float(row["benchmark_forward_return"])
        for _, row in market.iterrows()
    }


def _select_rebalance_dates(
    dates: list[pd.Timestamp],
    frequency: str,
    holding_period_business_days: int,
    allow_overlap_positions: bool,
) -> list[pd.Timestamp]:
    date_index = pd.Index(sorted(dates))
    if frequency != "monthly":
        raise ConfigError(f"Unsupported rebalance_frequency: {frequency}")

    candidates = (
        pd.Series(date_index, index=date_index)
        .groupby([date_index.year, date_index.month])
        .min()
        .tolist()
    )
    if allow_overlap_positions:
        return candidates

    selected: list[pd.Timestamp] = []
    next_allowed_idx = 0
    for rebalance_date in candidates:
        date_pos = int(date_index.get_indexer([rebalance_date])[0])
        if date_pos < next_allowed_idx:
            continue
        selected.append(rebalance_date)
        next_allowed_idx = date_pos + holding_period_business_days
    return selected


def _max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = (equity_curve / running_max) - 1.0
    return float(drawdown.min())


def _summarize_backtest(equity_frame: pd.DataFrame, trades_frame: pd.DataFrame) -> dict[str, float | int]:
    benchmark_curve = (1.0 + equity_frame["benchmark_return"].fillna(0.0)).cumprod()
    return {
        "total_return": float(equity_frame["portfolio_value"].iloc[-1] - 1.0),
        "benchmark_total_return": float(benchmark_curve.iloc[-1] - 1.0),
        "rebalance_count": int(len(equity_frame)),
        "trade_count": int(len(trades_frame)),
        "avg_portfolio_return": float(equity_frame["portfolio_return"].mean()),
        "win_rate": float((trades_frame["net_return"] > 0).mean()) if not trades_frame.empty else float("nan"),
        "max_drawdown": _max_drawdown(equity_frame["portfolio_value"]),
    }


def _save_backtest_outputs(
    command_name: str,
    backtest_name: str,
    timestamp: str,
    payload: dict[str, Any],
    trades_frame: pd.DataFrame,
    equity_frame: pd.DataFrame,
) -> tuple[Path, Path, Path, Path]:
    trades_output_path = project_path(f"reports/tables/backtest_{backtest_name}_trades_{timestamp}.csv")
    equity_output_path = project_path(f"reports/tables/backtest_{backtest_name}_equity_{timestamp}.csv")
    report_output_path = project_path(f"reports/tables/backtest_{backtest_name}_{timestamp}.json")
    metadata_path = project_path(f"data/metadata/backtest_{backtest_name}_{timestamp}.json")

    ensure_directory(trades_output_path.parent)
    trades_frame.to_csv(trades_output_path, index=False)
    equity_frame.to_csv(equity_output_path, index=False)

    report_payload = {
        "command": command_name,
        "timestamp_utc": timestamp,
        **payload,
        "trades_output_path": str(trades_output_path),
        "equity_output_path": str(equity_output_path),
    }
    write_json_file(report_output_path, report_payload)
    write_json_file(metadata_path, report_payload)
    return report_output_path, trades_output_path, equity_output_path, metadata_path


def _select_top_n_portfolio(
    scored: pd.DataFrame,
    rebalance_dates: list[pd.Timestamp],
    top_n: int,
    round_trip_cost: float,
    benchmark_lookup: dict[pd.Timestamp, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trades: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []
    portfolio_value = 1.0

    for rebalance_date in rebalance_dates:
        day_slice = scored[scored["date"] == rebalance_date].sort_values("probability", ascending=False)
        selected = day_slice.head(top_n).copy()
        if selected.empty:
            continue

        selected["gross_return"] = pd.to_numeric(selected["future_return_60bd"], errors="coerce")
        selected = selected.dropna(subset=["gross_return"])
        if selected.empty:
            continue

        selected["net_return"] = selected["gross_return"] - round_trip_cost
        portfolio_return = float(selected["net_return"].mean())
        portfolio_value *= 1.0 + portfolio_return
        benchmark_return = benchmark_lookup.get(pd.Timestamp(rebalance_date), np.nan)

        equity_rows.append(
            {
                "rebalance_date": str(pd.Timestamp(rebalance_date).date()),
                "portfolio_return": portfolio_return,
                "benchmark_return": benchmark_return,
                "portfolio_value": portfolio_value,
                "selected_count": int(len(selected)),
            }
        )

        for _, row in selected.iterrows():
            trades.append(
                {
                    "rebalance_date": str(pd.Timestamp(rebalance_date).date()),
                    "ticker": row["ticker"],
                    "probability": float(row["probability"]),
                    "gross_return": float(row["gross_return"]),
                    "net_return": float(row["net_return"]),
                    "label": int(row["label"]),
                }
            )

    if not equity_rows:
        raise ConfigError("Backtest produced no rebalance points. Check dataset dates and config.")

    return pd.DataFrame(trades), pd.DataFrame(equity_rows)


def run_backtest(
    backtest_config_name: str,
    train_config_name: str,
    dataset_input_path: str | None = None,
    model_input_path: str | None = None,
) -> BacktestResult:
    """Run a simple top-N backtest using model probabilities."""
    backtest_config = load_config("backtest", backtest_config_name)
    _ = load_config("train", train_config_name)

    resolved_dataset_path = _resolve_input(dataset_input_path, "data/processed/datasets", "dataset_*.csv")
    resolved_model_path = _resolve_input(model_input_path, "models", "*.joblib")

    artifact = joblib.load(resolved_model_path)
    pipeline = artifact.get("pipeline")
    feature_columns = artifact.get("feature_columns")
    if pipeline is None or not feature_columns:
        raise ConfigError(f"Model artifact is missing required fields: {resolved_model_path}")

    dataset = pd.read_csv(resolved_dataset_path)
    dataset["date"] = pd.to_datetime(dataset["date"], errors="coerce")
    dataset.sort_values(["date", "ticker"], inplace=True)

    missing_features = [column for column in feature_columns if column not in dataset.columns]
    if missing_features:
        raise ConfigError(f"Dataset is missing model feature columns: {missing_features}")

    x = prepare_feature_matrix(dataset, feature_columns)
    probabilities = pipeline.predict_proba(x)[:, 1]
    scored = dataset.copy()
    scored["probability"] = probabilities

    config = backtest_config["backtest"]
    portfolio_config = backtest_config["portfolio"]
    costs_config = backtest_config["costs"]
    execution_config = backtest_config["execution"]

    holding_period_business_days = int(config["holding_period_business_days"])
    benchmark_name = str(config.get("benchmark", "topix"))
    top_n = int(portfolio_config["top_n"])
    allow_overlap_positions = bool(execution_config.get("allow_overlap_positions", False))
    rebalance_dates = _select_rebalance_dates(
        dates=scored["date"].dropna().unique().tolist(),
        frequency=str(config["rebalance_frequency"]),
        holding_period_business_days=holding_period_business_days,
        allow_overlap_positions=allow_overlap_positions,
    )

    fee_bps = float(costs_config.get("fee_bps", 0.0))
    slippage_bps = float(costs_config.get("slippage_bps", 0.0))
    round_trip_cost = 2.0 * (fee_bps + slippage_bps) / 10000.0

    benchmark_lookup = _load_market_benchmark_returns(benchmark_name, holding_period_business_days)
    trades_frame, equity_frame = _select_top_n_portfolio(
        scored=scored,
        rebalance_dates=rebalance_dates,
        top_n=top_n,
        round_trip_cost=round_trip_cost,
        benchmark_lookup=benchmark_lookup,
    )
    summary = _summarize_backtest(equity_frame, trades_frame)

    timestamp = timestamp_for_filename()
    backtest_name = backtest_config["backtest"]["name"]
    report_output_path, trades_output_path, equity_output_path, metadata_path = _save_backtest_outputs(
        command_name="backtest run",
        backtest_name=backtest_name,
        timestamp=timestamp,
        payload={
            "backtest_config_name": backtest_config_name,
            "train_config_name": train_config_name,
            "dataset_input_path": str(resolved_dataset_path),
            "model_input_path": str(resolved_model_path),
            "summary": summary,
        },
        trades_frame=trades_frame,
        equity_frame=equity_frame,
    )

    return BacktestResult(
        report_output_path=report_output_path,
        trades_output_path=trades_output_path,
        equity_output_path=equity_output_path,
        metadata_path=metadata_path,
        dataset_input_path=resolved_dataset_path,
        model_input_path=resolved_model_path,
        rebalance_count=int(len(equity_frame)),
        trade_count=int(len(trades_frame)),
    )


def run_walk_forward_backtest(
    backtest_config_name: str,
    train_config_name: str,
    dataset_input_path: str | None = None,
) -> WalkForwardResult:
    """Run expanding-window walk-forward evaluation with monthly retraining."""
    backtest_config = load_config("backtest", backtest_config_name)
    train_config = load_config("train", train_config_name)

    resolved_dataset_path = _resolve_input(dataset_input_path, "data/processed/datasets", "dataset_*.csv")
    dataset = pd.read_csv(resolved_dataset_path)
    dataset["date"] = pd.to_datetime(dataset["date"], errors="coerce")
    dataset.sort_values(["date", "ticker"], inplace=True)

    config = backtest_config["backtest"]
    portfolio_config = backtest_config["portfolio"]
    costs_config = backtest_config["costs"]
    execution_config = backtest_config["execution"]
    walk_forward_config = backtest_config.get("walk_forward", {})

    holding_period_business_days = int(config["holding_period_business_days"])
    benchmark_name = str(config.get("benchmark", "topix"))
    top_n = int(portfolio_config["top_n"])
    allow_overlap_positions = bool(execution_config.get("allow_overlap_positions", False))
    prediction_start_date = pd.to_datetime(
        walk_forward_config.get("prediction_start_date", train_config["dataset"]["test_start_date"])
    )
    training_start_date = pd.to_datetime(
        walk_forward_config.get("training_start_date", train_config["dataset"]["train_start_date"])
    )
    min_training_rows = int(walk_forward_config.get("min_training_rows", 252))

    candidate_dates = dataset[dataset["date"] >= prediction_start_date]["date"].dropna().unique().tolist()
    rebalance_dates = _select_rebalance_dates(
        dates=candidate_dates,
        frequency=str(config["rebalance_frequency"]),
        holding_period_business_days=holding_period_business_days,
        allow_overlap_positions=allow_overlap_positions,
    )

    fee_bps = float(costs_config.get("fee_bps", 0.0))
    slippage_bps = float(costs_config.get("slippage_bps", 0.0))
    round_trip_cost = 2.0 * (fee_bps + slippage_bps) / 10000.0
    benchmark_lookup = _load_market_benchmark_returns(benchmark_name, holding_period_business_days)

    scored_rows: list[pd.DataFrame] = []
    training_windows: list[dict[str, Any]] = []

    for rebalance_date in rebalance_dates:
        train_window = dataset[(dataset["date"] >= training_start_date) & (dataset["date"] < rebalance_date)].copy()
        if len(train_window) < min_training_rows:
            continue
        if train_window["label"].nunique(dropna=True) < 2:
            continue

        feature_columns = get_numeric_feature_columns(train_window)
        feature_columns = [column for column in feature_columns if prepare_feature_matrix(train_window, [column])[column].notna().any()]
        if not feature_columns:
            continue

        pipeline = build_training_pipeline(train_config)
        x_train = prepare_feature_matrix(train_window, feature_columns)
        y_train = train_window["label"].astype(int)
        pipeline.fit(x_train, y_train)

        day_slice = dataset[dataset["date"] == rebalance_date].copy()
        if day_slice.empty:
            continue
        if any(column not in day_slice.columns for column in feature_columns):
            continue

        x_day = prepare_feature_matrix(day_slice, feature_columns)
        day_slice["probability"] = pipeline.predict_proba(x_day)[:, 1]
        day_slice["feature_count"] = len(feature_columns)
        day_slice["train_row_count"] = len(train_window)
        scored_rows.append(day_slice)
        training_windows.append(
            {
                "rebalance_date": str(pd.Timestamp(rebalance_date).date()),
                "train_start_date": str(pd.Timestamp(train_window["date"].min()).date()),
                "train_end_date": str(pd.Timestamp(train_window["date"].max()).date()),
                "train_row_count": int(len(train_window)),
                "feature_count": int(len(feature_columns)),
            }
        )

    if not scored_rows:
        raise ConfigError("Walk-forward evaluation produced no trained windows. Check config and dataset history.")

    scored = pd.concat(scored_rows, ignore_index=True)
    selected_rebalance_dates = sorted(pd.to_datetime(scored["date"], errors="coerce").dropna().unique().tolist())
    trades_frame, equity_frame = _select_top_n_portfolio(
        scored=scored,
        rebalance_dates=selected_rebalance_dates,
        top_n=top_n,
        round_trip_cost=round_trip_cost,
        benchmark_lookup=benchmark_lookup,
    )
    summary = _summarize_backtest(equity_frame, trades_frame)
    summary["trained_window_count"] = int(len(training_windows))

    timestamp = timestamp_for_filename()
    backtest_name = f'{backtest_config["backtest"]["name"]}_walk_forward'
    report_output_path, trades_output_path, equity_output_path, metadata_path = _save_backtest_outputs(
        command_name="backtest walk-forward",
        backtest_name=backtest_name,
        timestamp=timestamp,
        payload={
            "backtest_config_name": backtest_config_name,
            "train_config_name": train_config_name,
            "dataset_input_path": str(resolved_dataset_path),
            "summary": summary,
            "walk_forward": {
                "prediction_start_date": str(prediction_start_date.date()),
                "training_start_date": str(training_start_date.date()),
                "min_training_rows": min_training_rows,
                "trained_windows": training_windows,
            },
        },
        trades_frame=trades_frame,
        equity_frame=equity_frame,
    )

    return WalkForwardResult(
        report_output_path=report_output_path,
        trades_output_path=trades_output_path,
        equity_output_path=equity_output_path,
        metadata_path=metadata_path,
        dataset_input_path=resolved_dataset_path,
        rebalance_count=int(len(equity_frame)),
        trade_count=int(len(trades_frame)),
        trained_window_count=int(len(training_windows)),
    )
