"""Backtest command handlers."""

from __future__ import annotations

from typing import Any

from stock_ai.utils import ConfigError


def run_backtest_command(
    backtest_config_name: str,
    train_config_name: str,
    dataset_input_path: str | None = None,
    model_input_path: str | None = None,
) -> dict[str, Any]:
    """Run a real backtest using the trained model and processed dataset."""
    try:
        from stock_ai.backtest.run import run_backtest
    except ImportError as exc:
        raise ConfigError("Backtest dependencies are missing. Install project dependencies first.") from exc

    result = run_backtest(
        backtest_config_name=backtest_config_name,
        train_config_name=train_config_name,
        dataset_input_path=dataset_input_path,
        model_input_path=model_input_path,
    )
    return {
        "command": "backtest run",
        "dataset_input_path": str(result.dataset_input_path),
        "model_input_path": str(result.model_input_path),
        "report_output_path": str(result.report_output_path),
        "trades_output_path": str(result.trades_output_path),
        "equity_output_path": str(result.equity_output_path),
        "metadata_path": str(result.metadata_path),
        "rebalance_count": result.rebalance_count,
        "trade_count": result.trade_count,
    }


def run_walk_forward_command(
    backtest_config_name: str,
    train_config_name: str,
    dataset_input_path: str | None = None,
) -> dict[str, Any]:
    """Run walk-forward evaluation with repeated retraining."""
    try:
        from stock_ai.backtest.run import run_walk_forward_backtest
    except ImportError as exc:
        raise ConfigError("Backtest dependencies are missing. Install project dependencies first.") from exc

    result = run_walk_forward_backtest(
        backtest_config_name=backtest_config_name,
        train_config_name=train_config_name,
        dataset_input_path=dataset_input_path,
    )
    return {
        "command": "backtest walk-forward",
        "dataset_input_path": str(result.dataset_input_path),
        "report_output_path": str(result.report_output_path),
        "trades_output_path": str(result.trades_output_path),
        "equity_output_path": str(result.equity_output_path),
        "metadata_path": str(result.metadata_path),
        "rebalance_count": result.rebalance_count,
        "trade_count": result.trade_count,
        "trained_window_count": result.trained_window_count,
    }
