"""Common CLI entry point for the stock_ai project."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from stock_ai.backtest.commands import run_backtest_command, run_walk_forward_command
from stock_ai.data.commands import (
    run_build_universe_command,
    run_fetch_command,
    run_fetch_fundamentals_command,
    run_fetch_macro_command,
    run_fetch_prices_command,
    run_normalize_fundamentals_command,
    run_normalize_macro_command,
    run_normalize_prices_command,
)
from stock_ai.features.commands import run_build_dataset_command
from stock_ai.features.commands import run_build_labels_command
from stock_ai.inference.commands import run_predict_command
from stock_ai.modeling.commands import run_train_command
from stock_ai.reporting.commands import run_compare_models_command, run_evaluate_prediction_command
from stock_ai.utils import ConfigError, get_configs_root, list_config_files, load_config


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser."""
    parser = argparse.ArgumentParser(
        prog="stock_ai",
        description="Common CLI for stock_ai project tasks.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    config_parser = subparsers.add_parser("config", help="Inspect project config files.")
    config_subparsers = config_parser.add_subparsers(dest="config_command", required=True)

    config_list_parser = config_subparsers.add_parser("list", help="List config files.")
    config_list_parser.add_argument(
        "section",
        nargs="?",
        help="Optional config section such as data, features, train, inference, or backtest.",
    )
    config_list_parser.set_defaults(handler=handle_config_list)

    config_show_parser = config_subparsers.add_parser("show", help="Show a config file.")
    config_show_parser.add_argument("section", help="Config section name.")
    config_show_parser.add_argument("name", help="Config file name without .yaml suffix.")
    config_show_parser.set_defaults(handler=handle_config_show)

    data_parser = subparsers.add_parser("data", help="Run data pipeline commands.")
    data_subparsers = data_parser.add_subparsers(dest="data_command", required=True)

    fetch_prices_parser = data_subparsers.add_parser(
        "fetch-prices",
        help="Fetch and persist daily price data.",
    )
    fetch_prices_parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Optional ticker override list such as 7203.T 6758.T.",
    )
    fetch_prices_parser.add_argument("--start-date", default=None, help="Override start date YYYY-MM-DD.")
    fetch_prices_parser.add_argument("--end-date", default=None, help="Override end date YYYY-MM-DD.")
    fetch_prices_parser.set_defaults(handler=handle_data_fetch_prices)

    fetch_fundamentals_parser = data_subparsers.add_parser(
        "fetch-fundamentals",
        help="Fetch and persist raw EDINET fundamentals documents.",
    )
    fetch_fundamentals_parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Optional ticker override list such as 7203.T 6758.T.",
    )
    fetch_fundamentals_parser.add_argument(
        "--start-date",
        default=None,
        help="Override start date YYYY-MM-DD.",
    )
    fetch_fundamentals_parser.add_argument(
        "--end-date",
        default=None,
        help="Override end date YYYY-MM-DD.",
    )
    fetch_fundamentals_parser.set_defaults(handler=handle_data_fetch_fundamentals)

    fetch_macro_parser = data_subparsers.add_parser(
        "fetch-macro",
        help="Fetch and persist raw market and macro series.",
    )
    fetch_macro_parser.add_argument("--start-date", default=None, help="Override start date YYYY-MM-DD.")
    fetch_macro_parser.add_argument("--end-date", default=None, help="Override end date YYYY-MM-DD.")
    fetch_macro_parser.set_defaults(handler=handle_data_fetch_macro)

    build_universe_parser = data_subparsers.add_parser(
        "build-universe",
        help="Build a liquidity-filtered investable universe from price history.",
    )
    build_universe_parser.add_argument(
        "--input-path",
        default=None,
        help="Optional normalized/raw price CSV path. Defaults to the latest interim or raw price file.",
    )
    build_universe_parser.set_defaults(handler=handle_data_build_universe)

    normalize_prices_parser = data_subparsers.add_parser(
        "normalize-prices",
        help="Normalize raw daily price data into data/interim.",
    )
    normalize_prices_parser.add_argument(
        "--input-path",
        default=None,
        help="Optional raw price CSV path. Defaults to the latest prices_yfinance file.",
    )
    normalize_prices_parser.set_defaults(handler=handle_data_normalize_prices)

    normalize_fundamentals_parser = data_subparsers.add_parser(
        "normalize-fundamentals",
        help="Normalize raw fundamentals summary into data/interim.",
    )
    normalize_fundamentals_parser.add_argument(
        "--input-path",
        default=None,
        help="Optional fundamentals summary JSON path. Defaults to the latest summary file.",
    )
    normalize_fundamentals_parser.set_defaults(handler=handle_data_normalize_fundamentals)

    normalize_macro_parser = data_subparsers.add_parser(
        "normalize-macro",
        help="Normalize raw market data into data/interim.",
    )
    normalize_macro_parser.add_argument(
        "--input-path",
        default=None,
        help="Optional raw market CSV path. Defaults to the latest file in data/raw/market/.",
    )
    normalize_macro_parser.set_defaults(handler=handle_data_normalize_macro)

    features_parser = subparsers.add_parser("features", help="Run feature pipeline commands.")
    features_subparsers = features_parser.add_subparsers(dest="features_command", required=True)
    build_labels_parser = features_subparsers.add_parser(
        "build-labels",
        help="Build future_return_60bd and label columns from normalized prices.",
    )
    build_labels_parser.add_argument(
        "--input-path",
        default=None,
        help="Optional normalized price CSV path. Defaults to the latest file in data/interim/prices/.",
    )
    build_labels_parser.add_argument(
        "--label-config",
        default="labels",
        help="Label config name under configs/features/ without suffix.",
    )
    build_labels_parser.set_defaults(handler=handle_build_labels)

    build_dataset_parser = features_subparsers.add_parser(
        "build-dataset", help="Build a processed dataset from features and labels."
    )
    build_dataset_parser.add_argument(
        "--feature-config",
        default="feature_set_baseline",
        help="Feature config name under configs/features/ without suffix.",
    )
    build_dataset_parser.add_argument(
        "--label-config",
        default="labels",
        help="Label config name under configs/features/ without suffix.",
    )
    build_dataset_parser.add_argument(
        "--price-input-path",
        default=None,
        help="Optional normalized price CSV path. Defaults to the latest file in data/interim/prices/.",
    )
    build_dataset_parser.add_argument(
        "--label-input-path",
        default=None,
        help="Optional label CSV path. Defaults to the latest file in data/processed/labels/.",
    )
    build_dataset_parser.add_argument(
        "--macro-input-path",
        default=None,
        help="Optional normalized market CSV path. Defaults to the latest file in data/interim/market/.",
    )
    build_dataset_parser.add_argument(
        "--fundamentals-input-path",
        default=None,
        help="Optional fundamentals feature CSV path. Defaults to the latest file in data/interim/fundamentals/.",
    )
    build_dataset_parser.set_defaults(handler=handle_build_dataset)

    train_parser = subparsers.add_parser("train", help="Run training commands.")
    train_subparsers = train_parser.add_subparsers(dest="train_command", required=True)
    train_run_parser = train_subparsers.add_parser("run", help="Train a model and save artifacts.")
    train_run_parser.add_argument(
        "--config",
        default="baseline_logreg",
        help="Train config name under configs/train/ without suffix.",
    )
    train_run_parser.add_argument(
        "--dataset-input-path",
        default=None,
        help="Optional dataset CSV path. Defaults to the latest file in data/processed/datasets/.",
    )
    train_run_parser.set_defaults(handler=handle_train_run)

    inference_parser = subparsers.add_parser("inference", help="Run inference commands.")
    inference_subparsers = inference_parser.add_subparsers(dest="inference_command", required=True)
    inference_predict_parser = inference_subparsers.add_parser(
        "predict", help="Run model inference and save predictions."
    )
    inference_predict_parser.add_argument(
        "--config",
        default="default",
        help="Inference config name under configs/inference/ without suffix.",
    )
    inference_predict_parser.add_argument(
        "--train-config",
        default="baseline_logreg",
        help="Train config name under configs/train/ without suffix.",
    )
    inference_predict_parser.add_argument(
        "--dataset-input-path",
        default=None,
        help="Optional dataset CSV path. Defaults to the latest file in data/processed/datasets/.",
    )
    inference_predict_parser.add_argument(
        "--model-input-path",
        default=None,
        help="Optional model artifact path. Defaults to the latest .joblib file in models/.",
    )
    inference_predict_parser.add_argument(
        "--prediction-date",
        default=None,
        help="Optional prediction date YYYY-MM-DD. Defaults to config value or latest date in dataset.",
    )
    inference_predict_parser.set_defaults(handler=handle_inference_predict)

    backtest_parser = subparsers.add_parser("backtest", help="Run backtest commands.")
    backtest_subparsers = backtest_parser.add_subparsers(dest="backtest_command", required=True)
    backtest_run_parser = backtest_subparsers.add_parser(
        "run", help="Run a model-driven backtest and save reports."
    )
    backtest_run_parser.add_argument(
        "--config",
        default="default",
        help="Backtest config name under configs/backtest/ without suffix.",
    )
    backtest_run_parser.add_argument(
        "--train-config",
        default="baseline_logreg",
        help="Train config name under configs/train/ without suffix.",
    )
    backtest_run_parser.add_argument(
        "--dataset-input-path",
        default=None,
        help="Optional dataset CSV path. Defaults to the latest file in data/processed/datasets/.",
    )
    backtest_run_parser.add_argument(
        "--model-input-path",
        default=None,
        help="Optional model artifact path. Defaults to the latest .joblib file in models/.",
    )
    backtest_run_parser.set_defaults(handler=handle_backtest_run)

    backtest_walk_parser = backtest_subparsers.add_parser(
        "walk-forward", help="Run expanding-window walk-forward evaluation."
    )
    backtest_walk_parser.add_argument(
        "--config",
        default="default",
        help="Backtest config name under configs/backtest/ without suffix.",
    )
    backtest_walk_parser.add_argument(
        "--train-config",
        default="baseline_logreg",
        help="Train config name under configs/train/ without suffix.",
    )
    backtest_walk_parser.add_argument(
        "--dataset-input-path",
        default=None,
        help="Optional dataset CSV path. Defaults to the latest file in data/processed/datasets/.",
    )
    backtest_walk_parser.set_defaults(handler=handle_backtest_walk_forward)

    report_parser = subparsers.add_parser("report", help="Generate reporting outputs.")
    report_subparsers = report_parser.add_subparsers(dest="report_command", required=True)
    report_compare_parser = report_subparsers.add_parser(
        "compare-models", help="Compare two model variants using train and walk-forward reports."
    )
    report_compare_parser.add_argument("--left-train-report-path", default=None)
    report_compare_parser.add_argument("--right-train-report-path", default=None)
    report_compare_parser.add_argument("--left-walk-forward-report-path", default=None)
    report_compare_parser.add_argument("--right-walk-forward-report-path", default=None)
    report_compare_parser.add_argument("--left-name", default="baseline_logreg")
    report_compare_parser.add_argument("--right-name", default="baseline_lightgbm")
    report_compare_parser.set_defaults(handler=handle_report_compare_models)
    report_eval_parser = report_subparsers.add_parser(
        "evaluate-prediction", help="Evaluate a saved prediction file against realized returns."
    )
    report_eval_parser.add_argument("--prediction-input-path", default=None)
    report_eval_parser.add_argument("--dataset-input-path", default=None)
    report_eval_parser.set_defaults(handler=handle_report_evaluate_prediction)

    return parser


def handle_config_list(args: argparse.Namespace) -> int:
    """Handle config list subcommand."""
    if args.section:
        files = list_config_files(args.section)
        for path in files:
            print(path.name)
        return 0

    configs_root = get_configs_root()
    sections = sorted(path.name for path in configs_root.iterdir() if path.is_dir())
    for section in sections:
        print(section)
    return 0


def handle_config_show(args: argparse.Namespace) -> int:
    """Handle config show subcommand."""
    config = load_config(args.section, args.name)
    print(json.dumps(config, indent=2, ensure_ascii=False))
    return 0


def handle_data_fetch(args: argparse.Namespace) -> int:
    """Handle data fetch subcommands."""
    result = run_fetch_command(args.dataset_kind)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def handle_data_build_universe(args: argparse.Namespace) -> int:
    """Handle liquidity-filtered universe build subcommand."""
    result = run_build_universe_command(input_path=args.input_path)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def handle_data_fetch_macro(args: argparse.Namespace) -> int:
    """Handle macro fetch subcommand."""
    result = run_fetch_macro_command(start_date=args.start_date, end_date=args.end_date)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def handle_data_fetch_prices(args: argparse.Namespace) -> int:
    """Handle price fetch subcommand."""
    result = run_fetch_prices_command(
        tickers=args.tickers,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def handle_data_fetch_fundamentals(args: argparse.Namespace) -> int:
    """Handle fundamentals fetch subcommand."""
    result = run_fetch_fundamentals_command(
        tickers=args.tickers,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def handle_data_normalize_prices(args: argparse.Namespace) -> int:
    """Handle price normalization subcommand."""
    result = run_normalize_prices_command(input_path=args.input_path)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def handle_data_normalize_fundamentals(args: argparse.Namespace) -> int:
    """Handle fundamentals normalization subcommand."""
    result = run_normalize_fundamentals_command(input_path=args.input_path)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def handle_data_normalize_macro(args: argparse.Namespace) -> int:
    """Handle macro normalization subcommand."""
    result = run_normalize_macro_command(input_path=args.input_path)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def handle_build_dataset(args: argparse.Namespace) -> int:
    """Handle feature dataset build subcommand."""
    result = run_build_dataset_command(
        feature_config_name=args.feature_config,
        label_config_name=args.label_config,
        price_input_path=args.price_input_path,
        label_input_path=args.label_input_path,
        macro_input_path=args.macro_input_path,
        fundamentals_input_path=args.fundamentals_input_path,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def handle_build_labels(args: argparse.Namespace) -> int:
    """Handle feature label build subcommand."""
    result = run_build_labels_command(
        input_path=args.input_path,
        label_config_name=args.label_config,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def handle_train_run(args: argparse.Namespace) -> int:
    """Handle train run subcommand."""
    result = run_train_command(args.config, dataset_input_path=args.dataset_input_path)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def handle_inference_predict(args: argparse.Namespace) -> int:
    """Handle inference predict subcommand."""
    result = run_predict_command(
        args.config,
        args.train_config,
        dataset_input_path=args.dataset_input_path,
        model_input_path=args.model_input_path,
        prediction_date=args.prediction_date,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def handle_backtest_run(args: argparse.Namespace) -> int:
    """Handle backtest run subcommand."""
    result = run_backtest_command(
        args.config,
        args.train_config,
        dataset_input_path=args.dataset_input_path,
        model_input_path=args.model_input_path,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def handle_backtest_walk_forward(args: argparse.Namespace) -> int:
    """Handle backtest walk-forward subcommand."""
    result = run_walk_forward_command(
        args.config,
        args.train_config,
        dataset_input_path=args.dataset_input_path,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def handle_report_compare_models(args: argparse.Namespace) -> int:
    """Handle report compare-models subcommand."""
    result = run_compare_models_command(
        left_train_report_path=args.left_train_report_path,
        right_train_report_path=args.right_train_report_path,
        left_walk_forward_report_path=args.left_walk_forward_report_path,
        right_walk_forward_report_path=args.right_walk_forward_report_path,
        left_name=args.left_name,
        right_name=args.right_name,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def handle_report_evaluate_prediction(args: argparse.Namespace) -> int:
    """Handle report evaluate-prediction subcommand."""
    result = run_evaluate_prediction_command(
        prediction_input_path=args.prediction_input_path,
        dataset_input_path=args.dataset_input_path,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the common CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return args.handler(args)
    except ConfigError as exc:
        parser.exit(status=1, message=f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
